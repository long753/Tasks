"""读取 detect.py 输出，用局部几何和一次视觉 API 提取稀疏正向关系。"""

import argparse
import base64
import csv
import json
import mimetypes
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter

import yaml

GEOMETRIC_PREDICATES = {"left_of", "right_of"}


def load_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "relations.yaml",
        help="关系定义与提示词 YAML；默认使用项目 config/relations.yaml",
    )
    parser.add_argument("--detections", type=Path, required=True, help="detect.py 生成的 *_objects.json")
    parser.add_argument("--relations", nargs="+", help="要提取的关系名称；省略时使用配置中的全部关系")
    parser.add_argument("--output", type=Path, default=Path("outputs/relations"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=None, help="覆盖 .env 中的 QWEN_MODEL")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true", help="只生成配置快照、提示词和请求记录，不调用 API")
    return parser.parse_args()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _semantic_relation_types(config):
    return [name for name in config["relations"] if name not in GEOMETRIC_PREDICATES]


def geometric_relations(objects, relation_types):
    """每个无序对象对只保存一条成立的横向关系，不重复输出其逆关系。"""
    if not GEOMETRIC_PREDICATES.intersection(relation_types):
        return []
    predicate = "left_of" if "left_of" in relation_types else "right_of"
    relations = []
    for first, second in combinations(objects, 2):
        first_x = (first["bbox_xyxy"][0] + first["bbox_xyxy"][2]) / 2
        second_x = (second["bbox_xyxy"][0] + second["bbox_xyxy"][2]) / 2
        if first_x == second_x:
            continue
        left, right = (first, second) if first_x < second_x else (second, first)
        subject, obj = (left, right) if predicate == "left_of" else (right, left)
        relations.append(
            {
                "subject_id": subject["object_id"],
                "predicate": predicate,
                "object_id": obj["object_id"],
                "source": "geometry",
                "reason": "主体框中心在客体框中心左侧" if predicate == "left_of" else "主体框中心在客体框中心右侧",
            }
        )
    return relations


def build_prompt(objects, config):
    definitions = {name: config["relations"][name] for name in _semantic_relation_types(config)}
    return "\n".join(
        [
            config["prompt"]["task"].strip(),
            config["prompt"]["guidelines"].strip(),
            "只提取有明确画面证据的语义关系；不确定或不成立的关系省略，不必遍历所有对象对。",
            "按 subject_id → object_id 判断方向；编号必须使用当前对象的整数 object_id，不能写类别名。",
            "不要返回左右关系、status 或重复的关系。",
            '只输出 JSON 对象，顶层 relations 为列表；没有明确关系时输出 {"relations": []}。',
            "每项只包含 subject_id（整数）、predicate（关系名称）、object_id（整数）、reason（简短中文证据）。",
            "语义关系定义：" + json.dumps(definitions, ensure_ascii=False),
            "当前对象：" + json.dumps(objects, ensure_ascii=False),
        ]
    )


def image_data_url(path):
    mime = mimetypes.guess_type(path.name)[0]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def validate_response(text, objects, relation_types):
    """只检查模型输出的有效性，不要求对未提取的关系作答。"""
    candidates = json.loads(text)["relations"]
    if not isinstance(candidates, list):
        raise TypeError("relations 必须是稀疏关系列表，不能返回 q1 等查询编号映射")
    object_ids = {obj["object_id"] for obj in objects}
    allowed = set(relation_types) - GEOMETRIC_PREDICATES
    relations = []
    seen = set()
    for item in candidates:
        if "status" in item:
            raise ValueError("只提取成立关系，不接受旧协议的 status 字段")
        subject_id, object_id = item["subject_id"], item["object_id"]
        if (
            type(subject_id) is not int
            or type(object_id) is not int
            or subject_id not in object_ids
            or object_id not in object_ids
            or subject_id == object_id
        ):
            raise ValueError(f"无效关系对象对 {subject_id!r} → {object_id!r}；当前 ID：{sorted(object_ids)}")
        predicate = item["predicate"]
        if predicate not in allowed:
            raise ValueError(f"无效语义关系 {predicate!r}；允许：{sorted(allowed)}")
        reason = item["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("关系 reason 必须包含非空的可见证据")
        triple = (subject_id, predicate, object_id)
        if triple in seen:
            raise ValueError(f"重复关系：{triple}")
        seen.add(triple)
        relations.append(
            {
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "source": "qwen",
                "reason": reason.strip(),
            }
        )
    return relations


def request_relations(client, model, objects, config, images, max_tokens=4096):
    """用原图和编号图进行一次稀疏语义关系提取。"""
    prompt = build_prompt(objects, config)
    image_content = [{"type": "image_url", "image_url": {"url": image_data_url(Path(image))}} for image in images]
    start = perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_content,
                ],
            }
        ],
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
        temperature=0,
        max_tokens=max_tokens,
    )
    elapsed_ms = round((perf_counter() - start) * 1000, 2)
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError(f"响应未正常完成：{choice.finish_reason}")

    relations = validate_response(
        choice.message.content,
        objects,
        _semantic_relation_types(config),
    )
    return {
        "model_requested": model,
        "model_returned": response.model,
        "elapsed_ms": elapsed_ms,
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
        "relations": relations,
    }


def analyze_scene(client, model, scene, config, max_tokens=4096):
    """合并局部几何和一次可选的语义 API 调用，分析完整场景。"""
    objects = scene["objects"]
    relation_types = list(config["relations"])
    geometric = geometric_relations(objects, relation_types)
    semantic_types = _semantic_relation_types(config)
    should_request = len(objects) >= 2 and bool(semantic_types)

    if should_request:
        semantic_result = request_relations(
            client,
            model,
            objects,
            config,
            [scene["original_image"], scene["annotated_image"]],
            max_tokens=max_tokens,
        )
        semantic = semantic_result["relations"]
    else:
        semantic_result = {
            "model_requested": model,
            "model_returned": None,
            "elapsed_ms": 0,
            "usage": None,
        }
        semantic = []

    relations = geometric + semantic
    return {
        "status": "ok" if len(objects) >= 2 else "no_pairs",
        "mode": "sparse_positive",
        "api_call_count": int(should_request),
        "geometric_relation_count": len(geometric),
        "semantic_relation_count": len(semantic),
        "relation_count": len(relations),
        "model_requested": semantic_result["model_requested"],
        "model_returned": semantic_result["model_returned"],
        "elapsed_ms": semantic_result["elapsed_ms"],
        "usage": semantic_result["usage"],
        "relations": relations,
    }


def save_relations(output, data):
    write_json(output / "relations.json", data)
    with (output / "relations.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_id", "predicate", "object_id", "source", "reason"],
        )
        writer.writeheader()
        writer.writerows(data["relations"])


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    selected_types = args.relations or list(config["relations"])
    config = {**config, "relations": {name: config["relations"][name] for name in selected_types}}

    if not args.dry_run:
        from dotenv import load_dotenv

        load_dotenv(args.env_file.expanduser(), override=False)
    model = args.model or os.getenv("QWEN_MODEL")
    base_url = os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    detections = args.detections.expanduser().resolve()
    scene = json.loads(detections.read_text(encoding="utf-8"))
    for field in ("original_image", "annotated_image"):
        image = Path(scene[field]).expanduser()
        scene[field] = image if image.is_absolute() else detections.parent / image
    objects = scene["objects"]
    prompt = build_prompt(objects, config)

    run_name = detections.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output.expanduser().resolve() / run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_json(
        output / "request.json",
        {
            "detections_json": str(detections),
            "config_path": str(config_path),
            "original_image": str(scene["original_image"]),
            "annotated_image": str(scene["annotated_image"]),
            "objects": objects,
            "relation_types": selected_types,
            "mode": "sparse_positive",
            "model": model,
            "base_url": base_url,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "dry_run": args.dry_run,
        },
    )
    if args.dry_run:
        print(f"已保存配置快照、prompt.txt 和 request.json；结果目录：{output}")
        return

    if len(objects) >= 2 and _semantic_relation_types(config):
        api_key = os.environ["DASHSCOPE_API_KEY"]
        from openai import OpenAI

        with OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout, max_retries=0) as client:
            result = analyze_scene(client, model, scene, config, max_tokens=args.max_tokens)
    else:
        result = analyze_scene(None, model, scene, config, max_tokens=args.max_tokens)

    save_relations(
        output,
        {
            "detections_json": str(detections),
            "objects": objects,
            **result,
        },
    )
    print(json.dumps(result["relations"], ensure_ascii=False, indent=2))
    print("已保存关系 JSON / CSV、提示词和调用耗时。")


if __name__ == "__main__":
    main()
