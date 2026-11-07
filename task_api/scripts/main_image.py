"""依次检测 img01—img10，再为每张图提取稀疏的正向关系。"""

import argparse
import csv
import os
from pathlib import Path

import yaml
from detect import detect_image
from dotenv import load_dotenv
from openai import OpenAI
from relations import analyze_scene, load_config, save_relations, write_json
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        nargs="+",
        type=int,
        choices=range(1, 11),
        default=list(range(1, 11)),
        help="图片编号，默认 1—10",
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/images")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/image_main")
    parser.add_argument("--yolo-model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--model", help="覆盖 .env 中的 QWEN_MODEL")
    parser.add_argument("--config", type=Path, default=ROOT / "config/relations.yaml")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="例如 cpu 或 0")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120)
    return parser.parse_args()


def main():
    # 1. 一次加载配置、YOLO 和 API 客户端。
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    model = args.model or os.environ["QWEN_MODEL"]
    config = load_config(args.config.expanduser())
    detector = YOLO(str(args.yolo_model.expanduser()))
    input_dir = args.input_dir.expanduser().resolve()
    images = [next(input_dir.glob(f"img{number:02d}_*")) for number in args.images]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    summary = []
    with OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=args.timeout,
        max_retries=0,
    ) as client:
        for index, source in enumerate(images, start=1):
            print(f"[{index}/{len(images)}] {source.name}", flush=True)

            # 2. 检测物体，保存带编号的检测图和位置 JSON。
            image_output = output / source.stem
            scene = detect_image(
                detector,
                source,
                image_output,
                conf=args.conf,
                device=args.device,
                filenames=("original.png", "detected.png", "objects.json"),
            )
            objects = scene["objects"]

            # 3. 保存这张图的稀疏关系请求说明，再由核心一次完成几何和语义关系提取。
            write_json(
                image_output / "request.json",
                {
                    "mode": "sparse_positive",
                    "source": str(source),
                    "objects": objects,
                    "model": model,
                    "max_tokens": args.max_tokens,
                },
            )
            result = analyze_scene(
                client,
                model,
                scene,
                config,
                max_tokens=args.max_tokens,
            )
            save_relations(
                image_output,
                {
                    "source": str(source),
                    "objects": objects,
                    **result,
                },
            )
            summary.append(
                {
                    "image": source.name,
                    "object_count": len(objects),
                    "geometric_relation_count": result["geometric_relation_count"],
                    "semantic_relation_count": result["semantic_relation_count"],
                    "relation_count": result["relation_count"],
                    "api_call_count": result["api_call_count"],
                    "status": result["status"],
                }
            )
            with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image",
                        "object_count",
                        "geometric_relation_count",
                        "semantic_relation_count",
                        "relation_count",
                        "api_call_count",
                        "status",
                    ],
                )
                writer.writeheader()
                writer.writerows(summary)
            print(
                f"  {len(objects)} 个物体，{result['relation_count']} 项关系已保存",
                flush=True,
            )
    print(f"全部完成：{output}")


if __name__ == "__main__":
    main()
