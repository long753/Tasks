"""逐帧 YOLO + ByteTrack 跟踪，按视频时间采样后调用 Qwen 判断关系。"""

import argparse
import json
import os
from math import floor, isfinite
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import cv2
import yaml
from detect import draw_object
from dotenv import load_dotenv
from openai import OpenAI
from relations import analyze_scene, load_config, save_relations, write_json
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


class TrackHistory(TypedDict):
    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    frames_seen: int


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(ROOT / "data/videos/video01_book_place.mp4"),
        help="视频路径，或摄像头编号，例如 0",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo26n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/video_main")
    parser.add_argument("--relations-only", type=Path, help="仅重分析已有视频输出目录中的采样快照，不运行跟踪")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="例如 cpu 或 0")
    parser.add_argument("--show", action="store_true", help="显示实时画面，按 q 退出")
    parser.add_argument("--max-frames", type=int, help="只处理前 N 帧，省略则处理到结束")
    parser.add_argument("--fps", type=float, help="输出视频帧率；默认使用输入帧率")
    parser.add_argument("--relation-interval", type=float, default=1.0, help="关系采样间隔，单位秒")
    parser.add_argument("--relation-model", help="覆盖 .env 中的 QWEN_MODEL；--model 仍是 YOLO 权重")
    parser.add_argument("--config", type=Path, default=ROOT / "config/relations.yaml")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames 必须大于 0")
    if not isfinite(args.relation_interval) or args.relation_interval <= 0:
        parser.error("--relation-interval 必须为有限正数")
    return args


def track_video(args: argparse.Namespace):
    # 第一阶段：逐帧跟踪，采样帧只保存图像和对象信息，不请求 API。
    model = YOLO(str(args.model.expanduser()))
    is_camera = args.source.isdecimal()
    if is_camera:
        source = int(args.source)
        name = f"camera_{source}"
    else:
        source = str(Path(args.source).expanduser().resolve())
        name = Path(source).stem
    output = args.output.expanduser().resolve() / name
    cap = cv2.VideoCapture(source)
    writer = None
    frame_count = 0
    track_history: dict[int, TrackHistory] = {}
    sample_dirs: list[Path] = []
    next_sample_index = 0
    started = perf_counter()

    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源：{source}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = args.fps if args.fps is not None else source_fps
        if fps <= 0:
            raise ValueError("无法获得有效帧率，请用 --fps 指定输出帧率")
        output.mkdir(parents=True, exist_ok=True)
        with (output / "tracks.jsonl").open("w", encoding="utf-8") as tracks:
            while args.max_frames is None or frame_count < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = perf_counter() - started if is_camera else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                sample_time = next_sample_index * args.relation_interval
                sample_due = timestamp >= sample_time
                original = frame.copy() if sample_due else frame
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output / "tracked.mp4"),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("无法创建输出视频")

                # 2. persist=True 保留前一帧的轨迹，由 ByteTrack 分配 ID。
                results = model.track(
                    frame,
                    tracker="bytetrack.yaml",
                    persist=True,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    save=False,
                    verbose=False,
                )
                result = next(iter(results))
                frame_count += 1
                objects = []
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for box in boxes.cpu():
                        track_id = int(box.id.item())
                        class_id = int(box.cls.item())
                        obj = {
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": result.names[class_id],
                            "confidence": float(box.conf.item()),
                            "bbox_xyxy": box.xyxy[0].tolist(),
                        }
                        objects.append(obj)

                        # 3. 复用图片绘图函数，但编号来自轨迹，不是本帧检测顺序。
                        draw_object(frame, {**obj, "object_id": track_id})
                        if track_id not in track_history:
                            track_history[track_id] = {
                                "track_id": track_id,
                                "class_name": obj["class_name"],
                                "first_frame": frame_count,
                                "last_frame": frame_count,
                                "frames_seen": 0,
                            }
                        history = track_history[track_id]
                        history["last_frame"] = frame_count
                        history["frames_seen"] += 1

                if sample_due:
                    sample_dir = output / "samples" / f"sample_{len(sample_dirs):03d}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    original_path = sample_dir / "original.png"
                    annotated_path = sample_dir / "detected.png"
                    for path, image in ((original_path, original), (annotated_path, frame)):
                        if not cv2.imwrite(str(path), image):
                            raise OSError(f"图片保存失败：{path}")
                    write_json(
                        sample_dir / "objects.json",
                        {
                            "source": source,
                            "sample_index": len(sample_dirs),
                            "scheduled_timestamp_seconds": sample_time,
                            "timestamp_seconds": timestamp,
                            "frame_index": frame_count,
                            "image_width": width,
                            "image_height": height,
                            "original_image": str(original_path),
                            "annotated_image": str(annotated_path),
                            "model": str(model.ckpt_path),
                            "objects": [{**obj, "object_id": obj["track_id"]} for obj in objects],
                        },
                    )
                    sample_dirs.append(sample_dir)
                    # 跨过多个采样时刻时，只保存当前帧，不用它补造历史快照。
                    next_sample_index = max(next_sample_index + 1, floor(timestamp / args.relation_interval) + 1)

                # 4. 每帧立即写入，未确认轨迹或没有对象时保存空列表。
                tracks.write(
                    json.dumps(
                        {
                            "frame_index": frame_count,
                            "timestamp_seconds": timestamp,
                            "objects": objects,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                writer.write(frame)
                if frame_count % 25 == 0:
                    speed = frame_count / (perf_counter() - started)
                    print(
                        f"已处理 {frame_count} 帧，当前 {len(objects)} 个对象，平均 {speed:.1f} FPS",
                        flush=True,
                    )
                if args.show:
                    cv2.imshow("YOLO + ByteTrack", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frame_count == 0:
        raise RuntimeError("未读取到任何视频帧")

    # 5. 汇总实际处理速度和各条轨迹；轨迹数量不等于真实物体数量。
    elapsed = perf_counter() - started
    summary = {
        "source": source,
        "source_kind": "camera" if is_camera else "video",
        "model": str(model.ckpt_path),
        "tracker": "bytetrack.yaml",
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "source_fps": source_fps,
        "output_fps": fps,
        "output_duration_seconds": frame_count / fps,
        "elapsed_seconds": elapsed,
        "processing_fps": frame_count / elapsed,
        "track_count": len(track_history),
        "tracks": list(track_history.values()),
        "relation_interval_seconds": args.relation_interval,
        "sample_count": len(sample_dirs),
        "samples": [str(path.relative_to(output)) for path in sample_dirs],
    }
    write_json(output / "summary.json", summary)
    print(
        f"跟踪完成：{frame_count} 帧，{len(track_history)} 条轨迹，"
        f"{len(sample_dirs)} 个采样点，平均 {summary['processing_fps']:.1f} FPS"
    )
    return output, sample_dirs, summary


def analyze_samples(output: Path, sample_dirs: list[Path], args: argparse.Namespace):
    # 第二阶段：只处理本次采集的快照，按时间顺序查询当前对象之间的关系。
    load_dotenv(ROOT / ".env", override=False)
    relation_model = args.relation_model or os.environ["QWEN_MODEL"]
    config = load_config(args.config.expanduser())
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    api_call_count = 0
    geometric_count = 0
    semantic_count = 0
    print(f"开始稀疏关系分析：{relation_model}，共 {len(sample_dirs)} 个采样点", flush=True)
    with OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=args.timeout,
        max_retries=0,
    ) as client:
        for sample_dir in sample_dirs:
            scene = json.loads((sample_dir / "objects.json").read_text(encoding="utf-8"))
            result = analyze_scene(client, relation_model, scene, config, max_tokens=args.max_tokens)
            save_relations(sample_dir, {**scene, **result})
            api_call_count += result["api_call_count"]
            geometric_count += result["geometric_relation_count"]
            semantic_count += result["semantic_relation_count"]
            print(
                f"  {scene['timestamp_seconds']:.3f}s，第 {scene['frame_index']} 帧："
                f"{len(scene['objects'])} 个对象，{result['geometric_relation_count']} 条几何关系，"
                f"{result['semantic_relation_count']} 条语义关系",
                flush=True,
            )
    return {
        "relation_model": relation_model,
        "relation_sample_count": len(sample_dirs),
        "relation_mode": "sparse_positive",
        "relation_api_call_count": api_call_count,
        "geometric_relation_count": geometric_count,
        "semantic_relation_count": semantic_count,
        "relation_count": geometric_count + semantic_count,
    }


def main():
    args = parse_args()
    if args.relations_only is not None:
        output = args.relations_only.expanduser().resolve()
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        sample_dirs = [output / path for path in summary["samples"]]
        print(f"复用已有 {len(sample_dirs)} 个快照，不加载 YOLO，不改写跟踪视频", flush=True)
    else:
        output, sample_dirs, summary = track_video(args)
    for field in ("relation_query_count", "relation_batch_count"):
        summary.pop(field, None)
    summary.update(analyze_samples(output, sample_dirs, args))
    write_json(output / "summary.json", summary)
    print(f"全部完成：{output}")


if __name__ == "__main__":
    main()
