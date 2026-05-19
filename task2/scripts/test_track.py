import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO video multi-object tracking with configurable parameters."
    )
    parser.add_argument(
        "--model",
        default="/mnt/disk/maoqijun/runs/detect/MLLM/yolov8s_e80_img640_b32_lr001/weights/best.pt",
        help="Path to trained detector weights, usually weights/best.pt.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to input video, image folder, stream URL, or camera index.",
    )
    parser.add_argument(
        "--tracker",
        default="botsort",
        choices=["botsort", "bytetrack", "botsort.yaml", "bytetrack.yaml"],
        help="Tracker used by Ultralytics.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device id, for example 0, 0,1, cpu.",
    )
    parser.add_argument(
        "--project",
        default="/mnt/disk/maoqijun/runs/yolov8s_track",
        help="Directory for tracking outputs.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Experiment name. Defaults to input filename plus tracker name.",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated output video.",
    )
    parser.add_argument(
        "--save-txt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save tracking results as label txt files.",
    )
    parser.add_argument(
        "--save-conf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save confidence scores when --save-txt is enabled.",
    )
    parser.add_argument(
        "--show-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw class labels on output video.",
    )
    parser.add_argument(
        "--show-conf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw confidence scores on output video.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting/reusing an existing output directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output video with tracking and line-crossing count.",
    )
    parser.add_argument(
        "--line-orientation",
        default="vertical",
        choices=["vertical", "horizontal"],
        help="Counting line orientation.",
    )
    parser.add_argument(
        "--line-ratio",
        type=float,
        default=0.5,
        help="Line position ratio on x or y axis.",
    )
    parser.add_argument(
        "--direction",
        default="positive",
        choices=["positive", "negative"],
        help="positive: left-to-right or top-to-bottom; negative: reverse direction.",
    )
    return parser.parse_args()


def default_run_name(source: str, tracker: str) -> str:
    return f"{Path(source).stem}_{Path(tracker).stem}"


def tracker_config(tracker: str) -> str:
    return tracker if tracker.endswith(".yaml") else f"{tracker}.yaml"


def main() -> None:
    args = parse_args()
    tracker = tracker_config(args.tracker)
    run_name = args.name or default_run_name(args.source, tracker)
    output = args.output or str(Path(args.project) / run_name / f"{Path(args.source).stem}_count.mp4")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    writer = None
    prev_pos = {}
    counted_ids = set()
    count = 0

    model = YOLO(args.model)
    results = model.track(
        source=args.source,
        tracker=tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        project=args.project,
        name=run_name,
        exist_ok=args.exist_ok,
        verbose=True,
        stream=True,
    )

    for result in results:
        frame = result.plot()
        height, width = frame.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        if args.line_orientation == "vertical":
            line_pos = int(width * args.line_ratio)
            cv2.line(frame, (line_pos, 0), (line_pos, height), (0, 255, 255), 3)
        else:
            line_pos = int(height * args.line_ratio)
            cv2.line(frame, (0, line_pos), (width, line_pos), (0, 255, 255), 3)

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                pos = cx if args.line_orientation == "vertical" else cy

                if track_id in prev_pos and track_id not in counted_ids:
                    if args.direction == "positive":
                        crossed = prev_pos[track_id] <= line_pos and pos > line_pos
                    else:
                        crossed = prev_pos[track_id] >= line_pos and pos < line_pos
                    if crossed:
                        count += 1
                        counted_ids.add(track_id)

                prev_pos[track_id] = pos

        cv2.putText(frame, f"Count: {count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        writer.write(frame)

    writer.release()
    print(f"Output: {output}")
    print(f"Count: {count}")



if __name__ == "__main__":
    main()
