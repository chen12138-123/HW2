import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract consecutive frames from a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted frames.")
    parser.add_argument("--start-second", type=float, default=None, help="Start time in seconds.")
    parser.add_argument("--start-frame", type=int, default=None, help="Start frame index.")
    parser.add_argument("--num-frames", type=int, default=4, help="Number of consecutive frames.")
    parser.add_argument("--prefix", default="frame", help="Output filename prefix.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = args.start_frame
    if start_frame is None:
        start_frame = int(args.start_second * fps)

    for i in range(args.num_frames):
        frame_id = start_frame + i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if ok:
            out = output_dir / f"{args.prefix}_{frame_id:06d}.jpg"
            cv2.imwrite(str(out), frame)
            print(out)

    cap.release()


if __name__ == "__main__":
    main()
