#!/usr/bin/env python3
"""Convert a video into a pose-skeleton + depth-map visualization.

For each frame:
  1. Estimate monocular depth (Depth Anything V2) and render it as grayscale
     (brighter = closer).
  2. Detect people's poses (MediaPipe, up to --max-people) and draw
     OpenPose-style colored skeletons on top of the depth map.
  3. Optionally stack the original frame above the result (like the reference
     layout) or place them side by side.

Usage:
  python video2posedepth.py input.mp4
  python video2posedepth.py input.mp4 -o out.mp4 --layout depth-only
"""

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pose skeleton rendering (OpenPose-style colors on MediaPipe's 33 landmarks)
# ---------------------------------------------------------------------------

# (start_landmark, end_landmark, BGR color)
SKELETON = [
    # Head
    (0, 2, (255, 0, 255)), (0, 5, (200, 0, 255)),
    (2, 7, (255, 0, 200)), (5, 8, (170, 0, 255)),
    # Torso
    (11, 12, (0, 0, 255)), (11, 23, (0, 80, 255)), (12, 24, (255, 80, 0)),
    (23, 24, (0, 160, 255)),
    # Left arm (viewer right)
    (11, 13, (0, 255, 255)), (13, 15, (0, 255, 170)),
    # Right arm
    (12, 14, (0, 165, 255)), (14, 16, (0, 255, 85)),
    # Left leg
    (23, 25, (255, 255, 0)), (25, 27, (255, 170, 0)), (27, 31, (255, 85, 0)),
    # Right leg
    (24, 26, (85, 255, 0)), (26, 28, (170, 255, 0)), (28, 32, (255, 0, 85)),
]

JOINTS = sorted({i for a, b, _ in SKELETON for i in (a, b)})


def draw_skeleton(image, landmarks, visibility_thresh=0.5):
    """Draw the colored skeleton in place. `landmarks` is the MediaPipe list."""
    h, w = image.shape[:2]
    scale = max(1, round(min(h, w) / 250))

    def pt(i):
        lm = landmarks[i]
        if lm.visibility < visibility_thresh:
            return None
        return int(lm.x * w), int(lm.y * h)

    for a, b, color in SKELETON:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(image, pa, pb, color, 2 * scale, cv2.LINE_AA)
    for i in JOINTS:
        p = pt(i)
        if p:
            cv2.circle(image, p, 3 * scale, (0, 0, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

class DepthEstimator:
    """Depth Anything V2 via transformers; returns brighter-is-closer uint8."""

    def __init__(self, model_id, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        # Running min/max (EMA) so brightness doesn't flicker frame to frame.
        self._lo = None
        self._hi = None

    def __call__(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            depth = self.model(**inputs).predicted_depth  # inverse depth: high = close
        depth = self.torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=frame_bgr.shape[:2],
            mode="bicubic", align_corners=False,
        )[0, 0].float().cpu().numpy()

        lo, hi = float(depth.min()), float(depth.max())
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            alpha = 0.1
            self._lo = (1 - alpha) * self._lo + alpha * lo
            self._hi = (1 - alpha) * self._hi + alpha * hi
        span = max(self._hi - self._lo, 1e-6)
        gray = np.clip((depth - self._lo) / span, 0, 1)
        return (gray * 255).astype(np.uint8)


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Pose detection (MediaPipe Tasks API)
# ---------------------------------------------------------------------------

POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_full/float16/latest/pose_landmarker_full.task")


def make_pose_landmarker(max_people):
    import urllib.request
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "pose_landmarker_full.task")
    if not os.path.isfile(model_path):
        print("downloading pose model ...")
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=max_people,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    return mp, vision.PoseLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compose(original, depth_vis, layout):
    if layout == "depth-only":
        return depth_vis
    if layout == "side-by-side":
        return np.hstack([original, depth_vis])
    return np.vstack([original, depth_vis])  # stacked


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="input video file")
    ap.add_argument("-o", "--output", help="output video (default: <input>_posedepth.mp4)")
    ap.add_argument("--layout", choices=["stacked", "side-by-side", "depth-only"],
                    default="stacked",
                    help="stacked = original on top, depth+pose below (default)")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="HF depth-estimation model id")
    ap.add_argument("--no-pose", action="store_true", help="skip the pose skeleton")
    ap.add_argument("--max-people", type=int, default=4,
                    help="max number of people to skeleton (default 4)")
    ap.add_argument("--max-frames", type=int, default=0, help="limit frames (0 = all)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"error: input not found: {args.input}")
    output = args.output or os.path.splitext(args.input)[0] + "_posedepth.mp4"
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"error: cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)

    device = pick_device()
    print(f"loading depth model ({args.depth_model}) on {device} ...")
    depth_est = DepthEstimator(args.depth_model, device)

    mp = pose = None
    if not args.no_pose:
        mp, pose = make_pose_landmarker(args.max_people)

    writer = None
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and n >= args.max_frames):
                break

            depth_gray = depth_est(frame)
            depth_vis = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

            if pose is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = pose.detect_for_video(mp_image, int(n * 1000 / fps))
                for person in result.pose_landmarks:
                    draw_skeleton(depth_vis, person)

            out_frame = compose(frame, depth_vis, args.layout)
            if writer is None:
                h, w = out_frame.shape[:2]
                writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
            writer.write(out_frame)
            n += 1
            if n % 24 == 0 or n == total:
                print(f"\r{n}/{total} frames", end="", flush=True)
        print()
    finally:
        cap.release()
        if writer:
            writer.release()
        if pose:
            pose.close()

    if n == 0:
        sys.exit("error: no frames processed")

    # Re-encode to H.264 for broad playback compatibility.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_out,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output],
        check=True)
    os.unlink(tmp_out)
    print(f"wrote {output} ({n} frames)")


if __name__ == "__main__":
    main()
