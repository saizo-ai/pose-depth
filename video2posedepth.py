#!/usr/bin/env python3
"""Convert a video into a pose-skeleton + depth-map visualization.

For each frame:
  1. Estimate monocular depth and render it as grayscale (brighter = closer).
     Backends: per-frame Depth Anything V2 (default) or temporally consistent
     Video Depth Anything (--depth-backend video).
  2. Detect people's poses (YOLOv8-pose by default, up to --max-people) and
     draw OpenPose-style colored skeletons on top of the depth map.
  3. Output the depth+pose video on its own (default), or composite it with
     the original frame stacked on top or side by side.

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Pose skeleton rendering (OpenPose-style colors)
# ---------------------------------------------------------------------------

# (start_joint, end_joint, BGR color) over MediaPipe's 33 landmarks
MEDIAPIPE_SKELETON = [
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

# Same styling over the 17 COCO keypoints YOLO-pose predicts
COCO_SKELETON = [
    # Head
    (0, 1, (255, 0, 255)), (0, 2, (200, 0, 255)),
    (1, 3, (255, 0, 200)), (2, 4, (170, 0, 255)),
    # Torso
    (5, 6, (0, 0, 255)), (5, 11, (0, 80, 255)), (6, 12, (255, 80, 0)),
    (11, 12, (0, 160, 255)),
    # Left arm
    (5, 7, (0, 255, 255)), (7, 9, (0, 255, 170)),
    # Right arm
    (6, 8, (0, 165, 255)), (8, 10, (0, 255, 85)),
    # Left leg
    (11, 13, (255, 255, 0)), (13, 15, (255, 170, 0)),
    # Right leg
    (12, 14, (85, 255, 0)), (14, 16, (170, 255, 0)),
]


def draw_skeleton(image, points, skeleton):
    """Draw a colored skeleton in place.

    `points` is a list of (x, y) pixel tuples or None for low-confidence
    joints; `skeleton` is a list of (start, end, BGR color) bones.
    """
    h, w = image.shape[:2]
    scale = max(1, round(min(h, w) / 250))
    joints = sorted({i for a, b, _ in skeleton for i in (a, b)})

    for a, b, color in skeleton:
        if points[a] and points[b]:
            cv2.line(image, points[a], points[b], color, 2 * scale, cv2.LINE_AA)
    for i in joints:
        if points[i]:
            cv2.circle(image, points[i], 3 * scale, (0, 0, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

class VideoDepthEstimator:
    """Video Depth Anything: temporally consistent depth for a whole clip.

    Unlike the per-frame backend it needs every frame up front (the model
    attends across a sliding temporal window), so call `infer(frames)` once
    with the full list of BGR frames.
    """

    REPO_URL = "https://github.com/DepthAnything/Video-Depth-Anything"
    CKPT_URL = ("https://huggingface.co/depth-anything/Video-Depth-Anything-Small/"
                "resolve/main/video_depth_anything_vits.pth")

    def __init__(self, device):
        import subprocess as sp
        import urllib.request

        vendor = os.path.join(SCRIPT_DIR, "vendor", "Video-Depth-Anything")
        if not os.path.isdir(vendor):
            print("cloning Video-Depth-Anything ...")
            sp.run(["git", "clone", "--depth", "1", self.REPO_URL, vendor], check=True)
        ckpt = os.path.join(vendor, "checkpoints", "video_depth_anything_vits.pth")
        if not os.path.isfile(ckpt):
            print("downloading Video Depth Anything checkpoint (~116 MB) ...")
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            urllib.request.urlretrieve(self.CKPT_URL, ckpt)

        import torch
        sys.path.insert(0, vendor)
        from video_depth_anything.video_depth import VideoDepthAnything

        self.device = device
        self.model = VideoDepthAnything(encoder="vits", features=64,
                                        out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model = self.model.to(device).eval()

    def infer(self, frames_bgr, fps):
        """BGR frame list -> list of brighter-is-closer uint8 depth maps."""
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
        # MPS lacks some fp16 ops the model uses, so run fp32 off-CUDA.
        depths, _ = self.model.infer_video_depth(
            rgb, fps, input_size=518, device=self.device,
            fp32=(self.device != "cuda"))
        depths = np.asarray(depths)
        # One global mapping keeps brightness consistent across the video;
        # percentiles stop a single extreme close-up from crushing contrast.
        lo, hi = np.percentile(depths, [1, 99])
        span = max(hi - lo, 1e-6)
        return [(np.clip((d - lo) / span, 0, 1) * 255).astype(np.uint8)
                for d in depths]


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
# Pose detection backends
#
# Each backend is a callable: frame_bgr -> list of poses, where a pose is a
# list of (x, y) pixel tuples or None per joint. `.skeleton` gives the bones.
# ---------------------------------------------------------------------------


class YoloPose:
    """YOLOv8-pose: robust multi-person detection (17 COCO keypoints)."""

    skeleton = COCO_SKELETON

    def __init__(self, device, max_people, model_name="yolov8s-pose.pt"):
        from ultralytics import YOLO
        self.model = YOLO(os.path.join(SCRIPT_DIR, model_name))
        self.device = device
        self.max_people = max_people

    def __call__(self, frame_bgr):
        r = self.model(frame_bgr, verbose=False, device=self.device)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        order = r.boxes.conf.argsort(descending=True)
        poses = []
        for i in order[:self.max_people]:
            if float(r.boxes.conf[i]) < 0.5:
                break
            xy = r.keypoints.xy[i].cpu().numpy()
            conf = (r.keypoints.conf[i].cpu().numpy()
                    if r.keypoints.conf is not None else np.ones(len(xy)))
            poses.append([
                (int(x), int(y)) if c >= 0.5 and (x, y) != (0, 0) else None
                for (x, y), c in zip(xy, conf)])
        return poses

    def close(self):
        pass


class MediaPipePose:
    """MediaPipe PoseLandmarker: denser landmarks, best for a single person."""

    skeleton = MEDIAPIPE_SKELETON

    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

    def __init__(self, fps, max_people):
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = os.path.join(SCRIPT_DIR, "pose_landmarker_full.task")
        if not os.path.isfile(model_path):
            print("downloading pose model ...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)

        self.mp = mp
        self.fps = fps
        self.n = 0
        self.landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=max_people,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5))

    def __call__(self, frame_bgr):
        mp = self.mp
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(
            image, int(self.n * 1000 / self.fps))
        self.n += 1
        return [[(int(lm.x * w), int(lm.y * h)) if lm.visibility >= 0.5 else None
                 for lm in person]
                for person in result.pose_landmarks]

    def close(self):
        self.landmarker.close()


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
    ap.add_argument("--layout", choices=["depth-only", "stacked", "side-by-side"],
                    default="depth-only",
                    help="depth-only = just the depth+pose video (default); "
                         "stacked/side-by-side composite the original with it")
    ap.add_argument("--depth-backend", choices=["image", "video"], default="image",
                    help="image = per-frame Depth Anything V2 (default, streams); "
                         "video = Video Depth Anything, temporally consistent but "
                         "loads the whole clip into memory")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="HF depth-estimation model id (image backend only)")
    ap.add_argument("--no-pose", action="store_true", help="skip the pose skeleton")
    ap.add_argument("--pose-backend", choices=["yolo", "mediapipe"], default="yolo",
                    help="yolo = robust multi-person (default); "
                         "mediapipe = denser landmarks, single person")
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

    def frame_depth_pairs():
        """Yield (frame_bgr, depth_gray) for every frame of the input."""
        if args.depth_backend == "video":
            print(f"loading Video Depth Anything on {device} ...")
            est = VideoDepthEstimator(device)
            frames = []
            while not (args.max_frames and len(frames) >= args.max_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            print(f"estimating depth for {len(frames)} frames ...")
            yield from zip(frames, est.infer(frames, fps))
        else:
            print(f"loading depth model ({args.depth_model}) on {device} ...")
            est = DepthEstimator(args.depth_model, device)
            read = 0
            while not (args.max_frames and read >= args.max_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                read += 1
                yield frame, est(frame)

    pose = None
    if not args.no_pose:
        if args.pose_backend == "yolo":
            pose = YoloPose(device, args.max_people)
        else:
            pose = MediaPipePose(fps, args.max_people)

    writer = None
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    n = 0
    try:
        for frame, depth_gray in frame_depth_pairs():
            depth_vis = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

            if pose is not None:
                for person in pose(frame):
                    draw_skeleton(depth_vis, person, pose.skeleton)

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
