# pose-depth

Convert a video into a **pose skeleton + depth map** visualization: for every
frame it estimates monocular depth (grayscale, brighter = closer) and overlays
OpenPose-style colored skeletons for the people in frame, then stacks the
original frame above the result.

Useful for driving video-generation models (e.g. Seedance) with structural
conditioning, or for motion/staging reference.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The pose model (`pose_landmarker_full.task`, ~9 MB) and the depth model
(Depth Anything V2 Small, ~100 MB from Hugging Face) are downloaded
automatically on first run.

## Usage

```sh
.venv/bin/python video2posedepth.py input.mp4
```

Writes `input_posedepth.mp4` next to the input. Options:

| Flag | Description |
| --- | --- |
| `-o OUT.mp4` | output path |
| `--layout stacked\|side-by-side\|depth-only` | `stacked` (default) puts the original on top like the reference layout |
| `--depth-model ID` | any HF depth-estimation model (default `depth-anything/Depth-Anything-V2-Small-hf`; use `...-Large-hf` for higher quality) |
| `--no-pose` | depth map only, skip the skeleton |
| `--max-people N` | max number of people to skeleton (default 4) |
| `--max-frames N` | process only the first N frames (quick preview) |

## Notes

- Runs on Apple Silicon GPU (MPS) or CUDA automatically; falls back to CPU.
  ~7–8 fps on an M-series Mac for a 496×864 clip.
- Depth normalization uses a temporally smoothed min/max so brightness doesn't
  flicker between frames.
- Skeletons are drawn for up to `--max-people` people (default 4); people
  who are mostly out of frame are skipped automatically.
- Output is H.264/yuv420p, playable everywhere.

## License

[MIT](LICENSE)
