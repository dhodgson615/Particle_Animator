# ParticleAnimatorRust

A high-performance particle simulator and renderer written in Rust. The program simulates particles moving inside a parametric (super-ellipse style) boundary, accumulates a 2D histogram of particle density per frame, rasterizes frames to PNG images and streams them into `ffmpeg` (image2pipe) to encode MP4 video. Runs are resumable and metadata for each run is stored in `mp4/<index>/meta.json`.

## Overview

- Simulates many particles inside a parametric boundary and reflects particles at the implicit boundary.
- Per-frame particle positions are accumulated into a 2D histogram (density) which is mapped to a spectral palette and rasterized.
- PNG frames are written to ffmpeg stdin (image2pipe) on worker threads, and ffmpeg encodes the MP4.
- Resume capability: each run writes `mp4/<index>/meta.json` and can resume from `last_frame`.
- Heavy parallelization: per-task Rayon ThreadPools, per-thread buffers, and performance-oriented crates (ahash, parking_lot).

## Build (prerequisites)

- Rust (stable) toolchain — install via https://rustup.rs.
- `ffmpeg` on PATH for encoding frames into MP4 (`ffmpeg -version` to verify).
- Build release binary for best performance:
  cargo build --release

## Run / Usage

- On start the program prompts for a video index:
  - Press Enter to use the next available index (creates `mp4/<index>/frames` and `mp4/<index>.mp4`).
  - Or type a numeric index to use or resume an existing directory.
- Example invocations:
  - cargo run --release --
  - Quick test (fewer particles, lower res):
    cargo run --release -- --n-particles 2000 --res 128 --duration-s 4 --fps 24
  - Full quality render:
    cargo run --release -- --n-particles 20000 --res 512 --duration-s 20 --fps 60
- The program streams PNG frames into ffmpeg via stdin; the final MP4 is written to `mp4/<index>.mp4`.

### CLI flags (defaults shown)

- --a <f32> (default: 2.5)
- --b <f32> (default: 1.0)
- --n-exp <f32> (default: 4.0)
- --m-exp <f32> (default: 3.0)
- --n-particles <u64> (default: 10000)
- --dt <f32> (default: 0.001)
- --epsilon <f32> (default: 1e-8)
- --center-x <f32> (default: 0.75)
- --center-y <f32> (default: 0.25)
- --radius <f32> (default: 0.5)
- --vx0 <f32> (default: 3.0)
- --vy0 <f32> (default: 0.0)
- --fps <u64> (default: 60)
- --duration-s <u64> (default: 10)
- --steps-per-frame <u64> (default: 30)
- --res <u32> (default: 256) — histogram bins per dimension
- --dpi <u32> (default: 300) — output DPI; output image size (px) = FIG_INCHES * DPI (FIG_INCHES is 8.0 by default)
- --video-filename <string> (optional) — stored in meta.json but the binary saves to `mp4/<index>.mp4` by default

### Resume behavior and output layout

- Metadata is stored at: `mp4/<index>/meta.json`. Important fields:
  - `constants` — run parameters (Config constants).
  - `date` — run start time.
  - `last_frame` — last completed frame index (used for resuming).
  - `compute_time` — cumulative compute time (seconds).
  - `resolution` — histogram resolution (`res`).
- When resuming, the simulation state is advanced by `last_frame * steps_per_frame` steps before generating new frames.
- Output layout:
  - mp4/
    - <index>/
      - frames/ — optional saved PNG frames (useful for debugging)
      - meta.json — run metadata
    - <index>.mp4 — final encoded MP4 in the project root `mp4/`

## Performance & tuning notes

- The program creates dedicated Rayon ThreadPools for simulation and rendering (sized from detected CPU count) to improve locality and throughput.
- Per-thread buffers reduce contention (fewer atomics) and are reduced/merged after accumulation.
- To speed experiments: reduce `--n-particles`, `--res`, `--steps-per-frame`, or `--fps`/`--duration-s`.
- Build with --release for best performance.

## Troubleshooting

- ffmpeg errors / missing:
  - Ensure `ffmpeg` is installed and on PATH: `ffmpeg -version`.
- Permissions writing to `mp4/`:
  - Ensure the process has write permission to the project directory.
- Out-of-memory or crashes:
  - Reduce particles, resolution, or steps-per-frame; run fewer threads or smaller pools.
- If encoder fails:
  - Check ffmpeg stderr (run without `-loglevel error` or capture logs).

## Developer notes (internals)

- Particle state stored as contiguous Vec<Particle> (x, y, vx, vy).
- Simulation steps:
  - `step` advances positions and applies implicit-boundary reflection using analytic normal computation.
  - `advance` calls `step` repeatedly for steps_per_frame.
- Histogram:
  - Implemented as per-thread AtomicU32 buffers that are merged into a final histogram and converted to f32 for rendering.
- Rendering:
  - Histogram values are log-scaled and mapped to a spectral palette; boundary is rasterized using a Bresenham iterator and drawn as white pixels.
- IO:
  - Frames are encoded to PNG and streamed into an ffmpeg child process via stdin (image2pipe).
- Progress:
  - The program parses ffmpeg progress lines to display an on-screen spinner/progress.

If you want the README focused on a specific subsystem (rendering, histogram, threading, or build flags), tell me which and I will add a concise focused section.
