# ParticleAnimatorRust

A high-performance particle simulator and renderer written in Rust. It
simulates particles moving inside a parametric superellipse-like boundary,
accumulates a 2D density histogram per frame, rasterizes frames to PNG, and
streams them to `ffmpeg` (image2pipe) to encode MP4 video. Runs are resumable
and metadata for each run is stored in `mp4/<index>/meta.json`.

## Overview

- Simulation inside an implicit parametric boundary with analytic reflections
  at the boundary.
- Per-frame particle positions are accumulated into a 2D histogram and mapped
  to a spectral palette for rendering.
- PNG frames are produced on worker threads and streamed to `ffmpeg` via stdin;
  `ffmpeg` writes `mp4/<index>.mp4`.
- Resumable runs: `mp4/<index>/meta.json` tracks `last_frame`, `constants`,
  timing, etc.
- Parallelized with Rayon and uses `mimalloc` as the global allocator.

## Stack

- Language: Rust (edition 2024)
- Package manager/build: Cargo
- Key dependencies (see `Cargo.toml`): `rayon`, `crossbeam-channel`, `image`,
  `indicatif`, `serde`/`serde_json`, `chrono`, `ahash`, `mimalloc`
- External tools: `ffmpeg` (required at runtime for encoding)
- Entry point: `src/main.rs`

## Requirements

- Rust toolchain via [`rustup`](https://rustup.rs) (stable is fine)
- `ffmpeg` available on PATH (`ffmpeg -version` to verify)
- Optional (for developer script): Python 3.8+ to run
  `scripts/inline_sweeper.py`

## Setup

```bash
# Build optimized binary
cargo build --release
```

## Run / Usage

On start, the program asks for a video index:
- Press Enter to use the next available index (creates `mp4/<index>/frames` and
  writes `mp4/<index>.mp4`).
- Or type a numeric index to use/resume an existing directory.

Examples:
- Default run (interactive):
  ```bash
  cargo run --release
  ```
- Quick test (fewer particles, lower res):
  ```bash
  cargo run --release -- \
    --n-particles 2000 --res 128 --duration-s 4 --fps 24 --steps-per-frame 50
  ```
- Higher quality render:
  ```bash
  cargo run --release -- \
    --n-particles 20000 \
    --res 512 \
    --duration-s 20 \
    --fps 60 \
    --steps-per-frame 150
  ```

Frames are streamed to `ffmpeg` via stdin; the final MP4 is at
`mp4/<index>.mp4`.

### CLI flags (current defaults)

The binary uses `clap` and exposes these options; defaults reflect the current
`src/main.rs` constants at the time of writing.

- `--a <f32>` (default: 1.0)
- `--b <f32>` (default: 1.0)
- `--n-exp <f32>` (default: 2.0)
- `--m-exp <f32>` (default: 2.0)
- `--n-particles <u64>` (default: 1000)
- `--dt <f32>` (default: 0.0001)
- `--epsilon <f32>` (default: 1e-8)
- `--center-x <f32>` (default: 0.1)
- `--center-y <f32>` (default: -0.1)
- `--radius <f32>` (default: 0.1)
- `--vx0 <f32>` (default: 1.0)
- `--vy0 <f32>` (default: 0.0)
- `--fps <u64>` (default: 60)
- `--duration-s <u64>` (default: 10)
- `--steps-per-frame <u64>` (default: 300)
- `--res <u32>` (default: 932) — histogram bins per dimension
- `--dpi <u32>` (default: 300) — output DPI; output image size (px) =
  `FIG_INCHES * dpi` (with `FIG_INCHES = 8.0`)
- `--sim-threads <usize>` (optional) — override auto-detected simulation thread
  pool size
- `--render-threads <usize>` (optional) — override auto-detected render thread
  pool size
- `--video-filename <string>` (optional) — stored in `meta.json`; output MP4
  path is `mp4/<index>.mp4`

### Resume behavior and output layout

- Metadata at `mp4/<index>/meta.json` includes:
  - `constants` — effective run parameters
  - `date` — run start time
  - `last_frame` — last completed frame index (used to resume)
  - `compute_time` — cumulative compute time (seconds)
  - `resolution` — histogram resolution (`res`)
- When resuming, state is advanced by `last_frame * steps_per_frame` before new
  frames are generated.
- Output tree:
  - `mp4/`
    - `<index>/`
      - `frames/` — optional saved PNGs (useful for debugging)
      - `meta.json` — run metadata
    - `<index>.mp4` — final encoded video

## Scripts

- `scripts/inline_sweeper.py` — experiments with LLVM inlining thresholds by
  editing `.cargo/config.toml`, then running the program repeatedly and logging
  timings.
  - Requirements: Python 3.8+, Cargo, `ffmpeg` on PATH.
  - It will temporarily modify `.cargo/config.toml` and restore a backup when
    done. Inspect `START/END/STEP/REPEATS` constants at the top of the script.
  - Usage (from repo root):
    ```bash
    python3 scripts/inline_sweeper.py
    ```

## Environment variables

- `RUST_BACKTRACE=1` — useful for debugging; the script enables it during runs.
- CPU thread detection is automatic; override via `--sim-threads` /
  `--render-threads` flags.

## Tests

- No Rust tests are currently present. TODO: add unit tests/benchmarks for core
  simulation and rendering components.

## Performance & tuning notes

- Dedicated Rayon thread pools for simulation and rendering (sized from CPU
  count) to improve locality and throughput.
- Per-thread buffers reduce contention and are merged after accumulation.
- For faster iterations: reduce `--n-particles`, `--res`, `--steps-per-frame`,
  or `--fps`/`--duration-s`.
- Build with `--release` for best performance.
- Global allocator: `mimalloc`.
- Additional per-target optimization flags may be set in `.cargo/config.toml`
  (see that file for current values).

## Troubleshooting

- ffmpeg errors / missing: ensure `ffmpeg` is installed and on PATH (`ffmpeg
  -version`).
- Permission errors writing to `mp4/`: ensure the process can write to the repo
  directory.
- Out-of-memory or crashes: reduce particles, resolution, steps-per-frame, or
  thread counts.
- Encoder issues: check `ffmpeg` stderr; run without reduced log-levels to
  capture more detail.

## Project structure

```
ParticleAnimatorRust/
├─ Cargo.toml
├─ Cargo.lock
├─ src/
│  └─ main.rs
├─ scripts/
│  └─ inline_sweeper.py
├─ .cargo/
│  └─ config.toml
├─ mp4/                # run outputs: per-index subdirs and final MP4s
├─ target/             # Cargo build artifacts
├─ rustfmt.toml
├─ todo.txt
└─ README.md
```

---