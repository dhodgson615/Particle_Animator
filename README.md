# ParticleAnimatorRust

A high-performance particle simulator and renderer written in Rust. The program simulates particles moving inside a parametric boundary, accumulates a 2D histogram of particle density, renders frames as PNG images, and uses `ffmpeg` to encode the frames to an MP4 video.

---

## Features

- Parallel simulation and rendering using Rayon.
- Histogram-based rendering with a spectral palette.
- Multithreaded image saving (worker threads) to avoid blocking the renderer.
- Resume capability: existing `meta.json` under a video directory is used to continue generation from the last saved frame.
- Produces a PNG sequence (frames) and encodes the sequence to MP4 via `ffmpeg`.

---

## Prerequisites

- Rust toolchain (stable). Install from https://rustup.rs if you don't have it.
- `ffmpeg` (available on PATH) for encoding frames into MP4. On macOS you can install with Homebrew: `brew install ffmpeg`.
- Enough disk space for frame sequences (frames are saved as PNG files before encoding).

---

## Build

Build the release binary for best performance:

```bash
cargo build --release
```

Or run directly from Cargo (slower due to debug profile):

```bash
cargo run --release -- [flags]
```

The produced release binary is available at `target/release/ParticleAnimatorRust`.

---

## Run / Usage

When you run the program it will prompt for a video index. Press Enter to use the next available index (the program will create `mp4/<index>/frames` and a `meta.json`), or type a specific numeric index to use an existing directory.

Example (interactive):

```bash
# Build then run with defaults
cargo run --release --

# Run with custom options (example: reduce particles and resolution for quick testing)
cargo run --release -- --n-particles 2000 --res 128 --duration-s 5 --fps 30
```

You can also run the built binary directly:

```bash
./target/release/ParticleAnimatorRust --fps 60 --duration-s 10
```

After frames are generated the program will call `ffmpeg` to encode `mp4/<index>.mp4` from the produced frame sequence.

Note: The program saves frames to `mp4/<index>/frames/<frame>.png` and writes `mp4/<index>/meta.json` with run metadata and constants; the final encoded video is written to `mp4/<index>.mp4`.

---

## CLI flags

The program exposes a set of long-form flags (clap-derived). Below are the available flags and their default values (defaults come from constants in the source):

- `--a <f32> (default: 2.5)`
- `--b <f32> (default: 1.0)`
- `--n-exp <f32> (default: 4.0)`
- `--m-exp <f32> (default: 3.0)`
- `--n-particles <u64> (default: 10000)`
- `--dt <f32> (default: 0.001)`
- `--epsilon <f32> (default: 1e-8)`
- `--center-x <f32> (default: 0.75)`
- `--center-y <f32> (default: 0.25)`
- `--radius <f32> (default: 0.5)`
- `--vx0 <f32> (default: 3.0)`
- `--vy0 <f32> (default: 0.0)`
- `--fps <u64> (default: 60)`
- `--duration-s <u64> (default: 10)`
- `--steps-per-frame <u64> (default: 30)`
- `--res <u32> (default: 256)           # histogram bins per dimension`
- `--dpi <u32> (default: 300)           # output image DPI (affects image pixel size)`
- `--video-filename <string> (optional) # included in meta, not currently used as output path`

Examples:

```bash
# Generate a short, low-res preview (fast)
cargo run --release -- --n-particles 2000 --res 128 --duration-s 4 --fps 24

# Full quality render
cargo run --release -- --n-particles 20000 --res 512 --duration-s 20 --fps 60
```

---

## How resume works

- When you supply a numeric index or press Enter to use the next available index, the program creates (or opens) `mp4/<index>/meta.json`.
- If `meta.json` exists and contains a `last_frame` entry, the simulator will advance the internal system by `last_frame * steps_per_frame` and then continue generating frames from that frame index, so you can resume an interrupted run.

---

## Output layout

- `mp4/` - root output directory
  - `<index>/` - per-run directory created by the program
    - `frames/` - PNG frames written here (named `<frame>.png`)
    - `meta.json` - JSON with run metadata and constants
  - `<index>.mp4` - final encoded MP4 written to the `mp4/` root

You can inspect `meta.json` to see the constants used for the run and the `last_frame` and `compute_time` fields.

---

## Performance notes & tuning

- The program is heavily parallelized using Rayon for CPU-bound work. To control the number of threads used by Rayon, set the environment variable `RAYON_NUM_THREADS` or use the `RAYON_NUM_THREADS` mechanism appropriate for your environment.

- Build with `--release` to get optimized code and better performance.

- To speed up a run for experimentation, reduce:
  - `--n-particles` (fewer particles)
  - `--res` (fewer histogram bins)
  - `--steps-per-frame` (fewer internal steps per frame)
  - `--fps` or `--duration-s` (fewer total frames)

- Frames are written as PNGs before encoding. For long runs you may need hundreds or thousands of PNGs. Remove or compress old runs if disk space is a concern.

---

## Troubleshooting

- `ffmpeg` not found or errors during encoding:
  - Ensure `ffmpeg` is installed and available on your PATH.
  - Run `ffmpeg -version` to verify installation.

- Permission errors writing to `mp4/`:
  - Ensure you have write permissions in the project directory.

- Out of memory or poor performance:
  - Reduce particle count, resolution, or run shorter durations.
  - Ensure other heavy processes aren't contending for CPU.

- If an interrupted run produced partial frames, you can resume by reusing the same numeric index; the program will detect `last_frame` in metadata and continue rendering from that frame.

---

## Internals

- Particles are stored in a contiguous `Vec<Particle>`. The `step` function advances particle positions, checks against an implicit parametric boundary, and reflects velocities when particles hit the boundary.

- A 2D histogram over the simulation domain is computed and then converted to an RGB image using a wavelength-like spectral palette.

- Rayon is used for compute parallelism, and a small set of saver threads write images to disk to reduce blocking on I/O.


---
