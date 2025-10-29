use Value::Null;
use chrono::Utc;
use clap::Parser;
use collections::HashMap;
use crossbeam_channel::bounded;
use fs::{create_dir_all, read_dir, write};
use image::ColorType;
use image::ImageEncoder;
use image::RgbImage;
use image::codecs::png::PngEncoder;
use indicatif::{ProgressBar, ProgressStyle};
use io::BufReader;
use process::Stdio;
use rand::{RngCore, SeedableRng, prelude::StdRng};
use rayon::{current_num_threads, prelude::*};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, from_str, to_string_pretty};
use std::io::Write;
use std::{
    collections::{self, HashSet},
    error::Error,
    f32::consts::PI,
    fs,
    io::{self, BufRead, stdin},
    path::{Path, PathBuf},
    process::{self, Command},
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use thread::spawn;

fn main() -> Result<(), Box<dyn Error>> {
    let program_start = Instant::now();

    let config = Config::parse();
    let n_frames = config.fps * config.duration_s;
    let index = choose_video_index()?;
    let dirs = prepare_video_dirs_and_meta(index, &config)?;

    println!("\n--- Video Stats ---");
    if let Some(map) = dirs.meta.as_object() {
        for (key, value) in map {
            if key != "constants" {
                println!("{}: {}", key, value);
            }
        }
    }

    let entries: Vec<fs::DirEntry> = read_dir(&dirs.frames_dir)?.collect::<Result<Vec<_>, _>>()?;
    let (frame_count_res, total_size_bytes_res) = entries
        .par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "png") {
                match entry.metadata() {
                    Ok(m) => Some((1u64, m.len())),
                    Err(_) => None,
                }
            } else {
                None
            }
        })
        .reduce(|| (0u64, 0u64), |a, b| (a.0 + b.0, a.1 + b.1));

    let frame_count = frame_count_res;
    let total_size_bytes = total_size_bytes_res;

    println!("Frames saved: {}", frame_count);
    println!("Date: {}", dirs.meta.get("date").unwrap_or(&Null));
    println!("Total compute time: {}s", dirs.meta.get("compute_time").unwrap_or(&Value::from(0.0)));
    println!("Video dir size: {:.2} MB", total_size_bytes as f64 / 1_000_000.0);
    println!("Continue generating frames? (y/n): ");

    let mut response = String::new();
    stdin().read_line(&mut response)?;

    if response.trim().to_lowercase() != "y" {
        println!("Total elapsed time: {:.2}s", program_start.elapsed().as_secs_f64());
        return Ok(());
    }

    let sim_data = SimulationData::new(&config, dirs.start_frame);

    println!("Running simulation...");

    let start_time = Instant::now();
    let output_path = format!("mp4/{}.mp4", index);
    let compute_time = run_frame_generation(
        sim_data,
        &config,
        n_frames,
        dirs.start_frame,
        &dirs.frames_dir,
        &dirs.meta,
        &output_path,
        config.fps,
        n_frames,
    )?;
    let total_compute_time =
        dirs.meta.get("compute_time").and_then(|v| v.as_f64()).unwrap_or(0.0) + compute_time;
    let mut updated_meta = dirs.meta.clone();
    updated_meta["compute_time"] = Value::from(total_compute_time);
    updated_meta["last_frame"] = Value::from(n_frames);

    write(dirs.video_dir.join("meta.json"), to_string_pretty(&updated_meta)?)?;
    println!("Video saved to `mp4/{}`", index);

    let _total_elapsed = start_time.elapsed();
    println!("Total elapsed time: {:.2}s", program_start.elapsed().as_secs_f64());
    Ok(())
}

const A: f32 = 2.5;
const B: f32 = 1.0;
const N_EXP: f32 = 4.0;
const M_EXP: f32 = 3.0;
const DT: f32 = 0.001;
const EPSILON: f32 = 1e-8;
const CEN_X: f32 = 0.75;
const CEN_Y: f32 = 0.25;
const RADIUS: f32 = 0.5;
const VX0: f32 = 3.0;
const VY0: f32 = 0.0;
const N_PARTICLES: u64 = 10000;
const FPS: u64 = 60;
const DURATION_S: u64 = 10;
const STEPS_PER_FRAME: u64 = 30;
const RES: u32 = 256;
const DPI: u32 = 300;

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[clap(long, default_value_t = A)]
    pub a: f32,

    #[clap(long, default_value_t = B)]
    pub b: f32,

    #[clap(long, default_value_t = N_EXP)]
    pub n_exp: f32,

    #[clap(long, default_value_t = M_EXP)]
    pub m_exp: f32,

    #[clap(long, default_value_t = N_PARTICLES)]
    pub n_particles: u64,

    #[clap(long, default_value_t = DT)]
    pub dt: f32,

    #[clap(long, default_value_t = EPSILON)]
    pub epsilon: f32,

    #[clap(long, default_value_t = CEN_X)]
    pub center_x: f32,

    #[clap(long, default_value_t = CEN_Y)]
    pub center_y: f32,

    #[clap(long, default_value_t = RADIUS)]
    pub radius: f32,

    #[clap(long, default_value_t = VX0)]
    pub vx0: f32,

    #[clap(long, default_value_t = VY0)]
    pub vy0: f32,

    #[clap(long, default_value_t = FPS)]
    pub fps: u64,

    #[clap(long, default_value_t = DURATION_S)]
    pub duration_s: u64,

    #[clap(long, default_value_t = STEPS_PER_FRAME)]
    pub steps_per_frame: u64,

    #[clap(long, default_value_t = RES)]
    pub res: u32,

    #[clap(long, default_value_t = DPI)]
    pub dpi: u32,

    #[clap(long)]
    pub video_filename: Option<String>,
}

impl Config {
    pub fn constants(&self) -> Value {
        serde_json::json!({
            "A": self.a,
            "B": self.b,
            "N_EXP": self.n_exp,
            "M_EXP": self.m_exp,
            "n_particles": self.n_particles,
            "dt": self.dt,
            "epsilon": self.epsilon,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "vx0": self.vx0,
            "vy0": self.vy0,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "steps_per_frame": self.steps_per_frame,
            "res": self.res,
            "dpi": self.dpi,
            "video_filename": self.video_filename.as_deref().unwrap_or(""),
        })
    }
}

const PALETTE_SIZE: usize = 256;

pub type Vector3D<T> = [T; 3];

#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn from_rgb_u8(c: Vector3D<u8>) -> Self {
        Self { x: c[0] as f32 / 255.0, y: c[1] as f32 / 255.0, z: c[2] as f32 / 255.0 }
    }

    pub fn to_rgb_u8(self) -> Vector3D<u8> {
        [
            (self.x.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.y.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.z.clamp(0.0, 1.0) * 255.0).round() as u8,
        ]
    }

    pub fn clamp01(self) -> Self {
        Self { x: self.x.clamp(0.0, 1.0), y: self.y.clamp(0.0, 1.0), z: self.z.clamp(0.0, 1.0) }
    }
}

fn rgb_from_wavelength(wl: f32, gamma: f32) -> Vec3 {
    let (r, g, b) = if (380.0..=440.0).contains(&wl) {
        let t = (wl - 380.0) / 60.0;
        ((-t + 1.0).clamp(0.0, 1.0), 0.0, 1.0)
    } else if (440.0..=490.0).contains(&wl) {
        let t = (wl - 440.0) / 50.0;
        (0.0, t, 1.0)
    } else if (490.0..=510.0).contains(&wl) {
        let t = (wl - 490.0) / 20.0;
        (0.0, 1.0, (-t + 1.0).clamp(0.0, 1.0))
    } else if (510.0..=580.0).contains(&wl) {
        let t = (wl - 510.0) / 70.0;
        (t, 1.0, 0.0)
    } else if (580.0..=645.0).contains(&wl) {
        let t = (wl - 580.0) / 65.0;
        (1.0, (-t + 1.0).clamp(0.0, 1.0), 0.0)
    } else if (645.0..=750.0).contains(&wl) {
        let _t = (wl - 645.0) / 105.0;
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 0.0, 0.0)
    };

    let s = if wl <= 420.0 {
        0.3 + 0.7 * (wl - 380.0) / 40.0
    } else if wl > 700.0 {
        0.3 + 0.7 * (750.0 - wl) / 50.0
    } else {
        1.0
    };

    let apply_gamma =
        |c: f32| -> f32 { if c <= 0.0 { 0.0 } else { c.powf(gamma).clamp(0.0, 1.0) } };

    Vec3::new(apply_gamma(r * s), apply_gamma(g * s), apply_gamma(b * s))
}

pub fn build_palette() -> Vec<Vector3D<u8>> {
    let mut palette = Vec::with_capacity(PALETTE_SIZE);
    palette.push([0, 0, 0]);

    let rest: Vec<[u8; 3]> = (1..PALETTE_SIZE)
        .into_par_iter()
        .map(|i| {
            let wavelength = 700.0 - (700.0 - 380.0) * (i - 1) as f32 / 254.0;
            let rgb = rgb_from_wavelength(wavelength, 0.8);
            rgb.to_rgb_u8()
        })
        .collect();

    palette.extend(rest);
    palette
}

pub fn shape_boundary(
    a: f32,
    b: f32,
    n_exp: f32,
    m_exp: f32,
    n_points: usize,
) -> (Vec<f32>, Vec<f32>) {
    let pts: Vec<(f32, f32)> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let t = (i as f32 / n_points as f32) * 2.0 * PI;
            let cos_t = t.cos();
            let sin_t = t.sin();
            let sx = cos_t.signum() * cos_t.abs().powf(2.0 / n_exp);
            let sy = sin_t.signum() * sin_t.abs().powf(2.0 / m_exp);
            (a * sx, b * sy)
        })
        .collect();

    let (x, y): (Vec<f32>, Vec<f32>) = pts.into_iter().unzip();
    (x, y)
}

pub fn histogram_edges(a: f32, b: f32, bins: u32, factor: f32) -> (Vec<f32>, Vec<f32>) {
    let x_min = -a * factor;
    let x_max = a * factor;
    let y_min = -b * factor;
    let y_max = b * factor;

    let bins = bins as usize;
    let x_step = (x_max - x_min) / bins as f32;
    let y_step = (y_max - y_min) / bins as f32;

    let x_edges: Vec<f32> = (0..=bins).into_par_iter().map(|i| x_min + x_step * i as f32).collect();

    let y_edges: Vec<f32> = (0..=bins).into_par_iter().map(|i| y_min + y_step * i as f32).collect();

    (x_edges, y_edges)
}

pub fn compute_histogram(system: &ParticleSystem, a: f32, b: f32, bins: u32) -> Vec<f32> {
    let bins_usize = bins as usize;
    let factor = 1.2;
    let (x_edges, y_edges) = histogram_edges(a, b, bins, factor);

    let x_min = x_edges[0];
    let x_max = x_edges[bins_usize];
    let y_min = y_edges[0];
    let y_max = y_edges[bins_usize];

    let dx = (x_max - x_min) / bins as f32;
    let dy = (y_max - y_min) / bins as f32;
    let dx_inv = 1.0 / dx;
    let dy_inv = 1.0 / dy;

    let total_bins = bins_usize * bins_usize;

    let combined_atomic = system
        .particles()
        .par_chunks(1024)
        .map(|chunk| {
            let local: Vec<AtomicU32> = (0..total_bins).map(|_| AtomicU32::new(0)).collect();

            chunk.par_iter().for_each(|particle| {
                let px = particle.x;
                let py = particle.y;

                let ix = ((px - x_min) * dx_inv) as i32;
                let iy = ((py - y_min) * dy_inv) as i32;

                if ix >= 0 && ix < bins as i32 && iy >= 0 && iy < bins as i32 {
                    let idx = (iy as usize) * bins_usize + (ix as usize);
                    local[idx].fetch_add(1, Ordering::Relaxed);
                }
            });

            local
        })
        .reduce(
            || (0..total_bins).map(|_| AtomicU32::new(0)).collect::<Vec<AtomicU32>>(),
            |a, b| {
                b.into_par_iter().enumerate().for_each(|(i, v)| {
                    let vval = v.load(Ordering::Relaxed);
                    a[i].fetch_add(vval, Ordering::Relaxed);
                });

                a
            },
        );

    combined_atomic.into_iter().map(|c| c.load(Ordering::Relaxed) as f32).collect()
}

fn bresenham_points(mut x0: i64, mut y0: i64, x1: i64, y1: i64) -> Vec<(i64, i64)> {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };

    let mut err = dx + dy;
    let mut pts = Vec::new();

    loop {
        pts.push((x0, y0));
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
    pts
}

fn draw_boundary_parallel(
    image: &mut RgbImage,
    points: &[(i64, i64)],
    width: u32,
    height: u32,
    linewidth: usize,
) {
    let seg_pixels: Vec<Vec<(u32, u32)>> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            let (x1, y1) = points[i];
            let (x2, y2) = points[(i + 1) % points.len()];
            let base_pts = bresenham_points(x1, y1, x2, y2);
            let _local: Vec<i64> = Vec::new();
            let lw_half = (linewidth as i64) / 2;

            let local: Vec<(u32, u32)> = base_pts
                .par_iter()
                .flat_map_iter(|&(x, y)| {
                    let out = Vec::with_capacity(((2 * lw_half + 1) * (2 * lw_half + 1)) as usize);
                    (-lw_half..=lw_half)
                        .into_par_iter()
                        .flat_map_iter(|wx| {
                            (-lw_half..=lw_half).into_iter().filter_map(move |wy| {
                                let px = x + wx;
                                let py = y + wy;
                                if px >= 0
                                    && py >= 0
                                    && (px as u64) < width as u64
                                    && (py as u64) < height as u64
                                {
                                    Some((px as u32, py as u32))
                                } else {
                                    None
                                }
                            })
                        })
                        .collect::<Vec<(u32, u32)>>();

                    out.into_iter()
                })
                .collect();

            local
        })
        .collect();

    let per_row_map: HashMap<u32, Vec<u32>> = seg_pixels
        .into_par_iter()
        .fold(
            || HashMap::new(),
            |mut local: HashMap<u32, Vec<u32>>, seg| {
                for (px, py) in seg {
                    local.entry(py).or_default().push(px);
                }
                local
            },
        )
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (py, mut v) in b {
                    a.entry(py).or_default().append(&mut v);
                }
                a
            },
        );

    let mut per_row_vecs: Vec<Vec<u32>> = vec![Vec::new(); height as usize];

    for (py, v) in per_row_map {
        if (py as usize) < per_row_vecs.len() {
            per_row_vecs[py as usize] = v;
        }
    }

    let row_bytes = (width * 3) as usize;
    image.par_chunks_mut(row_bytes).enumerate().for_each(|(y, row)| {
        let pixels = &per_row_vecs[y];
        for &px in pixels {
            let base = (px as usize) * 3;
            if base + 2 < row.len() {
                row[base] = 255;
                row[base + 1] = 255;
                row[base + 2] = 255;
            }
        }
    });
}

pub fn render(
    hist: &[f32],
    x_edges: &[f32],
    y_edges: &[f32],
    bx: &[f32],
    by: &[f32],
    palette: &[Vector3D<u8>],
    out_px: (u32, u32),
    bins: u32,
) -> RgbImage {
    let (width, height) = out_px;
    let bins = bins as usize;
    let _total_bins = bins * bins;

    let (vmin, vmax) = hist
        .par_chunks(1024)
        .fold(
            || (f32::MAX, f32::MIN),
            |(min, max), chunk| {
                let chunk_min = chunk.iter().fold(f32::MAX, |a, &b| a.min(b));
                let chunk_max = chunk.iter().fold(f32::MIN, |a, &b| a.max(b));
                (min.min(chunk_min), max.max(chunk_max))
            },
        )
        .reduce(
            || (f32::MAX, f32::MIN),
            |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2)),
        );

    let vmin_clamped = vmin.max(1e-12);
    let vmax_clamped = vmax.max(vmin_clamped);

    let ln_min = vmin_clamped.ln();
    let ln_max = vmax_clamped.ln();
    let ln_range_inv = if (ln_max - ln_min).abs() > 0.0 { 1.0 / (ln_max - ln_min) } else { 0.0 };

    let palette_max = palette.len() - 1;
    let palette_scale = palette_max as f32;

    let small: Vec<[u8; 3]> = hist
        .par_iter()
        .map(|&v| {
            let nval =
                if v <= 0.0 { 0.0 } else { ((v.ln() - ln_min) * ln_range_inv).clamp(0.0, 1.0) };
            let palette_idx = (nval * palette_scale).round() as usize;
            let palette_idx = palette_idx.clamp(0, palette_max);
            palette[palette_idx]
        })
        .collect();

    let mut image = RgbImage::new(width, height);

    image.par_chunks_mut((width * 3) as usize).enumerate().for_each(|(y, row)| {
        let bin_y = (y * bins) / height as usize;

        for x in 0..width as usize {
            let bin_x = (x * bins) / width as usize;
            let small_idx = bin_y * bins + bin_x;
            let [r, g, b] = small[small_idx];

            let base = x * 3;
            if base + 2 < row.len() {
                row[base] = r;
                row[base + 1] = g;
                row[base + 2] = b;
            }
        }
    });

    use rayon::prelude::*;
    let points: Vec<(i64, i64)> = bx
        .par_iter()
        .zip(by.par_iter())
        .map(|(&bx_val, &by_val)| {
            let x = ((bx_val - x_edges[0]) / (x_edges[bins] - x_edges[0]) * (width - 1) as f32)
                .round() as i64;
            let y = height as i64
                - 1
                - ((by_val - y_edges[0]) / (y_edges[bins] - y_edges[0]) * (height - 1) as f32)
                    .round() as i64;
            (x, y)
        })
        .collect();

    let linewidth = (width / 150).max(1) as usize;

    draw_boundary_parallel(&mut image, &points, width, height, linewidth);

    image
}

pub struct SimulationData {
    pub palette: Arc<Vec<[u8; 3]>>,
    pub bx: Arc<Vec<f32>>,
    pub by: Arc<Vec<f32>>,
    pub system: ParticleSystem,
    pub x_edges: Arc<Vec<f32>>,
    pub y_edges: Arc<Vec<f32>>,
}

impl SimulationData {
    pub fn new(config: &Config, start_frame: u64) -> Self {
        let palette = Arc::new(build_palette());
        let (bx, by) = shape_boundary(config.a, config.b, config.n_exp, config.m_exp, 400);
        let (x_edges, y_edges) = histogram_edges(config.a, config.b, config.res, 1.2);

        let mut system = init_cluster(
            config.n_particles,
            config.radius,
            config.center_x,
            config.center_y,
            config.vx0,
            config.vy0,
        );

        if start_frame > 0 {
            advance(
                &mut system,
                start_frame * config.steps_per_frame,
                config.dt,
                config.epsilon,
                config.a,
                config.b,
                config.n_exp,
                config.m_exp,
            );
        }

        Self {
            palette,
            bx: Arc::new(bx),
            by: Arc::new(by),
            system,
            x_edges: Arc::new(x_edges),
            y_edges: Arc::new(y_edges),
        }
    }
}

pub fn run_frame_generation(
    mut sim_data: SimulationData,
    config: &Config,
    n_frames: u64,
    start_frame: u64,
    _frames_dir: &Path,
    _meta: &Value,
    output_path: &str,
    fps: u64,
    _total_frames: u64,
) -> Result<f64, Box<dyn Error>> {
    let (width, height) = compute_out_px(config.dpi);
    let total_to_generate = if n_frames > start_frame { n_frames - start_frame } else { 0 };
    let pb = ProgressBar::new(total_to_generate);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{prefix} {bar:40.cyan/blue} {pos:>7}/{len:7} {percent:>3}% ({eta})")?
            .progress_chars("##-"),
    );
    pb.set_prefix("Generating frames");

    let bins = config.res as usize;
    let total_bins = bins * bins;

    let mut histogram_buf = vec![0f32; total_bins];
    let mut h_log_flat = vec![0f32; total_bins];

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-f")
        .arg("image2pipe")
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg("-")
        .arg("-vf")
        .arg("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        .arg("-c:v")
        .arg("libx264")
        .arg("-crf")
        .arg("12")
        .arg("-preset")
        .arg("slow")
        .arg("-profile:v")
        .arg("high")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-movflags")
        .arg("+faststart")
        .arg("-progress")
        .arg("pipe:1")
        .arg(output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let mut child = cmd.spawn()?;
    let child_stdin = child.stdin.take().ok_or("Failed to open ffmpeg stdin")?;

    // writer thread for piping PNGs to ffmpeg stdin
    let (tx, rx) = bounded::<RgbImage>(16);
    let writer_handle = spawn(move || {
        let mut writer = child_stdin;
        while let Ok(img) = rx.recv() {
            let width = img.width();
            let height = img.height();
            let raw = img.into_raw();
            if let Err(e) = PngEncoder::new(&mut writer).write_image(
                &raw,
                width,
                height,
                ColorType::Rgb8.into(),
            ) {
                let _ = eprintln!("ffmpeg stdin write error: {}", e);
                break;
            }
            let _ = writer.flush();
        }
        drop(writer);
    });

    let start_time = Instant::now();

    for _frame_idx in start_frame..n_frames {
        advance(
            &mut sim_data.system,
            config.steps_per_frame,
            config.dt,
            config.epsilon,
            config.a,
            config.b,
            config.n_exp,
            config.m_exp,
        );

        let hist = compute_histogram(&sim_data.system, config.a, config.b, config.res);
        histogram_buf.copy_from_slice(&hist);

        h_log_flat.par_iter_mut().zip(&histogram_buf).for_each(|(h, &v)| {
            *h = v + config.epsilon;
        });

        let frame_image = render(
            &h_log_flat,
            &sim_data.x_edges,
            &sim_data.y_edges,
            &sim_data.bx,
            &sim_data.by,
            &sim_data.palette,
            (width, height),
            config.res,
        );

        tx.send(frame_image).map_err(|e| format!("ffmpeg writer channel send failed: {}", e))?;

        pb.inc(1);
    }

    // finished generating frames -> close sender and wait for writer to finish
    drop(tx);
    let _ = writer_handle.join();

    // now start spinner/parser to show ffmpeg progress while it finalises the video
    let stdout = child.stdout.take().ok_or("Failed to capture ffmpeg stdout")?;
    let reader = BufReader::new(stdout);

    let spinner = ProgressBar::new_spinner();
    spinner
        .set_style(ProgressStyle::default_spinner().template("{prefix} {spinner} {msg}").unwrap());
    spinner.set_prefix("Generating video");
    spinner.enable_steady_tick(Duration::from_millis(100));

    let spinner_clone = spinner.clone();
    let parser_handle = spawn(move || {
        let mut last_msg = String::new();
        for line_res in reader.lines() {
            if let Ok(line) = line_res {
                if line.is_empty() {
                    continue;
                }

                let mut kv_map = HashMap::new();
                let parts: Vec<&str> = line.split_whitespace().collect();
                let mut i = 0;
                while i < parts.len() {
                    let part = parts[i];
                    if let Some(eq_pos) = part.find('=') {
                        let key = part[..eq_pos].trim().to_string();
                        let mut val = part[eq_pos + 1..].trim();
                        if val.is_empty() && (i + 1) < parts.len() && !parts[i + 1].contains('=') {
                            i += 1;
                            val = parts[i].trim();
                        }
                        kv_map.insert(key, val.to_string());
                    }
                    i += 1;
                }

                let mut parts_msg = Vec::new();
                if let Some(size) = kv_map.get("total_size").or_else(|| kv_map.get("size")) {
                    if let Some(bytes) = size_to_bytes(size) {
                        parts_msg.push(format!("{:.2}MB", bytes as f64 / 1e6));
                    }
                }

                if let Some(fps_s) = kv_map.get("fps") {
                    parts_msg.push(format!("fps:{}", fps_s.trim()));
                }

                if let Some(speed) = kv_map.get("speed") {
                    parts_msg.push(format!("speed:{}", speed.trim()));
                }

                if let Some(out_time) = kv_map.get("out_time").or_else(|| kv_map.get("time")) {
                    parts_msg.push(format!("time:{}", out_time.trim()));
                }

                let msg = parts_msg.join(" | ");
                if msg != last_msg {
                    spinner_clone.set_message(msg.clone());
                    last_msg = msg;
                }

                if let Some(progress) = kv_map.get("progress") {
                    if progress.trim() == "end" {
                        break;
                    }
                }
            }
        }
    });

    let _status = child.wait()?;
    let _ = parser_handle.join();

    spinner.finish_and_clear();

    pb.finish_with_message("Frame generation complete");

    let elapsed = start_time.elapsed().as_secs_f64();
    Ok(elapsed)
}

fn compute_out_px(dpi: u32) -> (u32, u32) {
    let fig_inches = 8.0;
    let size = (fig_inches * dpi as f32).round() as u32;
    (size, size)
}

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
}

#[derive(Clone)]
pub struct ParticleSystem {
    particles: Vec<Particle>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self { particles: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { particles: Vec::with_capacity(capacity) }
    }

    pub fn resize(&mut self, size: usize) {
        self.particles.resize(size, Particle { x: 0.0, y: 0.0, vx: 0.0, vy: 0.0 });
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn particles_mut(&mut self) -> &mut [Particle] {
        &mut self.particles
    }
}

pub fn init_cluster(
    n: u64,
    radius: f32,
    center_x: f32,
    center_y: f32,
    vx0: f32,
    vy0: f32,
) -> ParticleSystem {
    let n = n as usize;
    let mut system = ParticleSystem::with_capacity(n);
    system.resize(n);

    let num_threads = current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads;

    system.particles_mut().par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
        let _rng = StdRng::seed_from_u64(chunk_idx as u64);

        chunk.par_iter_mut().enumerate().for_each(|(i, particle)| {
            let seed = (chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(i as u64);
            let mut local_rng = StdRng::seed_from_u64(seed);

            let u1 = (local_rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0);
            let u2 = (local_rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0);

            let r = radius * (u1.sqrt() as f32);
            let theta = (u2 as f32) * 2.0 * PI;

            particle.x = r * theta.cos() + center_x;
            particle.y = r * theta.sin() + center_y;
            particle.vx = vx0;
            particle.vy = vy0;
        });
    });

    system
}

fn pow_fast(x: f32, e: f32) -> f32 {
    const EPS: f32 = 1e-6;
    if (e - 1.0).abs() < EPS {
        x
    } else if (e - 2.0).abs() < EPS {
        x * x
    } else if (e - 3.0).abs() < EPS {
        x * x * x
    } else if (e - 4.0).abs() < EPS {
        let x2 = x * x;
        x2 * x2
    } else {
        x.powf(e)
    }
}

pub fn step(
    system: &mut ParticleSystem,
    dt: f32,
    epsilon: f32,
    a: f32,
    b: f32,
    n_exp: f32,
    m_exp: f32,
) {
    if system.is_empty() {
        return;
    }

    let inv_a = 1.0 / a;
    let inv_b = 1.0 / b;
    let eps2 = epsilon * epsilon;

    system.particles_mut().par_chunks_mut(1024).for_each(|chunk| {
        for particle in chunk {
            let px = particle.x + particle.vx * dt;
            let py = particle.y + particle.vy * dt;
            let xna = px.abs() * inv_a;
            let ynb = py.abs() * inv_b;

            let val = pow_fast(xna, n_exp) + pow_fast(ynb, m_exp) - 1.0;

            if val <= 0.0 {
                particle.x = px;
                particle.y = py;
                continue;
            }

            let sign_x = px.signum();
            let sign_y = py.signum();

            let xpow = pow_fast(px.abs() * inv_a, n_exp - 1.0);
            let ypow = pow_fast(py.abs() * inv_b, m_exp - 1.0);

            let df_dx = n_exp * inv_a * xpow * sign_x;
            let df_dy = m_exp * inv_b * ypow * sign_y;
            let len2 = df_dx * df_dx + df_dy * df_dy;

            if len2 <= eps2 || len2 == 0.0 {
                particle.x = px;
                particle.y = py;
                continue;
            }

            let inv_len = len2.sqrt().recip();
            let nx = df_dx * inv_len;
            let ny = df_dy * inv_len;
            let vx = particle.vx;
            let vy = particle.vy;
            let vxn = vx * nx + vy * ny;
            let rx = vx - 2.0 * vxn * nx;
            let ry = vy - 2.0 * vxn * ny;

            particle.vx = rx;
            particle.vy = ry;
            particle.x = px - rx * epsilon;
            particle.y = py - ry * epsilon;
        }
    });
}

pub fn advance(
    system: &mut ParticleSystem,
    steps: u64,
    dt: f32,
    epsilon: f32,
    a: f32,
    b: f32,
    n_exp: f32,
    m_exp: f32,
) {
    (0..steps).for_each(|_| step(system, dt, epsilon, a, b, n_exp, m_exp));
}

pub struct VideoDirs {
    pub video_dir: PathBuf,
    pub frames_dir: PathBuf,
    pub meta: Value,
    pub start_frame: u64,
}

fn try_insert_numeric(name: &str, used_indices: &mut HashSet<u64>) {
    if name.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(index) = name.parse::<u64>() {
            used_indices.insert(index);
        }
    }
}

pub fn next_available_index() -> Result<u64, Box<dyn Error>> {
    let mp4_dir = Path::new("mp4");
    if !mp4_dir.exists() {
        create_dir_all(mp4_dir)?;
        return Ok(1);
    }

    let mut used_indices = HashSet::new();

    for entry in read_dir(mp4_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "mp4") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                try_insert_numeric(stem, &mut used_indices);
            }
        } else if path.is_dir() {
            if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
                try_insert_numeric(dir_name, &mut used_indices);
            }
        }
    }

    for i in 1.. {
        if !used_indices.contains(&i) {
            return Ok(i);
        }
    }

    Ok(1)
}

pub fn choose_video_index() -> Result<u64, Box<dyn Error>> {
    println!("Enter video index (or Enter for next available): ");
    let mut input = String::new();
    stdin().read_line(&mut input)?;

    let input = input.trim();
    if input.is_empty() {
        let index = next_available_index()?;
        println!("Using next available index: {}", index);
        Ok(index)
    } else {
        input.parse::<u64>().map_err(|e| e.into())
    }
}

pub fn prepare_video_dirs_and_meta(
    index: u64,
    config: &Config,
) -> Result<VideoDirs, Box<dyn Error>> {
    let video_dir = Path::new("mp4").join(index.to_string());
    let frames_dir = video_dir.join("frames");

    create_dir_all(&frames_dir)?;

    let meta_path = video_dir.join("meta.json");
    let meta = if meta_path.exists() {
        let meta_content = fs::read_to_string(&meta_path)?;
        from_str(&meta_content)?
    } else {
        let mut meta = Map::new();
        meta.insert("constants".to_string(), config.constants());
        meta.insert("date".to_string(), Value::String(Utc::now().to_rfc3339()));
        meta.insert("last_frame".to_string(), Value::from(0));
        meta.insert("compute_time".to_string(), Value::from(0.0));
        meta.insert("resolution".to_string(), Value::from(config.res));
        Value::Object(meta)
    };

    let start_frame = meta.get("last_frame").and_then(Value::as_u64).unwrap_or(0);

    Ok(VideoDirs { video_dir, frames_dir, meta, start_frame })
}

pub fn generate_video(
    frames_dir: &Path,
    output_path: &str,
    fps: u64,
    total_frames: u64,
) -> Result<(), Box<dyn Error>> {
    let frames_pattern = frames_dir.join("%d.png").to_string_lossy().to_string();

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg(&frames_pattern)
        .arg("-vf")
        .arg("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        .arg("-c:v")
        .arg("libx264")
        .arg("-crf")
        .arg("12")
        .arg("-preset")
        .arg("slow")
        .arg("-profile:v")
        .arg("high")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-movflags")
        .arg("+faststart")
        .arg("-progress")
        .arg("pipe:1")
        .arg(output_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take().ok_or("Failed to capture ffmpeg stdout")?;
    let reader = BufReader::new(stdout);

    let spinner = ProgressBar::new_spinner();
    spinner
        .set_style(ProgressStyle::default_spinner().template("{prefix} {spinner} {msg}").unwrap());
    spinner.set_prefix("Generating video");
    spinner.enable_steady_tick(Duration::from_millis(100));

    let total = total_frames;
    let mut last_msg = String::new();
    let mut _seen_end = false;

    for line_res in reader.lines() {
        let line = line_res?;
        if line.is_empty() {
            continue;
        }

        let mut kv_map = HashMap::new();
        let parts: Vec<&str> = line.split_whitespace().collect();
        let mut i = 0;
        while i < parts.len() {
            let part = parts[i];
            if let Some(eq_pos) = part.find('=') {
                let key = part[..eq_pos].trim().to_string();
                let mut val = part[eq_pos + 1..].trim();
                if val.is_empty() && (i + 1) < parts.len() && !parts[i + 1].contains('=') {
                    i += 1;
                    val = parts[i].trim();
                }
                kv_map.insert(key, val.to_string());
            }
            i += 1;
        }

        let mut parts_msg = Vec::new();

        if let Some(size) = kv_map.get("total_size").or_else(|| kv_map.get("size")) {
            if let Some(bytes) = size_to_bytes(size) {
                parts_msg.push(format!("{:.2}MB", bytes as f64 / 1e6));
            }
        }

        if let Some(fps_s) = kv_map.get("fps") {
            parts_msg.push(format!("fps:{}", fps_s.trim()));
        }

        if let Some(speed) = kv_map.get("speed") {
            parts_msg.push(format!("speed:{}", speed.trim()));
        }

        if let Some(out_time) = kv_map.get("out_time").or_else(|| kv_map.get("time")) {
            parts_msg.push(format!("time:{}", out_time.trim()));
        }

        let msg = parts_msg.join(" | ");
        if msg != last_msg {
            spinner.set_message(msg.clone());
            last_msg = msg;
        }

        if let Some(progress) = kv_map.get("progress") {
            if progress.trim() == "end" {
                _seen_end = true;
                break;
            }
        }
    }

    let status = child.wait()?;

    spinner.finish_and_clear();

    let final_pb = ProgressBar::new(total);
    final_pb.set_style(
        ProgressStyle::default_bar()
            .template("{prefix} {bar:40.green/white} {pos:>7}/{len:7} {percent:>3}% ({elapsed})")
            .unwrap()
            .progress_chars("=>-"),
    );

    final_pb.set_prefix("Generating video");
    final_pb.set_position(total);
    final_pb.finish_with_message("Video generation complete");

    if !status.success() {
        return Err("ffmpeg failed to generate video".into());
    }

    Ok(())
}

fn size_to_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("N/A") {
        return None;
    }

    if let Ok(v) = s.parse::<u64>() {
        return Some(v);
    }

    let s_upper = s.to_uppercase();
    let units = [
        ("B", 1u64),
        ("KB", 1_000u64),
        ("KIB", 1024u64),
        ("MB", 1_000_000u64),
        ("MIB", 1024u64 * 1024u64),
        ("GB", 1_000_000_000u64),
        ("GIB", 1024u64 * 1024u64 * 1024u64),
        ("TB", 1_000_000_000_000u64),
        ("TIB", 1024u64.pow(4)),
        ("K", 1_000u64),
        ("M", 1_000_000u64),
        ("G", 1_000_000_000u64),
        ("T", 1_000_000_000_000u64),
    ];

    for (unit, mul) in &units {
        if s_upper.ends_with(unit) {
            let num = s_upper.trim_end_matches(unit).trim();
            if let Ok(f) = num.parse::<f64>() {
                return Some((f * (*mul as f64)).round() as u64);
            }
        }
    }

    None
}
