use ahash::AHashSet;
use chrono::Utc;
use clap::Parser;
use collections::HashMap;
use crossbeam_channel::bounded;
use image::RgbImage;
use indicatif::{ProgressBar, ProgressStyle};
use mimalloc::MiMalloc;
use num_cpus;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use serde::{Deserialize, Serialize};
use serde_json::{
    Map,
    Value::{self, Null, Object},
    from_str, json, to_string_pretty,
};
use std::{
    collections,
    error::Error,
    f32::consts::PI,
    fs::{DirEntry, create_dir_all, read_dir, read_to_string, write},
    io::{BufRead, BufReader, Write, stdin},
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use thread::scope;

#[global_allocator]
static GLOBAL_ALLOC: MiMalloc = MiMalloc; // Do not change

const A: f32 = 1.0;
const B: f32 = 1.0;
const N_EXP: f32 = 2.0;
const M_EXP: f32 = 2.0;
const DT: f32 = 0.0001; // change back to 0.001
const EPSILON: f32 = 1e-8;
const CEN_X: f32 = 0.1;
const CEN_Y: f32 = -0.1;
const RADIUS: f32 = 0.1;
const VX0: f32 = 1.0;
const VY0: f32 = 0.0;
const N_PARTICLES: u64 = 1000;
const FPS: u64 = 60;
const DURATION_S: u64 = 10;
const STEPS_PER_FRAME: u64 = 300; // change back to 30
const RES: u32 = 932;
const DPI: u32 = 300;
const HISTOGRAM_FACTOR: f32 = 1.25;

const SHAPE_SAMPLE_POINTS: usize = 1000;
const SEED_MULTIPLIER: u64 = 0x9E3779B97F4A7C15;
const FIG_INCHES: f32 = 8.0;
const BOUNDARY_THICKNESS: usize = 2;

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
    pub sim_threads: Option<usize>,

    #[clap(long)]
    pub render_threads: Option<usize>,

    #[clap(long)]
    pub video_filename: Option<String>,
}

impl Config {
    pub fn constants(&self) -> Value {
        json!({
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

fn precompute_pixel_bin_map(out_px: (u32, u32), bins: usize) -> Vec<usize> {
    let (width, height) = out_px;
    let w = width as usize;
    let h = height as usize;
    let total = w.saturating_mul(h);

    if total == 0 || bins == 0 {
        return Vec::new();
    }

    (0..total)
        .into_par_iter()
        .map(|i| {
            let y = i / w;
            let x = i % w;
            let bin_y = (y * bins) / h;
            let bin_x = (x * bins) / w;
            bin_y * bins + bin_x
        })
        .collect()
}

fn precompute_thickness_offsets(thickness: usize) -> Vec<(i64, i64)> {
    let r = thickness as i64;
    let rr = r * r;

    let per_rows: Vec<Vec<(i64, i64)>> = (-r..=r)
        .into_par_iter()
        .map(|dy| {
            let rem = rr - dy * dy;
            if rem >= 0 {
                let dx_lim = (rem as f64).sqrt() as i64;
                (-dx_lim..=dx_lim).into_par_iter().map(move |dx| (dx, dy)).collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    per_rows.into_par_iter().flat_map(|row| row.into_par_iter()).collect()
}

pub fn histogram_counts(
    system: &ParticleSystem,
    x_min: f32,
    y_min: f32,
    dx_inv: f32,
    dy_inv: f32,
    bins: u32,
    total_bins: usize,
    pool: &ThreadPool,
) -> Vec<u32> {
    let bins_usize = bins as usize;
    let n = system.len();

    pool.install(|| {
        (0..n)
            .into_par_iter()
            .fold(
                || vec![0u32; total_bins],
                |mut local, idx| {
                    let px = system.x[idx];
                    let py = system.y[idx];

                    let ix = ((px - x_min) * dx_inv) as i32;
                    let iy = ((py - y_min) * dy_inv) as i32;

                    if ix >= 0 && ix < bins as i32 && iy >= 0 && iy < bins as i32 {
                        let id = (iy as usize) * bins_usize + (ix as usize);
                        local[id] = local[id].wrapping_add(1);
                    }
                    local
                },
            )
            .reduce(
                || vec![0u32; total_bins],
                |mut a, b| {
                    for (i, v) in b.into_iter().enumerate() {
                        a[i] = a[i].wrapping_add(v);
                    }
                    a
                },
            )
    })
}

pub fn compute_histogram(
    system: &ParticleSystem,
    x_edges: &[f32],
    y_edges: &[f32],
    bins: u32,
    pool: &ThreadPool,
) -> Vec<f32> {
    let bins_usize = bins as usize;
    let x_min = x_edges[0];
    let x_max = *x_edges.last().unwrap_or(&x_min);
    let y_min = y_edges[0];
    let y_max = *y_edges.last().unwrap_or(&y_min);

    let dx = (x_max - x_min) / bins as f32;
    let dy = (y_max - y_min) / bins as f32;
    let dx_inv = if dx != 0.0 { 1.0 / dx } else { 0.0 };
    let dy_inv = if dy != 0.0 { 1.0 / dy } else { 0.0 };

    let total_bins = bins_usize * bins_usize;

    let combined_u32 =
        histogram_counts(system, x_min, y_min, dx_inv, dy_inv, bins, total_bins, pool);

    combined_u32.into_iter().map(|c| c as f32).collect()
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

fn precompute_boundary_pixels(
    bx: &[f32],
    by: &[f32],
    x_edges: &[f32],
    y_edges: &[f32],
    out_px: (u32, u32),
    sample_n: usize,
) -> Vec<(i64, i64)> {
    let (width, height) = out_px;
    let bins = std::cmp::max(sample_n, 3);
    let x_span = (x_edges.last().cloned().unwrap_or(0.0) - x_edges[0]).abs().max(1e-6);
    let y_span = (y_edges.last().cloned().unwrap_or(0.0) - y_edges[0]).abs().max(1e-6);

    let orig_n = std::cmp::max(bx.len(), 1);

    let sampled_pts: Vec<(i64, i64)> = (0..bins)
        .into_par_iter()
        .map(|k| {
            let idxf = (k as f32) * (orig_n as f32) / (bins as f32);
            let i0 = idxf.floor() as usize % orig_n;
            let i1 = (i0 + 1) % orig_n;
            let frac = idxf - idxf.floor();
            let bx_val = bx[i0] * (1.0 - frac) + bx[i1] * frac;
            let by_val = by[i0] * (1.0 - frac) + by[i1] * frac;

            let mut x = ((bx_val - x_edges[0]) / x_span * (width - 1) as f32).round() as i64;
            let mut y = height as i64
                - 1
                - ((by_val - y_edges[0]) / y_span * (height - 1) as f32).round() as i64;

            if x < 0 {
                x = 0
            }
            if y < 0 {
                y = 0
            }
            if x >= width as i64 {
                x = (width - 1) as i64
            }
            if y >= height as i64 {
                y = (height - 1) as i64
            }

            (x, y)
        })
        .collect();

    let n = sampled_pts.len();
    let mut all_pts: Vec<(i64, i64)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let a = sampled_pts[i];
            let b = sampled_pts[(i + 1) % n];
            bresenham_points(a.0, a.1, b.0, b.1)
        })
        .collect();

    all_pts.sort_unstable();
    all_pts.dedup();
    all_pts
}

fn draw_boundary(
    image: &mut RgbImage,
    bresenham_px: &[(i64, i64)],
    width: u32,
    height: u32,
    offsets: &[(i64, i64)],
    pool: &ThreadPool,
) {
    if bresenham_px.is_empty() || offsets.is_empty() {
        return;
    }

    let w = width as usize;
    let h = height as usize;
    let total = w.saturating_mul(h);

    let atoms_vec: Vec<AtomicU8> = (0..total).map(|_| AtomicU8::new(0)).collect();
    let atoms = Arc::new(atoms_vec);

    pool.install(|| {
        bresenham_px.par_iter().for_each(|&(lx, ly)| {
            for &(dx, dy) in offsets {
                let px = lx + dx;
                let py = ly + dy;

                if px < 0 || py < 0 {
                    continue;
                }

                let pxu = px as usize;
                let pyu = py as usize;

                if pxu >= w || pyu >= h {
                    continue;
                }

                let idx = pyu * w + pxu;
                atoms[idx].store(1, Ordering::Relaxed);
            }
        });
    });

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if atoms[idx].load(Ordering::Relaxed) != 0 {
                image.put_pixel(x as u32, y as u32, image::Rgb([255u8, 255u8, 255u8]));
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
}

pub struct ParticleSystem {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self { x: Vec::new(), y: Vec::new(), vx: Vec::new(), vy: Vec::new() }
    }
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            vx: Vec::with_capacity(capacity),
            vy: Vec::with_capacity(capacity),
        }
    }

    pub fn resize(&mut self, size: usize) {
        self.x.resize(size, 0.0);
        self.y.resize(size, 0.0);
        self.vx.resize(size, 0.0);
        self.vy.resize(size, 0.0);
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

const LANES: usize = 8;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

fn pow_fast(v: f32, e: f32) -> f32 {
    if approx_eq(e, 1.0) {
        v
    } else if approx_eq(e, 2.0) {
        v * v
    } else if approx_eq(e, 3.0) {
        v * v * v
    } else if approx_eq(e, 4.0) {
        let v2 = v * v;
        v2 * v2
    } else {
        v.powf(e)
    }
}

pub fn step_simd(
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

    let n = system.len();

    let mut i = 0usize;
    while i < n {
        let remaining = n - i;
        if remaining >= LANES {
            let mut ax = [0.0f32; LANES];
            let mut ay = [0.0f32; LANES];
            let mut avx = [0.0f32; LANES];
            let mut avy = [0.0f32; LANES];

            // TODO: load with SIMD
            for j in 0..LANES {
                ax[j] = system.x[i + j];
                ay[j] = system.y[i + j];
                avx[j] = system.vx[i + j];
                avy[j] = system.vy[i + j];
            }

            // TODO: parallelize with SIMD
            for j in 0..LANES {
                let px = ax[j] + avx[j] * dt;
                let py = ay[j] + avy[j] * dt;

                let xna = px.abs() * inv_a;
                let ynb = py.abs() * inv_b;
                let val = pow_fast(xna, n_exp) + pow_fast(ynb, m_exp) - 1.0;

                if val <= 0.0 {
                    system.x[i + j] = px;
                    system.y[i + j] = py;
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
                    system.x[i + j] = px;
                    system.y[i + j] = py;
                    continue;
                }

                let inv_len = 1.0 / len2.sqrt();
                let nx = df_dx * inv_len;
                let ny = df_dy * inv_len;

                let vx = avx[j];
                let vy = avy[j];
                let vxn = vx * nx + vy * ny;
                let rx = vx - 2.0 * vxn * nx;
                let ry = vy - 2.0 * vxn * ny;

                system.vx[i + j] = rx;
                system.vy[i + j] = ry;
                system.x[i + j] = px - rx * epsilon;
                system.y[i + j] = py - ry * epsilon;
            }

            i += LANES;
        } else {
            for j in i..n {
                unsafe {
                    let xj = *system.x.as_mut_ptr().add(j);
                    let yj = *system.y.as_mut_ptr().add(j);
                    let vxj = *system.vx.as_mut_ptr().add(j);
                    let vyj = *system.vy.as_mut_ptr().add(j);

                    let px = xj + vxj * dt;
                    let py = yj + vyj * dt;
                    let xna = px.abs() * inv_a;
                    let ynb = py.abs() * inv_b;
                    let val = pow_fast(xna, n_exp) + pow_fast(ynb, m_exp) - 1.0;

                    if val <= 0.0 {
                        *system.x.as_mut_ptr().add(j) = px;
                        *system.y.as_mut_ptr().add(j) = py;
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
                        *system.x.as_mut_ptr().add(j) = px;
                        *system.y.as_mut_ptr().add(j) = py;
                        continue;
                    }

                    let inv_len = 1.0 / len2.sqrt();
                    let nx = df_dx * inv_len;
                    let ny = df_dy * inv_len;

                    let vxn = vxj * nx + vyj * ny;
                    let rx = vxj - 2.0 * vxn * nx;
                    let ry = vyj - 2.0 * vxn * ny;

                    *system.vx.as_mut_ptr().add(j) = rx;
                    *system.vy.as_mut_ptr().add(j) = ry;
                    *system.x.as_mut_ptr().add(j) = px - rx * epsilon;
                    *system.y.as_mut_ptr().add(j) = py - ry * epsilon;
                }
            }
            break;
        }
    }
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
    (0..steps).for_each(|_| step_simd(system, dt, epsilon, a, b, n_exp, m_exp));
}

pub fn init_cluster(
    n: u64,
    radius: f32,
    center_x: f32,
    center_y: f32,
    vx0: f32,
    vy0: f32,
) -> ParticleSystem {
    let n_usize = n as usize;
    let mut system = ParticleSystem::with_capacity(n_usize);
    system.resize(n_usize);

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    let num_threads = std::cmp::min(num_cpus::get(), std::cmp::max(n_usize, 1));
    let chunk = (n_usize + num_threads - 1) / num_threads;

    let x_addr = system.x.as_mut_ptr() as usize;
    let y_addr = system.y.as_mut_ptr() as usize;
    let vx_addr = system.vx.as_mut_ptr() as usize;
    let vy_addr = system.vy.as_mut_ptr() as usize;

    scope(|s| {
        for t in 0..num_threads {
            let start = t * chunk;
            if start >= n_usize {
                break;
            }
            let end = (start + chunk).min(n_usize);
            let len = end - start;
            let seed0 = SEED_MULTIPLIER ^ (start as u64);

            s.spawn(move || {
                let x_ptr = x_addr as *mut f32;
                let y_ptr = y_addr as *mut f32;
                let vx_ptr = vx_addr as *mut f32;
                let vy_ptr = vy_addr as *mut f32;

                let mut seed = seed0;
                let mut i = 0usize;

                while i + 4 <= len {
                    // TODO: parallelize with SIMD
                    for k in 0..4 {
                        let s1 = splitmix64(seed);
                        seed = s1;
                        let s2 = splitmix64(seed);
                        seed = s2;

                        let u1 = (s1 as f64) / ((u64::MAX as f64) + 1.0);
                        let u2 = (s2 as f64) / ((u64::MAX as f64) + 1.0);

                        let r = radius * (u1.sqrt() as f32);
                        let theta = (u2 as f32) * 2.0 * PI;

                        unsafe {
                            let idx = start + i + k;
                            *x_ptr.add(idx) = r * theta.cos() + center_x;
                            *y_ptr.add(idx) = r * theta.sin() + center_y;
                            *vx_ptr.add(idx) = vx0;
                            *vy_ptr.add(idx) = vy0;
                        }
                    }
                    i += 4;
                }

                // TODO: parallelize remaining with SIMD
                for j in i..len {
                    let s1 = splitmix64(seed);
                    seed = s1;
                    let s2 = splitmix64(seed);
                    seed = s2;

                    let u1 = (s1 as f64) / ((u64::MAX as f64) + 1.0);
                    let u2 = (s2 as f64) / ((u64::MAX as f64) + 1.0);

                    let r = radius * (u1.sqrt() as f32);
                    let theta = (u2 as f32) * 2.0 * PI;

                    unsafe {
                        let idx = start + j;
                        *x_ptr.add(idx) = r * theta.cos() + center_x;
                        *y_ptr.add(idx) = r * theta.sin() + center_y;
                        *vx_ptr.add(idx) = vx0;
                        *vy_ptr.add(idx) = vy0;
                    }
                }
            });
        }
    });

    system
}

pub struct SimulationData {
    pub palette: Arc<Vec<[u8; 3]>>,
    pub bx: Arc<Vec<f32>>,
    pub by: Arc<Vec<f32>>,
    pub system: ParticleSystem,
    pub x_edges: Arc<Vec<f32>>,
    pub y_edges: Arc<Vec<f32>>,
    pub sim_pool: Arc<ThreadPool>,
    pub render_pool: Arc<ThreadPool>,
    pub boundary_pixels: Arc<Vec<(i64, i64)>>,
    pub pixel_bin_map: Arc<Vec<usize>>,
    pub thickness_offsets: Arc<Vec<(i64, i64)>>,
}

impl SimulationData {
    pub fn new(config: &Config, start_frame: u64) -> Self {
        let default_cpus = num_cpus::get();
        let sim_threads = config.sim_threads.unwrap_or(default_cpus);
        let render_threads =
            config.render_threads.unwrap_or(std::cmp::max(1usize, default_cpus / 2));

        let sim_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(sim_threads)
                .build()
                .expect("Failed to build sim thread pool"),
        );

        let render_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(render_threads)
                .build()
                .expect("Failed to build render thread pool"),
        );

        let palette = Arc::new(build_palette());

        let (bx, by) =
            shape_boundary(config.a, config.b, config.n_exp, config.m_exp, SHAPE_SAMPLE_POINTS);

        let (x_edges, y_edges) = histogram_edges(config.a, config.b, config.res, HISTOGRAM_FACTOR);

        let out_px = compute_out_px(config.dpi);
        let sample_n = std::cmp::max(config.res as usize, 3);
        let boundary_pixels =
            Arc::new(precompute_boundary_pixels(&bx, &by, &x_edges, &y_edges, out_px, sample_n));

        let pixel_bin_map = Arc::new(precompute_pixel_bin_map(out_px, config.res as usize));
        let thickness_offsets = Arc::new(precompute_thickness_offsets(BOUNDARY_THICKNESS));

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
            sim_pool,
            render_pool,
            boundary_pixels,
            pixel_bin_map,
            thickness_offsets,
        }
    }
}

fn compute_out_px(dpi: u32) -> (u32, u32) {
    let size = (FIG_INCHES * dpi as f32).round() as u32;
    (size, size)
}

fn build_ffmpeg_args(width: u32, height: u32, fps: u64, output_path: &str) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    args.push("-y".to_string());
    args.push("-hide_banner".to_string());
    args.push("-loglevel".to_string());
    args.push("error".to_string());
    args.push("-f".to_string());
    args.push("rawvideo".to_string());
    args.push("-pix_fmt".to_string());
    args.push("rgb24".to_string());
    args.push("-video_size".to_string());
    args.push(format!("{}x{}", width, height));
    args.push("-framerate".to_string());
    args.push(fps.to_string());
    args.push("-i".to_string());
    args.push("-".to_string());
    args.push("-vf".to_string());
    args.push("pad=ceil(iw/2)*2:ceil(ih/2)*2".to_string());
    args.push("-c:v".to_string());
    args.push("libx264".to_string());
    args.push("-crf".to_string());
    args.push("12".to_string());
    args.push("-preset".to_string());
    args.push("slow".to_string());
    args.push("-profile:v".to_string());
    args.push("high".to_string());
    args.push("-pix_fmt".to_string());
    args.push("yuv420p".to_string());
    args.push("-movflags".to_string());
    args.push("+faststart".to_string());
    args.push("-progress".to_string());
    args.push("pipe:1".to_string());
    args.push(output_path.to_string());
    args
}

pub fn run_frame_generation(
    mut sim_data: SimulationData,
    config: &Config,
    n_frames: u64,
    start_frame: u64,
    output_path: &str,
    fps: u64,
) -> Result<f64, Box<dyn Error>> {
    let (width, height) = compute_out_px(config.dpi);
    let start_time = Instant::now();
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
    let args = build_ffmpeg_args(width, height, fps, output_path);

    cmd.args(&args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());

    let mut child = cmd.spawn()?;
    let child_stdin = child.stdin.take().ok_or_else(|| {
        Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Failed to open ffmpeg stdin"))
            as Box<dyn Error>
    })?;
    let (tx, rx) = bounded::<RgbImage>(12);

    scope(|s| -> Result<(), Box<dyn Error>> {
        let writer_handle = s.spawn(move || {
            let mut writer = child_stdin;
            while let Ok(img) = rx.recv() {
                let raw = img.into_raw();
                if let Err(e) = writer.write_all(&raw) {
                    eprintln!("ffmpeg stdin write error: {}", e);
                    break;
                }
            }
            drop(writer);
        });

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

            compute_histogram_inplace(
                &sim_data.system,
                &sim_data.x_edges,
                &sim_data.y_edges,
                config.res,
                sim_data.sim_pool.as_ref(),
                &mut histogram_buf,
            );

            h_log_flat.par_iter_mut().zip(&histogram_buf).for_each(|(h, &v)| {
                *h = v + config.epsilon;
            });

            let frame_image = render(
                &h_log_flat,
                &sim_data.boundary_pixels,
                &sim_data.palette,
                (width, height),
                &sim_data.render_pool,
                &sim_data.pixel_bin_map,
                &sim_data.thickness_offsets,
            );

            if tx.send(frame_image).is_err() {
                return Err("ffmpeg writer channel send failed".into());
            }

            pb.inc(1);
        }

        drop(tx);
        let _ = writer_handle.join();
        Ok(())
    })?;

    let stdout = child.stdout.take().ok_or("Failed to capture ffmpeg stdout")?;
    let reader = BufReader::new(stdout);
    let spinner = ProgressBar::new_spinner();

    spinner
        .set_style(ProgressStyle::default_spinner().template("{prefix} {spinner} {msg}").unwrap());

    spinner.set_prefix("Generating video");
    spinner.enable_steady_tick(Duration::from_millis(100));

    let _status = scope(|s| -> Result<process::ExitStatus, Box<dyn Error>> {
        let spinner_clone = spinner.clone();

        let parser_handle = s.spawn(move || {
            let mut last_msg = String::new();
            for line_res in reader.lines() {
                if let Ok(line) = line_res {
                    if line.is_empty() {
                        continue;
                    }

                    let parts: Vec<&str> = line.split_whitespace().collect();
                    let kv_map = parse_kv_from_parts(&parts);
                    let (msg, is_end) = build_progress_msg(&kv_map);

                    if msg != last_msg {
                        spinner_clone.set_message(msg.clone());
                        last_msg = msg;
                    }

                    if is_end {
                        break;
                    }
                }
            }
        });

        let status = child.wait()?;
        let _ = parser_handle.join();
        Ok(status)
    })?;

    spinner.finish_and_clear();
    pb.finish_with_message("Frame generation complete");

    let elapsed = start_time.elapsed().as_secs_f64();

    Ok(elapsed)
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

    let mut last_msg = String::new();

    for line_res in reader.lines() {
        let line = line_res?;
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let kv_map = parse_kv_from_parts(&parts);

        let (msg, is_end) = build_progress_msg(&kv_map);
        if msg != last_msg {
            spinner.set_message(msg.clone());
            last_msg = msg;
        }

        if is_end {
            break;
        }
    }

    let status = child.wait()?;

    spinner.finish_and_clear();

    let final_pb = ProgressBar::new(total_frames);

    final_pb.set_style(
        ProgressStyle::default_bar()
            .template("{prefix} {bar:40.green/white} {pos:>7}/{len:7} {percent:>3}% ({elapsed})")
            .unwrap()
            .progress_chars("=>-"),
    );

    final_pb.set_prefix("Generating video");
    final_pb.set_position(total_frames);
    final_pb.finish_with_message("Video generation complete");

    if !status.success() {
        return Err(format!("ffmpeg failed to generate video").into());
    }

    Ok(())
}

fn try_insert_numeric(name: &str, used_indices: &mut AHashSet<u64>) {
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

    let paths: Vec<PathBuf> =
        read_dir(mp4_dir)?.filter_map(Result::ok).map(|entry| entry.path()).collect();

    let used_indices = Arc::new(Mutex::new(AHashSet::new()));

    paths.par_iter().for_each(|path| {
        if path.extension().map_or(false, |ext| ext == "mp4") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if let Ok(mut set) = used_indices.lock() {
                    try_insert_numeric(stem, &mut set);
                }
            }
        } else if path.is_dir() {
            if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
                if let Ok(mut set) = used_indices.lock() {
                    try_insert_numeric(dir_name, &mut set);
                }
            }
        }
    });

    let used_snapshot = {
        let guard = used_indices.lock().map_err(|e| format!("Mutex poisoned: {}", e))?;
        guard.clone()
    };

    for i in 1u64.. {
        if !used_snapshot.contains(&i) {
            return Ok(i);
        }
    }

    Ok(1)
}

pub fn choose_video_index() -> Result<u64, Box<dyn Error>> {
    println!("Enter video index (or Enter for next available): ");
    let mut input = String::new();
    stdin().read_line(&mut input)?;

    let input_trimmed = input.trim();

    if input_trimmed.is_empty() {
        let index = next_available_index()?;
        println!("Using next available index: {}", index);
        Ok(index)
    } else {
        input_trimmed.parse::<u64>().map_err(|e| e.into())
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
        let meta_content = read_to_string(&meta_path)?;
        from_str(&meta_content)?
    } else {
        let mut meta = Map::new();
        meta.insert("constants".to_string(), config.constants());
        meta.insert("date".to_string(), Value::String(Utc::now().to_rfc3339()));
        meta.insert("last_frame".to_string(), json!(0));
        meta.insert("compute_time".to_string(), json!(0.0));
        meta.insert("resolution".to_string(), Value::from(config.res));
        Object(meta)
    };

    let start_frame = meta.get("last_frame").and_then(Value::as_u64).unwrap_or(0);

    Ok(VideoDirs { video_dir, frames_dir, meta, start_frame })
}

pub struct VideoDirs {
    pub video_dir: PathBuf,
    pub frames_dir: PathBuf,
    pub meta: Value,
    pub start_frame: u64,
}

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

    let entries: Vec<DirEntry> = read_dir(&dirs.frames_dir)?.collect::<Result<Vec<_>, _>>()?;

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

    println!("Frames saved: {}", frame_count_res);
    println!("Date: {}", dirs.meta.get("date").unwrap_or(&Null));
    println!("Total compute time: {}s", dirs.meta.get("compute_time").unwrap_or(&Value::from(0.0)));
    println!("Video dir size: {:.2} MB", total_size_bytes_res as f64 / 1_000_000.0);
    println!("Continue generating frames? (y/n): ");

    let mut response = String::new();
    stdin().read_line(&mut response)?;

    if response.trim().to_lowercase() != "y" {
        println!("Total elapsed time: {:.2}s", program_start.elapsed().as_secs_f64());
        return Ok(());
    }

    let sim_data = SimulationData::new(&config, dirs.start_frame);

    println!("Running simulation...");

    let output_path = format!("mp4/{}.mp4", index);

    let compute_time = run_frame_generation(
        sim_data,
        &config,
        n_frames,
        dirs.start_frame,
        &output_path,
        config.fps,
    )?;

    let total_compute_time =
        dirs.meta.get("compute_time").and_then(|v| v.as_f64()).unwrap_or(0.0) + compute_time;

    let mut updated_meta = dirs.meta.clone();
    updated_meta["compute_time"] = json!(total_compute_time);
    updated_meta["last_frame"] = json!(n_frames);
    write(dirs.video_dir.join("meta.json"), to_string_pretty(&updated_meta)?)?;

    println!("Video saved to `mp4/{}`", index);
    println!("Total elapsed time: {:.2}s", program_start.elapsed().as_secs_f64());

    Ok(())
}

fn parse_kv_from_parts(parts: &[&str]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for part in parts {
        if let Some((k, v)) = part.split_once('=') {
            out.insert(k.to_string(), v.to_string());
        }
    }
    out
}

fn build_progress_msg(kv: &HashMap<String, String>) -> (String, bool) {
    let frame = kv.get("frame").cloned().unwrap_or_default();
    let speed = kv.get("speed").cloned().unwrap_or_default();
    let total_size = kv.get("total_size").cloned().unwrap_or_default();
    let progress = kv.get("progress").cloned().unwrap_or_default();

    let mut parts = Vec::new();
    if !frame.is_empty() {
        parts.push(format!("frame={}", frame));
    }
    if !total_size.is_empty() {
        parts.push(format!("size={}", total_size));
    }
    if !speed.is_empty() {
        parts.push(format!("speed={}", speed));
    }

    let msg = parts.join(" ");
    let is_end = progress == "end";
    (msg, is_end)
}

fn render(
    h_log_flat: &[f32],
    boundary_pixels: &Arc<Vec<(i64, i64)>>,
    palette: &Arc<Vec<[u8; 3]>>,
    out_px: (u32, u32),
    pool: &Arc<ThreadPool>,
    pixel_bin_map: &Arc<Vec<usize>>,
    thickness_offsets: &Arc<Vec<(i64, i64)>>,
) -> RgbImage {
    let (width, height) = out_px;
    let w = width as usize;

    let mut raw = vec![0u8; (width as usize) * (height as usize) * 3usize];

    let palette_len = palette.len();
    let max_v = h_log_flat.par_iter().cloned().reduce(|| 0.0f32, f32::max).max(0.0);
    let scale =
        if max_v > 0.0 { (palette_len as f32 - 1.0) / (max_v.sqrt() * 1.05f32) } else { 1.0 };

    let palette_ref: &[[u8; 3]] = palette.as_ref().as_slice();
    let pixel_bin_map_slice: &[usize] = pixel_bin_map.as_ref().as_slice();

    let pool_ref: &ThreadPool = pool.as_ref();

    pool_ref.install(|| {
        raw.par_chunks_mut((width as usize) * 3).enumerate().for_each(|(y, row)| {
            let base_idx = y * w;
            for (x, pixel) in row.chunks_mut(3).enumerate() {
                let idx = base_idx + x;

                if idx >= pixel_bin_map_slice.len() {
                    pixel[0] = 0;
                    pixel[1] = 0;
                    pixel[2] = 0;
                    continue;
                }

                let bin = pixel_bin_map_slice[idx];

                if bin >= h_log_flat.len() {
                    pixel[0] = 0;
                    pixel[1] = 0;
                    pixel[2] = 0;
                    continue;
                }

                let v = h_log_flat[bin].max(0.0);
                let pi = ((v.sqrt() * scale) as usize).min(palette_len - 1);
                let col = palette_ref[pi];

                pixel[0] = col[0];
                pixel[1] = col[1];
                pixel[2] = col[2];
            }
        });
    });

    let mut img =
        RgbImage::from_raw(width, height, raw).expect("Failed to construct image from raw buffer");

    let pool_for_boundary: &ThreadPool = pool.as_ref();
    draw_boundary(
        &mut img,
        &boundary_pixels,
        width,
        height,
        thickness_offsets.as_ref(),
        pool_for_boundary,
    );

    img
}

pub fn compute_histogram_inplace(
    system: &ParticleSystem,
    x_edges: &[f32],
    y_edges: &[f32],
    bins: u32,
    pool: &ThreadPool,
    out: &mut [f32],
) {
    let bins_usize = bins as usize;
    let total_bins = bins_usize * bins_usize;

    let x_min = x_edges[0];
    let x_max = *x_edges.last().unwrap_or(&x_min);
    let y_min = y_edges[0];
    let y_max = *y_edges.last().unwrap_or(&y_min);

    let dx = (x_max - x_min) / bins as f32;
    let dy = (y_max - y_min) / bins as f32;
    let dx_inv = if dx != 0.0 { 1.0 / dx } else { 0.0 };
    let dy_inv = if dy != 0.0 { 1.0 / dy } else { 0.0 };

    let combined: Vec<u32> =
        histogram_counts(system, x_min, y_min, dx_inv, dy_inv, bins, total_bins, pool);

    for (i, &c) in combined.iter().enumerate().take(total_bins) {
        out[i] = c as f32;
    }
}
