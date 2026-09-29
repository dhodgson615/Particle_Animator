use std::{
    f32::consts::E,
    fs::{OpenOptions, create_dir_all},
    hint::black_box,
    io::Write,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use particleanimatorrust::{
    BOUNDARY_THICKNESS, ParticleSystem, Vector3D, bresenham_points,
    build_palette, compute_histogram, compute_out_px, histogram_edges,
    init_cluster, pow_fast, precompute_boundary_pixels,
    precompute_pixel_bin_map, precompute_thickness_offsets, render,
    shape_boundary, step_simd,
};
use rayon::{ThreadPool, ThreadPoolBuilder};

fn bench_pow_fast(c: &mut Criterion) {
    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("pow_fast");
    let base: f32 = black_box(1.2345f32);

    for &exp in &[1.0f32, 2.0f32, 3.0f32, 4.0f32, E] {
        g.bench_with_input(BenchmarkId::from_parameter(exp), &exp, |b, &e| {
            b.iter(|| pow_fast(base, black_box(e)));
        });
    }
    g.finish();
}

fn bench_build_palette(c: &mut Criterion) {
    c.bench_function("build_palette", |b| b.iter(|| build_palette()));
}

fn bench_shape_boundary(c: &mut Criterion) {
    c.bench_function("shape_boundary_1000", |b| {
        b.iter(|| {
            shape_boundary(
                black_box(1.0f32),
                black_box(1.0f32),
                black_box(2.0f32),
                black_box(2.0f32),
                black_box(1000usize),
            )
        })
    });
}

fn bench_init_cluster(c: &mut Criterion) {
    c.bench_function("init_cluster_4096", |b| {
        b.iter(|| {
            init_cluster(
                black_box(4096u64),
                black_box(0.1f32),
                black_box(0.0f32),
                black_box(0.0f32),
                black_box(1.0f32),
                black_box(0.0f32),
            )
        })
    });
}

fn bench_step_simd_small(c: &mut Criterion) {
    let src: ParticleSystem =
        init_cluster(2048u64, 0.1f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32);

    c.bench_function("step_simd_2048_one_step", |b| {
        b.iter(|| {
            let mut sys = ParticleSystem {
                x:  src.x.clone(),
                y:  src.y.clone(),
                vx: src.vx.clone(),
                vy: src.vy.clone(),
            };

            step_simd(
                &mut sys,
                black_box(0.0001f32),
                black_box(1e-8f32),
                black_box(1.0f32),
                black_box(1.0f32),
                black_box(2.0f32),
                black_box(2.0f32),
            );
            black_box(sys)
        })
    });
}

fn bench_compute_histogram(c: &mut Criterion) {
    let sys: ParticleSystem =
        init_cluster(4096u64, 0.1f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32);

    let (x_edges, y_edges): (Vec<f32>, Vec<f32>) =
        histogram_edges(1.0f32, 1.0f32, 128, 1.25f32);

    let out: Vec<f32> = vec![0f32; 128 * 128];

    let pool: ThreadPool =
        ThreadPoolBuilder::new().num_threads(1).build().unwrap();

    c.bench_function("compute_histogram_128", |b| {
        b.iter(|| {
            let system_copy = ParticleSystem {
                x:  sys.x.clone(),
                y:  sys.y.clone(),
                vx: sys.vx.clone(),
                vy: sys.vy.clone(),
            };

            compute_histogram(
                &system_copy,
                &x_edges,
                &y_edges,
                128,
                &pool,
                &mut out.clone(),
            );

            black_box(&out);
        })
    });
}

fn bench_bresenham_points(c: &mut Criterion) {
    c.bench_function("bresenham_points_1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(bresenham_points(i as i64, 0, (i + 5) as i64, 5));
            }
        })
    });
}

fn bench_precompute_boundary_pixels(c: &mut Criterion) {
    let (bx, by): (Vec<f32>, Vec<f32>) =
        shape_boundary(1.0f32, 1.0f32, 2.0f32, 2.0f32, 1000usize);

    let (x_edges, y_edges): (Vec<f32>, Vec<f32>) =
        histogram_edges(1.0f32, 1.0f32, 256, 1.25f32);

    let out_px: (u32, u32) = compute_out_px(100);

    c.bench_function("precompute_boundary_pixels_256", |b| {
        b.iter(|| {
            precompute_boundary_pixels(
                &bx, &by, &x_edges, &y_edges, out_px, 256usize,
            );
        })
    });
}

fn bench_render_small(c: &mut Criterion) {
    let pool: Arc<ThreadPool> =
        Arc::new(ThreadPoolBuilder::new().num_threads(2).build().unwrap());

    let palette: Arc<Vec<Vector3D<u8>>> = Arc::new(build_palette());

    let (bx, by): (Vec<f32>, Vec<f32>) =
        shape_boundary(1.0f32, 1.0f32, 2.0f32, 2.0f32, 500usize);

    let (x_edges, y_edges): (Vec<f32>, Vec<f32>) =
        histogram_edges(1.0f32, 1.0f32, 128, 1.25f32);

    let out_px: (u32, u32) = compute_out_px(100);

    let boundary_pixels: Arc<Vec<(i64, i64)>> =
        Arc::new(precompute_boundary_pixels(
            &bx, &by, &x_edges, &y_edges, out_px, 128usize,
        ));

    let pixel_bin_map: Arc<Vec<usize>> =
        Arc::new(precompute_pixel_bin_map(out_px, 128usize));

    let thickness_offsets: Arc<Vec<(i64, i64)>> =
        Arc::new(precompute_thickness_offsets(BOUNDARY_THICKNESS));

    let h_log_flat: Vec<f32> = vec![1.0f32; 128 * 128];

    c.bench_function("render_128", |b| {
        b.iter(|| {
            let img: RgbImage = render(
                &h_log_flat,
                &boundary_pixels,
                &palette,
                out_px,
                &pool,
                &pixel_bin_map,
                &thickness_offsets,
            );
            black_box(img);
        })
    });
}

fn stats_from_samples(samples: &[f64]) -> (f64, f64, f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut s: Vec<f64> = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min: f64 = s.first().cloned().unwrap();
    let max: f64 = s.last().cloned().unwrap();

    let median = if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0
    };

    let mean: f64 = s.iter().sum::<f64>() / (s.len() as f64);

    let var: f64 = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
        / (s.len() as f64);

    let std: f64 = var.sqrt();
    (min, median, mean, max, std)
}

fn write_csv_header() {
    let _ = create_dir_all("bench");

    let mut f: File = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("bench/bench_report.csv")
        .expect("open report file");

    let _ = writeln!(f, "case,run_count,min_s,median_s,mean_s,max_s,std_s");
}

fn append_csv_line(
    case: &str,
    run_count: usize,
    stats: (f64, f64, f64, f64, f64),
) {
    let mut f: File = OpenOptions::new()
        .create(true)
        .append(true)
        .open("bench/bench_report.csv")
        .expect("open report file");

    let (min, median, mean, max, std): (f64, f64, f64, f64, f64) = stats;

    let _ = writeln!(
        f,
        "{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
        case, run_count, min, median, mean, max, std
    );
}

fn bench_manual_report(c: &mut Criterion) {
    c.bench_function("manual_verbose_report", |b| {
        b.iter_custom(|_iters| {
            let warmup: usize = 5usize;
            let runs: usize = 30usize;

            write_csv_header();

            {
                let case_prefix: &str = "pow_fast_manual";

                for &exp in &[1.0f32, 2.0f32, 3.0f32, 4.0f32, E] {
                    let mut samples: Vec<f64> = Vec::with_capacity(runs);

                    for _ in 0..warmup {
                        for _ in 0..10000 {
                            black_box(pow_fast(1.2345f32, exp));
                        }
                    }

                    for run_idx in 0..runs {
                        let start: Instant = Instant::now();

                        for _ in 0..10000 {
                            black_box(pow_fast(1.2345f32, exp));
                        }

                        let duration: Duration = start.elapsed();
                        let secs: f64 = duration.as_secs_f64();
                        samples.push(secs);

                        println!(
                            "{},exp={},run={} -> {:.6}s",
                            case_prefix,
                            exp,
                            run_idx + 1,
                            secs
                        );
                    }

                    let stats: (f64, f64, f64, f64, f64) =
                        stats_from_samples(&samples);

                    append_csv_line(
                        &format!("{}_exp_{:.6}", case_prefix, exp),
                        runs,
                        stats,
                    );
                }
            }

            {
                let case: &str = "build_palette_manual";
                let mut samples: Vec<f64> = Vec::with_capacity(runs);

                for _ in 0..warmup {
                    let _ = build_palette();
                }

                for run_idx in 0..runs {
                    let start: Instant = Instant::now();
                    let _ = build_palette();
                    let secs: f64 = start.elapsed().as_secs_f64();

                    samples.push(secs);
                    println!("{},run={} -> {:.6}s", case, run_idx + 1, secs);
                }

                let stats: (f64, f64, f64, f64, f64) =
                    stats_from_samples(&samples);

                append_csv_line(case, runs, stats);
            }

            {
                let case: &str = "shape_boundary_manual";
                let mut samples: Vec<f64> = Vec::with_capacity(runs);

                for _ in 0..warmup {
                    let _ = shape_boundary(
                        1.0f32, 1.0f32, 2.0f32, 2.0f32, 1000usize,
                    );
                }

                for run_idx in 0..runs {
                    let start: Instant = Instant::now();

                    let _ = shape_boundary(
                        1.0f32, 1.0f32, 2.0f32, 2.0f32, 1000usize,
                    );

                    let secs: f64 = start.elapsed().as_secs_f64();

                    samples.push(secs);
                    println!("{},run={} -> {:.6}s", case, run_idx + 1, secs);
                }

                let stats: (f64, f64, f64, f64, f64) =
                    stats_from_samples(&samples);

                append_csv_line(case, runs, stats);
            }

            {
                let case: &str = "init_cluster_manual";
                let mut samples: Vec<f64> = Vec::with_capacity(runs);

                for _ in 0..warmup {
                    let _ = init_cluster(
                        4096u64, 0.1f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32,
                    );
                }

                for run_idx in 0..runs {
                    let start: Instant = Instant::now();

                    let _ = init_cluster(
                        4096u64, 0.1f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32,
                    );

                    let secs: f64 = start.elapsed().as_secs_f64();

                    samples.push(secs);
                    println!("{},run={} -> {:.6}s", case, run_idx + 1, secs);
                }

                let stats: (f64, f64, f64, f64, f64) =
                    stats_from_samples(&samples);

                append_csv_line(case, runs, stats);
            }

            {
                let case: &str = "step_simd_manual_2048";
                let mut samples: Vec<f64> = Vec::with_capacity(runs);

                let src: ParticleSystem = init_cluster(
                    2048u64, 0.1f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32,
                );

                for _ in 0..warmup {
                    let mut sys: ParticleSystem = ParticleSystem {
                        x:  src.x.clone(),
                        y:  src.y.clone(),
                        vx: src.vx.clone(),
                        vy: src.vy.clone(),
                    };

                    step_simd(
                        &mut sys, 0.0001f32, 1e-8f32, 1.0f32, 1.0f32, 2.0f32,
                        2.0f32,
                    );
                }

                for run_idx in 0..runs {
                    let mut sys: ParticleSystem = ParticleSystem {
                        x:  src.x.clone(),
                        y:  src.y.clone(),
                        vx: src.vx.clone(),
                        vy: src.vy.clone(),
                    };

                    let start: Instant = Instant::now();

                    step_simd(
                        &mut sys, 0.0001f32, 1e-8f32, 1.0f32, 1.0f32, 2.0f32,
                        2.0f32,
                    );

                    let secs: f64 = start.elapsed().as_secs_f64();
                    samples.push(secs);
                    println!("{},run={} -> {:.6}s", case, run_idx + 1, secs);
                }

                let stats: (f64, f64, f64, f64, f64) =
                    stats_from_samples(&samples);

                append_csv_line(case, runs, stats);
            }

            {
                let case: &str = "compute_histogram_manual_128";
                let mut samples: Vec<f64> = Vec::with_capacity(runs);

                let sys: ParticleSystem = init_cluster(
                    4096u64, 0.1f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32,
                );

                let (x_edges, y_edges): (Vec<f32>, Vec<f32>) =
                    histogram_edges(1.0f32, 1.0f32, 128, 1.25f32);

                let out: Vec<f32> = vec![0f32; 128 * 128];

                let pool: ThreadPool =
                    ThreadPoolBuilder::new().num_threads(1).build().unwrap();

                for _ in 0..warmup {
                    let system_copy: ParticleSystem = ParticleSystem {
                        x:  sys.x.clone(),
                        y:  sys.y.clone(),
                        vx: sys.vx.clone(),
                        vy: sys.vy.clone(),
                    };

                    compute_histogram(
                        &system_copy,
                        &x_edges,
                        &y_edges,
                        128,
                        &pool,
                        &mut out.clone(),
                    );
                }

                for run_idx in 0..runs {
                    let system_copy: ParticleSystem = ParticleSystem {
                        x:  sys.x.clone(),
                        y:  sys.y.clone(),
                        vx: sys.vx.clone(),
                        vy: sys.vy.clone(),
                    };

                    let start: Instant = Instant::now();

                    compute_histogram(
                        &system_copy,
                        &x_edges,
                        &y_edges,
                        128,
                        &pool,
                        &mut out.clone(),
                    );

                    let secs: f64 = start.elapsed().as_secs_f64();

                    samples.push(secs);
                    println!("{},run={} -> {:.6}s", case, run_idx + 1, secs);
                }

                let stats: (f64, f64, f64, f64, f64) =
                    stats_from_samples(&samples);

                append_csv_line(case, runs, stats);
            }

            {
                let case: &str = "bresenham_points_manual_1000";
                let mut samples: Vec<f64> = Vec::with_capacity(runs);

                for _ in 0..warmup {
                    for i in 0..1000 {
                        black_box(bresenham_points(
                            i as i64,
                            0,
                            (i + 5) as i64,
                            5,
                        ));
                    }
                }

                for run_idx in 0..runs {
                    let start: Instant = Instant::now();

                    for i in 0..1000 {
                        black_box(bresenham_points(
                            i as i64,
                            0,
                            (i + 5) as i64,
                            5,
                        ));
                    }

                    let secs: f64 = start.elapsed().as_secs_f64();
                    samples.push(secs);
                    println!("{},run={} -> {:.6}s", case, run_idx + 1, secs);
                }

                let stats: (f64, f64, f64, f64, f64) =
                    stats_from_samples(&samples);

                append_csv_line(case, runs, stats);
            }

            {
                let case: &str = "precompute_boundary_pixels_manual_256";
                let mut samples: Vec<f64> = Vec::with_capacity(runs);

                let (bx, by): (Vec<f32>, Vec<f32>) =
                    shape_boundary(1.0f32, 1.0f32, 2.0f32, 2.0f32, 1000usize);

                let (x_edges, y_edges): (Vec<f32>, Vec<f32>) =
                    histogram_edges(1.0f32, 1.0f32, 256, 1.25f32);

                let out_px: (u32, u32) = compute_out_px(100);

                for _ in 0..warmup {
                    let _ = precompute_boundary_pixels(
                        &bx, &by, &x_edges, &y_edges, out_px, 256usize,
                    );
                }

                for run_idx in 0..runs {
                    let start: Instant = Instant::now();

                    let _ = precompute_boundary_pixels(
                        &bx, &by, &x_edges, &y_edges, out_px, 256usize,
                    );

                    let secs: f64 = start.elapsed().as_secs_f64();
                    samples.push(secs);
                    println!("{},run={} -> {:.6}s", case, run_idx + 1, secs);
                }

                let stats: (f64, f64, f64, f64, f64) =
                    stats_from_samples(&samples);

                append_csv_line(case, runs, stats);
            }

            {
                let case: &str = "render_manual_128";
                let mut samples: Vec<f64> = Vec::with_capacity(runs);

                let pool: Arc<ThreadPool> = Arc::new(
                    ThreadPoolBuilder::new().num_threads(2).build().unwrap(),
                );

                let palette: Arc<Vec<Vector3D<u8>>> = Arc::new(build_palette());

                let (bx, by): (Vec<f32>, Vec<f32>) =
                    shape_boundary(1.0f32, 1.0f32, 2.0f32, 2.0f32, 500usize);

                let (x_edges, y_edges): (Vec<f32>, Vec<f32>) =
                    histogram_edges(1.0f32, 1.0f32, 128, 1.25f32);

                let out_px: (u32, u32) = compute_out_px(100);

                let boundary_pixels: Arc<Vec<(i64, i64)>> =
                    Arc::new(precompute_boundary_pixels(
                        &bx, &by, &x_edges, &y_edges, out_px, 128usize,
                    ));

                let pixel_bin_map: Arc<Vec<usize>> =
                    Arc::new(precompute_pixel_bin_map(out_px, 128usize));

                let thickness_offsets: Arc<Vec<(i64, i64)>> =
                    Arc::new(precompute_thickness_offsets(BOUNDARY_THICKNESS));

                let h_log_flat: Vec<f32> = vec![1.0f32; 128 * 128];

                for _ in 0..warmup {
                    let _ = render(
                        &h_log_flat,
                        &boundary_pixels,
                        &palette,
                        out_px,
                        &pool,
                        &pixel_bin_map,
                        &thickness_offsets,
                    );
                }

                for run_idx in 0..runs {
                    let start: Instant = Instant::now();

                    let _ = render(
                        &h_log_flat,
                        &boundary_pixels,
                        &palette,
                        out_px,
                        &pool,
                        &pixel_bin_map,
                        &thickness_offsets,
                    );

                    let secs = start.elapsed().as_secs_f64();
                    samples.push(secs);
                    println!("{},run={} -> {:.6}s", case, run_idx + 1, secs);
                }

                let stats = stats_from_samples(&samples);
                append_csv_line(case, runs, stats);
            }

            Duration::from_secs(0)
        })
    });
}

criterion_group!(
    benches,
    bench_pow_fast,
    bench_build_palette,
    bench_shape_boundary,
    bench_init_cluster,
    bench_step_simd_small,
    bench_compute_histogram,
    bench_bresenham_points,
    bench_precompute_boundary_pixels,
    bench_render_small,
    bench_manual_report
);
criterion_main!(benches);
