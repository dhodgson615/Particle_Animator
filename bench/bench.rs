use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

use particleanimatorrust::*;

fn bench_pow_fast(c: &mut Criterion) {
    let mut g = c.benchmark_group("pow_fast");
    let base = black_box(1.2345f32);
    for &exp in &[1.0f32, 2.0f32, 3.0f32, 4.0f32, 2.71828f32] {
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
        b.iter(|| shape_boundary(black_box(1.0f32), black_box(1.0f32), black_box(2.0f32), black_box(2.0f32), black_box(1000usize)))
    });
}

fn bench_init_cluster(c: &mut Criterion) {
    c.bench_function("init_cluster_4096", |b| {
        b.iter(|| {
            init_cluster(black_box(4096u64), black_box(0.1f32), black_box(0.0f32), black_box(0.0f32), black_box(1.0f32), black_box(0.0f32))
        })
    });
}

fn bench_step_simd_small(c: &mut Criterion) {
    let src = init_cluster(2048u64, 0.1f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32);

    c.bench_function("step_simd_2048_one_step", |b| {
        b.iter(|| {
            let mut sys = ParticleSystem {
                x: src.x.clone(),
                y: src.y.clone(),
                vx: src.vx.clone(),
                vy: src.vy.clone(),
            };
            step_simd(&mut sys, black_box(0.0001f32), black_box(1e-8f32), black_box(1.0f32), black_box(1.0f32), black_box(2.0f32), black_box(2.0f32));
            black_box(sys)
        })
    });
}

fn bench_compute_histogram(c: &mut Criterion) {
    let mut sys = init_cluster(4096u64, 0.1f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32);
    let (x_edges, y_edges) = histogram_edges(1.0f32, 1.0f32, 128, 1.25f32);
    let mut out = vec![0f32; 128 * 128];
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

    c.bench_function("compute_histogram_128", |b| {
        b.iter(|| {
            let system_copy = ParticleSystem {
                x: sys.x.clone(),
                y: sys.y.clone(),
                vx: sys.vx.clone(),
                vy: sys.vy.clone(),
            };
            compute_histogram(&system_copy, &x_edges, &y_edges, 128, &pool, &mut out.clone());
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
    let (bx, by) = shape_boundary(1.0f32, 1.0f32, 2.0f32, 2.0f32, 1000usize);
    let (x_edges, y_edges) = histogram_edges(1.0f32, 1.0f32, 256, 1.25f32);
    let out_px = compute_out_px(100);
    c.bench_function("precompute_boundary_pixels_256", |b| {
        b.iter(|| {
            precompute_boundary_pixels(&bx, &by, &x_edges, &y_edges, out_px, 256usize);
        })
    });
}

fn bench_render_small(c: &mut Criterion) {
    let pool = Arc::new(ThreadPoolBuilder::new().num_threads(2).build().unwrap());
    let palette = Arc::new(build_palette());
    let (bx, by) = shape_boundary(1.0f32, 1.0f32, 2.0f32, 2.0f32, 500usize);
    let (x_edges, y_edges) = histogram_edges(1.0f32, 1.0f32, 128, 1.25f32);
    let out_px = compute_out_px(100);
    let boundary_pixels = Arc::new(precompute_boundary_pixels(&bx, &by, &x_edges, &y_edges, out_px, 128usize));
    let pixel_bin_map = Arc::new(precompute_pixel_bin_map(out_px, 128usize));
    let thickness_offsets = Arc::new(precompute_thickness_offsets(BOUNDARY_THICKNESS));
    let mut h_log_flat = vec![1.0f32; 128 * 128];

    c.bench_function("render_128", |b| {
        b.iter(|| {
            let img = render(&h_log_flat, &boundary_pixels, &palette, out_px, &pool, &pixel_bin_map, &thickness_offsets);
            black_box(img);
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
    bench_render_small
);
criterion_main!(benches);
