use rayon::ThreadPoolBuilder;

use super::*; /* Imports say they are unused but the code doesn't compile
              without them */

#[test]
fn test_build_palette_length_and_bounds() {
    let palette = build_palette();
    assert_eq!(palette.len(), 256);
    assert_eq!(palette[0], [0u8, 0u8, 0u8]);
    assert_ne!(palette[128], [0u8, 0u8, 0u8]);
}

#[test]
fn test_precompute_pixel_bin_map_small() {
    let out_px = (4u32, 2u32);
    let map = precompute_pixel_bin_map(out_px, 2usize);
    assert_eq!(map.len(), 8);
    let mut seen = vec![false; 4];

    for &b in &map {
        assert!(b < 4);
        seen[b] = true;
    }

    assert!(seen.iter().any(|&s| s));
}

#[test]
fn test_precompute_thickness_offsets_contains_center() {
    let offs = precompute_thickness_offsets(2);
    assert!(offs.contains(&(0, 0)));
    assert!(offs.len() > 1);
}

#[test]
fn test_compute_histogram_simple() {
    let mut sys = ParticleSystem::with_capacity(4);
    sys.resize(4);
    sys.x = vec![-0.5, 0.5, -0.5, 0.5];
    sys.y = vec![-0.5, -0.5, 0.5, 0.5];
    sys.vx = vec![0.0; 4];
    sys.vy = vec![0.0; 4];

    let (x_edges, y_edges) = histogram_edges(1.0, 1.0, 2, 1.0);
    let mut out = vec![0f32; 4];

    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

    compute_histogram(&sys, &x_edges, &y_edges, 2, &pool, &mut out);

    let counts: Vec<u32> = out.iter().map(|&v| v as u32).collect();
    assert_eq!(counts.iter().sum::<u32>(), 4);
    assert!(counts.into_iter().all(|c| c >= 1));
}

#[test]
fn test_pow_fast_special_cases() {
    let v = 2.0f32;

    assert_eq!(pow_fast(v, 1.0), 2.0);
    assert_eq!(pow_fast(v, 2.0), 4.0);
    assert_eq!(pow_fast(v, 3.0), 8.0);
    assert_eq!(pow_fast(1.5, 4.0), (1.5f32 * 1.5 * 1.5 * 1.5));
}

#[test]
fn test_approx_eq() {
    assert!(approx_eq(1.000000, 1.0000005));
    assert!(!approx_eq(1.0, 1.0001));
}

#[test]
fn test_rgb_from_wavelength_out_of_range() {
    let vec = rgb_from_wavelength(1000.0, 0.8);
    let rgb = vec.to_rgb_u8();

    assert_eq!(rgb, [0, 0, 0]);
}

#[test]
fn test_rgb_from_wavelength_middle() {
    let vec = rgb_from_wavelength(550.0, 0.8);
    let rgb = vec.to_rgb_u8();

    assert!(rgb[1] > 0);
}

#[test]
fn test_compute_out_px_scaling() {
    let dpi = 100u32;
    let (w, h) = compute_out_px(dpi);

    assert_eq!(w, h);
    assert_eq!(w, (8.0 * dpi as f32).round() as u32);
}

#[test]
fn test_bresenham_points_line() {
    let pts = bresenham_points(0, 0, 3, 3);

    assert!(pts.len() >= 4);
    assert!(pts.contains(&(0, 0)));
    assert!(pts.contains(&(3, 3)));
}

#[test]
fn test_parse_kv_from_parts_and_build_progress_msg() {
    let parts = vec!["frame=1", "fps=30", "progress=end", "size=1.5MB"];
    let kv =
        parse_kv_from_parts(&parts.iter().map(|s| *s).collect::<Vec<&str>>());

    assert_eq!(kv.get("frame").map(|s| s.as_str()), Some("1"));
    assert_eq!(kv.get("fps").map(|s| s.as_str()), Some("30"));

    let (msg, is_end) = build_progress_msg(&kv);

    assert!(
        msg.contains("fps:30") || msg.contains("1.5MB") || msg.contains("time")
    );

    assert!(is_end);
}

#[test]
fn test_size_to_bytes_various() {
    assert_eq!(size_to_bytes("1024"), Some(1024));
    assert_eq!(size_to_bytes("1KB"), Some(1_000));
    assert_eq!(size_to_bytes("1KiB"), Some(1024));
    assert_eq!(size_to_bytes("1.5MB"), Some(1_500_000));
    assert_eq!(size_to_bytes("N/A"), None);
    assert_eq!(size_to_bytes(""), None);
}

#[test]
fn test_histogram_edges_lengths() {
    let (x_edges, y_edges) = histogram_edges(1.0, 2.0, 10, 1.25);
    assert_eq!(x_edges.len(), 11);
    assert_eq!(y_edges.len(), 11);
    assert!(x_edges[0] < *x_edges.last().unwrap());
    assert!(y_edges[0] < *y_edges.last().unwrap());
}

#[test]
fn test_init_cluster_basic() {
    let n = 8u64;
    let radius = 0.5f32;
    let center_x = 1.0f32;
    let center_y = -1.0f32;
    let vx0 = 0.25f32;
    let vy0 = -0.25f32;

    let sys = init_cluster(n, radius, center_x, center_y, vx0, vy0);
    assert_eq!(sys.len(), n as usize);
    assert!(sys.x.iter().all(|&xx| xx.is_finite()));
    assert!(sys.y.iter().all(|&yy| yy.is_finite()));
    assert!(sys.vx.iter().all(|&vv| (vv - vx0).abs() < 1e-6));
    assert!(sys.vy.iter().all(|&vv| (vv - vy0).abs() < 1e-6));

    for (&xx, &yy) in sys.x.iter().zip(sys.y.iter()) {
        let dx = xx - center_x;
        let dy = yy - center_y;
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist <= radius + 1e-3);
    }
}

#[test]
fn test_step_simd_reflection_single_particle() {
    let mut sys = ParticleSystem::with_capacity(1);
    sys.resize(1);

    sys.x[0] = 0.9;
    sys.y[0] = 0.0;
    sys.vx[0] = 1.0;
    sys.vy[0] = 0.0;

    step_simd(&mut sys, 0.5, 1e-6, 1.0, 1.0, 2.0, 2.0);

    assert!((sys.vx[0] + 1.0).abs() < 1e-3);
    assert!(sys.vy[0].abs() < 1e-3);
}

#[test]
fn test_precompute_boundary_pixels_simple_square() {
    let bx = vec![-0.5f32, 0.5, 0.5, -0.5];
    let by = vec![-0.5f32, -0.5, 0.5, 0.5];

    let (x_edges, y_edges) = histogram_edges(1.0, 1.0, 4, 1.0);
    let out_px = (8u32, 8u32);

    let pixels =
        precompute_boundary_pixels(&bx, &by, &x_edges, &y_edges, out_px, 4);
    assert!(!pixels.is_empty());

    let (w, h) = out_px;
    for &(px, py) in &pixels {
        assert!(px >= 0 && px < w as i64);
        assert!(py >= 0 && py < h as i64);
    }
}

#[test]
fn test_draw_boundary_sets_pixels_white() {
    let width = 4u32;
    let height = 4u32;
    let mut img = RgbImage::new(width, height);

    let bres_px = vec![(1i64, 1i64)];
    let offsets = vec![(0i64, 0i64)];

    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

    draw_boundary(&mut img, &bres_px, width, height, &offsets, &pool);

    let _idx = 1 * width + 1;
    let pixel = img.get_pixel(1, 1);
    assert_eq!(pixel[0], 255);
    assert_eq!(pixel[1], 255);
    assert_eq!(pixel[2], 255);
}

#[test]
fn test_render_basic_nonblack() {
    let out_px = (4u32, 4u32);
    let (w, h) = out_px;
    let total_pixels = (w * h) as usize;

    let h_log_flat = vec![1.0f32];
    let pixel_bin_map = Arc::new(vec![0usize; total_pixels]);

    let palette = Arc::new(build_palette());
    let boundary_pixels = Arc::new(Vec::<(i64, i64)>::new());
    let thickness_offsets = Arc::new(vec![(0i64, 0i64)]);

    let pool =
        Arc::new(ThreadPoolBuilder::new().num_threads(1).build().unwrap());

    let img = render(
        &h_log_flat,
        &boundary_pixels,
        &palette,
        out_px,
        &pool,
        &pixel_bin_map,
        &thickness_offsets,
    );

    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);

    let raw = img.into_raw();
    assert!(raw.iter().any(|&b| b != 0));
}

#[test]
fn test_simulationdata_new_small_config() {
    let config = Config {
        a:               1.0,
        b:               1.0,
        n_exp:           2.0,
        m_exp:           2.0,
        n_particles:     16,
        dt:              0.001,
        epsilon:         1e-6,
        center_x:        0.0,
        center_y:        0.0,
        radius:          0.1,
        vx0:             0.5,
        vy0:             -0.5,
        fps:             30,
        duration_s:      1,
        steps_per_frame: 10,
        res:             64,
        dpi:             50,
        sim_threads:     Some(1),
        render_threads:  Some(1),
        video_filename:  None,
    };

    let sim = SimulationData::new(&config, 0);
    assert_eq!(sim.system.len(), config.n_particles as usize);
    assert_eq!(sim.palette.len(), PALETTE_SIZE);
    assert!(sim.bx.len() > 0);
    assert!(sim.by.len() > 0);

    let out_px = compute_out_px(config.dpi);
    let expected_map_len = (out_px.0 as usize) * (out_px.1 as usize);
    assert_eq!(sim.pixel_bin_map.len(), expected_map_len);
}
