use super::*;
use rayon::ThreadPoolBuilder; /* Imports say they are unused but the code
                                 doesn't compile without them */

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
    assert!(pts.contains(&(0,0)));
    assert!(pts.contains(&(3,3)));
}

#[test]
fn test_parse_kv_from_parts_and_build_progress_msg() {
    let parts = vec!["frame=1", "fps=30", "progress=end", "size=1.5MB"];
    let kv = parse_kv_from_parts(&parts.iter().map(|s| *s).collect::<Vec<&str>>());
    assert_eq!(kv.get("frame").map(|s| s.as_str()), Some("1"));
    assert_eq!(kv.get("fps").map(|s| s.as_str()), Some("30"));

    let (msg, is_end) = build_progress_msg(&kv);
    assert!(msg.contains("fps:30") || msg.contains("1.5MB") || msg.contains("time"));
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
