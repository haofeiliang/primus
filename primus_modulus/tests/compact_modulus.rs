//! Tests for `CompactModulus` — basic ops cross-validated against `UintModulus`.

use primus_modulus::{CompactModulus, UintModulus};
use primus_reduce::prelude::*;
use rand::{RngExt, distr::Uniform, prelude::*};

const MODULUS: u32 = 536_813_569;

#[test]
fn constructor_bounds() {
    assert!(std::panic::catch_unwind(|| CompactModulus::<u32>::new(0)).is_err());
    let limit = (1u32) << (u32::BITS - 2);
    assert_eq!(CompactModulus::new(limit - 1).0, limit - 1);
    assert!(std::panic::catch_unwind(|| CompactModulus::new(limit)).is_err());
}

#[test]
fn scalar_ops_against_uint() {
    let cm = CompactModulus::new(MODULUS);
    let u = UintModulus(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..64 {
        let a: u32 = distr.sample(&mut rng);
        let b: u32 = distr.sample(&mut rng);

        assert_eq!(cm.reduce_add(a, b), u.reduce_add(a, b));
        assert_eq!(cm.reduce_double(a), u.reduce_double(a));
        assert_eq!(cm.reduce_sub(a, b), u.reduce_sub(a, b));
        assert_eq!(cm.reduce_neg(a), u.reduce_neg(a));

        let once_input = if rng.random_bool(0.5) {
            a
        } else {
            a.wrapping_add(MODULUS)
        };
        assert_eq!(cm.reduce_once(once_input), u.reduce_once(once_input));
    }
}

#[test]
fn slice_ops_against_uint() {
    let cm = CompactModulus::new(MODULUS);
    let u = UintModulus(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for &len in &[0usize, 1, 3, 7, 8, 15, 16, 17, 31, 33, 64, 65] {
        let a: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let b: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();

        for op in &["once", "neg", "add", "sub"] {
            let a_in = match *op {
                "once" => a.iter().map(|&x| x.wrapping_add(MODULUS)).collect(),
                _ => a.clone(),
            };
            let mut cm_res = a_in.clone();
            let mut u_res = a_in;
            match *op {
                "once" => {
                    cm.reduce_once_slice_assign(&mut cm_res);
                    u.reduce_once_slice_assign(&mut u_res);
                }
                "neg" => {
                    cm.reduce_neg_slice_assign(&mut cm_res);
                    u.reduce_neg_slice_assign(&mut u_res);
                }
                "add" => {
                    cm.reduce_add_slice_assign(&mut cm_res, &b);
                    u.reduce_add_slice_assign(&mut u_res, &b);
                }
                "sub" => {
                    cm.reduce_sub_slice_assign(&mut cm_res, &b);
                    u.reduce_sub_slice_assign(&mut u_res, &b);
                }
                _ => {}
            }
            assert_eq!(cm_res, u_res, "{op} len={len}");
        }
    }
}

#[cfg(feature = "simd")]
#[test]
fn simd_slice_ops_against_uint() {
    let cm = CompactModulus::new(MODULUS);
    let u = UintModulus(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for &len in &[
        0usize, 1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ] {
        let a: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let b: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();

        for op in &["add", "sub", "neg", "once"] {
            let a_in = match *op {
                "once" => a.iter().map(|&x| x.wrapping_add(MODULUS)).collect(),
                _ => a.clone(),
            };
            let mut cm_res = a_in.clone();
            let mut u_res = a_in;
            match *op {
                "once" => {
                    cm.reduce_once_slice_assign(&mut cm_res);
                    u.reduce_once_slice_assign(&mut u_res);
                }
                "neg" => {
                    cm.reduce_neg_slice_assign(&mut cm_res);
                    u.reduce_neg_slice_assign(&mut u_res);
                }
                "add" => {
                    cm.reduce_add_slice_assign(&mut cm_res, &b);
                    u.reduce_add_slice_assign(&mut u_res, &b);
                }
                "sub" => {
                    cm.reduce_sub_slice_assign(&mut cm_res, &b);
                    u.reduce_sub_slice_assign(&mut u_res, &b);
                }
                _ => {}
            }
            assert_eq!(cm_res, u_res, "simd {op} len={len}");
        }
    }
}

#[test]
fn u128_constructor_bounds() {
    assert!(std::panic::catch_unwind(|| CompactModulus::<u128>::new(0)).is_err());
    let limit = 1u128 << (u128::BITS - 2);
    assert_eq!(CompactModulus::new(limit - 1).0, limit - 1);
    assert!(std::panic::catch_unwind(|| CompactModulus::new(limit)).is_err());
}

#[test]
fn implements_modulus_trait() {
    use primus_reduce::Modulus;
    let m = CompactModulus::new(536_813_569u32);
    assert_eq!(m.value(), Some(536_813_569));
    assert_eq!(m.minus_one(), 536_813_568);
}
