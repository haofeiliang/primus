//! Tests for `PowOf2Modulus` — basic ops cross-validated against `UintModulus`.

use primus_modulus::{PowOf2Modulus, UintModulus};
use primus_reduce::prelude::*;
use rand::{RngExt, distr::Uniform, prelude::*};

const MODULUS: u32 = 16_777_216; // 2^24

#[test]
fn constructor_bounds() {
    assert!(std::panic::catch_unwind(|| PowOf2Modulus::<u32>::new(0)).is_err());
    assert!(std::panic::catch_unwind(|| PowOf2Modulus::<u32>::new(3)).is_err());
    assert!(std::panic::catch_unwind(|| PowOf2Modulus::<u32>::new(MODULUS)).is_ok());
}

#[test]
fn scalar_ops_against_uint() {
    let p = PowOf2Modulus::<u32>::new(MODULUS);
    let u = UintModulus(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..20 {
        let a: u32 = distr.sample(&mut rng);
        let b: u32 = distr.sample(&mut rng);

        assert_eq!(p.reduce_add(a, b), u.reduce_add(a, b));
        assert_eq!(p.reduce_sub(a, b), u.reduce_sub(a, b));
        assert_eq!(p.reduce_neg(a), u.reduce_neg(a));
        assert_eq!(p.reduce_double(a), u.reduce_double(a));

        let v = if rng.random_bool(0.5) {
            a
        } else {
            a.wrapping_add(MODULUS)
        };
        assert_eq!(p.reduce_once(v), u.reduce_once(v));
    }
}

#[test]
fn reduce_any_value() {
    let p = PowOf2Modulus::<u32>::new(MODULUS);
    let mut rng = rand::rng();

    for _ in 0..20 {
        let v: u32 = rng.random();
        assert_eq!(p.reduce(v), v & (MODULUS - 1));
    }
}

#[test]
fn mul_ops() {
    let p = PowOf2Modulus::<u32>::new(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..20 {
        let a: u32 = distr.sample(&mut rng);
        let b: u32 = distr.sample(&mut rng);

        let expected_mul = ((a as u64 * b as u64) % MODULUS as u64) as u32;
        assert_eq!(p.reduce_mul(a, b), expected_mul);
        assert_eq!(
            p.reduce_square(a),
            ((a as u64 * a as u64) % MODULUS as u64) as u32
        );
    }
}

#[test]
fn slice_ops_against_uint() {
    let p = PowOf2Modulus::<u32>::new(MODULUS);
    let u = UintModulus(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for &len in &[0usize, 1, 3, 7, 8, 15, 16, 17, 31, 33, 64, 65] {
        let a: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let b: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();

        for op in &["add", "sub", "neg"] {
            let mut p_res = a.clone();
            let mut u_res = a.clone();
            match *op {
                "add" => {
                    p.reduce_add_slice_assign(&mut p_res, &b);
                    u.reduce_add_slice_assign(&mut u_res, &b);
                }
                "sub" => {
                    p.reduce_sub_slice_assign(&mut p_res, &b);
                    u.reduce_sub_slice_assign(&mut u_res, &b);
                }
                "neg" => {
                    p.reduce_neg_slice_assign(&mut p_res);
                    u.reduce_neg_slice_assign(&mut u_res);
                }
                _ => {}
            }
            assert_eq!(p_res, u_res, "{op} len={len}");
        }
    }
}

#[cfg(feature = "simd")]
#[test]
fn simd_slice_ops_against_uint() {
    let p = PowOf2Modulus::<u32>::new(MODULUS);
    let u = UintModulus(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for &len in &[0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129] {
        let a: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let b: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();

        for op in &["add", "sub"] {
            let mut p_res = a.clone();
            let mut u_res = a.clone();
            match *op {
                "add" => {
                    p.reduce_add_slice_assign(&mut p_res, &b);
                    u.reduce_add_slice_assign(&mut u_res, &b);
                }
                "sub" => {
                    p.reduce_sub_slice_assign(&mut p_res, &b);
                    u.reduce_sub_slice_assign(&mut u_res, &b);
                }
                _ => {}
            }
            assert_eq!(p_res, u_res, "simd {op} len={len}");
        }
    }
}
