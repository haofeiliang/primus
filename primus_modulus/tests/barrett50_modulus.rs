//! Cross-validation tests for `Barrett50Modulus`.

use primus_modulus::{Barrett50Modulus, BarrettModulus};
use primus_reduce::prelude::*;
use rand::{distr::Uniform, prelude::*};

const MODULUS: u64 = 281_474_976_710_657; // 2^48 + 1

fn mul_mod(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % MODULUS as u128) as u64
}

#[test]
fn constructor_bounds() {
    assert!(std::panic::catch_unwind(|| Barrett50Modulus::new(0)).is_err());
    assert!(std::panic::catch_unwind(|| Barrett50Modulus::new(1u64 << 47)).is_err());
    assert!(Barrett50Modulus::try_new(1u64 << 48).is_some());
    assert!(Barrett50Modulus::try_new((1u64 << 50) - 1).is_some());
    assert!(Barrett50Modulus::try_new(1u64 << 50).is_none());
}

#[test]
fn scalar_ops_against_barrett() {
    let b50 = Barrett50Modulus::new(MODULUS);
    let barrett = BarrettModulus::<u64>::new(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..20 {
        let a: u64 = distr.sample(&mut rng);
        let b: u64 = distr.sample(&mut rng);

        assert_eq!(b50.reduce_add(a, b), barrett.reduce_add(a, b));
        assert_eq!(b50.reduce_sub(a, b), barrett.reduce_sub(a, b));
        assert_eq!(b50.reduce_neg(a), barrett.reduce_neg(a));
        assert_eq!(b50.reduce_once(a), barrett.reduce_once(a));
        assert_eq!(b50.reduce_double(a), barrett.reduce_double(a));
    }
}

#[test]
fn mul_ops() {
    let b50 = Barrett50Modulus::new(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..20 {
        let a: u64 = distr.sample(&mut rng);
        let b: u64 = distr.sample(&mut rng);
        let c: u64 = distr.sample(&mut rng);

        assert_eq!(b50.reduce_mul(a, b), mul_mod(a, b));
        assert_eq!(b50.reduce_square(a), mul_mod(a, a));

        let expected_fma = ((a as u128 * b as u128 + c as u128) % MODULUS as u128) as u64;
        assert_eq!(b50.reduce_mul_add(a, b, c), expected_fma);

        let lazy = b50.lazy_reduce_mul(a, b);
        assert!(lazy < MODULUS * 2);
        assert_eq!(b50.reduce_once(lazy), mul_mod(a, b));
    }
}

#[test]
fn dot_product() {
    let b50 = Barrett50Modulus::new(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for &len in &[0usize, 1, 7, 15, 16, 17, 31, 32, 33, 127, 128, 129] {
        let a: Vec<u64> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let b: Vec<u64> = (0..len).map(|_| distr.sample(&mut rng)).collect();

        let expected = a.iter().zip(&b).fold(0u128, |acc, (&x, &y)| {
            (acc + x as u128 * y as u128) % MODULUS as u128
        }) as u64;
        assert_eq!(
            b50.reduce_dot_product(&a, &b),
            expected,
            "dot_product len={len}"
        );
        assert_eq!(
            b50.reduce_dot_product_iter(a.iter().copied(), b.iter().copied()),
            expected,
            "iter len={len}"
        );
    }
}

#[test]
fn mul_slice_ops() {
    let b50 = Barrett50Modulus::new(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for &len in &[0usize, 1, 3, 7, 8, 15, 16, 17, 31, 33, 64, 65] {
        let a: Vec<u64> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let b: Vec<u64> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let c: Vec<u64> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let scalar: u64 = distr.sample(&mut rng);
        let expected_mul: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| mul_mod(x, y)).collect();

        let mut assign = a.clone();
        b50.reduce_mul_slice_assign(&mut assign, &b);
        assert_eq!(assign, expected_mul, "mul_slice_assign len={len}");
        let mut to = vec![0; len];
        b50.reduce_mul_slice_to(&a, &b, &mut to);
        assert_eq!(to, expected_mul, "mul_slice_to len={len}");

        let expected_scalar: Vec<u64> = a.iter().map(|&x| mul_mod(x, scalar)).collect();
        let mut assign = a.clone();
        b50.reduce_scalar_mul_slice_assign(&mut assign, scalar);
        assert_eq!(assign, expected_scalar, "scalar_mul_slice_assign len={len}");

        let mut lazy_assign = a.clone();
        b50.lazy_reduce_mul_slice_assign(&mut lazy_assign, &b);
        for v in lazy_assign.iter_mut() {
            assert!(*v < MODULUS * 2);
            *v = b50.reduce_once(*v);
        }
        assert_eq!(lazy_assign, expected_mul, "lazy_mul_slice_assign len={len}");

        let expected_acc: Vec<u64> = c
            .iter()
            .zip(&a)
            .zip(&b)
            .map(|((&acc, &x), &y)| {
                ((acc as u128 + x as u128 * y as u128) % MODULUS as u128) as u64
            })
            .collect();
        let mut acc = c.clone();
        b50.reduce_add_mul_slice_assign(&mut acc, &a, &b);
        assert_eq!(acc, expected_acc, "add_mul_slice_assign len={len}");

        let expected_sub: Vec<u64> = c
            .iter()
            .zip(&a)
            .zip(&b)
            .map(|((&acc, &x), &y)| {
                let prod = mul_mod(x, y);
                if acc >= prod {
                    acc - prod
                } else {
                    acc + MODULUS - prod
                }
            })
            .collect();
        let mut acc = c.clone();
        b50.reduce_sub_mul_slice_assign(&mut acc, &a, &b);
        assert_eq!(acc, expected_sub, "sub_mul_slice_assign len={len}");

        let expected_abc: Vec<u64> = a
            .iter()
            .zip(&b)
            .zip(&c)
            .map(|((&x, &y), &z)| ((x as u128 * y as u128 + z as u128) % MODULUS as u128) as u64)
            .collect();
        let mut out = vec![0; len];
        b50.reduce_mul_add_slice_to(&a, &b, &c, &mut out);
        assert_eq!(out, expected_abc, "mul_add_slice_to len={len}");

        let expected_sbc: Vec<u64> = b
            .iter()
            .zip(&c)
            .map(|(&y, &z)| ((scalar as u128 * y as u128 + z as u128) % MODULUS as u128) as u64)
            .collect();
        let mut out = vec![0; len];
        b50.reduce_scalar_mul_add_slice_to(scalar, &b, &c, &mut out);
        assert_eq!(out, expected_sbc, "scalar_mul_add_slice_to len={len}");
    }
}

#[test]
fn slice_ops_against_barrett() {
    let b50 = Barrett50Modulus::new(MODULUS);
    let barrett = BarrettModulus::<u64>::new(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for &len in &[0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65] {
        let a: Vec<u64> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let b: Vec<u64> = (0..len).map(|_| distr.sample(&mut rng)).collect();

        for op in &["add", "sub", "neg", "once"] {
            let in_b = match *op {
                "once" => a.iter().map(|&x| x.wrapping_add(MODULUS)).collect(),
                _ => a.clone(),
            };
            let mut b50_res = in_b.clone();
            let mut barrett_res = in_b;
            match *op {
                "add" => {
                    b50.reduce_add_slice_assign(&mut b50_res, &b);
                    barrett.reduce_add_slice_assign(&mut barrett_res, &b);
                }
                "sub" => {
                    b50.reduce_sub_slice_assign(&mut b50_res, &b);
                    barrett.reduce_sub_slice_assign(&mut barrett_res, &b);
                }
                "neg" => {
                    b50.reduce_neg_slice_assign(&mut b50_res);
                    barrett.reduce_neg_slice_assign(&mut barrett_res);
                }
                "once" => {
                    b50.reduce_once_slice_assign(&mut b50_res);
                    barrett.reduce_once_slice_assign(&mut barrett_res);
                }
                _ => {}
            }
            assert_eq!(b50_res, barrett_res, "{op} len={len}");
        }
    }
}
