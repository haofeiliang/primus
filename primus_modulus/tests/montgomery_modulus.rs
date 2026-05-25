//! Tests for `MontgomeryModulus` — basic ops cross-validated against `UintModulus`.

use primus_modulus::{MontgomeryModulus, UintModulus};
use primus_reduce::prelude::*;
use rand::{RngExt, distr::Uniform, prelude::*};

const MODULUS: u32 = 536_813_569;

#[test]
fn constructor_bounds() {
    assert!(std::panic::catch_unwind(|| MontgomeryModulus::<u32>::new(0)).is_err());
    assert!(std::panic::catch_unwind(|| MontgomeryModulus::<u32>::new(2)).is_err());
    assert!(MontgomeryModulus::<u32>::try_new(MODULUS).is_some());
    assert!(MontgomeryModulus::<u32>::try_new(MODULUS + 1).is_none());
}

#[test]
fn n_prime_property() {
    let m = MontgomeryModulus::<u32>::new(MODULUS);
    assert_eq!(MODULUS.wrapping_mul(m.n_prime()), u32::MAX);
}

#[test]
fn to_from_montgomery() {
    let m = MontgomeryModulus::<u32>::new(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..20 {
        let v: u32 = distr.sample(&mut rng);
        assert_eq!(m.from_montgomery(m.to_montgomery(v)), v);
    }
}

#[test]
fn add_sub_neg_against_uint() {
    let m = MontgomeryModulus::<u32>::new(MODULUS);
    let u = UintModulus(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..20 {
        let a: u32 = distr.sample(&mut rng);
        let b: u32 = distr.sample(&mut rng);
        let a_m = m.to_montgomery(a);
        let b_m = m.to_montgomery(b);

        assert_eq!(
            m.from_montgomery(m.reduce_add(a_m, b_m)),
            u.reduce_add(a, b)
        );
        assert_eq!(
            m.from_montgomery(m.reduce_sub(a_m, b_m)),
            u.reduce_sub(a, b)
        );
        assert_eq!(m.from_montgomery(m.reduce_neg(a_m)), u.reduce_neg(a));
    }
}

#[test]
fn mul_ops() {
    let m = MontgomeryModulus::<u32>::new(MODULUS);
    let distr = Uniform::new(0, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..20 {
        let a: u32 = distr.sample(&mut rng);
        let b: u32 = distr.sample(&mut rng);
        let expected = ((a as u64 * b as u64) % MODULUS as u64) as u32;

        let a_m = m.to_montgomery(a);
        let b_m = m.to_montgomery(b);
        assert_eq!(m.from_montgomery(m.reduce_mul(a_m, b_m)), expected);
        assert_eq!(
            m.from_montgomery(m.reduce_once(m.lazy_reduce_mul(a_m, b_m))),
            expected
        );
    }
}

#[test]
fn exp_ops() {
    let m = MontgomeryModulus::<u32>::new(MODULUS);
    let distr = Uniform::new(1, MODULUS).unwrap();
    let mut rng = rand::rng();

    for _ in 0..10 {
        let base: u32 = distr.sample(&mut rng);
        let exp: u32 = rng.random_range(0..32);
        let pow_m = m.reduce_exp(m.to_montgomery(base), exp);

        let expected = (0..exp).fold(1u64, |acc, _| (acc * base as u64) % MODULUS as u64) as u32;
        assert_eq!(m.from_montgomery(pow_m), expected);
    }
}
