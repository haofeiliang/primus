//! Tests for `UintModulus` — scalar ops, slice ops, and (when the
//! `simd` feature is enabled) SIMD slice ops.

// ===========================================================================
// Constrained modulus tests (modulus < 2^{BITS-1})
// ===========================================================================

mod constrained {
    use primus_modulus::UintModulus;
    use primus_reduce::prelude::*;
    use rand::{RngExt, distr::Uniform, prelude::*};

    const MODULUS: u32 = 536_813_569;

    fn wide_add(a: u32, b: u32) -> u32 {
        let s = a as u64 + b as u64;
        if s >= MODULUS as u64 {
            (s - MODULUS as u64) as u32
        } else {
            s as u32
        }
    }
    fn wide_sub(a: u32, b: u32) -> u32 {
        if a >= b {
            a - b
        } else {
            (a as u64 + MODULUS as u64 - b as u64) as u32
        }
    }
    fn wide_neg(v: u32) -> u32 {
        if v == 0 { 0 } else { MODULUS - v }
    }
    fn wide_once(v: u32) -> u32 {
        if v >= MODULUS { v - MODULUS } else { v }
    }

    #[test]
    fn scalar_ops() {
        let m = UintModulus(MODULUS);
        let distr = Uniform::new(0, MODULUS).unwrap();
        let mut rng = rand::rng();

        for _ in 0..20 {
            let a: u32 = distr.sample(&mut rng);
            let b: u32 = distr.sample(&mut rng);

            assert_eq!(m.reduce_add(a, b), wide_add(a, b));
            let mut assign = a;
            m.reduce_add_assign(&mut assign, b);
            assert_eq!(assign, wide_add(a, b));

            assert_eq!(m.reduce_sub(a, b), wide_sub(a, b));
            let mut sub_assign = a;
            m.reduce_sub_assign(&mut sub_assign, b);
            assert_eq!(sub_assign, wide_sub(a, b));

            let v = if rng.random_bool(0.5) {
                a
            } else {
                a.wrapping_add(MODULUS)
            };
            assert_eq!(m.reduce_once(v), wide_once(v));
            let mut once_assign = v;
            m.reduce_once_assign(&mut once_assign);
            assert_eq!(once_assign, wide_once(v));

            assert_eq!(m.reduce_neg(a), wide_neg(a));
            let mut neg_assign = a;
            m.reduce_neg_assign(&mut neg_assign);
            assert_eq!(neg_assign, wide_neg(a));
        }
    }

    #[test]
    fn slice_ops() {
        let m = UintModulus(MODULUS);
        let distr = Uniform::new(0, MODULUS).unwrap();
        let mut rng = rand::rng();

        for &len in &[0usize, 1, 3, 7, 8, 15, 16, 17, 31, 33, 64, 65] {
            let a: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
            let b: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();

            let once_in: Vec<u32> = a
                .iter()
                .map(|&x| {
                    if rng.random_bool(0.5) {
                        x
                    } else {
                        x.wrapping_add(MODULUS)
                    }
                })
                .collect();
            let expected_once: Vec<u32> = once_in.iter().map(|&x| wide_once(x)).collect();

            let mut assign = once_in.clone();
            m.reduce_once_slice_assign(&mut assign);
            assert_eq!(assign, expected_once, "once_slice_assign len={len}");
            let mut to = vec![0; len];
            m.reduce_once_slice_to(&once_in, &mut to);
            assert_eq!(to, expected_once, "once_slice_to len={len}");

            let expected_neg: Vec<u32> = a.iter().map(|&x| wide_neg(x)).collect();
            let mut assign = a.clone();
            m.reduce_neg_slice_assign(&mut assign);
            assert_eq!(assign, expected_neg, "neg_slice_assign len={len}");
            let mut to = vec![0; len];
            m.reduce_neg_slice_to(&a, &mut to);
            assert_eq!(to, expected_neg, "neg_slice_to len={len}");

            let expected_add: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| wide_add(x, y)).collect();
            let mut assign = a.clone();
            m.reduce_add_slice_assign(&mut assign, &b);
            assert_eq!(assign, expected_add, "add_slice_assign len={len}");
            let mut to = vec![0; len];
            m.reduce_add_slice_to(&a, &b, &mut to);
            assert_eq!(to, expected_add, "add_slice_to len={len}");

            let expected_sub: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| wide_sub(x, y)).collect();
            let mut assign = a.clone();
            m.reduce_sub_slice_assign(&mut assign, &b);
            assert_eq!(assign, expected_sub, "sub_slice_assign len={len}");
            let mut to = vec![0; len];
            m.reduce_sub_slice_to(&a, &b, &mut to);
            assert_eq!(to, expected_sub, "sub_slice_to len={len}");

            let mut rev = b.clone();
            m.reduce_sub_slice_rev_assign(&a, &mut rev);
            assert_eq!(rev, expected_sub, "sub_slice_rev_assign len={len}");
        }
    }

    #[test]
    fn cross_validate_against_barrett() {
        use primus_modulus::BarrettModulus;

        let uint = UintModulus(MODULUS);
        let barrett = BarrettModulus::<u32>::new(MODULUS);
        let distr = Uniform::new(0, MODULUS).unwrap();
        let mut rng = rand::rng();

        for &len in &[8, 16, 32, 64] {
            let a: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
            let b: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();

            for op in &["add", "sub", "neg", "once"] {
                let mut uint_res = a.clone();
                let mut barrett_res = a.clone();
                match *op {
                    "add" => {
                        uint.reduce_add_slice_assign(&mut uint_res, &b);
                        barrett.reduce_add_slice_assign(&mut barrett_res, &b);
                    }
                    "sub" => {
                        uint.reduce_sub_slice_assign(&mut uint_res, &b);
                        barrett.reduce_sub_slice_assign(&mut barrett_res, &b);
                    }
                    "neg" => {
                        uint.reduce_neg_slice_assign(&mut uint_res);
                        barrett.reduce_neg_slice_assign(&mut barrett_res);
                    }
                    "once" => {
                        let once_in: Vec<u32> =
                            a.iter().map(|&x| x.wrapping_add(MODULUS)).collect();
                        uint_res = once_in.clone();
                        barrett_res = once_in.clone();
                        uint.reduce_once_slice_assign(&mut uint_res);
                        barrett.reduce_once_slice_assign(&mut barrett_res);
                    }
                    _ => {}
                }
                assert_eq!(uint_res, barrett_res, "cross {op} len={len}");
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_slice_ops() {
        let m = UintModulus(MODULUS);
        let distr = Uniform::new(0, MODULUS).unwrap();
        let mut rng = rand::rng();

        for &len in &[
            0usize, 1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
        ] {
            let a: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
            let b: Vec<u32> = (0..len).map(|_| distr.sample(&mut rng)).collect();

            let mut add = a.clone();
            m.reduce_add_slice_assign(&mut add, &b);
            let add_exp: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| wide_add(x, y)).collect();
            assert_eq!(add, add_exp, "simd add len={len}");

            let mut sub = a.clone();
            m.reduce_sub_slice_assign(&mut sub, &b);
            let sub_exp: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| wide_sub(x, y)).collect();
            assert_eq!(sub, sub_exp, "simd sub len={len}");
        }
    }
}

// ===========================================================================
// Arbitrary modulus — exercises the full-range code path (modulus >= 2^{BITS-1})
// ===========================================================================

mod arbitrary_u64 {
    use primus_modulus::UintModulus;
    use primus_reduce::prelude::*;

    const MODULUS: u64 = 18_446_744_073_709_551_557;

    fn wide_add(a: u64, b: u64) -> u64 {
        let s = a as u128 + b as u128;
        if s >= MODULUS as u128 {
            (s - MODULUS as u128) as u64
        } else {
            s as u64
        }
    }
    fn wide_sub(a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            (a as u128 + MODULUS as u128 - b as u128) as u64
        }
    }
    fn wide_neg(v: u64) -> u64 {
        if v == 0 { 0 } else { MODULUS - v }
    }
    fn wide_once(v: u64) -> u64 {
        if v >= MODULUS { v - MODULUS } else { v }
    }

    fn residues() -> Vec<u64> {
        vec![0, 1, MODULUS / 2 - 1, MODULUS / 2, MODULUS - 2, MODULUS - 1]
    }
    fn once_values() -> Vec<u64> {
        vec![0, 1, MODULUS - 1, MODULUS, u64::MAX - 1, u64::MAX]
    }

    #[test]
    fn scalar_ops_support_large_modulus() {
        let m = UintModulus::new(MODULUS);

        for a in residues() {
            assert_eq!(m.reduce_double(a), wide_add(a, a), "double {a:?}");
            let mut double_assign = a;
            m.reduce_double_assign(&mut double_assign);
            assert_eq!(double_assign, wide_add(a, a));

            assert_eq!(m.reduce_neg(a), wide_neg(a), "neg {a:?}");
            let mut neg_assign = a;
            m.reduce_neg_assign(&mut neg_assign);
            assert_eq!(neg_assign, wide_neg(a));

            for b in residues() {
                assert_eq!(m.reduce_add(a, b), wide_add(a, b), "add");
                let mut add_assign = a;
                m.reduce_add_assign(&mut add_assign, b);
                assert_eq!(add_assign, wide_add(a, b));

                assert_eq!(m.reduce_sub(a, b), wide_sub(a, b), "sub");
                let mut sub_assign = a;
                m.reduce_sub_assign(&mut sub_assign, b);
                assert_eq!(sub_assign, wide_sub(a, b));
            }
        }

        for value in once_values() {
            assert_eq!(m.reduce_once(value), wide_once(value), "once {value:?}");
            let mut once_assign = value;
            m.reduce_once_assign(&mut once_assign);
            assert_eq!(once_assign, wide_once(value));
        }
    }

    #[test]
    fn slice_ops_support_large_modulus() {
        let m = UintModulus::new(MODULUS);
        let a = residues();
        let mut b = residues();
        b.reverse();

        let expected_add: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| wide_add(x, y)).collect();
        let mut add_assign = a.clone();
        m.reduce_add_slice_assign(&mut add_assign, &b);
        assert_eq!(add_assign, expected_add);
        let mut add_to = vec![0; a.len()];
        m.reduce_add_slice_to(&a, &b, &mut add_to);
        assert_eq!(add_to, expected_add);

        let expected_sub: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| wide_sub(x, y)).collect();
        let mut sub_assign = a.clone();
        m.reduce_sub_slice_assign(&mut sub_assign, &b);
        assert_eq!(sub_assign, expected_sub);

        let once_input = once_values();
        let expected_once: Vec<u64> = once_input.iter().map(|&x| wide_once(x)).collect();
        let mut once_assign = once_input.clone();
        m.reduce_once_slice_assign(&mut once_assign);
        assert_eq!(once_assign, expected_once);

        let mut once_to = vec![0; once_input.len()];
        m.reduce_once_slice_to(&once_input, &mut once_to);
        assert_eq!(once_to, expected_once);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_slice_ops_support_large_modulus() {
        let m = UintModulus::new(MODULUS);
        let seed = residues();

        for &len in &[0usize, 1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
            let a: Vec<u64> = (0..len).map(|i| seed[i % seed.len()]).collect();
            let b: Vec<u64> = (0..len)
                .map(|i| seed[seed.len() - 1 - (i % seed.len())])
                .collect();

            let mut add_assign = a.clone();
            m.reduce_add_slice_assign(&mut add_assign, &b);
            let expected_add: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| wide_add(x, y)).collect();
            assert_eq!(add_assign, expected_add, "simd add len={len}");

            let mut sub_assign = a.clone();
            m.reduce_sub_slice_assign(&mut sub_assign, &b);
            let expected_sub: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| wide_sub(x, y)).collect();
            assert_eq!(sub_assign, expected_sub, "simd sub len={len}");
        }
    }
}
