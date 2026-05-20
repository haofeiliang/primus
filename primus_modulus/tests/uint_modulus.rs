//! Tests for `UintModulus` — scalar ops, scalar slice ops, and (when the
//! `simd` feature is enabled) SIMD slice ops.
//!
//! Every test cross-checks results against a reference implementation computed
//! with wide-integer arithmetic so that the test is independent of the
//! Barrett / Shoup code paths.

// ===========================================================================
// Test macro — generates per-type scalar + slice tests.
// ===========================================================================

macro_rules! test_uint_modulus {
    ($ty:ty, $wide:ty, $modulus_val:expr, $mod_name:ident) => {
        mod $mod_name {
            use primus_modulus::UintModulus;
            use primus_reduce::prelude::*;
            use rand::{RngExt, distr::Uniform, prelude::*};

            const MODULUS: $ty = $modulus_val;

            fn wide_add(a: $ty, b: $ty) -> $ty {
                let s = a as $wide + b as $wide;
                if s >= MODULUS as $wide {
                    (s - MODULUS as $wide) as $ty
                } else {
                    s as $ty
                }
            }

            fn wide_sub(a: $ty, b: $ty) -> $ty {
                if a >= b {
                    a - b
                } else {
                    (a as $wide + MODULUS as $wide - b as $wide) as $ty
                }
            }

            fn wide_neg(v: $ty) -> $ty {
                if v == 0 { 0 } else { MODULUS - v }
            }

            fn wide_once(v: $ty) -> $ty {
                if v >= MODULUS { v - MODULUS } else { v }
            }

            // -----------------------------------------------------------------
            // Scalar (per-element) tests
            // -----------------------------------------------------------------

            #[test]
            fn scalar_ops() {
                let m = UintModulus(MODULUS);
                let distr = Uniform::new(0, MODULUS).unwrap();
                let mut rng = rand::rng();

                for _ in 0..20 {
                    let a: $ty = distr.sample(&mut rng);
                    let b: $ty = distr.sample(&mut rng);

                    // reduce_add
                    {
                        let expected = wide_add(a, b);
                        assert_eq!(m.reduce_add(a, b), expected);
                        let mut assign = a;
                        m.reduce_add_assign(&mut assign, b);
                        assert_eq!(assign, expected);
                    }

                    // reduce_sub
                    {
                        let expected = wide_sub(a, b);
                        assert_eq!(m.reduce_sub(a, b), expected);
                        let mut assign = a;
                        m.reduce_sub_assign(&mut assign, b);
                        assert_eq!(assign, expected);
                    }

                    // reduce_once
                    {
                        let v = if rng.random_bool(0.5) {
                            a
                        } else {
                            a.wrapping_add(MODULUS)
                        };
                        let expected = wide_once(v);
                        let mut assign = v;
                        m.reduce_once_assign(&mut assign);
                        assert_eq!(assign, expected);
                        assert_eq!(m.reduce_once(v), expected);
                    }

                    // reduce_neg
                    {
                        let expected = wide_neg(a);
                        assert_eq!(m.reduce_neg(a), expected);
                        let mut assign = a;
                        m.reduce_neg_assign(&mut assign);
                        assert_eq!(assign, expected);
                    }
                }
            }

            // -----------------------------------------------------------------
            // Scalar slice tests — exercises both in-place and out-of-place
            // forms with a range of lengths to hit tail handling.
            // -----------------------------------------------------------------

            #[test]
            fn slice_ops() {
                let m = UintModulus(MODULUS);
                let distr = Uniform::new(0, MODULUS).unwrap();
                let mut rng = rand::rng();

                for &len in &[0usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 65, 67] {
                    let a: Vec<$ty> = (0..len).map(|_| distr.sample(&mut rng)).collect();
                    let b: Vec<$ty> = (0..len).map(|_| distr.sample(&mut rng)).collect();

                    // ReduceOnceSlice
                    {
                        let once_input: Vec<$ty> = a
                            .iter()
                            .map(|&x| {
                                if rng.random_bool(0.5) {
                                    x
                                } else {
                                    x.wrapping_add(MODULUS)
                                }
                            })
                            .collect();
                        let expected: Vec<$ty> = once_input.iter().map(|&x| wide_once(x)).collect();

                        let mut assign = once_input.clone();
                        m.reduce_once_slice_assign(&mut assign);
                        assert_eq!(assign, expected, "reduce_once_slice_assign len={len}");

                        let mut to = vec![0; len];
                        m.reduce_once_slice_to(&once_input, &mut to);
                        assert_eq!(to, expected, "reduce_once_slice_to len={len}");
                    }

                    // ReduceNegSlice
                    {
                        let expected: Vec<$ty> = a.iter().map(|&x| wide_neg(x)).collect();

                        let mut assign = a.clone();
                        m.reduce_neg_slice_assign(&mut assign);
                        assert_eq!(assign, expected, "reduce_neg_slice_assign len={len}");

                        let mut to = vec![0; len];
                        m.reduce_neg_slice_to(&a, &mut to);
                        assert_eq!(to, expected, "reduce_neg_slice_to len={len}");
                    }

                    // ReduceAddSlice
                    {
                        let expected: Vec<$ty> =
                            a.iter().zip(&b).map(|(&x, &y)| wide_add(x, y)).collect();

                        let mut assign = a.clone();
                        m.reduce_add_slice_assign(&mut assign, &b);
                        assert_eq!(assign, expected, "reduce_add_slice_assign len={len}");

                        let mut to = vec![0; len];
                        m.reduce_add_slice_to(&a, &b, &mut to);
                        assert_eq!(to, expected, "reduce_add_slice_to len={len}");
                    }

                    // ReduceSubSlice
                    {
                        let expected: Vec<$ty> =
                            a.iter().zip(&b).map(|(&x, &y)| wide_sub(x, y)).collect();

                        let mut assign = a.clone();
                        m.reduce_sub_slice_assign(&mut assign, &b);
                        assert_eq!(assign, expected, "reduce_sub_slice_assign len={len}");

                        let mut to = vec![0; len];
                        m.reduce_sub_slice_to(&a, &b, &mut to);
                        assert_eq!(to, expected, "reduce_sub_slice_to len={len}");

                        let mut rev = b.clone();
                        m.reduce_sub_slice_rev_assign(&a, &mut rev);
                        assert_eq!(rev, expected, "reduce_sub_slice_rev_assign len={len}");
                    }
                }
            }

            // -----------------------------------------------------------------
            // SIMD slice tests — only compiled when `simd` feature is on.
            // Exercises the same API with heavier lane-boundary coverage.
            // -----------------------------------------------------------------

            #[cfg(feature = "simd")]
            #[test]
            fn simd_slice_ops() {
                let m = UintModulus(MODULUS);
                let distr = Uniform::new(0, MODULUS).unwrap();
                let mut rng = rand::rng();

                for &len in &[
                    0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128,
                    129, 255, 256, 257, 511, 512, 513,
                ] {
                    let a: Vec<$ty> = (0..len).map(|_| distr.sample(&mut rng)).collect();
                    let b: Vec<$ty> = (0..len).map(|_| distr.sample(&mut rng)).collect();

                    // ReduceOnceSlice (SIMD)
                    {
                        let once_input: Vec<$ty> = a
                            .iter()
                            .map(|&x| {
                                if rng.random_bool(0.5) {
                                    x
                                } else {
                                    x.wrapping_add(MODULUS)
                                }
                            })
                            .collect();
                        let expected: Vec<$ty> = once_input.iter().map(|&x| wide_once(x)).collect();

                        let mut assign = once_input.clone();
                        m.reduce_once_slice_assign(&mut assign);
                        assert_eq!(assign, expected, "simd reduce_once_slice_assign len={len}");

                        let mut to = vec![0; len];
                        m.reduce_once_slice_to(&once_input, &mut to);
                        assert_eq!(to, expected, "simd reduce_once_slice_to len={len}");
                    }

                    // ReduceNegSlice (SIMD)
                    {
                        let expected: Vec<$ty> = a.iter().map(|&x| wide_neg(x)).collect();
                        let mut assign = a.clone();
                        m.reduce_neg_slice_assign(&mut assign);
                        assert_eq!(assign, expected, "simd reduce_neg_slice_assign len={len}");

                        let mut to = vec![0; len];
                        m.reduce_neg_slice_to(&a, &mut to);
                        assert_eq!(to, expected, "simd reduce_neg_slice_to len={len}");
                    }

                    // ReduceAddSlice (SIMD)
                    {
                        let expected: Vec<$ty> =
                            a.iter().zip(&b).map(|(&x, &y)| wide_add(x, y)).collect();

                        let mut assign = a.clone();
                        m.reduce_add_slice_assign(&mut assign, &b);
                        assert_eq!(assign, expected, "simd reduce_add_slice_assign len={len}");

                        let mut to = vec![0; len];
                        m.reduce_add_slice_to(&a, &b, &mut to);
                        assert_eq!(to, expected, "simd reduce_add_slice_to len={len}");
                    }

                    // ReduceSubSlice (SIMD)
                    {
                        let expected: Vec<$ty> =
                            a.iter().zip(&b).map(|(&x, &y)| wide_sub(x, y)).collect();

                        let mut assign = a.clone();
                        m.reduce_sub_slice_assign(&mut assign, &b);
                        assert_eq!(assign, expected, "simd reduce_sub_slice_assign len={len}");

                        let mut to = vec![0; len];
                        m.reduce_sub_slice_to(&a, &b, &mut to);
                        assert_eq!(to, expected, "simd reduce_sub_slice_to len={len}");

                        let mut rev = b.clone();
                        m.reduce_sub_slice_rev_assign(&a, &mut rev);
                        assert_eq!(rev, expected, "simd reduce_sub_slice_rev_assign len={len}");
                    }
                }
            }

            // -----------------------------------------------------------------
            // Cross-validation against BarrettModulus.
            // -----------------------------------------------------------------

            #[test]
            fn cross_validate_against_barrett() {
                use primus_modulus::BarrettModulus;

                let uint_m = UintModulus(MODULUS);
                let barrett_m = BarrettModulus::<$ty>::new(MODULUS);

                let distr = Uniform::new(0, MODULUS).unwrap();
                let mut rng = rand::rng();

                for &len in &[8, 16, 32, 64, 100, 128] {
                    let a: Vec<$ty> = (0..len).map(|_| distr.sample(&mut rng)).collect();
                    let b: Vec<$ty> = (0..len).map(|_| distr.sample(&mut rng)).collect();

                    // add
                    {
                        let mut uint_res = a.clone();
                        uint_m.reduce_add_slice_assign(&mut uint_res, &b);
                        let mut barrett_res = a.clone();
                        barrett_m.reduce_add_slice_assign(&mut barrett_res, &b);
                        assert_eq!(uint_res, barrett_res, "cross add len={len}");
                    }

                    // sub
                    {
                        let mut uint_res = a.clone();
                        uint_m.reduce_sub_slice_assign(&mut uint_res, &b);
                        let mut barrett_res = a.clone();
                        barrett_m.reduce_sub_slice_assign(&mut barrett_res, &b);
                        assert_eq!(uint_res, barrett_res, "cross sub len={len}");
                    }

                    // once
                    {
                        let once_input: Vec<$ty> =
                            a.iter().map(|&x| x.wrapping_add(MODULUS)).collect();
                        let mut uint_res = once_input.clone();
                        uint_m.reduce_once_slice_assign(&mut uint_res);
                        let mut barrett_res = once_input.clone();
                        barrett_m.reduce_once_slice_assign(&mut barrett_res);
                        assert_eq!(uint_res, barrett_res, "cross once len={len}");
                    }

                    // neg
                    {
                        let mut uint_res = a.clone();
                        uint_m.reduce_neg_slice_assign(&mut uint_res);
                        let mut barrett_res = a.clone();
                        barrett_m.reduce_neg_slice_assign(&mut barrett_res);
                        assert_eq!(uint_res, barrett_res, "cross neg len={len}");
                    }
                }
            }
        }
    };
}

// ===========================================================================
// Instantiate tests for each primitive width.
// ===========================================================================

test_uint_modulus!(u8, u16, 61, u8_tests);
test_uint_modulus!(u16, u32, 12289, u16_tests);
test_uint_modulus!(u32, u64, 536_813_569, u32_tests);

#[cfg(not(target_pointer_width = "32"))]
test_uint_modulus!(u64, u128, 4_611_686_018_427_322_369, u64_tests);

#[cfg(target_pointer_width = "32")]
test_uint_modulus!(usize, u64, 536_813_569, usize_tests);
#[cfg(target_pointer_width = "64")]
test_uint_modulus!(usize, u128, 4_611_686_018_427_322_369, usize_tests);
