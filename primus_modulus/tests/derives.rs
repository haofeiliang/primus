macro_rules! test_modulus {
    () => {
        fn field_trait<M: FieldContext<ValueT>>(_modulus: M) {}

        #[test]
        fn test_scalar_ops() {
            field_trait(Modulus);
            let modulus_value = Modulus::value();
            let distribution = Uniform::new(0, modulus_value).unwrap();

            let mut rng = rand::rng();
            let a = distribution.sample(&mut rng);
            let b = distribution.sample(&mut rng);
            let c = distribution.sample(&mut rng);

            let d = Modulus.reduce_add(a, b);
            assert_eq!(d, (a + b) % modulus_value);

            let d = Modulus.reduce_sub(a, b);
            assert_eq!(d, (a + modulus_value - b) % modulus_value);

            let d = Modulus.reduce_neg(a);
            assert_eq!(0, Modulus.reduce_add(a, d));

            let d = Modulus.reduce_mul(a, b);
            assert_eq!(
                d,
                (a as WideT * b as WideT % modulus_value as WideT) as ValueT
            );

            let d = Modulus.reduce_square(a);
            assert_eq!(
                d,
                (a as WideT * a as WideT % modulus_value as WideT) as ValueT
            );

            let d = Modulus.reduce_mul_add(a, b, c);
            assert_eq!(
                d,
                ((a as WideT * b as WideT + c as WideT) % modulus_value as WideT) as ValueT
            );

            if a != 0 {
                let d = Modulus.reduce_inv(a);
                assert_eq!(1, Modulus.reduce_mul(a, d));
            }

            if b != 0 {
                let d = Modulus.reduce_div(a, b);
                assert_eq!(a, Modulus.reduce_mul(b, d));
            }
        }

        #[test]
        fn test_slice_ops() {
            use primus_modulus::BarrettModulus;

            let modulus_value = Modulus::value();
            let distr = Uniform::new(0, modulus_value).unwrap();
            let mut rng = rand::rng();

            // Exercise a range of lengths: small, medium, odd, and values not
            // divisible by typical SIMD lane counts to hit the scalar tail.
            for &len in &[1usize, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 65, 67] {
                let a: Vec<ValueT> = distr.sample_iter(&mut rng).take(len).collect();
                let b: Vec<ValueT> = distr.sample_iter(&mut rng).take(len).collect();
                let c: Vec<ValueT> = distr.sample_iter(&mut rng).take(len).collect();
                let scalar: ValueT = distr.sample(&mut rng);

                // -------------------------------------------------------------
                // ReduceOnceSlice
                // -------------------------------------------------------------
                {
                    let once_input: Vec<ValueT> = a
                        .iter()
                        .map(|&x| {
                            if rng.random_bool(0.5) {
                                x
                            } else {
                                x.wrapping_add(modulus_value)
                            }
                        })
                        .collect();

                    let mut assign = once_input.clone();
                    Modulus.reduce_once_slice_assign(&mut assign);
                    assert_eq!(assign, a, "reduce_once_slice_assign len={len}");

                    let mut to = vec![0; len];
                    Modulus.reduce_once_slice_to(&once_input, &mut to);
                    assert_eq!(to, a, "reduce_once_slice_to len={len}");
                }

                // -------------------------------------------------------------
                // ReduceNegSlice
                // -------------------------------------------------------------
                {
                    let expected: Vec<ValueT> = a
                        .iter()
                        .map(|&x| if x == 0 { 0 } else { modulus_value - x })
                        .collect();

                    let mut assign = a.clone();
                    Modulus.reduce_neg_slice_assign(&mut assign);
                    assert_eq!(assign, expected, "reduce_neg_slice_assign len={len}");

                    let mut to = vec![0; len];
                    Modulus.reduce_neg_slice_to(&a, &mut to);
                    assert_eq!(to, expected, "reduce_neg_slice_to len={len}");
                }

                // -------------------------------------------------------------
                // ReduceAddSlice
                // -------------------------------------------------------------
                {
                    let expected: Vec<ValueT> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            let sum = x + y;
                            if sum >= modulus_value {
                                sum - modulus_value
                            } else {
                                sum
                            }
                        })
                        .collect();

                    let mut assign = a.clone();
                    Modulus.reduce_add_slice_assign(&mut assign, &b);
                    assert_eq!(assign, expected, "reduce_add_slice_assign len={len}");

                    let mut to = vec![0; len];
                    Modulus.reduce_add_slice_to(&a, &b, &mut to);
                    assert_eq!(to, expected, "reduce_add_slice_to len={len}");
                }

                // -------------------------------------------------------------
                // ReduceSubSlice
                // -------------------------------------------------------------
                {
                    let expected: Vec<ValueT> = a
                        .iter()
                        .zip(&b)
                        .map(
                            |(&x, &y)| {
                                if x >= y { x - y } else { x + modulus_value - y }
                            },
                        )
                        .collect();

                    let mut assign = a.clone();
                    Modulus.reduce_sub_slice_assign(&mut assign, &b);
                    assert_eq!(assign, expected, "reduce_sub_slice_assign len={len}");

                    let mut to = vec![0; len];
                    Modulus.reduce_sub_slice_to(&a, &b, &mut to);
                    assert_eq!(to, expected, "reduce_sub_slice_to len={len}");
                }

                // -------------------------------------------------------------
                // ReduceMulSlice
                // -------------------------------------------------------------
                {
                    let expected: Vec<ValueT> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (x as WideT * y as WideT % modulus_value as WideT) as ValueT
                        })
                        .collect();

                    let mut assign = a.clone();
                    Modulus.reduce_mul_slice_assign(&mut assign, &b);
                    assert_eq!(assign, expected, "reduce_mul_slice_assign len={len}");

                    let mut to = vec![0; len];
                    Modulus.reduce_mul_slice_to(&a, &b, &mut to);
                    assert_eq!(to, expected, "reduce_mul_slice_to len={len}");

                    // reduce_scalar_mul variant
                    let expected_scalar: Vec<ValueT> = a
                        .iter()
                        .map(|&x| (x as WideT * scalar as WideT % modulus_value as WideT) as ValueT)
                        .collect();
                    let mut assign = a.clone();
                    Modulus.reduce_scalar_mul_slice_assign(&mut assign, scalar);
                    assert_eq!(
                        assign, expected_scalar,
                        "reduce_scalar_mul_slice_assign len={len}"
                    );
                    let mut to = vec![0; len];
                    Modulus.reduce_scalar_mul_slice_to(&a, scalar, &mut to);
                    assert_eq!(to, expected_scalar, "reduce_scalar_mul_slice_to len={len}");
                }

                // -------------------------------------------------------------
                // LazyReduceMulSlice
                // -------------------------------------------------------------
                {
                    let expected_canonical: Vec<ValueT> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (x as WideT * y as WideT % modulus_value as WideT) as ValueT
                        })
                        .collect();

                    let mut assign = a.clone();
                    Modulus.lazy_reduce_mul_slice_assign(&mut assign, &b);
                    for (&v, &lazy) in expected_canonical.iter().zip(assign.iter()) {
                        assert!(
                            lazy < modulus_value.wrapping_mul(2),
                            "lazy_reduce_mul_slice_assign: lazy={lazy} >= 2m"
                        );
                        let folded = if lazy >= modulus_value {
                            lazy - modulus_value
                        } else {
                            lazy
                        };
                        assert_eq!(folded, v, "lazy_reduce_mul_slice_assign len={len}");
                    }

                    let mut to = vec![0; len];
                    Modulus.lazy_reduce_mul_slice_to(&a, &b, &mut to);
                    for (&v, &lazy) in expected_canonical.iter().zip(to.iter()) {
                        assert!(lazy < modulus_value.wrapping_mul(2));
                        let folded = if lazy >= modulus_value {
                            lazy - modulus_value
                        } else {
                            lazy
                        };
                        assert_eq!(folded, v, "lazy_reduce_mul_slice_to len={len}");
                    }

                    // lazy_reduce_scalar_mul variant
                    let expected_canonical_scalar: Vec<ValueT> = a
                        .iter()
                        .map(|&x| (x as WideT * scalar as WideT % modulus_value as WideT) as ValueT)
                        .collect();
                    let mut assign = a.clone();
                    Modulus.lazy_reduce_scalar_mul_slice_assign(&mut assign, scalar);
                    for (&v, &lazy) in expected_canonical_scalar.iter().zip(assign.iter()) {
                        assert!(
                            lazy < modulus_value.wrapping_mul(2),
                            "lazy_reduce_scalar_mul_slice_assign: out of range"
                        );
                        let folded = if lazy >= modulus_value {
                            lazy - modulus_value
                        } else {
                            lazy
                        };
                        assert_eq!(folded, v, "lazy_reduce_scalar_mul_slice_assign len={len}");
                    }
                    let mut to = vec![0; len];
                    Modulus.lazy_reduce_scalar_mul_slice_to(&a, scalar, &mut to);
                    for (&v, &lazy) in expected_canonical_scalar.iter().zip(to.iter()) {
                        assert!(
                            lazy < modulus_value.wrapping_mul(2),
                            "lazy_reduce_scalar_mul_slice_to: out of range"
                        );
                        let folded = if lazy >= modulus_value {
                            lazy - modulus_value
                        } else {
                            lazy
                        };
                        assert_eq!(folded, v, "lazy_reduce_scalar_mul_slice_to len={len}");
                    }
                }

                // -------------------------------------------------------------
                // ReduceMulAddSlice
                // -------------------------------------------------------------
                {
                    // reduce_add_mul_slice_assign: acc += a * b
                    {
                        let expected: Vec<ValueT> = c
                            .iter()
                            .zip(&a)
                            .zip(&b)
                            .map(|((&acc, &x), &y)| {
                                let prod = (x as WideT * y as WideT) % modulus_value as WideT;
                                let sum = acc as WideT + prod;
                                (sum % modulus_value as WideT) as ValueT
                            })
                            .collect();
                        let mut acc = c.clone();
                        Modulus.reduce_add_mul_slice_assign(&mut acc, &a, &b);
                        assert_eq!(acc, expected, "reduce_add_mul_slice_assign len={len}");
                    }

                    // reduce_sub_mul_slice_assign: acc -= a * b
                    {
                        let expected: Vec<ValueT> = c
                            .iter()
                            .zip(&a)
                            .zip(&b)
                            .map(|((&acc, &x), &y)| {
                                let prod = (x as WideT * y as WideT) % modulus_value as WideT;
                                if acc as WideT >= prod {
                                    (acc as WideT - prod) as ValueT
                                } else {
                                    (acc as WideT + modulus_value as WideT - prod) as ValueT
                                }
                            })
                            .collect();
                        let mut acc = c.clone();
                        Modulus.reduce_sub_mul_slice_assign(&mut acc, &a, &b);
                        assert_eq!(acc, expected, "reduce_sub_mul_slice_assign len={len}");
                    }

                    // reduce_mul_add_slice_to: output = a * b + c
                    {
                        let expected: Vec<ValueT> = a
                            .iter()
                            .zip(&b)
                            .zip(&c)
                            .map(|((&x, &y), &z)| {
                                ((x as WideT * y as WideT + z as WideT) % modulus_value as WideT)
                                    as ValueT
                            })
                            .collect();
                        let mut out = vec![0; len];
                        Modulus.reduce_mul_add_slice_to(&a, &b, &c, &mut out);
                        assert_eq!(out, expected, "reduce_mul_add_slice_to len={len}");
                    }

                    // reduce_scalar_mul_add_slice_to: output = scalar * b + c
                    {
                        let expected: Vec<ValueT> = b
                            .iter()
                            .zip(&c)
                            .map(|(&y, &z)| {
                                ((scalar as WideT * y as WideT + z as WideT)
                                    % modulus_value as WideT) as ValueT
                            })
                            .collect();
                        let mut out = vec![0; len];
                        Modulus.reduce_scalar_mul_add_slice_to(scalar, &b, &c, &mut out);
                        assert_eq!(out, expected, "reduce_scalar_mul_add_slice_to len={len}");
                    }
                }

                // -------------------------------------------------------------
                // LazyReduceMulAddSlice
                // -------------------------------------------------------------
                {
                    let two_m = modulus_value.wrapping_mul(2);

                    // lazy_reduce_add_mul_slice_assign
                    {
                        let expected_canonical: Vec<ValueT> = c
                            .iter()
                            .zip(&a)
                            .zip(&b)
                            .map(|((&acc, &x), &y)| {
                                let prod = (x as WideT * y as WideT) % modulus_value as WideT;
                                let sum = acc as WideT + prod;
                                (sum % modulus_value as WideT) as ValueT
                            })
                            .collect();
                        let mut acc = c.clone();
                        Modulus.lazy_reduce_add_mul_slice_assign(&mut acc, &a, &b);
                        for (&v, &lazy) in expected_canonical.iter().zip(acc.iter()) {
                            assert!(
                                lazy < two_m,
                                "lazy_reduce_add_mul_slice_assign: lazy={lazy} >= 2m"
                            );
                            let folded = if lazy >= modulus_value {
                                lazy - modulus_value
                            } else {
                                lazy
                            };
                            assert_eq!(folded, v, "lazy_reduce_add_mul_slice_assign len={len}");
                        }
                    }

                    // lazy_reduce_sub_mul_slice_assign
                    {
                        let expected_canonical: Vec<ValueT> = c
                            .iter()
                            .zip(&a)
                            .zip(&b)
                            .map(|((&acc, &x), &y)| {
                                let prod = (x as WideT * y as WideT) % modulus_value as WideT;
                                if acc as WideT >= prod {
                                    (acc as WideT - prod) as ValueT
                                } else {
                                    (acc as WideT + modulus_value as WideT - prod) as ValueT
                                }
                            })
                            .collect();
                        let mut acc = c.clone();
                        Modulus.lazy_reduce_sub_mul_slice_assign(&mut acc, &a, &b);
                        for (&v, &lazy) in expected_canonical.iter().zip(acc.iter()) {
                            assert!(
                                lazy < two_m,
                                "lazy_reduce_sub_mul_slice_assign: lazy={lazy} >= 2m"
                            );
                            let folded = if lazy >= modulus_value {
                                lazy - modulus_value
                            } else {
                                lazy
                            };
                            assert_eq!(folded, v, "lazy_reduce_sub_mul_slice_assign len={len}");
                        }
                    }

                    // lazy_reduce_mul_add_slice_to
                    {
                        let expected_canonical: Vec<ValueT> = a
                            .iter()
                            .zip(&b)
                            .zip(&c)
                            .map(|((&x, &y), &z)| {
                                ((x as WideT * y as WideT + z as WideT) % modulus_value as WideT)
                                    as ValueT
                            })
                            .collect();
                        let mut out = vec![0; len];
                        Modulus.lazy_reduce_mul_add_slice_to(&a, &b, &c, &mut out);
                        for (&v, &lazy) in expected_canonical.iter().zip(out.iter()) {
                            assert!(
                                lazy < two_m,
                                "lazy_reduce_mul_add_slice_to: lazy={lazy} >= 2m"
                            );
                            let folded = if lazy >= modulus_value {
                                lazy - modulus_value
                            } else {
                                lazy
                            };
                            assert_eq!(folded, v, "lazy_reduce_mul_add_slice_to len={len}");
                        }
                    }

                    // lazy_reduce_scalar_mul_add_slice_to
                    {
                        let expected_canonical: Vec<ValueT> = b
                            .iter()
                            .zip(&c)
                            .map(|(&y, &z)| {
                                ((scalar as WideT * y as WideT + z as WideT)
                                    % modulus_value as WideT) as ValueT
                            })
                            .collect();
                        let mut out = vec![0; len];
                        Modulus.lazy_reduce_scalar_mul_add_slice_to(scalar, &b, &c, &mut out);
                        for (&v, &lazy) in expected_canonical.iter().zip(out.iter()) {
                            assert!(
                                lazy < two_m,
                                "lazy_reduce_scalar_mul_add_slice_to: lazy={lazy} >= 2m"
                            );
                            let folded = if lazy >= modulus_value {
                                lazy - modulus_value
                            } else {
                                lazy
                            };
                            assert_eq!(folded, v, "lazy_reduce_scalar_mul_add_slice_to len={len}");
                        }
                    }
                }

                // -------------------------------------------------------------
                // ReduceDotProduct
                // -------------------------------------------------------------
                {
                    // Use the runtime BarrettModulus as reference — it has a
                    // well-tested chunked accumulation that avoids overflow.
                    let expected = {
                        let ref_m = BarrettModulus::<ValueT>::new(modulus_value);
                        ref_m.reduce_dot_product(&a, &b)
                    };
                    let result = Modulus.reduce_dot_product(&a, &b);
                    assert_eq!(result, expected, "reduce_dot_product len={len}");

                    let result_iter =
                        Modulus.reduce_dot_product_iter(a.iter().copied(), b.iter().copied());
                    assert_eq!(result_iter, expected, "reduce_dot_product_iter len={len}");
                }
            }
        }
    };
}

#[cfg(all(test, feature = "derive"))]
mod u8tests {
    use primus_modulus::Barrett;
    use primus_reduce::FieldContext;
    use primus_reduce::prelude::*;
    use rand::{distr::Uniform, prelude::*};

    #[derive(Barrett)]
    #[modulus(ty = u8, value = 61)]
    struct Modulus;

    type ValueT = u8;
    type WideT = u16;

    test_modulus!();
}

#[cfg(all(test, feature = "derive"))]
mod u16tests {
    use primus_modulus::Barrett;
    use primus_reduce::FieldContext;
    use primus_reduce::prelude::*;
    use rand::{distr::Uniform, prelude::*};

    #[derive(Barrett)]
    #[modulus(ty = u16, value = 12289)]
    struct Modulus;

    type ValueT = u16;
    type WideT = u32;

    test_modulus!();
}

#[cfg(all(test, feature = "derive"))]
mod u32tests {
    use primus_modulus::Barrett;
    use primus_reduce::FieldContext;
    use primus_reduce::prelude::*;
    use rand::{distr::Uniform, prelude::*};

    #[derive(Barrett)]
    #[modulus(ty = u32, value = 536813569)]
    struct Modulus;

    type ValueT = u32;
    type WideT = u64;

    test_modulus!();
}

#[cfg(all(test, feature = "derive"))]
mod u64tests {
    use primus_modulus::Barrett;
    use primus_reduce::FieldContext;
    use primus_reduce::prelude::*;
    use rand::{distr::Uniform, prelude::*};

    #[derive(Barrett)]
    #[modulus(ty = u64, value = 4611686018427322369)]
    struct Modulus;

    type ValueT = u64;
    type WideT = u128;

    test_modulus!();
}
