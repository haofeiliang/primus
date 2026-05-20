use primus_factor::{FactorMul, LazyFactorMul};
use primus_gcd::Xgcd;
use primus_integer::UnsignedInteger;
use primus_reduce::ReduceError;
use primus_reduce::prelude::*;

use super::UintModulus;

impl<T: UnsignedInteger> ReduceOnce<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_once(self, value: T) -> Self::Output {
        value.min(value.wrapping_sub(self.0))
    }
}

impl<T: UnsignedInteger> ReduceOnceAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_once_assign(self, value: &mut T) {
        *value = (*value).min(value.wrapping_sub(self.0));
    }
}

impl<T: UnsignedInteger> ReduceAdd<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_add(self, a: T, b: T) -> Self::Output {
        let sum = a + b;
        sum.min(sum.wrapping_sub(self.0))
    }
}

impl<T: UnsignedInteger> ReduceAddAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_add_assign(self, a: &mut T, b: T) {
        let sum = *a + b;
        *a = sum.min(sum.wrapping_sub(self.0));
    }
}

impl<T: UnsignedInteger> ReduceDouble<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_double(self, value: T) -> Self::Output {
        let d = value.wrapping_shl(1);
        d.min(d.wrapping_sub(self.0))
    }
}

impl<T: UnsignedInteger> ReduceDoubleAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_double_assign(self, value: &mut T) {
        *value = self.reduce_double(*value);
    }
}

impl<T: UnsignedInteger> ReduceSub<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_sub(self, a: T, b: T) -> Self::Output {
        let diff = a.wrapping_sub(b);
        diff.min(diff.wrapping_add(self.0))
    }
}

impl<T: UnsignedInteger> ReduceSubAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_sub_assign(self, a: &mut T, b: T) {
        let diff = a.wrapping_sub(b);
        *a = diff.min(diff.wrapping_add(self.0));
    }
}

impl<T: UnsignedInteger> ReduceNeg<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_neg(self, value: T) -> Self::Output {
        if value.is_zero() {
            T::ZERO
        } else {
            self.0 - value
        }
    }
}

impl<T: UnsignedInteger> ReduceNegAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_neg_assign(self, value: &mut T) {
        if !value.is_zero() {
            *value = self.0 - *value;
        }
    }
}

impl<T: UnsignedInteger> ReduceInv<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_inv(self, value: T) -> Self::Output {
        debug_assert!(self.0 > value);

        let (inv, gcd) = Xgcd::gcdinv(value, self.0);
        assert_eq!(gcd, T::ONE, "No {value}^(-1) mod {}", self.0);

        inv
    }
}

impl<T: UnsignedInteger> ReduceInvAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_inv_assign(self, value: &mut T) {
        *value = self.reduce_inv(*value);
    }
}

impl<T: UnsignedInteger> TryReduceInv<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn try_reduce_inv(self, value: T) -> Result<Self::Output, ReduceError<T>> {
        debug_assert!(self.0 > value);

        let (inv, gcd) = Xgcd::gcdinv(value, self.0);

        if gcd.is_one() {
            Ok(inv)
        } else {
            Err(ReduceError::NoInverse {
                value,
                modulus: self.0,
            })
        }
    }
}

impl<T: UnsignedInteger, F> LazyReduceMul<T, F> for UintModulus<T>
where
    F: LazyFactorMul<T>,
{
    type Output = T;

    #[inline(always)]
    fn lazy_reduce_mul(self, a: T, b: F) -> Self::Output {
        b.lazy_factor_mul_modulo(a, self.0)
    }
}

impl<T: UnsignedInteger, F> ReduceMul<T, F> for UintModulus<T>
where
    F: FactorMul<T>,
{
    type Output = T;

    #[inline(always)]
    fn reduce_mul(self, a: T, b: F) -> Self::Output {
        b.factor_mul_modulo(a, self.0)
    }
}

// ===========================================================================
// Slice-level trait impls for UintModulus.
//
// When the `simd` feature is enabled, slice ops dispatch to SIMD kernels
// in `super::simd`. Without SIMD, they fall back to per-element scalar
// operations (which already exist on `UintModulus` via `Reduce*` traits).
// ===========================================================================

// `impl_uint_slice_scalar` must always be defined (not gated by `not(simd)`)
// because `u128` always uses scalar even when SIMD is enabled.
macro_rules! impl_uint_slice_scalar {
    ($($t:ty),* $(,)?) => {
        $(
            impl ReduceOnceSlice<$t> for UintModulus<$t> {
                #[inline]
                fn reduce_once_slice_assign(self, values: &mut [$t]) {
                    values.iter_mut().for_each(|v| self.reduce_once_assign(v));
                }
                #[inline]
                fn reduce_once_slice_to(self, input: &[$t], output: &mut [$t]) {
                    debug_assert_eq!(input.len(), output.len());
                    input.iter().zip(output).for_each(|(&i, o)| *o = self.reduce_once(i));
                }
            }

            impl ReduceNegSlice<$t> for UintModulus<$t> {
                #[inline]
                fn reduce_neg_slice_assign(self, values: &mut [$t]) {
                    values.iter_mut().for_each(|v| self.reduce_neg_assign(v));
                }
                #[inline]
                fn reduce_neg_slice_to(self, input: &[$t], output: &mut [$t]) {
                    debug_assert_eq!(input.len(), output.len());
                    input.iter().zip(output).for_each(|(&i, o)| *o = self.reduce_neg(i));
                }
            }

            impl ReduceAddSlice<$t> for UintModulus<$t> {
                #[inline]
                fn reduce_add_slice_assign(self, a: &mut [$t], b: &[$t]) {
                    a.iter_mut().zip(b).for_each(|(a, &b)| self.reduce_add_assign(a, b));
                }
                #[inline]
                fn reduce_add_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                    debug_assert_eq!(a.len(), b.len());
                    debug_assert_eq!(a.len(), output.len());
                    a.iter().zip(b).zip(output).for_each(|((&a_val, &b_val), o)| {
                        *o = self.reduce_add(a_val, b_val);
                    });
                }
            }

            impl ReduceSubSlice<$t> for UintModulus<$t> {
                #[inline]
                fn reduce_sub_slice_assign(self, a: &mut [$t], b: &[$t]) {
                    a.iter_mut().zip(b).for_each(|(a, &b)| self.reduce_sub_assign(a, b));
                }
                #[inline]
                fn reduce_sub_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                    debug_assert_eq!(a.len(), b.len());
                    debug_assert_eq!(a.len(), output.len());
                    a.iter().zip(b).zip(output).for_each(|((&a_val, &b_val), o)| {
                        *o = self.reduce_sub(a_val, b_val);
                    });
                }
                #[inline]
                fn reduce_sub_slice_rev_assign(self, a: &[$t], b: &mut [$t]) {
                    a.iter().zip(b).for_each(|(&a_val, b)| {
                        let diff = a_val.wrapping_sub(*b);
                        *b = diff.min(diff.wrapping_add(self.0));
                    });
                }
            }
        )*
    };
}

#[cfg(feature = "simd")]
macro_rules! impl_uint_slice_simd {
    ($t:ty, $lanes:expr) => {
        impl ReduceOnceSlice<$t> for UintModulus<$t> {
            #[inline]
            fn reduce_once_slice_assign(self, values: &mut [$t]) {
                super::simd::reduce_once_slice_assign::<$t, { $lanes }>(self.0, values)
            }
            #[inline]
            fn reduce_once_slice_to(self, input: &[$t], output: &mut [$t]) {
                super::simd::reduce_once_slice_to::<$t, { $lanes }>(self.0, input, output)
            }
        }

        impl ReduceNegSlice<$t> for UintModulus<$t> {
            #[inline]
            fn reduce_neg_slice_assign(self, values: &mut [$t]) {
                super::simd::reduce_neg_slice_assign::<$t, { $lanes }>(self.0, values)
            }
            #[inline]
            fn reduce_neg_slice_to(self, input: &[$t], output: &mut [$t]) {
                super::simd::reduce_neg_slice_to::<$t, { $lanes }>(self.0, input, output)
            }
        }

        impl ReduceAddSlice<$t> for UintModulus<$t> {
            #[inline]
            fn reduce_add_slice_assign(self, a: &mut [$t], b: &[$t]) {
                super::simd::reduce_add_slice_assign::<$t, { $lanes }>(self.0, a, b)
            }
            #[inline]
            fn reduce_add_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                super::simd::reduce_add_slice_to::<$t, { $lanes }>(self.0, a, b, output)
            }
        }

        impl ReduceSubSlice<$t> for UintModulus<$t> {
            #[inline]
            fn reduce_sub_slice_assign(self, a: &mut [$t], b: &[$t]) {
                super::simd::reduce_sub_slice_assign::<$t, { $lanes }>(self.0, a, b)
            }
            #[inline]
            fn reduce_sub_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                super::simd::reduce_sub_slice_to::<$t, { $lanes }>(self.0, a, b, output)
            }
            #[inline]
            fn reduce_sub_slice_rev_assign(self, a: &[$t], b: &mut [$t]) {
                super::simd::reduce_sub_slice_rev_assign::<$t, { $lanes }>(self.0, a, b)
            }
        }
    };
}

// u128 always falls back to scalar — `Simd<u128, _>` is not viable on most targets.
impl_uint_slice_scalar!(u128);

// When SIMD is off, every primitive width falls back to scalar.
#[cfg(not(feature = "simd"))]
impl_uint_slice_scalar!(u8, u16, u32, u64, usize);

#[cfg(feature = "simd")]
impl_uint_slice_simd!(u8, primus_integer::lanes::VECTOR_BITS / 8);
#[cfg(feature = "simd")]
impl_uint_slice_simd!(u16, primus_integer::lanes::VECTOR_BITS / 16);
#[cfg(feature = "simd")]
impl_uint_slice_simd!(u32, primus_integer::lanes::VECTOR_BITS / 32);
#[cfg(feature = "simd")]
impl_uint_slice_simd!(u64, primus_integer::lanes::VECTOR_BITS / 64);
#[cfg(all(feature = "simd", target_pointer_width = "64"))]
impl_uint_slice_simd!(usize, primus_integer::lanes::VECTOR_BITS / 64);
#[cfg(all(feature = "simd", target_pointer_width = "32"))]
impl_uint_slice_simd!(usize, primus_integer::lanes::VECTOR_BITS / 32);
