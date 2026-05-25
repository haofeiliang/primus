use primus_reduce::prelude::*;

use super::NativeModulus;

// ===========================================================================
// Basic three (neg / add / sub) — from shared macros
// ===========================================================================

macro_rules! native_scalar {
    ($($t:ty),*) => {
        $(
            impl_reduce_neg_slice_scalar!(impl ReduceNegSlice<$t> for NativeModulus<$t>);
            impl_reduce_add_slice_scalar!(impl ReduceAddSlice<$t> for NativeModulus<$t>);
            impl_reduce_sub_slice_scalar!(impl ReduceSubSlice<$t> for NativeModulus<$t>);
        )*
    };
}

#[cfg(feature = "simd")]
macro_rules! impl_basic_slice_simd {
    ($ModType:ty, $t:ty, $lanes:expr) => {
        impl ReduceNegSlice<$t> for $ModType {
            #[inline]
            fn reduce_neg_slice_assign(self, values: &mut [$t]) {
                super::simd::reduce_neg_slice_assign::<$t, { $lanes }>(values)
            }
            #[inline]
            fn reduce_neg_slice_to(self, input: &[$t], output: &mut [$t]) {
                super::simd::reduce_neg_slice_to::<$t, { $lanes }>(input, output)
            }
        }

        impl ReduceAddSlice<$t> for $ModType {
            #[inline]
            fn reduce_add_slice_assign(self, a: &mut [$t], b: &[$t]) {
                super::simd::reduce_add_slice_assign::<$t, { $lanes }>(a, b)
            }
            #[inline]
            fn reduce_add_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                super::simd::reduce_add_slice_to::<$t, { $lanes }>(a, b, output)
            }
        }

        impl ReduceSubSlice<$t> for $ModType {
            #[inline]
            fn reduce_sub_slice_assign(self, a: &mut [$t], b: &[$t]) {
                super::simd::reduce_sub_slice_assign::<$t, { $lanes }>(a, b)
            }
            #[inline]
            fn reduce_sub_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                super::simd::reduce_sub_slice_to::<$t, { $lanes }>(a, b, output)
            }
            #[inline]
            fn reduce_sub_slice_rev_assign(self, a: &[$t], b: &mut [$t]) {
                super::simd::reduce_sub_slice_rev_assign::<$t, { $lanes }>(a, b)
            }
        }
    };
}

#[cfg(feature = "simd")]
macro_rules! native_simd {
    ($($t:ty),*) => {
        $( impl_basic_slice_simd!(NativeModulus<$t>, $t, primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize); )*
    };
}

native_scalar!(u128);

#[cfg(not(feature = "simd"))]
native_scalar!(u8, u16, u32, u64, usize);

#[cfg(feature = "simd")]
native_simd!(u8, u16, u32, u64, usize);

// ===========================================================================
// ReduceOnceSlice — always no-op, no SIMD needed
// ===========================================================================

macro_rules! native_once {
    ($($t:ty),*) => {
        $(
            impl ReduceOnceSlice<$t> for NativeModulus<$t> {
                #[inline(always)]
                fn reduce_once_slice_assign(self, _values: &mut [$t]) {}
                #[inline]
                fn reduce_once_slice_to(self, input: &[$t], output: &mut [$t]) {
                    debug_assert_eq!(input.len(), output.len());
                    output.copy_from_slice(input);
                }
            }
        )*
    };
}

native_once!(u8, u16, u32, u64, u128, usize);

// ===========================================================================
// Extended ops (mul / lazy / dot product) — scalar
// ===========================================================================

macro_rules! native_ext_scalar {
    ($($t:ty),*) => {
        $(
            impl ReduceMulSlice<$t> for NativeModulus<$t> {
                #[inline]
                fn reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) {
                    debug_assert_eq!(a.len(), b.len());
                    a.iter_mut().zip(b).for_each(|(x, &y)| self.reduce_mul_assign(x, y));
                }
                #[inline]
                fn reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                    debug_assert_eq!(output.len(), a.len());
                    debug_assert_eq!(output.len(), b.len());
                    output.iter_mut().zip(a).zip(b).for_each(|((out, &x), &y)| {
                        *out = self.reduce_mul(x, y);
                    });
                }
                #[inline]
                fn reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) {
                    a.iter_mut().for_each(|x| self.reduce_mul_assign(x, scalar));
                }
                #[inline]
                fn reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) {
                    debug_assert_eq!(a.len(), output.len());
                    output.iter_mut().zip(a).for_each(|(out, &x)| *out = self.reduce_mul(x, scalar));
                }
            }

            impl ReduceMulAddSlice<$t> for NativeModulus<$t> {
                #[inline]
                fn reduce_add_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) {
                    debug_assert_eq!(acc.len(), a.len());
                    debug_assert_eq!(acc.len(), b.len());
                    acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
                        *acc = self.reduce_mul_add(a, b, *acc);
                    });
                }
                #[inline]
                fn reduce_sub_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) {
                    debug_assert_eq!(acc.len(), a.len());
                    debug_assert_eq!(acc.len(), b.len());
                    acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
                        *acc = acc.wrapping_sub(a.wrapping_mul(b));
                    });
                }
                #[inline]
                fn reduce_add_scalar_mul_slice_assign(self, acc: &mut [$t], scalar: $t, b: &[$t]) {
                    debug_assert_eq!(acc.len(), b.len());
                    acc.iter_mut().zip(b).for_each(|(acc, &b)| {
                        *acc = self.reduce_mul_add(scalar, b, *acc);
                    });
                }
                #[inline]
                fn reduce_mul_add_slice_to(
                    self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t],
                ) {
                    debug_assert_eq!(a.len(), b.len());
                    debug_assert_eq!(a.len(), c.len());
                    debug_assert_eq!(a.len(), output.len());
                    a.iter().zip(b).zip(c).zip(output).for_each(|(((&a, &b), &c), o)| {
                        *o = self.reduce_mul_add(a, b, c);
                    });
                }
                #[inline]
                fn reduce_scalar_mul_add_slice_to(
                    self, scalar: $t, b: &[$t], c: &[$t], output: &mut [$t],
                ) {
                    debug_assert_eq!(b.len(), c.len());
                    debug_assert_eq!(b.len(), output.len());
                    b.iter().zip(c).zip(output).for_each(|((&b, &c), o)| {
                        *o = self.reduce_mul_add(scalar, b, c);
                    });
                }
            }

            impl ReduceDotProduct<$t> for NativeModulus<$t> {
                type Output = $t;
                #[inline]
                fn reduce_dot_product(self, a: impl AsRef<[$t]>, b: impl AsRef<[$t]>) -> $t {
                    let a = a.as_ref();
                    let b = b.as_ref();
                    assert_eq!(a.len(), b.len(), "reduce_dot_product: length mismatch");
                    a.iter().zip(b).fold(0, |acc, (&x, &y)| {
                        x.wrapping_mul(y).wrapping_add(acc)
                    })
                }
                #[inline]
                fn reduce_dot_product_iter(
                    self, a: impl IntoIterator<Item = $t>, b: impl IntoIterator<Item = $t>,
                ) -> $t {
                    std::iter::zip(a, b).fold(0, |acc, (x, y)| {
                        x.wrapping_mul(y).wrapping_add(acc)
                    })
                }
            }

            impl LazyReduceMulSlice<$t> for NativeModulus<$t> {
                #[inline]
                fn lazy_reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) {
                    self.reduce_mul_slice_assign(a, b);
                }
                #[inline]
                fn lazy_reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                    self.reduce_mul_slice_to(a, b, output);
                }
                #[inline]
                fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) {
                    self.reduce_scalar_mul_slice_assign(a, scalar);
                }
                #[inline]
                fn lazy_reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) {
                    self.reduce_scalar_mul_slice_to(a, scalar, output);
                }
            }

            impl LazyReduceMulAddSlice<$t> for NativeModulus<$t> {
                #[inline]
                fn lazy_reduce_add_mul_slice_assign(
                    self, acc: &mut [$t], a: &[$t], b: &[$t],
                ) { self.reduce_add_mul_slice_assign(acc, a, b); }
                #[inline]
                fn lazy_reduce_sub_mul_slice_assign(
                    self, acc: &mut [$t], a: &[$t], b: &[$t],
                ) { self.reduce_sub_mul_slice_assign(acc, a, b); }
                #[inline]
                fn lazy_reduce_mul_add_slice_to(
                    self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t],
                ) { self.reduce_mul_add_slice_to(a, b, c, output); }
                #[inline]
                fn lazy_reduce_scalar_mul_add_slice_to(
                    self, scalar: $t, b: &[$t], c: &[$t], output: &mut [$t],
                ) { self.reduce_scalar_mul_add_slice_to(scalar, b, c, output); }
            }
        )*
    };
}

native_ext_scalar!(u128);

#[cfg(not(feature = "simd"))]
native_ext_scalar!(u8, u16, u32, u64, usize);

// ===========================================================================
// Extended ops — SIMD
// ===========================================================================

#[cfg(feature = "simd")]
macro_rules! native_ext_simd {
    ($($t:ty),*) => {
        $(
            impl ReduceMulSlice<$t> for NativeModulus<$t> {
                #[inline]
                fn reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) {
                    super::simd::reduce_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(a, b)
                }
                #[inline]
                fn reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                    super::simd::reduce_mul_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(a, b, output)
                }
                #[inline]
                fn reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) {
                    super::simd::reduce_scalar_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(a, scalar)
                }
                #[inline]
                fn reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) {
                    super::simd::reduce_scalar_mul_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(a, scalar, output)
                }
            }
            impl ReduceMulAddSlice<$t> for NativeModulus<$t> {
                #[inline]
                fn reduce_add_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) {
                    super::simd::reduce_add_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(acc, a, b)
                }
                #[inline]
                fn reduce_sub_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) {
                    super::simd::reduce_sub_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(acc, a, b)
                }
                #[inline]
                fn reduce_add_scalar_mul_slice_assign(
                    self, acc: &mut [$t], scalar: $t, b: &[$t],
                ) {
                    super::simd::reduce_add_scalar_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(acc, scalar, b)
                }
                #[inline]
                fn reduce_mul_add_slice_to(
                    self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t],
                ) {
                    super::simd::reduce_mul_add_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(a, b, c, output)
                }
                #[inline]
                fn reduce_scalar_mul_add_slice_to(
                    self, scalar: $t, b: &[$t], c: &[$t], output: &mut [$t],
                ) {
                    super::simd::reduce_scalar_mul_add_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(scalar, b, c, output)
                }
            }
            impl ReduceDotProduct<$t> for NativeModulus<$t> {
                type Output = $t;
                #[inline]
                fn reduce_dot_product(self, a: impl AsRef<[$t]>, b: impl AsRef<[$t]>) -> $t {
                    super::simd::reduce_dot_product::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(a.as_ref(), b.as_ref())
                }
                #[inline]
                fn reduce_dot_product_iter(
                    self, a: impl IntoIterator<Item = $t>, b: impl IntoIterator<Item = $t>,
                ) -> $t {
                    std::iter::zip(a, b).fold(0, |acc, (x, y)| {
                        x.wrapping_mul(y).wrapping_add(acc)
                    })
                }
            }
        )*
    };
}

#[cfg(feature = "simd")]
native_ext_simd!(u8, u16, u32, u64, usize);
