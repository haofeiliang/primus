use primus_reduce::{
    LazyReduceMulAddSlice, LazyReduceMulSlice, ReduceAdd, ReduceAddAssign, ReduceAddSlice,
    ReduceDotProduct, ReduceMul, ReduceMulAdd, ReduceMulAddSlice, ReduceMulAssign, ReduceMulSlice,
    ReduceNeg, ReduceNegAssign, ReduceNegSlice, ReduceOnce, ReduceOnceAssign, ReduceOnceSlice,
    ReduceSub, ReduceSubAssign, ReduceSubSlice,
};

use super::PowOf2Modulus;

macro_rules! pow2_scalar {
    ($($t:ty),*) => {
        $(
            impl_reduce_once_slice_scalar!(impl ReduceOnceSlice<$t> for PowOf2Modulus<$t>);
            impl_reduce_neg_slice_scalar!(impl ReduceNegSlice<$t> for PowOf2Modulus<$t>);
            impl_reduce_add_slice_scalar!(impl ReduceAddSlice<$t> for PowOf2Modulus<$t>);
            impl_reduce_sub_slice_scalar!(impl ReduceSubSlice<$t> for PowOf2Modulus<$t>);
        )*
    };
}

#[cfg(feature = "simd")]
macro_rules! pow2_simd {
    ($($t:ty),*) => {
        $(
            impl_reduce_once_slice_simd_with!(impl ReduceOnceSlice<$t> for PowOf2Modulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), super::simd, access = mask);
            impl_reduce_neg_slice_simd_with!(impl ReduceNegSlice<$t> for PowOf2Modulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), super::simd, access = mask);
            impl_reduce_add_slice_simd_with!(impl ReduceAddSlice<$t> for PowOf2Modulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), super::simd, access = mask);
            impl_reduce_sub_slice_simd_with!(impl ReduceSubSlice<$t> for PowOf2Modulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), super::simd, access = mask);
        )*
    };
}

pow2_scalar!(u128);

#[cfg(not(feature = "simd"))]
pow2_scalar!(u8, u16, u32, u64, usize);

#[cfg(feature = "simd")]
pow2_simd!(u8, u16, u32, u64, usize);

// ===========================================================================
// Extended ops (mul / lazy / dot product) — scalar
// ===========================================================================

macro_rules! pow2_ext_scalar {
    ($($t:ty),*) => {
        $(
            impl ReduceMulSlice<$t> for PowOf2Modulus<$t> {
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

            impl ReduceMulAddSlice<$t> for PowOf2Modulus<$t> {
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
                    let mask = self.mask();
                    acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
                        *acc = (*acc).wrapping_sub(a.wrapping_mul(b)) & mask;
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

            impl ReduceDotProduct<$t> for PowOf2Modulus<$t> {
                type Output = $t;
                #[inline]
                fn reduce_dot_product(self, a: impl AsRef<[$t]>, b: impl AsRef<[$t]>) -> $t {
                    let a = a.as_ref();
                    let b = b.as_ref();
                    debug_assert_eq!(a.len(), b.len(), "reduce_dot_product: length mismatch");
                    a.iter().zip(b).fold(0, |acc, (&x, &y)| {
                        x.wrapping_mul(y).wrapping_add(acc)
                    }) & self.mask()
                }
                #[inline]
                fn reduce_dot_product_iter(
                    self, a: impl IntoIterator<Item = $t>, b: impl IntoIterator<Item = $t>,
                ) -> $t {
                    std::iter::zip(a, b).fold(0, |acc, (x, y)| {
                        x.wrapping_mul(y).wrapping_add(acc)
                    }) & self.mask()
                }
            }

            impl LazyReduceMulSlice<$t> for PowOf2Modulus<$t> {
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

            impl LazyReduceMulAddSlice<$t> for PowOf2Modulus<$t> {
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

pow2_ext_scalar!(u128);

#[cfg(not(feature = "simd"))]
pow2_ext_scalar!(u8, u16, u32, u64, usize);

// ===========================================================================
// Extended ops — SIMD
// ===========================================================================

#[cfg(feature = "simd")]
macro_rules! pow2_ext_simd {
    ($($t:ty),*) => {
        $(
            impl ReduceMulSlice<$t> for PowOf2Modulus<$t> {
                #[inline]
                fn reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) {
                    super::simd::reduce_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self.mask(), a, b)
                }
                #[inline]
                fn reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                    super::simd::reduce_mul_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self.mask(), a, b, output)
                }
                #[inline]
                fn reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) {
                    super::simd::reduce_scalar_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self.mask(), a, scalar)
                }
                #[inline]
                fn reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) {
                    super::simd::reduce_scalar_mul_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self.mask(), a, scalar, output)
                }
            }
            impl ReduceMulAddSlice<$t> for PowOf2Modulus<$t> {
                #[inline]
                fn reduce_add_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) {
                    super::simd::reduce_add_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self.mask(), acc, a, b)
                }
                #[inline]
                fn reduce_sub_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) {
                    super::simd::reduce_sub_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self.mask(), acc, a, b)
                }
                #[inline]
                fn reduce_add_scalar_mul_slice_assign(
                    self, acc: &mut [$t], scalar: $t, b: &[$t],
                ) {
                    super::simd::reduce_add_scalar_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(
                        self.mask(), acc, scalar, b,
                    )
                }
                #[inline]
                fn reduce_mul_add_slice_to(
                    self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t],
                ) {
                    super::simd::reduce_mul_add_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(
                        self.mask(), a, b, c, output,
                    )
                }
                #[inline]
                fn reduce_scalar_mul_add_slice_to(
                    self, scalar: $t, b: &[$t], c: &[$t], output: &mut [$t],
                ) {
                    super::simd::reduce_scalar_mul_add_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(
                        self.mask(), scalar, b, c, output,
                    )
                }
            }
            impl ReduceDotProduct<$t> for PowOf2Modulus<$t> {
                type Output = $t;
                #[inline]
                fn reduce_dot_product(self, a: impl AsRef<[$t]>, b: impl AsRef<[$t]>) -> $t {
                    super::simd::reduce_dot_product::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self.mask(), a.as_ref(), b.as_ref())
                }
                #[inline]
                fn reduce_dot_product_iter(
                    self, a: impl IntoIterator<Item = $t>, b: impl IntoIterator<Item = $t>,
                ) -> $t {
                    let mask = self.mask();
                    std::iter::zip(a, b).fold(0, |acc, (x, y)| {
                        x.wrapping_mul(y).wrapping_add(acc)
                    }) & mask
                }
            }
        )*
    };
}

#[cfg(feature = "simd")]
pow2_ext_simd!(u8, u16, u32, u64, usize);
