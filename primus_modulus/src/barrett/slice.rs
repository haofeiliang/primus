//! Slice-level (bulk) Barrett operations.
//!
//! The scalar helpers in this module loop over a slice using the existing
//! scalar `Reduce*` impls on [`BarrettModulus`]. They are also used as
//! the tail of the SIMD kernels in [`super::simd`] when the
//! `simd` feature combo is enabled.

use primus_integer::UnsignedInteger;
use primus_reduce::prelude::*;

use super::BarrettModulus;

// ---------------------------------------------------------------------------
// Scalar slice helpers (also reused as the SIMD tail).
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn scalar_reduce_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &mut [T],
    b: &[T],
) {
    debug_assert_eq!(a.len(), b.len());
    a.iter_mut()
        .zip(b)
        .for_each(|(a, &b)| *a = modulus.reduce_mul(*a, b));
}

#[inline]
pub(super) fn scalar_reduce_mul_slice_to<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &[T],
    b: &[T],
    output: &mut [T],
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    a.iter()
        .zip(b)
        .zip(output)
        .for_each(|((&a, &b), o)| *o = modulus.reduce_mul(a, b));
}

#[inline]
pub(super) fn scalar_reduce_scalar_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &mut [T],
    scalar: T,
) {
    a.iter_mut()
        .for_each(|a| *a = modulus.reduce_mul(*a, scalar));
}

#[inline]
pub(super) fn scalar_reduce_scalar_mul_slice_to<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &[T],
    scalar: T,
    output: &mut [T],
) {
    debug_assert_eq!(a.len(), output.len());
    a.iter()
        .zip(output)
        .for_each(|(&a, o)| *o = modulus.reduce_mul(a, scalar));
}

#[inline]
pub(super) fn scalar_lazy_reduce_scalar_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &mut [T],
    scalar: T,
) {
    a.iter_mut()
        .for_each(|a| *a = modulus.lazy_reduce_mul(*a, scalar));
}

#[inline]
pub(super) fn scalar_lazy_reduce_scalar_mul_slice_to<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &[T],
    scalar: T,
    output: &mut [T],
) {
    debug_assert_eq!(a.len(), output.len());
    a.iter()
        .zip(output)
        .for_each(|(&a, o)| *o = modulus.lazy_reduce_mul(a, scalar));
}

#[inline]
pub(super) fn scalar_lazy_reduce_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &mut [T],
    b: &[T],
) {
    debug_assert_eq!(a.len(), b.len());
    a.iter_mut()
        .zip(b)
        .for_each(|(a, &b)| *a = modulus.lazy_reduce_mul(*a, b));
}

#[inline]
pub(super) fn scalar_lazy_reduce_mul_slice_to<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &[T],
    b: &[T],
    output: &mut [T],
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    a.iter()
        .zip(b)
        .zip(output)
        .for_each(|((&a, &b), o)| *o = modulus.lazy_reduce_mul(a, b));
}

#[inline]
pub(super) fn scalar_reduce_add_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    a: &[T],
    b: &[T],
) {
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    acc.iter_mut()
        .zip(a)
        .zip(b)
        .for_each(|((acc, &a), &b)| *acc = modulus.reduce_mul_add(a, b, *acc));
}

#[inline]
pub(super) fn scalar_reduce_sub_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    a: &[T],
    b: &[T],
) {
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
        let prod = modulus.reduce_mul(a, b);
        modulus.reduce_sub_assign(acc, prod);
    });
}

#[inline]
pub(super) fn scalar_reduce_mul_add_slice_to<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &[T],
    b: &[T],
    c: &[T],
    output: &mut [T],
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), output.len());
    a.iter()
        .zip(b)
        .zip(c)
        .zip(output)
        .for_each(|(((&a, &b), &c), o)| *o = modulus.reduce_mul_add(a, b, c));
}

#[inline]
pub(super) fn scalar_reduce_scalar_mul_add_slice_to<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    scalar: T,
    b: &[T],
    c: &[T],
    output: &mut [T],
) {
    debug_assert_eq!(b.len(), c.len());
    debug_assert_eq!(b.len(), output.len());
    b.iter()
        .zip(c)
        .zip(output)
        .for_each(|((&b, &c), o)| *o = modulus.reduce_mul_add(scalar, b, c));
}

#[inline]
pub(super) fn scalar_reduce_add_scalar_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    scalar: T,
    b: &[T],
) {
    debug_assert_eq!(acc.len(), b.len());
    acc.iter_mut()
        .zip(b)
        .for_each(|(acc, &b)| *acc = modulus.reduce_mul_add(scalar, b, *acc));
}

#[inline]
pub(super) fn scalar_lazy_reduce_add_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    a: &[T],
    b: &[T],
) {
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    acc.iter_mut()
        .zip(a)
        .zip(b)
        .for_each(|((acc, &a), &b)| *acc = modulus.lazy_reduce_mul_add(a, b, *acc));
}

#[inline]
pub(super) fn scalar_lazy_reduce_sub_mul_slice_assign<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    a: &[T],
    b: &[T],
) {
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    let m = modulus.value();
    acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
        let prod = modulus.reduce_mul(a, b);
        *acc = acc.wrapping_add(m - prod);
    });
}

#[inline]
pub(super) fn scalar_lazy_reduce_mul_add_slice_to<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &[T],
    b: &[T],
    c: &[T],
    output: &mut [T],
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), output.len());
    a.iter()
        .zip(b)
        .zip(c)
        .zip(output)
        .for_each(|(((&a, &b), &c), o)| *o = modulus.lazy_reduce_mul_add(a, b, c));
}

#[inline]
pub(super) fn scalar_lazy_reduce_scalar_mul_add_slice_to<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    scalar: T,
    b: &[T],
    c: &[T],
    output: &mut [T],
) {
    debug_assert_eq!(b.len(), c.len());
    debug_assert_eq!(b.len(), output.len());
    b.iter()
        .zip(c)
        .zip(output)
        .for_each(|((&b, &c), o)| *o = modulus.lazy_reduce_mul_add(scalar, b, c));
}

// ---------------------------------------------------------------------------
// dot_product helpers (used by the scalar trait impl AND as the tail of the
// SIMD kernel — both call into `multiply_add` to accumulate widening products
// into a double-word, then `BarrettModulus::reduce` to fold back to `[0, m)`).
//
// `DOT_PRODUCT_INNER_CHUNK = 16` is the maximum chunk size such that
// `K * m^2 < 2^128` for any valid `BarrettModulus<T>` (where `m < 2^(T::BITS - 1)`
// is enforced by `BarrettModulus::new`). On a `T = u64` modulus capped at
// `2^62`, accumulating 16 widening products keeps the high limb strictly
// below `2^64` and the low limb's carry strictly below `2^64`.
// ---------------------------------------------------------------------------

pub(super) const DOT_PRODUCT_INNER_CHUNK: usize = 16;

/// `c += a * b` on a double-word accumulator.
#[inline]
pub(super) fn multiply_add<T: UnsignedInteger>(c: &mut [T; 2], a: T, b: T) {
    let (lw, hw) = a.widening_mul(b);
    let carry;
    (c[0], carry) = c[0].overflowing_add(lw);
    (c[1], _) = c[1].carrying_add(hw, carry);
}

#[inline]
pub(super) fn scalar_reduce_dot_product<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: &[T],
    b: &[T],
) -> T {
    assert_eq!(a.len(), b.len(), "reduce_dot_product: length mismatch");

    let mut a_iter = a.chunks_exact(DOT_PRODUCT_INNER_CHUNK);
    let mut b_iter = b.chunks_exact(DOT_PRODUCT_INNER_CHUNK);

    let inter = (&mut a_iter)
        .zip(&mut b_iter)
        .map(|(a_s, b_s)| {
            let mut c: [T; 2] = [T::ZERO, T::ZERO];
            for (&a, &b) in a_s.iter().zip(b_s) {
                multiply_add(&mut c, a, b);
            }
            modulus.reduce(c)
        })
        .fold(T::ZERO, |acc: T, b| modulus.reduce_add(acc, b));

    let mut c: [T; 2] = [T::ZERO, T::ZERO];
    a_iter
        .remainder()
        .iter()
        .zip(b_iter.remainder())
        .for_each(|(&a, &b)| {
            multiply_add(&mut c, a, b);
        });
    modulus.reduce_add(modulus.reduce(c), inter)
}

#[inline]
pub(super) fn scalar_reduce_dot_product_iter<T: UnsignedInteger>(
    modulus: BarrettModulus<T>,
    a: impl IntoIterator<Item = T>,
    b: impl IntoIterator<Item = T>,
) -> T {
    let mut a_iter = a.into_iter();
    let mut b_iter = b.into_iter();

    let mut a_temp_array = [T::ZERO; DOT_PRODUCT_INNER_CHUNK];
    let mut b_temp_array = [T::ZERO; DOT_PRODUCT_INNER_CHUNK];

    let mut i = 0;
    let mut result = T::ZERO;

    while let (Some(a_next), Some(b_next)) = (a_iter.next(), b_iter.next()) {
        if i < DOT_PRODUCT_INNER_CHUNK {
            a_temp_array[i] = a_next;
            b_temp_array[i] = b_next;
            i += 1;
        } else {
            let mut c: [T; 2] = [T::ZERO, T::ZERO];
            for (&a, b) in a_temp_array.iter().zip(b_temp_array) {
                multiply_add(&mut c, a, b);
            }
            modulus.reduce_add_assign(&mut result, modulus.reduce(c));

            a_temp_array.fill(T::ZERO);
            b_temp_array.fill(T::ZERO);
            a_temp_array[0] = a_next;
            b_temp_array[0] = b_next;
            i = 1;
        }
    }

    let mut c: [T; 2] = [T::ZERO, T::ZERO];
    for (&a, &b) in a_temp_array[..i].iter().zip(b_temp_array[..i].iter()) {
        multiply_add(&mut c, a, b);
    }
    modulus.reduce_add_assign(&mut result, modulus.reduce(c));

    result
}

// ---------------------------------------------------------------------------
// Dispatch — basic four from shared macros, Barrett-specific for mul/lazy/dot
// ---------------------------------------------------------------------------

macro_rules! barrett_scalar {
    ($($t:ty),*) => {
        $(
            impl_reduce_once_slice_scalar!(impl ReduceOnceSlice<$t> for BarrettModulus<$t>);
            impl_reduce_neg_slice_scalar!(impl ReduceNegSlice<$t> for BarrettModulus<$t>);
            impl_reduce_add_slice_scalar!(impl ReduceAddSlice<$t> for BarrettModulus<$t>);
            impl_reduce_sub_slice_scalar!(impl ReduceSubSlice<$t> for BarrettModulus<$t>);
        )*
    };
}

#[cfg(feature = "simd")]
macro_rules! barrett_simd {
    ($($t:ty),*) => {
        $(
            impl_reduce_once_slice_simd_with!(impl ReduceOnceSlice<$t> for BarrettModulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), crate::common::uint::simd, access = value);
            impl_reduce_neg_slice_simd_with!(impl ReduceNegSlice<$t> for BarrettModulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), crate::common::compact::simd, access = value);
            impl_reduce_add_slice_simd_with!(impl ReduceAddSlice<$t> for BarrettModulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), crate::common::compact::simd, access = value);
            impl_reduce_sub_slice_simd_with!(impl ReduceSubSlice<$t> for BarrettModulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), crate::common::compact::simd, access = value);
        )*
    };
}

barrett_scalar!(u128);

#[cfg(not(feature = "simd"))]
barrett_scalar!(u8, u16, u32, u64, usize);

#[cfg(feature = "simd")]
barrett_simd!(u8, u16, u32, u64, usize);

// ---------------------------------------------------------------------------
// Mul / lazy / dot product — scalar
// ---------------------------------------------------------------------------

macro_rules! barrett_ext_scalar {
    ($($t:ty),*) => {
        $(
            impl ReduceMulSlice<$t> for BarrettModulus<$t> {
                #[inline]
                fn reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) { scalar_reduce_mul_slice_assign(self, a, b) }
                #[inline]
                fn reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) { scalar_reduce_mul_slice_to(self, a, b, output) }
                #[inline]
                fn reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) { scalar_reduce_scalar_mul_slice_assign(self, a, scalar) }
                #[inline]
                fn reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) { scalar_reduce_scalar_mul_slice_to(self, a, scalar, output) }
            }
            impl LazyReduceMulSlice<$t> for BarrettModulus<$t> {
                #[inline]
                fn lazy_reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) { scalar_lazy_reduce_mul_slice_assign(self, a, b) }
                #[inline]
                fn lazy_reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) { scalar_lazy_reduce_mul_slice_to(self, a, b, output) }
                #[inline]
                fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) { scalar_lazy_reduce_scalar_mul_slice_assign(self, a, scalar) }
                #[inline]
                fn lazy_reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) { scalar_lazy_reduce_scalar_mul_slice_to(self, a, scalar, output) }
            }
            impl ReduceMulAddSlice<$t> for BarrettModulus<$t> {
                #[inline]
                fn reduce_add_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) { scalar_reduce_add_mul_slice_assign(self, acc, a, b) }
                #[inline]
                fn reduce_sub_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) { scalar_reduce_sub_mul_slice_assign(self, acc, a, b) }
                #[inline]
                fn reduce_mul_add_slice_to(self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t]) { scalar_reduce_mul_add_slice_to(self, a, b, c, output) }
                #[inline]
                fn reduce_scalar_mul_add_slice_to(self, scalar: $t, b: &[$t], c: &[$t], output: &mut [$t]) { scalar_reduce_scalar_mul_add_slice_to(self, scalar, b, c, output) }
                #[inline]
                fn reduce_add_scalar_mul_slice_assign(self, acc: &mut [$t], scalar: $t, b: &[$t]) { scalar_reduce_add_scalar_mul_slice_assign(self, acc, scalar, b) }
            }
            impl LazyReduceMulAddSlice<$t> for BarrettModulus<$t> {
                #[inline]
                fn lazy_reduce_add_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) { scalar_lazy_reduce_add_mul_slice_assign(self, acc, a, b) }
                #[inline]
                fn lazy_reduce_sub_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) { scalar_lazy_reduce_sub_mul_slice_assign(self, acc, a, b) }
                #[inline]
                fn lazy_reduce_mul_add_slice_to(self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t]) { scalar_lazy_reduce_mul_add_slice_to(self, a, b, c, output) }
                #[inline]
                fn lazy_reduce_scalar_mul_add_slice_to(self, scalar: $t, b: &[$t], c: &[$t], output: &mut [$t]) { scalar_lazy_reduce_scalar_mul_add_slice_to(self, scalar, b, c, output) }
            }
            impl ReduceDotProduct<$t> for BarrettModulus<$t> {
                type Output = $t;
                #[inline]
                fn reduce_dot_product(self, a: impl AsRef<[$t]>, b: impl AsRef<[$t]>) -> $t { scalar_reduce_dot_product(self, a.as_ref(), b.as_ref()) }
                #[inline]
                fn reduce_dot_product_iter(self, a: impl IntoIterator<Item = $t>, b: impl IntoIterator<Item = $t>) -> $t { scalar_reduce_dot_product_iter(self, a, b) }
            }
        )*
    };
}

#[cfg(feature = "simd")]
macro_rules! barrett_ext_simd {
    ($($t:ty),*) => {
        $(
            impl ReduceMulSlice<$t> for BarrettModulus<$t> {
                #[inline]
                fn reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) { super::simd::reduce_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, b) }
                #[inline]
                fn reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) { super::simd::reduce_mul_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, b, output) }
                #[inline]
                fn reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) { super::simd::reduce_scalar_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, scalar) }
                #[inline]
                fn reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) { super::simd::reduce_scalar_mul_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, scalar, output) }
            }
            impl LazyReduceMulSlice<$t> for BarrettModulus<$t> {
                #[inline]
                fn lazy_reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) { super::simd::lazy_reduce_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, b) }
                #[inline]
                fn lazy_reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) { super::simd::lazy_reduce_mul_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, b, output) }
                #[inline]
                fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) { super::simd::lazy_reduce_scalar_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, scalar) }
                #[inline]
                fn lazy_reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) { super::simd::lazy_reduce_scalar_mul_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, scalar, output) }
            }
            impl ReduceMulAddSlice<$t> for BarrettModulus<$t> {
                #[inline]
                fn reduce_add_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) { super::simd::reduce_add_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, acc, a, b) }
                #[inline]
                fn reduce_sub_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) { super::simd::reduce_sub_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, acc, a, b) }
                #[inline]
                fn reduce_mul_add_slice_to(self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t]) { super::simd::reduce_mul_add_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, b, c, output) }
                #[inline]
                fn reduce_scalar_mul_add_slice_to(self, scalar: $t, b: &[$t], c: &[$t], output: &mut [$t]) { super::simd::reduce_scalar_mul_add_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, scalar, b, c, output) }
                #[inline]
                fn reduce_add_scalar_mul_slice_assign(self, acc: &mut [$t], scalar: $t, b: &[$t]) { super::simd::reduce_add_scalar_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, acc, scalar, b) }
            }
            impl LazyReduceMulAddSlice<$t> for BarrettModulus<$t> {
                #[inline]
                fn lazy_reduce_add_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) { super::simd::lazy_reduce_add_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, acc, a, b) }
                #[inline]
                fn lazy_reduce_sub_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) { super::simd::lazy_reduce_sub_mul_slice_assign::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, acc, a, b) }
                #[inline]
                fn lazy_reduce_mul_add_slice_to(self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t]) { super::simd::lazy_reduce_mul_add_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a, b, c, output) }
                #[inline]
                fn lazy_reduce_scalar_mul_add_slice_to(self, scalar: $t, b: &[$t], c: &[$t], output: &mut [$t]) { super::simd::lazy_reduce_scalar_mul_add_slice_to::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, scalar, b, c, output) }
            }
            impl ReduceDotProduct<$t> for BarrettModulus<$t> {
                type Output = $t;
                #[inline]
                fn reduce_dot_product(self, a: impl AsRef<[$t]>, b: impl AsRef<[$t]>) -> $t { super::simd::reduce_dot_product::<$t, { primus_integer::lanes::VECTOR_BITS / <$t>::BITS as usize }>(self, a.as_ref(), b.as_ref()) }
                #[inline]
                fn reduce_dot_product_iter(self, a: impl IntoIterator<Item = $t>, b: impl IntoIterator<Item = $t>) -> $t { scalar_reduce_dot_product_iter(self, a, b) }
            }
        )*
    };
}

barrett_ext_scalar!(u128);

#[cfg(not(feature = "simd"))]
barrett_ext_scalar!(u8, u16, u32, u64, usize);

#[cfg(feature = "simd")]
barrett_ext_simd!(u8, u16, u32, u64, usize);
