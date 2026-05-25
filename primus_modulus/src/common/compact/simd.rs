use std::simd::{Simd, cmp::SimdOrd};

use primus_integer::{SimdArray, SimdUnsignedInteger};

#[inline]
pub fn simd_reduce_add<T: SimdUnsignedInteger, const N: usize>(
    a: Simd<T, N>,
    b: Simd<T, N>,
    m: Simd<T, N>,
) -> Simd<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    let sum = a + b;
    sum.simd_min(sum - m)
}

#[inline]
pub fn simd_reduce_sub<T: SimdUnsignedInteger, const N: usize>(
    a: Simd<T, N>,
    b: Simd<T, N>,
    m: Simd<T, N>,
) -> Simd<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    // `a, b ∈ [0, m)`. When `a >= b`, `diff = a - b < m` and `diff + m < 2m`
    // does not wrap (provided `m < 2^{BITS-1}`), so `min` picks `diff`.
    // When `a < b`, `diff` wraps to a huge value and `diff + m` wraps back to
    // the canonical `(a - b) mod m`, so `min` picks the wrapped-back result.
    // Lowers to a single `vpminuq` on AVX-512.
    let diff = a - b;
    diff.simd_min(diff + m)
}

// ===========================================================================
// SIMD slice kernels.
// ===========================================================================

pub use crate::common::uint::simd::{
    reduce_neg_slice_assign, reduce_neg_slice_to, reduce_once_slice_assign, reduce_once_slice_to,
};

#[inline]
pub fn reduce_add_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: T,
    a: &mut [T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let m = Simd::splat(modulus);
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (ac, bc) in a_chunks.iter_mut().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *ac = simd_reduce_add(av, bv, m).to_array();
    }
    for (a, &b) in a_rem.iter_mut().zip(b_rem) {
        super::reduce_add_assign(modulus, a, b);
    }
}

#[inline]
pub fn reduce_add_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: T,
    a: &[T],
    b: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    let m = Simd::splat(modulus);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *oc = simd_reduce_add(av, bv, m).to_array();
    }
    for ((&a, &b), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        *o = super::reduce_add(modulus, a, b);
    }
}

#[inline]
pub fn reduce_sub_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: T,
    a: &mut [T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let m = Simd::splat(modulus);
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (ac, bc) in a_chunks.iter_mut().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *ac = simd_reduce_sub(av, bv, m).to_array();
    }
    for (a, &b) in a_rem.iter_mut().zip(b_rem) {
        super::reduce_sub_assign(modulus, a, b);
    }
}

#[inline]
pub fn reduce_sub_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: T,
    a: &[T],
    b: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    let m = Simd::splat(modulus);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *oc = simd_reduce_sub(av, bv, m).to_array();
    }
    for ((&a, &b), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        *o = super::reduce_sub(modulus, a, b);
    }
}

#[inline]
pub fn reduce_sub_slice_rev_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: T,
    a: &[T],
    b: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let m = Simd::splat(modulus);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks_mut::<N>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *bc = simd_reduce_sub(av, bv, m).to_array();
    }
    for (&a, b) in a_rem.iter().zip(b_rem) {
        *b = super::reduce_sub(modulus, a, *b);
    }
}
