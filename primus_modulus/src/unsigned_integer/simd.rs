use std::simd::{
    Simd,
    cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd},
};

use primus_integer::{SimdArray, SimdMaskArray, SimdUnsignedInteger};

#[inline]
fn simd_reduce_once<T: SimdUnsignedInteger, const N: usize>(
    v: Simd<T, N>,
    m: Simd<T, N>,
) -> Simd<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    v.simd_min(v - m)
}

#[inline]
fn simd_reduce_add<T: SimdUnsignedInteger, const N: usize>(
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
fn simd_reduce_sub<T: SimdUnsignedInteger, const N: usize>(
    a: Simd<T, N>,
    b: Simd<T, N>,
    m: Simd<T, N>,
) -> Simd<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    let diff = a - b;
    a.simd_lt(b).select(diff + m, diff)
}

#[inline]
fn simd_reduce_neg<T: SimdUnsignedInteger, const N: usize>(
    v: Simd<T, N>,
    m: Simd<T, N>,
) -> Simd<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    let zero = Simd::splat(T::ZERO);
    v.simd_eq(zero).select(zero, m - v)
}

// ===========================================================================
// SIMD slice kernels.
// ===========================================================================

#[inline]
pub fn reduce_once_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: T,
    values: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    let m = Simd::splat(modulus);
    let (chunks, rem) = values.as_chunks_mut::<N>();
    for chunk in chunks {
        let v = Simd::from_array(*chunk);
        *chunk = simd_reduce_once(v, m).to_array();
    }
    // Scalar fallback for remainder.
    for v in rem {
        if *v >= modulus {
            *v -= modulus;
        }
    }
}

#[inline]
pub fn reduce_once_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: T,
    input: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(input.len(), output.len());
    let m = Simd::splat(modulus);
    let (in_chunks, in_rem) = input.as_chunks::<N>();
    let (out_chunks, out_rem) = output.as_chunks_mut::<N>();
    for (i, o) in in_chunks.iter().zip(out_chunks) {
        let v = Simd::from_array(*i);
        *o = simd_reduce_once(v, m).to_array();
    }
    for (i, o) in in_rem.iter().zip(out_rem) {
        *o = if *i >= modulus { *i - modulus } else { *i };
    }
}

#[inline]
pub fn reduce_neg_slice_assign<T: SimdUnsignedInteger, const N: usize>(modulus: T, values: &mut [T])
where
    Simd<T, N>: SimdArray<T, N>,
{
    let m = Simd::splat(modulus);
    let (chunks, rem) = values.as_chunks_mut::<N>();
    for chunk in chunks {
        let v = Simd::from_array(*chunk);
        *chunk = simd_reduce_neg(v, m).to_array();
    }
    for v in rem {
        if !v.is_zero() {
            *v = modulus - *v;
        }
    }
}

#[inline]
pub fn reduce_neg_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: T,
    input: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(input.len(), output.len());
    let m = Simd::splat(modulus);
    let (in_chunks, in_rem) = input.as_chunks::<N>();
    let (out_chunks, out_rem) = output.as_chunks_mut::<N>();
    for (i, o) in in_chunks.iter().zip(out_chunks) {
        let v = Simd::from_array(*i);
        *o = simd_reduce_neg(v, m).to_array();
    }
    for (i, o) in in_rem.iter().zip(out_rem) {
        *o = if i.is_zero() { *i } else { modulus - *i };
    }
}

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
        let diff = modulus - b;
        if diff > *a {
            *a += b;
        } else {
            *a = a.wrapping_sub(diff);
        }
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
    for ((&a_val, &b_val), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        let sum = a_val + b_val;
        *o = if sum >= modulus { sum - modulus } else { sum };
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
        if b > *a {
            *a = a.wrapping_sub(b).wrapping_add(modulus);
        } else {
            *a -= b;
        }
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
    for ((&a_val, &b_val), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        *o = if b_val > a_val {
            a_val.wrapping_sub(b_val).wrapping_add(modulus)
        } else {
            a_val - b_val
        };
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
    for (&a_val, b) in a_rem.iter().zip(b_rem) {
        if a_val < *b {
            *b = a_val.wrapping_sub(*b).wrapping_add(modulus);
        } else {
            *b = a_val - *b;
        }
    }
}
