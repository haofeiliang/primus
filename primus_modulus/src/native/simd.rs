use std::simd::{Simd, num::SimdUint};

use primus_integer::{SimdArray, SimdUnsignedInteger};

// ===========================================================================
// SIMD slice kernels for NativeModulus.
//
// Every operation is just the corresponding wrapping Simd operation
// because `mod 2^BITS` is implicit in the arithmetic.
// ===========================================================================
//
// ── ReduceOnce / ReduceNeg / ReduceAdd / ReduceSub ──
// ===========================================================================

#[inline]
pub fn reduce_neg_slice_assign<T: SimdUnsignedInteger, const N: usize>(values: &mut [T])
where
    Simd<T, N>: SimdArray<T, N>,
{
    let (chunks, rem) = values.as_chunks_mut::<N>();
    for chunk in chunks {
        let v = Simd::from_array(*chunk);
        *chunk = v.wrapping_neg().to_array();
    }
    for v in rem {
        *v = v.wrapping_neg();
    }
}

#[inline]
pub fn reduce_neg_slice_to<T: SimdUnsignedInteger, const N: usize>(input: &[T], output: &mut [T])
where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(input.len(), output.len());
    let (in_chunks, in_rem) = input.as_chunks::<N>();
    let (out_chunks, out_rem) = output.as_chunks_mut::<N>();
    for (i, o) in in_chunks.iter().zip(out_chunks) {
        let v = Simd::from_array(*i);
        *o = v.wrapping_neg().to_array();
    }
    for (&i, o) in in_rem.iter().zip(out_rem) {
        *o = i.wrapping_neg();
    }
}

#[inline]
pub fn reduce_add_slice_assign<T: SimdUnsignedInteger, const N: usize>(a: &mut [T], b: &[T])
where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (ac, bc) in a_chunks.iter_mut().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *ac = (av + bv).to_array();
    }
    for (a, &b) in a_rem.iter_mut().zip(b_rem) {
        *a = a.wrapping_add(b);
    }
}

#[inline]
pub fn reduce_add_slice_to<T: SimdUnsignedInteger, const N: usize>(
    a: &[T],
    b: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *oc = (av + bv).to_array();
    }
    for ((&a_val, &b_val), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        *o = a_val.wrapping_add(b_val);
    }
}

#[inline]
pub fn reduce_sub_slice_assign<T: SimdUnsignedInteger, const N: usize>(a: &mut [T], b: &[T])
where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (ac, bc) in a_chunks.iter_mut().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *ac = (av - bv).to_array();
    }
    for (a, &b) in a_rem.iter_mut().zip(b_rem) {
        *a = a.wrapping_sub(b);
    }
}

#[inline]
pub fn reduce_sub_slice_to<T: SimdUnsignedInteger, const N: usize>(
    a: &[T],
    b: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *oc = (av - bv).to_array();
    }
    for ((&a_val, &b_val), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        *o = a_val.wrapping_sub(b_val);
    }
}

#[inline]
pub fn reduce_sub_slice_rev_assign<T: SimdUnsignedInteger, const N: usize>(a: &[T], b: &mut [T])
where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks_mut::<N>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *bc = (av - bv).to_array();
    }
    for (&a, b) in a_rem.iter().zip(b_rem) {
        *b = a.wrapping_sub(*b);
    }
}

// ===========================================================================
// ReduceDotProduct
// ===========================================================================

#[inline]
pub fn reduce_dot_product<T: SimdUnsignedInteger, const N: usize>(a: &[T], b: &[T]) -> T
where
    Simd<T, N>: SimdArray<T, N>,
{
    assert_eq!(a.len(), b.len(), "reduce_dot_product: length mismatch");
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let mut acc = Simd::<T, N>::splat(T::ZERO);
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        acc += av * bv;
    }
    let mut result = acc.reduce_sum();
    for (&a_val, &b_val) in a_rem.iter().zip(b_rem) {
        result = result.wrapping_add(a_val.wrapping_mul(b_val));
    }
    result
}

// ===========================================================================
// ReduceMulSlice
// ===========================================================================

#[inline]
pub fn reduce_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(a: &mut [T], b: &[T])
where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (ac, bc) in a_chunks.iter_mut().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *ac = (av * bv).to_array();
    }
    for (a, &b) in a_rem.iter_mut().zip(b_rem) {
        *a = a.wrapping_mul(b);
    }
}

#[inline]
pub fn reduce_mul_slice_to<T: SimdUnsignedInteger, const N: usize>(
    a: &[T],
    b: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *oc = (av * bv).to_array();
    }
    for ((&a, &b), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        *o = a.wrapping_mul(b);
    }
}

#[inline]
pub fn reduce_scalar_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    a: &mut [T],
    scalar: T,
) where
    Simd<T, N>: SimdArray<T, N>,
{
    let s = Simd::<T, N>::splat(scalar);
    let (chunks, rem) = a.as_chunks_mut::<N>();
    for chunk in chunks {
        let v = Simd::from_array(*chunk);
        *chunk = (v * s).to_array();
    }
    for v in rem {
        *v = v.wrapping_mul(scalar);
    }
}

#[inline]
pub fn reduce_scalar_mul_slice_to<T: SimdUnsignedInteger, const N: usize>(
    a: &[T],
    scalar: T,
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), output.len());
    let s = Simd::<T, N>::splat(scalar);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for (ac, oc) in a_chunks.iter().zip(o_chunks) {
        let v = Simd::from_array(*ac);
        *oc = (v * s).to_array();
    }
    for (&a, o) in a_rem.iter().zip(o_rem) {
        *o = a.wrapping_mul(scalar);
    }
}

// ===========================================================================
// ReduceMulAddSlice
// ===========================================================================

#[inline]
pub fn reduce_add_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    acc: &mut [T],
    a: &[T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    let (acc_chunks, acc_rem) = acc.as_chunks_mut::<N>();
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for ((ac, av_slice), bv_slice) in acc_chunks.iter_mut().zip(a_chunks).zip(b_chunks) {
        let acc_v = Simd::from_array(*ac);
        let a_v = Simd::from_array(*av_slice);
        let b_v = Simd::from_array(*bv_slice);
        *ac = (acc_v + a_v * b_v).to_array();
    }
    for ((acc, &a), &b) in acc_rem.iter_mut().zip(a_rem).zip(b_rem) {
        *acc = acc.wrapping_add(a.wrapping_mul(b));
    }
}

#[inline]
pub fn reduce_sub_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    acc: &mut [T],
    a: &[T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    let (acc_chunks, acc_rem) = acc.as_chunks_mut::<N>();
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for ((ac, av_slice), bv_slice) in acc_chunks.iter_mut().zip(a_chunks).zip(b_chunks) {
        let acc_v = Simd::from_array(*ac);
        let a_v = Simd::from_array(*av_slice);
        let b_v = Simd::from_array(*bv_slice);
        *ac = (acc_v - a_v * b_v).to_array();
    }
    for ((acc, &a), &b) in acc_rem.iter_mut().zip(a_rem).zip(b_rem) {
        *acc = acc.wrapping_sub(a.wrapping_mul(b));
    }
}

#[inline]
pub fn reduce_add_scalar_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    acc: &mut [T],
    scalar: T,
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(acc.len(), b.len());
    let s = Simd::<T, N>::splat(scalar);
    let (acc_chunks, acc_rem) = acc.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (ac, bc) in acc_chunks.iter_mut().zip(b_chunks) {
        let acc_v = Simd::from_array(*ac);
        let b_v = Simd::from_array(*bc);
        *ac = (acc_v + s * b_v).to_array();
    }
    for (acc, &b) in acc_rem.iter_mut().zip(b_rem) {
        *acc = acc.wrapping_add(scalar.wrapping_mul(b));
    }
}

#[inline]
pub fn reduce_mul_add_slice_to<T: SimdUnsignedInteger, const N: usize>(
    a: &[T],
    b: &[T],
    c: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), output.len());
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (c_chunks, c_rem) = c.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for (((ac, bc), cc), oc) in a_chunks.iter().zip(b_chunks).zip(c_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        let cv = Simd::from_array(*cc);
        *oc = (av * bv + cv).to_array();
    }
    for (((&a, &b), &c), o) in a_rem.iter().zip(b_rem).zip(c_rem).zip(o_rem) {
        *o = a.wrapping_mul(b).wrapping_add(c);
    }
}

#[inline]
pub fn reduce_scalar_mul_add_slice_to<T: SimdUnsignedInteger, const N: usize>(
    scalar: T,
    b: &[T],
    c: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(b.len(), c.len());
    debug_assert_eq!(b.len(), output.len());
    let s = Simd::<T, N>::splat(scalar);
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (c_chunks, c_rem) = c.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((bc, cc), oc) in b_chunks.iter().zip(c_chunks).zip(o_chunks) {
        let bv = Simd::from_array(*bc);
        let cv = Simd::from_array(*cc);
        *oc = (s * bv + cv).to_array();
    }
    for ((&b, &c), o) in b_rem.iter().zip(c_rem).zip(o_rem) {
        *o = scalar.wrapping_mul(b).wrapping_add(c);
    }
}
