use std::simd::Simd;

use primus_integer::{CarryingAdd, CarryingMul, SimdArray, SimdUnsignedInteger, WideningMul};
use primus_reduce::prelude::*;

use super::BarrettModulus;

use crate::common::{
    compact::simd::{simd_reduce_add, simd_reduce_sub},
    uint::simd::simd_reduce_once,
};

/// A modulus, using barrett reduction algorithm.
///
/// The struct stores the modulus number and some precomputed
/// data. Here, `b` = 2^T::BITS
///
/// It's efficient if many reductions are performed with a single modulus.
#[derive(Debug, Clone, Copy)]
pub struct SimdBarrettModulus<T: SimdUnsignedInteger, const N: usize>
where
    Simd<T, N>: SimdArray<T, N>,
{
    value: Simd<T, N>,
    ratio: [Simd<T, N>; 2],
}

impl<T: SimdUnsignedInteger, const N: usize> SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    #[inline]
    pub fn lazy_reduce_wide(&self, lo: Simd<T, N>, hi: Simd<T, N>) -> Simd<T, N> {
        let ah = lo.widening_mul_hw(self.ratio[0]);

        let b = lo.carrying_mul(self.ratio[1], ah);
        let c = hi.widening_mul(self.ratio[0]);

        let d = hi * self.ratio[1];

        let bch = b.1.carrying_add(c.1, b.0.overflowing_add(c.0).1).0;

        let q = d + bch;

        // Step 2.
        lo - (q * self.value)
    }
}

impl<T: SimdUnsignedInteger, const N: usize> From<BarrettModulus<T>> for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    #[inline]
    fn from(modulus: BarrettModulus<T>) -> Self {
        let ratio = modulus.ratio();
        Self {
            value: Simd::splat(modulus.value()),
            ratio: [Simd::splat(ratio[0]), Simd::splat(ratio[1])],
        }
    }
}

impl<T: SimdUnsignedInteger, const N: usize> LazyReduce<Simd<T, N>> for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn lazy_reduce(self, value: Simd<T, N>) -> Self::Output {
        let tmp = value.widening_mul_hw(self.ratio[0]); // tmp1
        let q = value.carrying_mul_hw(self.ratio[1], tmp); // q₃

        // Step 2.
        value - (q * self.value) // r = r₁ - r₂
    }
}

impl<T: SimdUnsignedInteger, const N: usize> LazyReduce<[Simd<T, N>; 2]>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn lazy_reduce(self, value: [Simd<T, N>; 2]) -> Self::Output {
        self.lazy_reduce_wide(value[0], value[1])
    }
}

impl<T: SimdUnsignedInteger, const N: usize> LazyReduce<(Simd<T, N>, Simd<T, N>)>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn lazy_reduce(self, value: (Simd<T, N>, Simd<T, N>)) -> Self::Output {
        self.lazy_reduce_wide(value.0, value.1)
    }
}

impl<T: SimdUnsignedInteger, const N: usize> LazyReduceAssign<Simd<T, N>>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    #[inline]
    fn lazy_reduce_assign(self, value: &mut Simd<T, N>) {
        *value = self.lazy_reduce(*value);
    }
}

impl<T: SimdUnsignedInteger, const N: usize> LazyReduceMul<Simd<T, N>> for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn lazy_reduce_mul(self, a: Simd<T, N>, b: Simd<T, N>) -> Self::Output {
        self.lazy_reduce(a.widening_mul(b))
    }
}

impl<T: SimdUnsignedInteger, const N: usize> LazyReduceMulAssign<Simd<T, N>>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    #[inline]
    fn lazy_reduce_mul_assign(self, a: &mut Simd<T, N>, b: Simd<T, N>) {
        *a = self.lazy_reduce(a.widening_mul(b));
    }
}

impl<T: SimdUnsignedInteger, const N: usize> LazyReduceMulAdd<Simd<T, N>>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn lazy_reduce_mul_add(self, a: Simd<T, N>, b: Simd<T, N>, c: Simd<T, N>) -> Self::Output {
        self.lazy_reduce(a.carrying_mul(b, c))
    }
}

impl<T: SimdUnsignedInteger, const N: usize> LazyReduceMulAddAssign<Simd<T, N>>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    #[inline]
    fn lazy_reduce_mul_add_assign(self, a: &mut Simd<T, N>, b: Simd<T, N>, c: Simd<T, N>) {
        *a = self.lazy_reduce(a.carrying_mul(b, c));
    }
}

impl<T: SimdUnsignedInteger, const N: usize> Reduce<Simd<T, N>> for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn reduce(self, value: Simd<T, N>) -> Self::Output {
        simd_reduce_once(self.lazy_reduce(value), self.value)
    }
}

impl<T: SimdUnsignedInteger, const N: usize> Reduce<[Simd<T, N>; 2]> for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn reduce(self, value: [Simd<T, N>; 2]) -> Self::Output {
        simd_reduce_once(self.lazy_reduce(value), self.value)
    }
}

impl<T: SimdUnsignedInteger, const N: usize> Reduce<(Simd<T, N>, Simd<T, N>)>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn reduce(self, value: (Simd<T, N>, Simd<T, N>)) -> Self::Output {
        simd_reduce_once(self.lazy_reduce(value), self.value)
    }
}

impl<T: SimdUnsignedInteger, const N: usize> ReduceAssign<Simd<T, N>> for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    #[inline]
    fn reduce_assign(self, value: &mut Simd<T, N>) {
        *value = self.reduce(*value);
    }
}

impl<T: SimdUnsignedInteger, const N: usize> ReduceMul<Simd<T, N>> for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn reduce_mul(self, a: Simd<T, N>, b: Simd<T, N>) -> Self::Output {
        self.reduce(a.widening_mul(b))
    }
}

impl<T: SimdUnsignedInteger, const N: usize> ReduceMulAssign<Simd<T, N>>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    #[inline]
    fn reduce_mul_assign(self, a: &mut Simd<T, N>, b: Simd<T, N>) {
        *a = self.reduce(a.widening_mul(b));
    }
}

impl<T: SimdUnsignedInteger, const N: usize> ReduceMulAdd<Simd<T, N>> for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    type Output = Simd<T, N>;

    #[inline]
    fn reduce_mul_add(self, a: Simd<T, N>, b: Simd<T, N>, c: Simd<T, N>) -> Self::Output {
        self.reduce(a.carrying_mul(b, c))
    }
}

impl<T: SimdUnsignedInteger, const N: usize> ReduceMulAddAssign<Simd<T, N>>
    for SimdBarrettModulus<T, N>
where
    Simd<T, N>: SimdArray<T, N>,
{
    #[inline]
    fn reduce_mul_add_assign(self, a: &mut Simd<T, N>, b: Simd<T, N>, c: Simd<T, N>) {
        *a = self.reduce(a.carrying_mul(b, c));
    }
}

// ===========================================================================
// SIMD slice kernels.
// ===========================================================================

#[inline]
pub fn lazy_reduce_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &mut [T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (ac, bc) in a_chunks.iter_mut().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *ac = sm.lazy_reduce_mul(av, bv).to_array();
    }

    for (a, &b) in a_rem.iter_mut().zip(b_rem) {
        *a = modulus.lazy_reduce_mul(*a, b)
    }
}

#[inline]
pub fn lazy_reduce_mul_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &[T],
    b: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *oc = sm.lazy_reduce_mul(av, bv).to_array();
    }

    for ((&a, &b), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        *o = modulus.lazy_reduce_mul(a, b)
    }
}

#[inline]
pub fn lazy_reduce_scalar_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &mut [T],
    scalar: T,
) where
    Simd<T, N>: SimdArray<T, N>,
{
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let sv = Simd::splat(scalar);
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    for ac in a_chunks {
        let av = Simd::from_array(*ac);
        *ac = sm.lazy_reduce_mul(av, sv).to_array();
    }

    for a in a_rem {
        *a = modulus.lazy_reduce_mul(*a, scalar)
    }
}

#[inline]
pub fn lazy_reduce_scalar_mul_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &[T],
    scalar: T,
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), output.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let sv = Simd::splat(scalar);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for (ac, oc) in a_chunks.iter().zip(o_chunks) {
        let av = Simd::from_array(*ac);
        *oc = sm.lazy_reduce_mul(av, sv).to_array();
    }

    for (&a, o) in a_rem.iter().zip(o_rem) {
        *o = modulus.lazy_reduce_mul(a, scalar)
    }
}

#[inline]
pub fn reduce_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &mut [T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (ac, bc) in a_chunks.iter_mut().zip(b_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *ac = sm.reduce_mul(av, bv).to_array();
    }

    for (a, &b) in a_rem.iter_mut().zip(b_rem) {
        *a = modulus.reduce_mul(*a, b)
    }
}

#[inline]
pub fn reduce_mul_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &[T],
    b: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), output.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *oc = sm.reduce_mul(av, bv).to_array();
    }

    for ((&a, &b), o) in a_rem.iter().zip(b_rem).zip(o_rem) {
        *o = modulus.reduce_mul(a, b)
    }
}

#[inline]
pub fn reduce_scalar_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &mut [T],
    scalar: T,
) where
    Simd<T, N>: SimdArray<T, N>,
{
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let sv = Simd::splat(scalar);
    let (a_chunks, a_rem) = a.as_chunks_mut::<N>();
    for ac in a_chunks {
        let av = Simd::from_array(*ac);
        *ac = sm.reduce_mul(av, sv).to_array();
    }

    for a in a_rem {
        *a = modulus.reduce_mul(*a, scalar)
    }
}

#[inline]
pub fn reduce_scalar_mul_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &[T],
    scalar: T,
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), output.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let sv = Simd::splat(scalar);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for (ac, oc) in a_chunks.iter().zip(o_chunks) {
        let av = Simd::from_array(*ac);
        *oc = sm.reduce_mul(av, sv).to_array();
    }

    for (&a, o) in a_rem.iter().zip(o_rem) {
        *o = modulus.reduce_mul(a, scalar)
    }
}

#[inline]
pub fn reduce_add_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    a: &[T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let (acc_chunks, acc_rem) = acc.as_chunks_mut::<N>();
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for ((accc, ac), bc) in acc_chunks.iter_mut().zip(a_chunks).zip(b_chunks) {
        let accv = Simd::from_array(*accc);
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *accc = sm.reduce_mul_add(av, bv, accv).to_array();
    }

    for ((acc, &a), &b) in acc_rem.iter_mut().zip(a_rem).zip(b_rem) {
        *acc = modulus.reduce_mul_add(a, b, *acc)
    }
}

#[inline]
pub fn reduce_add_scalar_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    scalar: T,
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(acc.len(), b.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let sv = Simd::splat(scalar);
    let (acc_chunks, acc_rem) = acc.as_chunks_mut::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for (accc, bc) in acc_chunks.iter_mut().zip(b_chunks) {
        let accv = Simd::from_array(*accc);
        let bv = Simd::from_array(*bc);
        *accc = sm.reduce_mul_add(sv, bv, accv).to_array();
    }

    for (acc, &b) in acc_rem.iter_mut().zip(b_rem) {
        *acc = modulus.reduce_mul_add(scalar, b, *acc)
    }
}

#[inline]
pub fn reduce_sub_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    a: &[T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let m = Simd::splat(modulus.value());
    let (acc_chunks, acc_rem) = acc.as_chunks_mut::<N>();
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for ((accc, ac), bc) in acc_chunks.iter_mut().zip(a_chunks).zip(b_chunks) {
        let accv = Simd::from_array(*accc);
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        let prod = sm.reduce_mul(av, bv);
        *accc = simd_reduce_sub(accv, prod, m).to_array();
    }

    for ((acc, &a), &b) in acc_rem.iter_mut().zip(a_rem).zip(b_rem) {
        let prod = modulus.reduce_mul(a, b);
        modulus.reduce_sub_assign(acc, prod);
    }
}

#[inline]
pub fn reduce_mul_add_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
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
    let sm = SimdBarrettModulus::<T, N>::from(modulus);

    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (c_chunks, c_rem) = c.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for (((ac, bc), cc), oc) in a_chunks.iter().zip(b_chunks).zip(c_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        let cv = Simd::from_array(*cc);
        *oc = sm.reduce_mul_add(av, bv, cv).to_array();
    }

    for (((&a, &b), &c), o) in a_rem.iter().zip(b_rem).zip(c_rem).zip(o_rem) {
        *o = modulus.reduce_mul_add(a, b, c)
    }
}

#[inline]
pub fn reduce_scalar_mul_add_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    scalar: T,
    b: &[T],
    c: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(b.len(), c.len());
    debug_assert_eq!(b.len(), output.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let sv = Simd::splat(scalar);
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (c_chunks, c_rem) = c.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((bc, cc), oc) in b_chunks.iter().zip(c_chunks).zip(o_chunks) {
        let bv = Simd::from_array(*bc);
        let cv = Simd::from_array(*cc);
        *oc = sm.reduce_mul_add(sv, bv, cv).to_array();
    }

    for ((&b, &c), o) in b_rem.iter().zip(c_rem).zip(o_rem) {
        *o = modulus.reduce_mul_add(scalar, b, c)
    }
}

#[inline]
pub fn lazy_reduce_add_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    a: &[T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let (acc_chunks, acc_rem) = acc.as_chunks_mut::<N>();
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for ((accc, ac), bc) in acc_chunks.iter_mut().zip(a_chunks).zip(b_chunks) {
        let accv = Simd::from_array(*accc);
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        *accc = sm.lazy_reduce_mul_add(av, bv, accv).to_array();
    }

    for ((acc, &a), &b) in acc_rem.iter_mut().zip(a_rem).zip(b_rem) {
        *acc = modulus.lazy_reduce_mul_add(a, b, *acc)
    }
}

#[inline]
pub fn lazy_reduce_sub_mul_slice_assign<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    acc: &mut [T],
    a: &[T],
    b: &[T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(acc.len(), a.len());
    debug_assert_eq!(acc.len(), b.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let (acc_chunks, acc_rem) = acc.as_chunks_mut::<N>();
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    for ((accc, ac), bc) in acc_chunks.iter_mut().zip(a_chunks).zip(b_chunks) {
        let accv = Simd::from_array(*accc);
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        let prod = sm.reduce_mul(av, bv);
        let diff = accv + sm.value - prod;
        *accc = diff.to_array();
    }

    for ((acc, &a), &b) in acc_rem.iter_mut().zip(a_rem).zip(b_rem) {
        let prod = modulus.reduce_mul(a, b);
        *acc = acc.wrapping_add(modulus.value).wrapping_sub(prod);
    }
}

#[inline]
pub fn lazy_reduce_mul_add_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
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
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let (a_chunks, a_rem) = a.as_chunks::<N>();
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (c_chunks, c_rem) = c.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for (((ac, bc), cc), oc) in a_chunks.iter().zip(b_chunks).zip(c_chunks).zip(o_chunks) {
        let av = Simd::from_array(*ac);
        let bv = Simd::from_array(*bc);
        let cv = Simd::from_array(*cc);
        *oc = sm.lazy_reduce_mul_add(av, bv, cv).to_array();
    }

    for (((&a, &b), &c), o) in a_rem.iter().zip(b_rem).zip(c_rem).zip(o_rem) {
        *o = modulus.lazy_reduce_mul_add(a, b, c)
    }
}

#[inline]
pub fn lazy_reduce_scalar_mul_add_slice_to<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    scalar: T,
    b: &[T],
    c: &[T],
    output: &mut [T],
) where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(b.len(), c.len());
    debug_assert_eq!(b.len(), output.len());
    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let sv = Simd::splat(scalar);
    let (b_chunks, b_rem) = b.as_chunks::<N>();
    let (c_chunks, c_rem) = c.as_chunks::<N>();
    let (o_chunks, o_rem) = output.as_chunks_mut::<N>();
    for ((bc, cc), oc) in b_chunks.iter().zip(c_chunks).zip(o_chunks) {
        let bv = Simd::from_array(*bc);
        let cv = Simd::from_array(*cc);
        *oc = sm.lazy_reduce_mul_add(sv, bv, cv).to_array();
    }

    for ((&b, &c), o) in b_rem.iter().zip(c_rem).zip(o_rem) {
        *o = modulus.lazy_reduce_mul_add(scalar, b, c)
    }
}

// ---------------------------------------------------------------------------
// SIMD dot_product
//
// Outer chunk size = `K * N`, where `K = super::slice::DOT_PRODUCT_INNER_CHUNK`
// (currently 16). Inside each outer chunk we accumulate `K` SIMD widening
// products into a `[Simd<T, N>; 2]` double-word per lane, then collapse the
// double-word back into a single SIMD word in `[0, m)` via Barrett + the
// `min(v, v - m)` reduce_once trick. Cross-chunk accumulation stays in `[0, m)`
// lane-wise via `simd_reduce_add`, so the running SIMD accumulator never grows.
// After the chunked loop, a horizontal modular sum collapses the N lanes to a
// scalar, and any tail shorter than `K * N` is handled by the scalar helper.
//
// Hi-limb safety: each scalar widening product has `hi < m^2 / 2^BITS`, and the
// lo-limb's running sum can carry at most `K - 1` extra units into hi. With
// `m < 2^(BITS - 1)` enforced by `BarrettModulus::new` and `K ≤ 16`, both
// limbs stay strictly below `2^BITS` — identical bound to the scalar path.
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn simd_multiply_add<T: SimdUnsignedInteger, const N: usize>(
    c: &mut [Simd<T, N>; 2],
    a: Simd<T, N>,
    b: Simd<T, N>,
) where
    Simd<T, N>: SimdArray<T, N>,
{
    let (lw, hw) = a.widening_mul(b);
    let carry;
    (c[0], carry) = c[0].overflowing_add(lw);
    (c[1], _) = c[1].carrying_add(hw, carry);
}

#[inline]
pub fn reduce_dot_product<T: SimdUnsignedInteger, const N: usize>(
    modulus: BarrettModulus<T>,
    a: &[T],
    b: &[T],
) -> T
where
    Simd<T, N>: SimdArray<T, N>,
{
    debug_assert_eq!(a.len(), b.len(), "reduce_dot_product: length mismatch");

    let k = super::slice::DOT_PRODUCT_INNER_CHUNK;
    let outer = k * N;

    let sm = SimdBarrettModulus::<T, N>::from(modulus);
    let m_simd = Simd::splat(modulus.value());

    let mut total_acc = Simd::<T, N>::splat(T::ZERO);

    let mut a_outer = a.chunks_exact(outer);
    let mut b_outer = b.chunks_exact(outer);

    for (a_chunk, b_chunk) in (&mut a_outer).zip(&mut b_outer) {
        // Each outer chunk is exactly `K * N` elements, so the inner
        // `as_chunks::<N>` always splits into `K` lane-wide subchunks with
        // an empty tail.
        let (a_lanes, _) = a_chunk.as_chunks::<N>();
        let (b_lanes, _) = b_chunk.as_chunks::<N>();
        let mut c = [Simd::<T, N>::splat(T::ZERO); 2];
        for (a_n, b_n) in a_lanes.iter().zip(b_lanes) {
            let av = Simd::<T, N>::from_array(*a_n);
            let bv = Simd::<T, N>::from_array(*b_n);
            simd_multiply_add(&mut c, av, bv);
        }
        let r = sm.reduce(c);
        total_acc = simd_reduce_add(total_acc, r, m_simd);
    }

    let tail_result = {
        let a: &[T] = a_outer.remainder();
        let b: &[T] = b_outer.remainder();

        let mut a_iter = a.chunks_exact(k);
        let mut b_iter = b.chunks_exact(k);
        let inter = (&mut a_iter)
            .zip(&mut b_iter)
            .map(|(a_s, b_s)| {
                let mut c: [T; 2] = [T::ZERO, T::ZERO];
                for (&a, &b) in a_s.iter().zip(b_s) {
                    super::slice::multiply_add(&mut c, a, b);
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
                super::slice::multiply_add(&mut c, a, b);
            });
        modulus.reduce_add(modulus.reduce(c), inter)
    };

    let lanes = total_acc.to_array();
    let mut result = T::ZERO;
    for v in lanes {
        result = modulus.reduce_add(result, v);
    }
    modulus.reduce_add(result, tail_result)
}
