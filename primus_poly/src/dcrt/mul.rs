use primus_data::{Data, DataMut, RawData};
use primus_factor::{FactorMul, FactorSliceOps, ShoupFactor};
use primus_integer::{FheUint, izip};
use primus_modulus::CompactModulus;
use primus_reduce::{ReduceAddAssign, ReduceMulAddSlice, ReduceMulSlice, ReduceSub};

#[cfg(feature = "simd")]
use primus_factor::SimdShoupFactor;
#[cfg(feature = "simd")]
use primus_integer::SimdUnsignedInteger;
#[cfg(feature = "simd")]
use std::simd::{Simd, cmp::SimdOrd};

use crate::ArrayBase;

use super::DcrtPolynomial;

// ===========================================================================
// Butterfly dispatch trait — scalar / SIMD (min_specialization).
// ===========================================================================

trait ButterflyDispatch: FheUint {
    fn butterfly_inner(
        lhs: &mut [Self],
        rhs: &[Self],
        w: &[ShoupFactor<Self>],
        result: &mut [Self],
        modulus: Self,
    );
}

macro_rules! impl_butterfly_blanket {
    ($($default_kw:ident)?) => {
        impl<T: FheUint> ButterflyDispatch for T {
            $($default_kw)? fn butterfly_inner(
                lhs: &mut [Self],
                rhs: &[Self],
                w: &[ShoupFactor<Self>],
                result: &mut [Self],
                modulus: Self,
            ) {
                izip!(lhs, rhs, w, result).for_each(|(a, &s, &w, b)| {
                    let a_orig = *a;
                    CompactModulus(modulus).reduce_add_assign(a, s);
                    let diff = CompactModulus(modulus).reduce_sub(a_orig, s);
                    *b = w.factor_mul_modulo(diff, modulus);
                });
            }
        }
    };
}

#[cfg(not(feature = "simd"))]
impl_butterfly_blanket!();

#[cfg(feature = "simd")]
impl_butterfly_blanket!(default);

#[cfg(feature = "simd")]
macro_rules! impl_butterfly_simd {
    ($t:ty, $lanes:expr) => {
        impl ButterflyDispatch for $t {
            fn butterfly_inner(
                lhs: &mut [Self],
                rhs: &[Self],
                w: &[ShoupFactor<Self>],
                result: &mut [Self],
                modulus: Self,
            ) {
                let m = Simd::splat(modulus);
                let (lhs_chunks, lhs_rem) = lhs.as_chunks_mut::<{ $lanes }>();
                let (rhs_chunks, rhs_rem) = rhs.as_chunks::<{ $lanes }>();
                let (w_chunks, w_rem) = w.as_chunks::<{ $lanes }>();
                let (res_chunks, res_rem) = result.as_chunks_mut::<{ $lanes }>();

                for (((l, r), w_arr), res) in lhs_chunks
                    .iter_mut()
                    .zip(rhs_chunks)
                    .zip(w_chunks)
                    .zip(res_chunks)
                {
                    let a = Simd::from_array(*l);
                    let s = Simd::from_array(*r);
                    let w_simd = SimdShoupFactor::<$t, { $lanes }>::from_array(*w_arr);

                    // diff = a - s (mod m)
                    let diff = a - s;
                    let diff = diff.simd_min(diff + m);

                    // a_new = a + s (mod m) — reuses original a from load
                    let sum = a + s;
                    *l = sum.simd_min(sum - m).to_array();

                    // b = w * diff (mod m)
                    *res = w_simd.factor_mul_modulo(diff, m).to_array();
                }

                // scalar remainder
                let m_ctx = CompactModulus(modulus);
                for (((a, &s), &w), b) in lhs_rem.iter_mut().zip(rhs_rem).zip(w_rem).zip(res_rem) {
                    let a_orig = *a;
                    m_ctx.reduce_add_assign(a, s);
                    let diff = m_ctx.reduce_sub(a_orig, s);
                    *b = w.factor_mul_modulo(diff, modulus);
                }
            }
        }
    };
}

#[cfg(feature = "simd")]
impl_butterfly_simd!(u16, u16::LANE_COUNT);
#[cfg(feature = "simd")]
impl_butterfly_simd!(u32, u32::LANE_COUNT);
#[cfg(feature = "simd")]
impl_butterfly_simd!(u64, u64::LANE_COUNT);

impl<S, T> DcrtPolynomial<S>
where
    S: RawData<Elem = T> + DataMut,
    T: FheUint,
{
    /// Performs `self * scalar` according to `moduli`.
    #[inline]
    pub fn mul_scalar<M>(mut self, scalar: &[T], poly_length: usize, moduli: &[M]) -> Self
    where
        M: Copy + ReduceMulSlice<T>,
    {
        self.mul_scalar_assign(scalar, poly_length, moduli);
        self
    }

    /// Performs `self *= scalar` according to `moduli`.
    #[inline]
    pub fn mul_scalar_assign<M>(&mut self, scalar: &[T], poly_length: usize, moduli: &[M])
    where
        M: Copy + ReduceMulSlice<T>,
    {
        izip!(self.iter_each_modulus_mut(poly_length), scalar, moduli).for_each(
            |(poly, &scalar, &modulus)| ArrayBase(poly).mul_scalar_assign(scalar, modulus),
        )
    }

    /// Performs `self += scalar * rhs` according to `moduli`.
    #[inline]
    pub fn add_mul_scalar_assign<M, A>(
        &mut self,
        rhs: &DcrtPolynomial<A>,
        scalar: &[T],
        poly_length: usize,
        moduli: &[M],
    ) where
        M: Copy + ReduceMulAddSlice<T>,
        A: RawData<Elem = T> + Data,
    {
        izip!(
            self.iter_each_modulus_mut(poly_length),
            rhs.iter_each_modulus(poly_length),
            scalar,
            moduli
        )
        .for_each(|(xs, ys, &scalar, &modulus)| {
            ArrayBase(xs).add_mul_scalar_assign(&ArrayBase(ys), scalar, modulus);
        });
    }

    /// Performs `self * scalar` according to `moduli`.
    #[inline]
    pub fn mul_factor<F>(mut self, factor: &[F], poly_length: usize, moduli: &[T]) -> Self
    where
        F: Copy + FactorSliceOps<T>,
    {
        self.mul_factor_assign(factor, poly_length, moduli);
        self
    }

    /// Performs `self *= scalar` according to `moduli`.
    #[inline]
    pub fn mul_factor_assign<F>(&mut self, factor: &[F], poly_length: usize, moduli: &[T])
    where
        F: Copy + FactorSliceOps<T>,
    {
        izip!(self.iter_each_modulus_mut(poly_length), factor, moduli)
            .for_each(|(poly, &f, &modulus)| ArrayBase(poly).mul_factor_assign(f, modulus))
    }

    /// Performs `self * rhs` according to `moduli`.
    #[inline]
    pub fn mul<M, A>(mut self, rhs: &DcrtPolynomial<A>, poly_length: usize, moduli: &[M]) -> Self
    where
        M: Copy + ReduceMulSlice<T>,
        A: RawData<Elem = T> + Data,
    {
        self.mul_assign(rhs, poly_length, moduli);
        self
    }

    /// Performs `self *= rhs` according to `moduli`.
    #[inline]
    pub fn mul_assign<M, A>(&mut self, rhs: &DcrtPolynomial<A>, poly_length: usize, moduli: &[M])
    where
        M: Copy + ReduceMulSlice<T>,
        A: RawData<Elem = T> + Data,
    {
        izip!(
            self.iter_each_modulus_mut(poly_length),
            rhs.iter_each_modulus(poly_length),
            moduli
        )
        .for_each(|(xs, ys, &modulus)| {
            ArrayBase(xs).mul_element_wise_assign(&ArrayBase(ys), modulus)
        })
    }

    /// Inverse butterfly with a Shoup-factor polynomial.
    ///
    /// `(self, result) = (self + rhs, (self_orig - rhs) * w)`.
    ///
    /// `self` and `rhs` are expected in `[0, q)`. Both outputs are written
    /// back in `[0, q)`.
    #[inline]
    pub fn butterfly_mul_factor_inplace<A, B>(
        &mut self,
        rhs: &DcrtPolynomial<A>,
        w: &[ShoupFactor<T>],
        result: &mut DcrtPolynomial<B>,
        poly_length: usize,
        moduli: &[T],
    ) where
        A: RawData<Elem = T> + Data,
        B: RawData<Elem = T> + DataMut,
    {
        let lhs = self.0.as_mut_slice();
        let rhs = rhs.0.as_slice();
        let result = result.0.as_mut_slice();

        debug_assert_eq!(lhs.len(), rhs.len());
        debug_assert_eq!(lhs.len(), result.len());
        debug_assert_eq!(lhs.len(), w.len());
        debug_assert_eq!(lhs.len(), poly_length * moduli.len());

        izip!(
            lhs.chunks_exact_mut(poly_length),
            rhs.chunks_exact(poly_length),
            w.chunks_exact(poly_length),
            result.chunks_exact_mut(poly_length),
            moduli
        )
        .for_each(|(lhs, rhs, w, result, &modulus)| {
            T::butterfly_inner(lhs, rhs, w, result, modulus);
        })
    }
}

impl<S, T> DcrtPolynomial<S>
where
    S: RawData<Elem = T> + Data,
    T: FheUint,
{
    /// Performs `result = self * rhs` according to `moduli`.
    #[inline]
    pub fn mul_inplace<M, A, B>(
        &self,
        rhs: &DcrtPolynomial<A>,
        result: &mut DcrtPolynomial<B>,
        poly_length: usize,
        moduli: &[M],
    ) where
        M: Copy + ReduceMulSlice<T>,
        A: RawData<Elem = T> + Data,
        B: RawData<Elem = T> + DataMut,
    {
        izip!(
            self.iter_each_modulus(poly_length),
            rhs.iter_each_modulus(poly_length),
            result.iter_each_modulus_mut(poly_length),
            moduli
        )
        .for_each(|(xs, ys, zs, modulus)| {
            ArrayBase(xs).mul_element_wise_inplace(&ArrayBase(ys), &mut ArrayBase(zs), *modulus);
        })
    }

    /// Performs `result = self * scalar` according to `moduli`.
    #[inline]
    pub fn mul_factor_inplace<F, A>(
        &self,
        factor: &[F],
        result: &mut DcrtPolynomial<A>,
        poly_length: usize,
        moduli: &[T],
    ) where
        F: Copy + FactorSliceOps<T>,
        A: RawData<Elem = T> + DataMut,
    {
        izip!(
            self.iter_each_modulus(poly_length),
            result.iter_each_modulus_mut(poly_length),
            factor,
            moduli
        )
        .for_each(|(in_poly, out_poly, &f, &modulus)| {
            ArrayBase(in_poly).mul_factor_inplace(f, &mut ArrayBase(out_poly), modulus)
        })
    }
}
