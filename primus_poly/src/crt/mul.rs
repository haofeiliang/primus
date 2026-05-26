use primus_data::{Data, DataMut, RawData};
use primus_factor::FactorSliceOps;
use primus_integer::{FheUint, izip};
use primus_reduce::{ReduceMulAddSlice, ReduceMulSlice, ReduceNegAssign};

use crate::ArrayBase;

use super::CrtPolynomial;

impl<S, T> CrtPolynomial<S>
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

    /// Performs `self += scalar * rhs` according to `moduli`.
    #[inline]
    pub fn add_mul_scalar_assign<M, A>(
        &mut self,
        rhs: &CrtPolynomial<A>,
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

    /// Performs `self += scalar * rhs` according to `moduli`.
    #[inline]
    pub fn add_mul_factor_assign<F, A>(
        &mut self,
        rhs: &CrtPolynomial<A>,
        factor: &[F],
        poly_length: usize,
        moduli_value: &[T],
    ) where
        F: Copy + FactorSliceOps<T>,
        A: RawData<Elem = T> + Data,
    {
        izip!(
            self.iter_each_modulus_mut(poly_length),
            rhs.iter_each_modulus(poly_length),
            factor,
            moduli_value
        )
        .for_each(|(xs, ys, &f, &modulus)| {
            ArrayBase(xs).add_mul_factor_assign(&ArrayBase(ys), f, modulus);
        });
    }

    pub fn mul_monomial_assign<M>(&mut self, r: usize, poly_length: usize, moduli: &[M])
    where
        M: Copy + ReduceNegAssign<T>,
    {
        if r < poly_length {
            let rotate = |poly: &mut [T], modulus: M| {
                poly.rotate_right(r);
                poly[0..r]
                    .iter_mut()
                    .for_each(|v| modulus.reduce_neg_assign(v));
            };

            self.iter_each_modulus_mut(poly_length)
                .zip(moduli)
                .for_each(|(poly, &modulus)| rotate(poly, modulus));
        } else {
            debug_assert!(r < poly_length * 2);
            let r = r - poly_length;

            let rotate = |poly: &mut [T], modulus: M| {
                poly.rotate_right(r);
                poly[r..]
                    .iter_mut()
                    .for_each(|v| modulus.reduce_neg_assign(v));
            };

            self.iter_each_modulus_mut(poly_length)
                .zip(moduli)
                .for_each(|(poly, &modulus)| rotate(poly, modulus));
        }
    }
}

impl<S, T> CrtPolynomial<S>
where
    S: RawData<Elem = T> + Data,
    T: FheUint,
{
    /// Performs `result = self * scalar` according to `moduli`.
    #[inline]
    pub fn mul_scalar_inplace<M, A>(
        &self,
        scalar: &[T],
        result: &mut CrtPolynomial<A>,
        poly_length: usize,
        moduli: &[M],
    ) where
        M: Copy + ReduceMulSlice<T>,
        A: RawData<Elem = T> + DataMut,
    {
        izip!(
            self.iter_each_modulus(poly_length),
            scalar,
            result.iter_each_modulus_mut(poly_length),
            moduli
        )
        .for_each(|(in_poly, &scalar, out_poly, &modulus)| {
            ArrayBase(in_poly).mul_scalar_inplace(scalar, &mut ArrayBase(out_poly), modulus)
        })
    }

    /// Performs `result = self * scalar` according to `moduli`.
    #[inline]
    pub fn mul_factor_inplace<F, A>(
        &self,
        factor: &[F],
        result: &mut CrtPolynomial<A>,
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
