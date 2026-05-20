use primus_factor::FactorSliceOps;
use primus_integer::{Data, DataMut, RawData, UnsignedInteger, izip};
use primus_reduce::{ReduceAdd, ReduceMul, ReduceMulAddSlice, ReduceMulSlice, ReduceSub};

use super::ArrayBase;

impl<S, T> ArrayBase<S>
where
    S: RawData<Elem = T> + DataMut,
    T: UnsignedInteger,
{
    /// Performs `self *= scalar` according to `modulus`.
    #[inline]
    pub fn mul_scalar_assign<M>(&mut self, scalar: T, modulus: M)
    where
        M: Copy + ReduceMulSlice<T>,
    {
        modulus.reduce_scalar_mul_slice_assign(self.as_mut(), scalar);
    }

    /// Performs `self += scalar * rhs` according to `modulus`.
    #[inline]
    pub fn add_mul_scalar_assign<M, A>(&mut self, rhs: &ArrayBase<A>, scalar: T, modulus: M)
    where
        M: Copy + ReduceMulAddSlice<T>,
        A: RawData<Elem = T> + Data,
    {
        modulus.reduce_add_scalar_mul_slice_assign(self.as_mut(), scalar, rhs.as_ref());
    }

    /// Performs `self *= scalar` according to `modulus`.
    #[inline]
    pub fn mul_factor_assign<F>(&mut self, factor: F, modulus: T)
    where
        F: FactorSliceOps<T>,
    {
        factor.factor_mul_slice_assign(self.as_mut(), modulus)
    }

    /// Performs `self += scalar * rhs` according to `modulus`.
    #[inline]
    pub fn add_mul_factor_assign<F, A>(&mut self, rhs: &ArrayBase<A>, factor: F, modulus: T)
    where
        F: FactorSliceOps<T>,
        A: RawData<Elem = T> + Data,
    {
        debug_assert_eq!(self.len(), rhs.len());
        factor.add_factor_mul_slice_assign(self.as_mut(), rhs.as_ref(), modulus);
    }

    #[inline]
    pub fn mul_element_wise_assign<M, A>(&mut self, rhs: &ArrayBase<A>, modulus: M)
    where
        M: Copy + ReduceMulSlice<T>,
        A: RawData<Elem = T> + Data,
    {
        modulus.reduce_mul_slice_assign(self.as_mut(), rhs.as_ref());
    }

    /// Inverse butterfly: `(self[i], result[i]) = (self[i] + rhs[i], (self[i] - rhs[i]) * w[i])`
    #[inline]
    pub fn butterfly_mul_element_wise_inplace<M, A, B, C>(
        &mut self,
        rhs: &ArrayBase<A>,
        w: &ArrayBase<B>,
        result: &mut ArrayBase<C>,
        modulus: M,
    ) where
        M: Copy + ReduceAdd<T, Output = T> + ReduceSub<T, Output = T> + ReduceMul<T, Output = T>,
        A: RawData<Elem = T> + Data,
        B: RawData<Elem = T> + Data,
        C: RawData<Elem = T> + DataMut,
    {
        debug_assert_eq!(self.len(), rhs.len());
        debug_assert_eq!(self.len(), w.len());
        debug_assert_eq!(self.len(), result.len());
        izip!(self, rhs, w, result.iter_mut()).for_each(|(a, &s, &w, b)| {
            let a_orig = *a;
            *a = modulus.reduce_add(a_orig, s);
            *b = modulus.reduce_mul(modulus.reduce_sub(a_orig, s), w);
        });
    }
}

impl<S, T> ArrayBase<S>
where
    S: RawData<Elem = T> + Data,
    T: UnsignedInteger,
{
    /// Performs element wise modular multiplication operation `result = self * rhs` according to `modulus`.
    #[inline]
    pub fn mul_element_wise_inplace<M, A, B>(
        &self,
        rhs: &ArrayBase<A>,
        result: &mut ArrayBase<B>,
        modulus: M,
    ) where
        M: Copy + ReduceMulSlice<T>,
        A: RawData<Elem = T> + Data,
        B: RawData<Elem = T> + DataMut,
    {
        modulus.reduce_mul_slice_to(self.as_ref(), rhs.as_ref(), result.as_mut());
    }

    /// Performs `result = scalar * self` according to `modulus`.
    #[inline]
    pub fn mul_scalar_inplace<M, A>(&self, scalar: T, result: &mut ArrayBase<A>, modulus: M)
    where
        M: Copy + ReduceMulSlice<T>,
        A: RawData<Elem = T> + DataMut,
    {
        modulus.reduce_scalar_mul_slice_to(self.as_ref(), scalar, result.as_mut());
    }

    /// Performs `result = scalar * self` according to `modulus`.
    #[inline]
    pub fn mul_factor_inplace<F, A>(&self, factor: F, result: &mut ArrayBase<A>, modulus: T)
    where
        F: FactorSliceOps<T>,
        A: RawData<Elem = T> + DataMut,
    {
        debug_assert_eq!(self.len(), result.len());
        factor.factor_mul_slice_to(self.as_ref(), result.as_mut(), modulus);
    }
}
