use primus_data::{Data, DataMut, RawData};
use primus_factor::FactorSliceOps;
use primus_integer::FheUint;
use primus_reduce::{
    ReduceMul, ReduceMulAdd, ReduceMulAddSlice, ReduceMulSlice, ReduceNegAssign, ReduceSubAssign,
};

use super::Polynomial;

impl<S, T> Polynomial<S>
where
    S: RawData<Elem = T> + DataMut,
    T: FheUint,
{
    /// Performs `self * scalar` according to `modulus`.
    #[inline]
    pub fn mul_scalar<M>(mut self, scalar: T, modulus: M) -> Self
    where
        M: Copy + ReduceMulSlice<T>,
    {
        self.mul_scalar_assign(scalar, modulus);
        self
    }

    /// Performs `self * scalar` according to `modulus`.
    #[inline]
    pub fn mul_factor<F>(mut self, scalar: F, modulus: T) -> Self
    where
        F: FactorSliceOps<T>,
    {
        self.mul_factor_assign(scalar, modulus);
        self
    }

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
    pub fn add_mul_scalar_assign<M, A>(&mut self, rhs: &Polynomial<A>, scalar: T, modulus: M)
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
    pub fn add_mul_factor_assign<F, A>(&mut self, rhs: &Polynomial<A>, factor: F, modulus: T)
    where
        F: FactorSliceOps<T>,
        A: RawData<Elem = T> + Data,
    {
        factor.add_factor_mul_slice_assign(self.as_mut(), rhs.as_ref(), modulus);
    }

    pub fn mul_monomial_assign<M>(&mut self, r: usize, modulus: M)
    where
        M: Copy + ReduceNegAssign<T>,
    {
        let poly_length = self.poly_length();

        if r < poly_length {
            let rotate = |poly: &mut [T], modulus: M| {
                poly.rotate_right(r);
                poly[0..r]
                    .iter_mut()
                    .for_each(|v| modulus.reduce_neg_assign(v));
            };

            rotate(self.as_mut_slice(), modulus)
        } else {
            debug_assert!(r < poly_length * 2);
            let r = r - poly_length;

            let rotate = |poly: &mut [T], modulus: M| {
                poly.rotate_right(r);
                poly[r..]
                    .iter_mut()
                    .for_each(|v| modulus.reduce_neg_assign(v));
            };

            rotate(self.as_mut_slice(), modulus)
        }
    }
}

impl<S, T> Polynomial<S>
where
    S: RawData<Elem = T> + Data,
    T: FheUint,
{
    /// A naive multiplication over polynomial.
    pub fn naive_mul_inplace<M, A, B>(
        &self,
        rhs: &Polynomial<A>,
        result: &mut Polynomial<B>,
        modulus: M,
    ) where
        M: Copy + ReduceSubAssign<T> + ReduceMul<T, Output = T> + ReduceMulAdd<T, Output = T>,
        A: RawData<Elem = T> + Data,
        B: RawData<Elem = T> + DataMut,
    {
        let a: &[T] = self.as_ref();
        let b: &[T] = rhs.as_ref();
        let c: &mut [T] = result.as_mut();

        let coeff_count = a.len();
        debug_assert_eq!(coeff_count, b.len());
        debug_assert_eq!(coeff_count, c.len());

        for i in 0..coeff_count {
            for j in 0..=i {
                c[i] = modulus.reduce_mul_add(a[j], b[i - j], c[i]);
            }
        }

        // mod (x^n + 1)
        for i in coeff_count..coeff_count * 2 - 1 {
            let k = i - coeff_count;
            for j in i - coeff_count + 1..coeff_count {
                modulus.reduce_sub_assign(&mut c[k], modulus.reduce_mul(a[j], b[i - j]));
            }
        }
    }
}
