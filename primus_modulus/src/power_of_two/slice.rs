//! Scalar slice impls for [`PowOf2Modulus`].
//!
//! Each operation reduces by masking with `self.mask`, which is just bit-AND
//! and lets the compiler auto-vectorize most loops on its own. The impls
//! below loop over the slice using the existing scalar `Reduce*` impls.

use primus_integer::UnsignedInteger;
use primus_reduce::{
    LazyReduceMulAddSlice, LazyReduceMulSlice, ReduceAdd, ReduceAddAssign, ReduceAddSlice,
    ReduceMul, ReduceMulAdd, ReduceMulAddSlice, ReduceMulAssign, ReduceMulSlice, ReduceNeg,
    ReduceNegAssign, ReduceNegSlice, ReduceOnce, ReduceOnceAssign, ReduceOnceSlice, ReduceSub,
    ReduceSubAssign, ReduceSubSlice,
};

use super::PowOf2Modulus;

impl<T: UnsignedInteger> ReduceOnceSlice<T> for PowOf2Modulus<T> {
    #[inline]
    fn reduce_once_slice_assign(self, values: &mut [T]) {
        values
            .iter_mut()
            .for_each(|value| self.reduce_once_assign(value));
    }

    #[inline]
    fn reduce_once_slice_to(self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(input.len(), output.len());
        output
            .iter_mut()
            .zip(input)
            .for_each(|(x, &y)| *x = self.reduce_once(y));
    }
}

impl<T: UnsignedInteger> ReduceNegSlice<T> for PowOf2Modulus<T> {
    #[inline]
    fn reduce_neg_slice_assign(self, values: &mut [T]) {
        values
            .iter_mut()
            .for_each(|value| self.reduce_neg_assign(value));
    }

    #[inline]
    fn reduce_neg_slice_to(self, input: &[T], output: &mut [T]) {
        debug_assert_eq!(input.len(), output.len());
        output
            .iter_mut()
            .zip(input)
            .for_each(|(x, &y)| *x = self.reduce_neg(y));
    }
}

impl<T: UnsignedInteger> ReduceAddSlice<T> for PowOf2Modulus<T> {
    #[inline]
    fn reduce_add_slice_assign(self, a: &mut [T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        a.iter_mut()
            .zip(b)
            .for_each(|(x, &y)| self.reduce_add_assign(x, y));
    }

    #[inline]
    fn reduce_add_slice_to(self, a: &[T], b: &[T], output: &mut [T]) {
        debug_assert_eq!(output.len(), a.len());
        debug_assert_eq!(output.len(), b.len());
        output
            .iter_mut()
            .zip(a)
            .zip(b)
            .for_each(|((out, &x), &y)| *out = self.reduce_add(x, y));
    }
}

impl<T: UnsignedInteger> ReduceSubSlice<T> for PowOf2Modulus<T> {
    #[inline]
    fn reduce_sub_slice_assign(self, a: &mut [T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        a.iter_mut()
            .zip(b)
            .for_each(|(x, &y)| self.reduce_sub_assign(x, y));
    }

    #[inline]
    fn reduce_sub_slice_to(self, a: &[T], b: &[T], output: &mut [T]) {
        debug_assert_eq!(output.len(), a.len());
        debug_assert_eq!(output.len(), b.len());
        output
            .iter_mut()
            .zip(a)
            .zip(b)
            .for_each(|((out, &x), &y)| *out = self.reduce_sub(x, y));
    }

    #[inline]
    fn reduce_sub_slice_rev_assign(self, a: &[T], b: &mut [T]) {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter_mut())
            .for_each(|(&x, y)| *y = self.reduce_sub(x, *y));
    }
}

impl<T: UnsignedInteger> ReduceMulSlice<T> for PowOf2Modulus<T> {
    #[inline]
    fn reduce_mul_slice_assign(self, a: &mut [T], b: &[T]) {
        debug_assert_eq!(a.len(), b.len());
        a.iter_mut()
            .zip(b)
            .for_each(|(x, &y)| self.reduce_mul_assign(x, y));
    }

    #[inline]
    fn reduce_mul_slice_to(self, a: &[T], b: &[T], output: &mut [T]) {
        debug_assert_eq!(output.len(), a.len());
        debug_assert_eq!(output.len(), b.len());
        output
            .iter_mut()
            .zip(a)
            .zip(b)
            .for_each(|((out, &x), &y)| *out = self.reduce_mul(x, y));
    }

    #[inline]
    fn reduce_scalar_mul_slice_assign(self, a: &mut [T], scalar: T) {
        a.iter_mut().for_each(|x| self.reduce_mul_assign(x, scalar));
    }

    #[inline]
    fn reduce_scalar_mul_slice_to(self, a: &[T], scalar: T, output: &mut [T]) {
        debug_assert_eq!(a.len(), output.len());
        output
            .iter_mut()
            .zip(a)
            .for_each(|(out, &x)| *out = self.reduce_mul(x, scalar));
    }
}

impl<T: UnsignedInteger> LazyReduceMulSlice<T> for PowOf2Modulus<T> {
    #[inline]
    fn lazy_reduce_mul_slice_assign(self, a: &mut [T], b: &[T]) {
        self.reduce_mul_slice_assign(a, b);
    }

    #[inline]
    fn lazy_reduce_mul_slice_to(self, a: &[T], b: &[T], output: &mut [T]) {
        self.reduce_mul_slice_to(a, b, output);
    }

    #[inline]
    fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [T], scalar: T) {
        self.reduce_scalar_mul_slice_assign(a, scalar);
    }

    #[inline]
    fn lazy_reduce_scalar_mul_slice_to(self, a: &[T], scalar: T, output: &mut [T]) {
        self.reduce_scalar_mul_slice_to(a, scalar, output);
    }
}

impl<T: UnsignedInteger> ReduceMulAddSlice<T> for PowOf2Modulus<T> {
    #[inline]
    fn reduce_add_mul_slice_assign(self, acc: &mut [T], a: &[T], b: &[T]) {
        debug_assert_eq!(acc.len(), a.len());
        debug_assert_eq!(acc.len(), b.len());
        acc.iter_mut()
            .zip(a)
            .zip(b)
            .for_each(|((acc, &a), &b)| *acc = self.reduce_mul_add(a, b, *acc));
    }

    #[inline]
    fn reduce_sub_mul_slice_assign(self, acc: &mut [T], a: &[T], b: &[T]) {
        debug_assert_eq!(acc.len(), a.len());
        debug_assert_eq!(acc.len(), b.len());
        let mask = self.mask();
        acc.iter_mut()
            .zip(a)
            .zip(b)
            .for_each(|((acc, &a), &b)| *acc = (*acc).wrapping_sub(a.wrapping_mul(b)) & mask);
    }

    #[inline]
    fn reduce_mul_add_slice_to(self, a: &[T], b: &[T], c: &[T], output: &mut [T]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), output.len());
        a.iter()
            .zip(b)
            .zip(c)
            .zip(output)
            .for_each(|(((&a, &b), &c), o)| *o = self.reduce_mul_add(a, b, c));
    }

    #[inline]
    fn reduce_scalar_mul_add_slice_to(self, scalar: T, b: &[T], c: &[T], output: &mut [T]) {
        debug_assert_eq!(b.len(), c.len());
        debug_assert_eq!(b.len(), output.len());
        b.iter()
            .zip(c)
            .zip(output)
            .for_each(|((&b, &c), o)| *o = self.reduce_mul_add(scalar, b, c));
    }
}

impl<T: UnsignedInteger> LazyReduceMulAddSlice<T> for PowOf2Modulus<T> {
    #[inline]
    fn lazy_reduce_add_mul_slice_assign(self, acc: &mut [T], a: &[T], b: &[T]) {
        self.reduce_add_mul_slice_assign(acc, a, b);
    }

    #[inline]
    fn lazy_reduce_sub_mul_slice_assign(self, acc: &mut [T], a: &[T], b: &[T]) {
        self.reduce_sub_mul_slice_assign(acc, a, b);
    }

    #[inline]
    fn lazy_reduce_mul_add_slice_to(self, a: &[T], b: &[T], c: &[T], output: &mut [T]) {
        self.reduce_mul_add_slice_to(a, b, c, output);
    }

    #[inline]
    fn lazy_reduce_scalar_mul_add_slice_to(self, scalar: T, b: &[T], c: &[T], output: &mut [T]) {
        self.reduce_scalar_mul_add_slice_to(scalar, b, c, output);
    }
}
