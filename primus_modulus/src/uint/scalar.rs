use primus_factor::{FactorMul, LazyFactorMul};
use primus_integer::UnsignedInteger;
use primus_reduce::{ReduceError, prelude::*};

use crate::common::uint;

use super::UintModulus;

impl<T: UnsignedInteger> ReduceOnce<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_once(self, value: T) -> Self::Output {
        uint::reduce_once(self.0, value)
    }
}

impl<T: UnsignedInteger> ReduceOnceAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_once_assign(self, value: &mut T) {
        uint::reduce_once_assign(self.0, value);
    }
}

impl<T: UnsignedInteger> ReduceAdd<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_add(self, a: T, b: T) -> Self::Output {
        uint::reduce_add(self.0, a, b)
    }
}

impl<T: UnsignedInteger> ReduceAddAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_add_assign(self, a: &mut T, b: T) {
        uint::reduce_add_assign(self.0, a, b);
    }
}

impl<T: UnsignedInteger> ReduceDouble<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_double(self, value: T) -> Self::Output {
        uint::reduce_double(self.0, value)
    }
}

impl<T: UnsignedInteger> ReduceDoubleAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_double_assign(self, value: &mut T) {
        uint::reduce_double_assign(self.0, value);
    }
}

impl<T: UnsignedInteger> ReduceSub<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_sub(self, a: T, b: T) -> Self::Output {
        uint::reduce_sub(self.0, a, b)
    }
}

impl<T: UnsignedInteger> ReduceSubAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_sub_assign(self, a: &mut T, b: T) {
        uint::reduce_sub_assign(self.0, a, b);
    }
}

impl<T: UnsignedInteger> ReduceNeg<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_neg(self, value: T) -> Self::Output {
        uint::reduce_neg(self.0, value)
    }
}

impl<T: UnsignedInteger> ReduceNegAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_neg_assign(self, value: &mut T) {
        uint::reduce_neg_assign(self.0, value);
    }
}

impl<T: UnsignedInteger> ReduceInv<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_inv(self, value: T) -> Self::Output {
        uint::reduce_inv(self.0, value)
    }
}

impl<T: UnsignedInteger> ReduceInvAssign<T> for UintModulus<T> {
    #[inline(always)]
    fn reduce_inv_assign(self, value: &mut T) {
        uint::reduce_inv_assign(self.0, value);
    }
}

impl<T: UnsignedInteger> TryReduceInv<T> for UintModulus<T> {
    type Output = T;

    #[inline(always)]
    fn try_reduce_inv(self, value: T) -> Result<Self::Output, ReduceError<T>> {
        uint::try_reduce_inv(self.0, value)
    }
}

/// Factor-based lazy multiplication for [`UintModulus`].
///
/// `UintModulus::new` accepts any non-zero modulus, but this implementation is
/// only as general as the supplied factor type. For the standard
/// `ShoupFactor`, the lazy product is represented in the same integer type and
/// therefore requires `2 * modulus` to fit in `T`; in practice use it only with
/// `modulus < 2^(T::BITS - 1)`. The plain add/sub/neg operations do not have
/// this extra restriction.
impl<T: UnsignedInteger, F> LazyReduceMul<T, F> for UintModulus<T>
where
    F: LazyFactorMul<T>,
{
    type Output = T;

    #[inline(always)]
    fn lazy_reduce_mul(self, a: T, b: F) -> Self::Output {
        b.lazy_factor_mul_modulo(a, self.0)
    }
}

/// Factor-based canonical multiplication for [`UintModulus`].
///
/// This delegates to the factor implementation. With `ShoupFactor`, the same
/// modulus restriction as [`LazyReduceMul`] applies: `modulus` must be small
/// enough that the factor's lazy intermediate range is representable in `T`.
impl<T: UnsignedInteger, F> ReduceMul<T, F> for UintModulus<T>
where
    F: FactorMul<T>,
{
    type Output = T;

    #[inline(always)]
    fn reduce_mul(self, a: T, b: F) -> Self::Output {
        b.factor_mul_modulo(a, self.0)
    }
}
