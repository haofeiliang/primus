use primus_integer::UnsignedInteger;
use primus_reduce::ReduceError;
use primus_reduce::prelude::*;

use crate::common::uint;

use super::MontgomeryModulus;

// ---------------------------------------------------------------------------
// LazyReduce — Montgomery form naturally produces values in [0, 2N)
// ---------------------------------------------------------------------------

impl<T: UnsignedInteger> LazyReduce<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn lazy_reduce(self, value: T) -> T {
        let (lo, hi) = value.widening_mul(self.r);
        self.montgomery_reduce([lo, hi])
    }
}

impl<T: UnsignedInteger> LazyReduceAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn lazy_reduce_assign(self, value: &mut T) {
        *value = self.lazy_reduce(*value);
    }
}

impl<T: UnsignedInteger> LazyReduceMul<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn lazy_reduce_mul(self, a: T, b: T) -> Self::Output {
        let (lo, hi) = a.widening_mul(b);
        self.montgomery_reduce([lo, hi])
    }
}

impl<T: UnsignedInteger> LazyReduceMulAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn lazy_reduce_mul_assign(self, a: &mut T, b: T) {
        let (lo, hi) = a.widening_mul(b);
        *a = self.montgomery_reduce([lo, hi]);
    }
}

impl<T: UnsignedInteger> LazyReduceMulAdd<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn lazy_reduce_mul_add(self, a: T, b: T, c: T) -> Self::Output {
        let (lo, hi) = a.widening_mul(b);
        let (sum_hi, carry) = hi.overflowing_add(c);
        let result = self.montgomery_reduce([lo, sum_hi]);
        if carry {
            let diff = self.value - self.r;
            if diff > result {
                result + self.r
            } else {
                result.wrapping_sub(diff)
            }
        } else {
            result
        }
    }
}

impl<T: UnsignedInteger> LazyReduceMulAddAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn lazy_reduce_mul_add_assign(self, a: &mut T, b: T, c: T) {
        *a = self.lazy_reduce_mul_add(*a, b, c);
    }
}

// ---------------------------------------------------------------------------
// Reduce — canonical results in [0, modulus)
// ---------------------------------------------------------------------------

impl<T: UnsignedInteger> Reduce<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce(self, value: T) -> Self::Output {
        let (lo, hi) = value.widening_mul(self.r);
        self.montgomery_reduce([lo, hi])
    }
}

impl<T: UnsignedInteger> ReduceAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn reduce_assign(self, value: &mut T) {
        *value = self.reduce(*value);
    }
}

impl<T: UnsignedInteger> ReduceOnce<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_once(self, value: T) -> Self::Output {
        if value >= self.value {
            value - self.value
        } else {
            value
        }
    }
}

impl<T: UnsignedInteger> ReduceOnceAssign<T> for MontgomeryModulus<T> {
    #[inline(always)]
    fn reduce_once_assign(self, value: &mut T) {
        if *value >= self.value {
            *value -= self.value;
        }
    }
}

// ---------------------------------------------------------------------------
// Basic arithmetic
// ---------------------------------------------------------------------------

impl<T: UnsignedInteger> ReduceAdd<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_add(self, a: T, b: T) -> Self::Output {
        uint::reduce_add(self.value, a, b)
    }
}

impl<T: UnsignedInteger> ReduceAddAssign<T> for MontgomeryModulus<T> {
    #[inline(always)]
    fn reduce_add_assign(self, a: &mut T, b: T) {
        uint::reduce_add_assign(self.value, a, b)
    }
}

impl<T: UnsignedInteger> ReduceDouble<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_double(self, value: T) -> Self::Output {
        uint::reduce_double(self.value, value)
    }
}

impl<T: UnsignedInteger> ReduceDoubleAssign<T> for MontgomeryModulus<T> {
    #[inline(always)]
    fn reduce_double_assign(self, value: &mut T) {
        uint::reduce_double_assign(self.value, value)
    }
}

impl<T: UnsignedInteger> ReduceSub<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_sub(self, a: T, b: T) -> Self::Output {
        uint::reduce_sub(self.value, a, b)
    }
}

impl<T: UnsignedInteger> ReduceSubAssign<T> for MontgomeryModulus<T> {
    #[inline(always)]
    fn reduce_sub_assign(self, a: &mut T, b: T) {
        uint::reduce_sub_assign(self.value, a, b)
    }
}

impl<T: UnsignedInteger> ReduceNeg<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline(always)]
    fn reduce_neg(self, value: T) -> Self::Output {
        uint::reduce_neg(self.value, value)
    }
}

impl<T: UnsignedInteger> ReduceNegAssign<T> for MontgomeryModulus<T> {
    #[inline(always)]
    fn reduce_neg_assign(self, value: &mut T) {
        uint::reduce_neg_assign(self.value, value)
    }
}

// ---------------------------------------------------------------------------
// Multiplication — single REDC per operation
// ---------------------------------------------------------------------------

impl<T: UnsignedInteger> ReduceMul<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn reduce_mul(self, a: T, b: T) -> Self::Output {
        let (lo, hi) = a.widening_mul(b);
        self.montgomery_reduce([lo, hi])
    }
}

impl<T: UnsignedInteger> ReduceMulAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn reduce_mul_assign(self, a: &mut T, b: T) {
        let (lo, hi) = a.widening_mul(b);
        *a = self.montgomery_reduce([lo, hi]);
    }
}

impl<T: UnsignedInteger> ReduceSquare<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn reduce_square(self, value: T) -> Self::Output {
        let (lo, hi) = value.widening_mul(value);
        self.montgomery_reduce([lo, hi])
    }
}

impl<T: UnsignedInteger> ReduceSquareAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn reduce_square_assign(self, value: &mut T) {
        let (lo, hi) = value.widening_mul(*value);
        *value = self.montgomery_reduce([lo, hi]);
    }
}

// ---------------------------------------------------------------------------
// Fused multiply-add — single REDC via (a*b + c*R) * R^(-1)
// ---------------------------------------------------------------------------

impl<T: UnsignedInteger> ReduceMulAdd<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn reduce_mul_add(self, a: T, b: T, c: T) -> Self::Output {
        let (lo, hi) = a.widening_mul(b);
        // (hi, lo) + (c, 0): the low word never carries (lo + 0 = lo).
        let (sum_hi, carry) = hi.overflowing_add(c);
        let result = self.montgomery_reduce([lo, sum_hi]);
        if carry {
            // The intermediate value exceeded 2^(2w); REDC of the overflow
            // is R * R^(-1) ≡ R (mod N), so add self.r once.
            self.reduce_add(result, self.r)
        } else {
            result
        }
    }
}

impl<T: UnsignedInteger> ReduceMulAddAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn reduce_mul_add_assign(self, a: &mut T, b: T, c: T) {
        *a = self.reduce_mul_add(*a, b, c);
    }
}

// ---------------------------------------------------------------------------
// Exponentiation
// ---------------------------------------------------------------------------

impl<T: UnsignedInteger> ReduceExp<T> for MontgomeryModulus<T> {
    #[inline]
    fn reduce_exp<E: UnsignedInteger>(self, base: T, mut exp: E) -> T {
        if exp.is_zero() {
            return self.to_montgomery(T::ONE);
        }

        if base.is_zero() {
            return T::ZERO;
        }

        debug_assert!(base < self.value);

        let mut power: T = base;

        let exp_trailing_zeros = exp.trailing_zeros();
        if exp_trailing_zeros > 0 {
            for _ in 0..exp_trailing_zeros {
                self.reduce_square_assign(&mut power);
            }
            exp >>= exp_trailing_zeros;
        }

        if exp.is_one() {
            return power;
        }

        let mut intermediate: T = power;
        for _ in 1..(E::BITS - exp.leading_zeros()) {
            exp >>= 1;
            self.reduce_square_assign(&mut power);
            if !(exp & E::ONE).is_zero() {
                self.reduce_mul_assign(&mut intermediate, power);
            }
        }
        intermediate
    }
}

impl<T: UnsignedInteger> ReduceExpPowOf2<T> for MontgomeryModulus<T> {
    #[inline]
    fn reduce_exp_power_of_2(self, base: T, exp_log: u32) -> T {
        if base.is_zero() {
            return T::ZERO;
        }

        let mut power = base;

        for _ in 0..exp_log {
            self.reduce_square_assign(&mut power);
        }

        power
    }
}

// ---------------------------------------------------------------------------
// Inverse and division — via conversion to standard form
// ---------------------------------------------------------------------------

impl<T: UnsignedInteger> TryReduceInv<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn try_reduce_inv(self, value: T) -> Result<T, ReduceError<T>> {
        let std_val = self.from_montgomery(value);
        uint::try_reduce_inv(self.value, std_val).map(|inv| self.to_montgomery(inv))
    }
}

impl<T: UnsignedInteger> ReduceInv<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn reduce_inv(self, value: T) -> Self::Output {
        let std_val = self.from_montgomery(value);
        let inv = uint::reduce_inv(self.value, std_val);
        self.to_montgomery(inv)
    }
}

impl<T: UnsignedInteger> ReduceInvAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn reduce_inv_assign(self, value: &mut T) {
        *value = self.reduce_inv(*value);
    }
}

impl<T: UnsignedInteger> ReduceDiv<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn reduce_div(self, a: T, b: T) -> Self::Output {
        self.reduce_mul(a, self.reduce_inv(b))
    }
}

impl<T: UnsignedInteger> ReduceDivAssign<T> for MontgomeryModulus<T> {
    #[inline]
    fn reduce_div_assign(self, a: &mut T, b: T) {
        self.reduce_mul_assign(a, self.reduce_inv(b));
    }
}

// ---------------------------------------------------------------------------
// Dot product — chunked accumulation to amortize REDC cost
// ---------------------------------------------------------------------------

/// `c += a * b` where `c` is a double-width accumulator `[lo, hi]`.
#[inline]
fn multiply_add<T: UnsignedInteger>(c: &mut [T; 2], a: T, b: T) {
    let (lw, hw) = a.widening_mul(b);
    let carry;
    (c[0], carry) = c[0].overflowing_add(lw);
    (c[1], _) = c[1].carrying_add(hw, carry);
}

impl<T: UnsignedInteger> ReduceDotProduct<T> for MontgomeryModulus<T> {
    type Output = T;

    #[inline]
    fn reduce_dot_product(self, a: impl AsRef<[T]>, b: impl AsRef<[T]>) -> T {
        let a = a.as_ref();
        let b = b.as_ref();

        assert_eq!(a.len(), b.len(), "reduce_dot_product: length mismatch");

        let mut a_iter = a.chunks_exact(16);
        let mut b_iter = b.chunks_exact(16);

        let inter = (&mut a_iter)
            .zip(&mut b_iter)
            .map(|(a_s, b_s)| {
                let mut c: [T; 2] = [T::ZERO, T::ZERO];
                for (&a, &b) in a_s.iter().zip(b_s) {
                    multiply_add(&mut c, a, b);
                }
                self.montgomery_reduce(c)
            })
            .fold(T::ZERO, |acc: T, b| self.reduce_add(acc, b));

        let mut c: [T; 2] = [T::ZERO, T::ZERO];
        a_iter
            .remainder()
            .iter()
            .zip(b_iter.remainder())
            .for_each(|(&a, &b)| {
                multiply_add(&mut c, a, b);
            });
        self.reduce_add(self.montgomery_reduce(c), inter)
    }

    #[inline]
    fn reduce_dot_product_iter(
        self,
        a: impl IntoIterator<Item = T>,
        b: impl IntoIterator<Item = T>,
    ) -> T {
        let mut a_iter = a.into_iter();
        let mut b_iter = b.into_iter();

        let mut a_temp = [T::ZERO; 16];
        let mut b_temp = [T::ZERO; 16];
        let mut i = 0;
        let mut result = T::ZERO;

        while let (Some(a_next), Some(b_next)) = (a_iter.next(), b_iter.next()) {
            if i < 16 {
                a_temp[i] = a_next;
                b_temp[i] = b_next;
                i += 1;
            } else {
                let mut c: [T; 2] = [T::ZERO, T::ZERO];
                for (&a, b) in a_temp.iter().zip(b_temp) {
                    multiply_add(&mut c, a, b);
                }
                self.reduce_add_assign(&mut result, self.montgomery_reduce(c));

                a_temp.fill(T::ZERO);
                b_temp.fill(T::ZERO);
                a_temp[0] = a_next;
                b_temp[0] = b_next;
                i = 1;
            }
        }

        let mut c: [T; 2] = [T::ZERO, T::ZERO];
        for (&a, &b) in a_temp[..i].iter().zip(b_temp[..i].iter()) {
            multiply_add(&mut c, a, b);
        }
        self.reduce_add_assign(&mut result, self.montgomery_reduce(c));

        result
    }
}
