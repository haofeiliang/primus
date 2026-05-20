use primus_integer::UnsignedInteger;

mod ops;
#[cfg(feature = "simd")]
mod simd;

/// Unsigned integer modulus.
///
/// Just store the modulus value and only support some basic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct UintModulus<T>(pub T);

impl<T: UnsignedInteger> UintModulus<T> {
    /// Creates a new [`UintModulus<T>`].
    ///
    /// # Panics
    ///
    /// Panics if `value >= 2^{T::BITS - 1}`. The SIMD `reduce_sub` kernel
    /// relies on `modulus < 2^{BITS-1}` to avoid overflow in the wrapping
    /// subtraction path. All FHE parameter sets satisfy this bound.
    #[inline(always)]
    pub fn new(value: T) -> Self {
        let limit = T::ONE << (T::BITS - 1);
        assert!(
            value < limit,
            "UintModulus value must be < 2^(T::BITS - 1), got {value:?}"
        );
        Self(value)
    }
}

impl<T: UnsignedInteger> primus_reduce::Modulus for UintModulus<T> {
    type ValueT = T;

    #[inline(always)]
    fn value(self) -> Option<Self::ValueT> {
        Some(self.0)
    }

    #[inline(always)]
    fn value_unchecked(self) -> Self::ValueT {
        self.0
    }

    #[inline(always)]
    fn minus_one(self) -> Self::ValueT {
        self.0 - T::ONE
    }
}
