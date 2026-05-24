#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "simd", feature(min_specialization))]

mod ops;

mod multiply;
mod shoup_factor;

pub use ops::*;

pub use multiply::MultiplyFactor;
pub use shoup_factor::ShoupFactor;

#[cfg(feature = "simd")]
pub use shoup_factor::{SimdShoupFactor, simd_kernel};

pub trait Factor<T>:
    Copy + LazyFactorMul<T> + FactorMul<T> + LazyFactorSliceOps<T> + FactorSliceOps<T>
{
}

impl<T, F> Factor<T> for F where
    F: Copy + LazyFactorMul<T> + FactorMul<T> + LazyFactorSliceOps<T> + FactorSliceOps<T>
{
}
