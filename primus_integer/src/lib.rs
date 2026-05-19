#![cfg_attr(feature = "simd", feature(portable_simd))]

mod macros;

mod integer_traits;

mod integer;
mod unsigned_integer;

mod big_integer;

mod data;

#[cfg(feature = "simd")]
mod simd;

mod size;
pub use size::Size;

pub use integer_traits::*;

pub use integer::Integer;
pub use unsigned_integer::UnsignedInteger;

pub use big_integer::{
    BigUint, BigUintIter, BigUintIterMut, BigUintMut, BigUintOwned, BigUintRef,
    multiply_many_values, multiply_many_values_except, multiply_many_values_except_inplace,
};

pub use data::{Data, DataMut, DataOwned, RawData};

#[cfg(feature = "simd")]
pub use simd::{SimdArray, SimdMaskArray, SimdUnsignedInteger, lanes};
