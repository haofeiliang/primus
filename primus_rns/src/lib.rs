#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "simd", feature(min_specialization))]

mod error;

mod base;
mod converter;

pub use error::RNSError;

pub use base::RNSBase;
pub use converter::BaseConverter;
