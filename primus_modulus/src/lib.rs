#![cfg_attr(feature = "simd", feature(portable_simd))]

//! Concrete modulus types implementing the [`primus_reduce`] traits.
//!
//! | Type | Reduction | Use case |
//! |------|-----------|----------|
//! | [`NativeModulus`] | Wrapping arithmetic (mod 2^BITS) | Native integer overflow |
//! | [`PowOf2Modulus`] | Bitwise mask (mod 2^k) | Explicit power-of-two |
//! | [`BarrettModulus`] | Barrett reduction | General prime modulus |
//! | [`MontgomeryModulus`] | Montgomery form | Repeated multiplication |
//! | [`UintModulus`] | Basic compare-and-subtract | Fallback / simple ops |
//!
//! The [`Barrett`] derive macro (feature-gated behind `derive`) creates
//! zero-sized Barrett modulus types at compile time.

pub use primus_integer as integer;
pub use primus_reduce as reduce;

#[macro_use]
mod macros;

pub mod common;

mod barrett;
mod barrett50;
mod compact;
mod montgomery;

mod native;
mod power_of_two;
mod uint;

#[cfg(feature = "derive")]
pub use primus_barrett_derive::Barrett;

pub use barrett::BarrettModulus;
pub use barrett50::Barrett50Modulus;
pub use compact::CompactModulus;
pub use montgomery::MontgomeryModulus;

pub use native::NativeModulus;
pub use power_of_two::PowOf2Modulus;
pub use uint::UintModulus;

#[cfg(feature = "simd")]
pub use barrett::SimdBarrettModulus;

#[cfg(feature = "simd")]
pub use barrett::simd_kernel as barrett_simd_kernel;
