//! Prelude: re-exports all operation traits but deliberately omits
//! [`ModuloError`](crate::ModuloError) — import that explicitly when needed.

pub use crate::lazy_ops::*;
pub use crate::lazy_slice_ops::*;
pub use crate::ops::*;
pub use crate::slice_ops::*;
