//! Compile-time SIMD vector width chosen for the current target.
//!
//! This is the single source of truth for the default lane count used by
//! SIMD slice kernels across the workspace. Downstream crates should read
//! [`VECTOR_BITS`] (or call [`lanes_for_bits`]) instead of re-defining their own
//! per-target constant.

/// Native SIMD vector width in bits.
///
/// * AVX-512 → 512 bits
/// * AVX2 → 256 bits
/// * other (NEON / SSE2 / no SIMD) → 256 bits as a "wide fallback".
///   `portable_simd` lowers oversize vectors to multiple native
///   instructions, which on 128-bit ISAs behaves like 2× loop unrolling
///   and is usually as fast or faster than a 128-bit default. Build with
///   `-C target-cpu=native` (or `-C target-feature=+avx512f`) to get a
///   wider native width when the host supports it.
#[cfg(target_feature = "avx512f")]
pub const VECTOR_BITS: usize = 512;
#[cfg(not(target_feature = "avx512f"))]
pub const VECTOR_BITS: usize = 256;

/// Default lane count for an integer type of `BITS` bits.
#[inline]
#[must_use]
pub const fn lanes_for_bits(bits: usize) -> usize {
    VECTOR_BITS / bits
}

/// 2× of [`VECTOR_BITS`]: forces `portable_simd` to emit two native
/// vectors per chunk. Use it on hot paths whose kernels are mul-heavy
/// or have long dependency chains, where the extra register pressure
/// buys instruction-level parallelism (measured 10–11 % on u64 Barrett
/// `reduce_mul_slice_to` / `lazy_reduce_mul_slice_to` /
/// `reduce_mul_add_slice_to`, ~4 % on `reduce_dot_product`, AVX-512
/// host — see `benches/barrett_lane_width.rs`).
///
/// Do **not** flip this on indiscriminately:
///
/// - small slice lengths (well under `LARGE_VECTOR_BITS / bits`)
///   spend more time in the scalar tail than they save in the
///   vector body;
/// - non-mul kernels (add/sub/once/neg) were not measured and are
///   unlikely to win;
/// - unverified on non-AVX-512 targets — bench before flipping.
pub const LARGE_VECTOR_BITS: usize = 2 * VECTOR_BITS;

/// Like [`lanes_for_bits`] but for the wider [`LARGE_VECTOR_BITS`].
#[inline]
#[must_use]
pub const fn large_lanes_for_bits(bits: usize) -> usize {
    LARGE_VECTOR_BITS / bits
}
