//! Slice trait wiring for [`Barrett50Modulus`].
//!
//! The mul-family slice traits dispatch to the IFMA kernels in
//! `super::simd_ifma` when the target feature combo
//! `avx512f + avx512dq + avx512ifma` is enabled at compile time;
//! otherwise (and for the non-mul traits in all configurations) they
//! delegate to the wrapped [`crate::BarrettModulus<u64>`], whose own
//! slice impls already SIMD-vectorize via portable_simd or fall back to
//! scalar.

use primus_reduce::{
    LazyReduceMulAddSlice, LazyReduceMulSlice, ReduceAddSlice, ReduceDotProduct, ReduceMulAddSlice,
    ReduceMulSlice, ReduceNegSlice, ReduceOnceSlice, ReduceSubSlice,
};

use super::Barrett50Modulus;

// ---------------------------------------------------------------------------
// Non-mul slice traits: always delegate to inner. IFMA gives no speedup on
// add/sub/neg/once, so don't duplicate kernels.
// ---------------------------------------------------------------------------

impl ReduceOnceSlice<u64> for Barrett50Modulus {
    #[inline]
    fn reduce_once_slice_assign(self, values: &mut [u64]) {
        self.inner.reduce_once_slice_assign(values)
    }
    #[inline]
    fn reduce_once_slice_to(self, input: &[u64], output: &mut [u64]) {
        self.inner.reduce_once_slice_to(input, output)
    }
}

impl ReduceNegSlice<u64> for Barrett50Modulus {
    #[inline]
    fn reduce_neg_slice_assign(self, values: &mut [u64]) {
        self.inner.reduce_neg_slice_assign(values)
    }
    #[inline]
    fn reduce_neg_slice_to(self, input: &[u64], output: &mut [u64]) {
        self.inner.reduce_neg_slice_to(input, output)
    }
}

impl ReduceAddSlice<u64> for Barrett50Modulus {
    #[inline]
    fn reduce_add_slice_assign(self, a: &mut [u64], b: &[u64]) {
        self.inner.reduce_add_slice_assign(a, b)
    }
    #[inline]
    fn reduce_add_slice_to(self, a: &[u64], b: &[u64], output: &mut [u64]) {
        self.inner.reduce_add_slice_to(a, b, output)
    }
}

impl ReduceSubSlice<u64> for Barrett50Modulus {
    #[inline]
    fn reduce_sub_slice_assign(self, a: &mut [u64], b: &[u64]) {
        self.inner.reduce_sub_slice_assign(a, b)
    }
    #[inline]
    fn reduce_sub_slice_to(self, a: &[u64], b: &[u64], output: &mut [u64]) {
        self.inner.reduce_sub_slice_to(a, b, output)
    }
    #[inline]
    fn reduce_sub_slice_rev_assign(self, a: &[u64], b: &mut [u64]) {
        self.inner.reduce_sub_slice_rev_assign(a, b)
    }
}

// ---------------------------------------------------------------------------
// Mul-family slice traits: IFMA fast path vs fallback.
//
// We use two mutually exclusive `#[cfg(...)]`-gated impl modules. Rust does
// not allow macros inside `#[cfg(...)]`, so the gate condition is repeated
// verbatim on both branches.
// ---------------------------------------------------------------------------

#[cfg(all(
    feature = "simd",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512ifma",
))]
mod ifma_impls {
    use super::*;

    impl ReduceMulSlice<u64> for Barrett50Modulus {
        #[inline]
        fn reduce_mul_slice_assign(self, a: &mut [u64], b: &[u64]) {
            unsafe { super::super::simd_ifma::reduce_mul_slice_assign(self, a, b) }
        }
        #[inline]
        fn reduce_mul_slice_to(self, a: &[u64], b: &[u64], output: &mut [u64]) {
            unsafe { super::super::simd_ifma::reduce_mul_slice_to(self, a, b, output) }
        }
        #[inline]
        fn reduce_scalar_mul_slice_assign(self, a: &mut [u64], scalar: u64) {
            unsafe { super::super::simd_ifma::reduce_scalar_mul_slice_assign(self, a, scalar) }
        }
        #[inline]
        fn reduce_scalar_mul_slice_to(self, a: &[u64], scalar: u64, output: &mut [u64]) {
            unsafe { super::super::simd_ifma::reduce_scalar_mul_slice_to(self, a, scalar, output) }
        }
    }

    impl LazyReduceMulSlice<u64> for Barrett50Modulus {
        #[inline]
        fn lazy_reduce_mul_slice_assign(self, a: &mut [u64], b: &[u64]) {
            unsafe { super::super::simd_ifma::lazy_reduce_mul_slice_assign(self, a, b) }
        }
        #[inline]
        fn lazy_reduce_mul_slice_to(self, a: &[u64], b: &[u64], output: &mut [u64]) {
            unsafe { super::super::simd_ifma::lazy_reduce_mul_slice_to(self, a, b, output) }
        }
        #[inline]
        fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [u64], scalar: u64) {
            unsafe { super::super::simd_ifma::lazy_reduce_scalar_mul_slice_assign(self, a, scalar) }
        }
        #[inline]
        fn lazy_reduce_scalar_mul_slice_to(self, a: &[u64], scalar: u64, output: &mut [u64]) {
            unsafe {
                super::super::simd_ifma::lazy_reduce_scalar_mul_slice_to(self, a, scalar, output)
            }
        }
    }

    impl ReduceMulAddSlice<u64> for Barrett50Modulus {
        #[inline]
        fn reduce_add_mul_slice_assign(self, acc: &mut [u64], a: &[u64], b: &[u64]) {
            unsafe { super::super::simd_ifma::reduce_add_mul_slice_assign(self, acc, a, b) }
        }
        #[inline]
        fn reduce_sub_mul_slice_assign(self, acc: &mut [u64], a: &[u64], b: &[u64]) {
            unsafe { super::super::simd_ifma::reduce_sub_mul_slice_assign(self, acc, a, b) }
        }
        #[inline]
        fn reduce_mul_add_slice_to(self, a: &[u64], b: &[u64], c: &[u64], output: &mut [u64]) {
            unsafe { super::super::simd_ifma::reduce_mul_add_slice_to(self, a, b, c, output) }
        }
        #[inline]
        fn reduce_scalar_mul_add_slice_to(
            self,
            scalar: u64,
            b: &[u64],
            c: &[u64],
            output: &mut [u64],
        ) {
            unsafe {
                super::super::simd_ifma::reduce_scalar_mul_add_slice_to(self, scalar, b, c, output)
            }
        }
        #[inline]
        fn reduce_add_scalar_mul_slice_assign(self, acc: &mut [u64], scalar: u64, b: &[u64]) {
            unsafe {
                super::super::simd_ifma::reduce_add_scalar_mul_slice_assign(self, acc, scalar, b)
            }
        }
    }

    impl LazyReduceMulAddSlice<u64> for Barrett50Modulus {
        #[inline]
        fn lazy_reduce_add_mul_slice_assign(self, acc: &mut [u64], a: &[u64], b: &[u64]) {
            unsafe { super::super::simd_ifma::lazy_reduce_add_mul_slice_assign(self, acc, a, b) }
        }
        #[inline]
        fn lazy_reduce_sub_mul_slice_assign(self, acc: &mut [u64], a: &[u64], b: &[u64]) {
            unsafe { super::super::simd_ifma::lazy_reduce_sub_mul_slice_assign(self, acc, a, b) }
        }
        #[inline]
        fn lazy_reduce_mul_add_slice_to(self, a: &[u64], b: &[u64], c: &[u64], output: &mut [u64]) {
            unsafe { super::super::simd_ifma::lazy_reduce_mul_add_slice_to(self, a, b, c, output) }
        }
        #[inline]
        fn lazy_reduce_scalar_mul_add_slice_to(
            self,
            scalar: u64,
            b: &[u64],
            c: &[u64],
            output: &mut [u64],
        ) {
            unsafe {
                super::super::simd_ifma::lazy_reduce_scalar_mul_add_slice_to(
                    self, scalar, b, c, output,
                )
            }
        }
    }

    impl ReduceDotProduct<u64> for Barrett50Modulus {
        type Output = u64;

        #[inline]
        fn reduce_dot_product(self, a: impl AsRef<[u64]>, b: impl AsRef<[u64]>) -> u64 {
            unsafe { super::super::simd_ifma::reduce_dot_product(self, a.as_ref(), b.as_ref()) }
        }

        #[inline]
        fn reduce_dot_product_iter(
            self,
            a: impl IntoIterator<Item = u64>,
            b: impl IntoIterator<Item = u64>,
        ) -> u64 {
            self.inner.reduce_dot_product_iter(a, b)
        }
    }
}

#[cfg(not(all(
    feature = "simd",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512ifma",
)))]
mod fallback_impls {
    use super::*;

    impl ReduceMulSlice<u64> for Barrett50Modulus {
        #[inline]
        fn reduce_mul_slice_assign(self, a: &mut [u64], b: &[u64]) {
            self.inner.reduce_mul_slice_assign(a, b)
        }

        #[inline]
        fn reduce_mul_slice_to(self, a: &[u64], b: &[u64], output: &mut [u64]) {
            self.inner.reduce_mul_slice_to(a, b, output)
        }

        #[inline]
        fn reduce_scalar_mul_slice_assign(self, a: &mut [u64], scalar: u64) {
            self.inner.reduce_scalar_mul_slice_assign(a, scalar)
        }

        #[inline]
        fn reduce_scalar_mul_slice_to(self, a: &[u64], scalar: u64, output: &mut [u64]) {
            self.inner.reduce_scalar_mul_slice_to(a, scalar, output);
        }
    }

    impl LazyReduceMulSlice<u64> for Barrett50Modulus {
        #[inline]
        fn lazy_reduce_mul_slice_assign(self, a: &mut [u64], b: &[u64]) {
            self.inner.lazy_reduce_mul_slice_assign(a, b)
        }

        #[inline]
        fn lazy_reduce_mul_slice_to(self, a: &[u64], b: &[u64], output: &mut [u64]) {
            self.inner.lazy_reduce_mul_slice_to(a, b, output)
        }

        #[inline]
        fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [u64], scalar: u64) {
            self.inner.lazy_reduce_scalar_mul_slice_assign(a, scalar);
        }

        #[inline]
        fn lazy_reduce_scalar_mul_slice_to(self, a: &[u64], scalar: u64, output: &mut [u64]) {
            self.inner
                .lazy_reduce_scalar_mul_slice_to(a, scalar, output);
        }
    }

    impl ReduceMulAddSlice<u64> for Barrett50Modulus {
        #[inline]
        fn reduce_add_mul_slice_assign(self, acc: &mut [u64], a: &[u64], b: &[u64]) {
            self.inner.reduce_add_mul_slice_assign(acc, a, b)
        }
        #[inline]
        fn reduce_sub_mul_slice_assign(self, acc: &mut [u64], a: &[u64], b: &[u64]) {
            self.inner.reduce_sub_mul_slice_assign(acc, a, b)
        }
        #[inline]
        fn reduce_mul_add_slice_to(self, a: &[u64], b: &[u64], c: &[u64], output: &mut [u64]) {
            self.inner.reduce_mul_add_slice_to(a, b, c, output)
        }
        #[inline]
        fn reduce_scalar_mul_add_slice_to(
            self,
            scalar: u64,
            b: &[u64],
            c: &[u64],
            output: &mut [u64],
        ) {
            self.inner
                .reduce_scalar_mul_add_slice_to(scalar, b, c, output)
        }
        #[inline]
        fn reduce_add_scalar_mul_slice_assign(self, acc: &mut [u64], scalar: u64, b: &[u64]) {
            self.inner
                .reduce_add_scalar_mul_slice_assign(acc, scalar, b)
        }
    }

    impl LazyReduceMulAddSlice<u64> for Barrett50Modulus {
        #[inline]
        fn lazy_reduce_add_mul_slice_assign(self, acc: &mut [u64], a: &[u64], b: &[u64]) {
            self.inner.lazy_reduce_add_mul_slice_assign(acc, a, b)
        }
        #[inline]
        fn lazy_reduce_sub_mul_slice_assign(self, acc: &mut [u64], a: &[u64], b: &[u64]) {
            self.inner.lazy_reduce_sub_mul_slice_assign(acc, a, b)
        }
        #[inline]
        fn lazy_reduce_mul_add_slice_to(self, a: &[u64], b: &[u64], c: &[u64], output: &mut [u64]) {
            self.inner.lazy_reduce_mul_add_slice_to(a, b, c, output)
        }
        #[inline]
        fn lazy_reduce_scalar_mul_add_slice_to(
            self,
            scalar: u64,
            b: &[u64],
            c: &[u64],
            output: &mut [u64],
        ) {
            self.inner
                .lazy_reduce_scalar_mul_add_slice_to(scalar, b, c, output)
        }
    }

    impl ReduceDotProduct<u64> for Barrett50Modulus {
        type Output = u64;

        #[inline]
        fn reduce_dot_product(self, a: impl AsRef<[u64]>, b: impl AsRef<[u64]>) -> u64 {
            self.inner.reduce_dot_product(a, b)
        }

        #[inline]
        fn reduce_dot_product_iter(
            self,
            a: impl IntoIterator<Item = u64>,
            b: impl IntoIterator<Item = u64>,
        ) -> u64 {
            self.inner.reduce_dot_product_iter(a, b)
        }
    }
}
