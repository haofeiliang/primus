//! Value-side mirror of `primus_reduce::slice_ops`.
//!
//! Each trait is implemented on `[T]` and delegates to the corresponding
//! modulus-receiver trait, mirroring the scalar `XxxModulo` / `ReduceXxx`
//! pairing in [`crate::ops`].
//!
//! See `primus_reduce::slice_ops` for the conventions on length checks
//! and value-range invariants.

use primus_reduce::prelude::*;

/// Value-side mirror of [`ReduceOnceSlice`].
pub trait OnceModuloSlice<M> {
    /// For each `v` in `self`: `v -= modulus` if `v >= modulus`.
    fn once_modulo_slice_assign(&mut self, modulus: M);

    /// Writes the once-reduced value into `output`.
    fn once_modulo_slice_to(&self, output: &mut Self, modulus: M);
}

impl<T, M> OnceModuloSlice<M> for [T]
where
    M: ReduceOnceSlice<T> + Copy,
{
    #[inline(always)]
    fn once_modulo_slice_assign(&mut self, modulus: M) {
        modulus.reduce_once_slice_assign(self);
    }

    #[inline(always)]
    fn once_modulo_slice_to(&self, output: &mut Self, modulus: M) {
        modulus.reduce_once_slice_to(self, output);
    }
}

/// Value-side mirror of [`ReduceNegSlice`].
pub trait NegModuloSlice<M> {
    /// Calculates `v = -v (mod modulus)` for each element in-place.
    fn neg_modulo_slice_assign(&mut self, modulus: M);

    /// Writes `-self[i] (mod modulus)` into `output[i]` for each element.
    fn neg_modulo_slice_to(&self, output: &mut Self, modulus: M);
}

impl<T, M> NegModuloSlice<M> for [T]
where
    M: ReduceNegSlice<T> + Copy,
{
    #[inline(always)]
    fn neg_modulo_slice_assign(&mut self, modulus: M) {
        modulus.reduce_neg_slice_assign(self);
    }

    #[inline(always)]
    fn neg_modulo_slice_to(&self, output: &mut Self, modulus: M) {
        modulus.reduce_neg_slice_to(self, output);
    }
}

/// Value-side mirror of [`ReduceAddSlice`].
pub trait AddModuloSlice<M, B: ?Sized = Self> {
    /// Calculates `self[i] = (self[i] + b[i]) (mod modulus)` element-wise.
    fn add_modulo_slice_assign(&mut self, b: &B, modulus: M);

    /// Writes `self[i] + b[i] (mod modulus)` into `output[i]`.
    fn add_modulo_slice_to(&self, b: &B, output: &mut Self, modulus: M);
}

impl<T, M, B> AddModuloSlice<M, [B]> for [T]
where
    M: ReduceAddSlice<T, B> + Copy,
{
    #[inline(always)]
    fn add_modulo_slice_assign(&mut self, b: &[B], modulus: M) {
        modulus.reduce_add_slice_assign(self, b);
    }

    #[inline(always)]
    fn add_modulo_slice_to(&self, b: &[B], output: &mut [T], modulus: M) {
        modulus.reduce_add_slice_to(self, b, output);
    }
}

/// Value-side mirror of [`ReduceSubSlice`].
pub trait SubModuloSlice<M, B: ?Sized = Self> {
    /// Calculates `self[i] = (self[i] - b[i]) (mod modulus)` element-wise.
    fn sub_modulo_slice_assign(&mut self, b: &B, modulus: M);

    /// Writes `self[i] - b[i] (mod modulus)` into `output[i]`.
    fn sub_modulo_slice_to(&self, b: &B, output: &mut Self, modulus: M);
}

impl<T, M, B> SubModuloSlice<M, [B]> for [T]
where
    M: ReduceSubSlice<T, B> + Copy,
{
    #[inline(always)]
    fn sub_modulo_slice_assign(&mut self, b: &[B], modulus: M) {
        modulus.reduce_sub_slice_assign(self, b);
    }

    #[inline(always)]
    fn sub_modulo_slice_to(&self, b: &[B], output: &mut [T], modulus: M) {
        modulus.reduce_sub_slice_to(self, b, output);
    }
}

/// Value-side mirror of [`ReduceMulSlice`].
pub trait MulModuloSlice<M, B: ?Sized = Self> {
    /// Calculates `self[i] = (self[i] * b[i]) (mod modulus)` element-wise.
    fn mul_modulo_slice_assign(&mut self, b: &B, modulus: M);

    /// Writes `self[i] * b[i] (mod modulus)` into `output[i]`.
    fn mul_modulo_slice_to(&self, b: &B, output: &mut Self, modulus: M);
}

impl<T, M, B> MulModuloSlice<M, [B]> for [T]
where
    M: ReduceMulSlice<T, B> + Copy,
{
    #[inline(always)]
    fn mul_modulo_slice_assign(&mut self, b: &[B], modulus: M) {
        modulus.reduce_mul_slice_assign(self, b);
    }

    #[inline(always)]
    fn mul_modulo_slice_to(&self, b: &[B], output: &mut [T], modulus: M) {
        modulus.reduce_mul_slice_to(self, b, output);
    }
}

/// Value-side mirror of [`ReduceMulAddSlice`].
///
/// The receiver `self` plays the role of the accumulator or first
/// multiplicand depending on the method (see method docs).
pub trait MulAddModuloSlice<M, T> {
    /// `self[i] += a[i] * b[i] (mod modulus)` — FMAC accumulate.
    fn add_mul_modulo_slice_assign(&mut self, a: &[T], b: &[T], modulus: M);

    /// `self[i] -= a[i] * b[i] (mod modulus)` — fused multiply-subtract.
    fn sub_mul_modulo_slice_assign(&mut self, a: &[T], b: &[T], modulus: M);

    /// `output[i] = self[i] * b[i] + c[i] (mod modulus)`.
    fn mul_add_modulo_slice_to(&self, b: &[T], c: &[T], output: &mut [T], modulus: M);

    /// `output[i] = scalar * self[i] + c[i] (mod modulus)`.
    ///
    /// Note: `self` is the slice playing the role of `b` in the
    /// modulus-side `reduce_scalar_mul_add_slice_to(scalar, b, c, out)`.
    fn scalar_mul_add_modulo_slice_to(&self, scalar: T, c: &[T], output: &mut [T], modulus: M);
}

impl<T, M> MulAddModuloSlice<M, T> for [T]
where
    M: ReduceMulAddSlice<T> + Copy,
{
    #[inline(always)]
    fn add_mul_modulo_slice_assign(&mut self, a: &[T], b: &[T], modulus: M) {
        modulus.reduce_add_mul_slice_assign(self, a, b);
    }

    #[inline(always)]
    fn sub_mul_modulo_slice_assign(&mut self, a: &[T], b: &[T], modulus: M) {
        modulus.reduce_sub_mul_slice_assign(self, a, b);
    }

    #[inline(always)]
    fn mul_add_modulo_slice_to(&self, b: &[T], c: &[T], output: &mut [T], modulus: M) {
        modulus.reduce_mul_add_slice_to(self, b, c, output);
    }

    #[inline(always)]
    fn scalar_mul_add_modulo_slice_to(&self, scalar: T, c: &[T], output: &mut [T], modulus: M) {
        modulus.reduce_scalar_mul_add_slice_to(scalar, self, c, output);
    }
}

/// The modular dot product.
///
/// This is always used for slice. For example, `u64` slice `[u64]`.
///
/// For two same length slice `a = (a₀, a₁, ..., an)` and `b = (b₀, b₁, ..., bn)`.
///
/// This trait will calculate `a₀×b₀ + a₁×b₁ + ... + an×bn mod modulus`.
pub trait DotProductModulo<M, T>
where
    Self: AsRef<[T]>,
{
    /// Calculate `∑a_i×b_i (mod modulus)`.
    fn dot_product_modulo<B>(self, b: B, modulus: M) -> T
    where
        B: AsRef<[T]>;
}

impl<M, T, A> DotProductModulo<M, T> for A
where
    A: AsRef<[T]>,
    M: ReduceDotProduct<T, Output = T>,
{
    #[inline(always)]
    fn dot_product_modulo<B>(self, b: B, modulus: M) -> T
    where
        B: AsRef<[T]>,
    {
        modulus.reduce_dot_product(self, b)
    }
}

/// The modular dot product.
///
/// This is always used for slice. For example, `u64` slice `[u64]`.
///
/// For two same length slice `a = (a₀, a₁, ..., an)` and `b = (b₀, b₁, ..., bn)`.
///
/// This trait will calculate `a₀×b₀ + a₁×b₁ + ... + an×bn mod modulus`.
pub trait DotProductModuloIter<M, T>
where
    Self: IntoIterator<Item = T>,
{
    /// Calculate `∑a_i×b_i (mod modulus)`.
    fn dot_product_modulo_iter<B>(self, b: B, modulus: M) -> T
    where
        B: IntoIterator<Item = T>;
}

impl<M, T, A> DotProductModuloIter<M, T> for A
where
    A: IntoIterator<Item = T>,
    M: ReduceDotProduct<T, Output = T>,
{
    #[inline(always)]
    fn dot_product_modulo_iter<B>(self, b: B, modulus: M) -> T
    where
        B: IntoIterator<Item = T>,
    {
        modulus.reduce_dot_product_iter(self, b)
    }
}
