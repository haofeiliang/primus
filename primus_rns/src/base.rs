use std::slice::Iter;

use itertools::Itertools;
use primus_data::{Data, DataMut, RawData};
use primus_factor::{FactorMul, FactorSliceOps, ShoupFactor};
use primus_integer::{
    BigUint, BigUintIter, BigUintIterMut, FheUint, izip, multiply_many_values,
    multiply_many_values_except_inplace,
};
use primus_modulo::prelude::*;
use primus_modulus::UintModulus;
use primus_poly::{BigUintPolynomial, CrtPolynomial, Polynomial};
use primus_reduce::{FieldContext, ReduceAddAssign};

#[cfg(feature = "simd")]
use primus_integer::{SimdArray, SimdMaskArray, SimdUnsignedInteger};
#[cfg(feature = "simd")]
use std::simd::{
    Simd,
    cmp::{SimdOrd, SimdPartialOrd},
};

use crate::RNSError;

/// A residue number system or residue numeral system (RNS) is a numeral system representing integers
/// by their values modulo several pairwise coprime integers called the moduli.
/// This representation is allowed by the Chinese remainder theorem,
/// which asserts that, if M is the product of the moduli, there is,
/// in an interval of length M, exactly one integer having any given set of modular values.
/// Using a residue numeral system for arithmetic operations is also called multi-modular arithmetic.
#[derive(Clone)]
pub struct RNSBase<T, M>
where
    T: FheUint,
    M: FieldContext<T>,
{
    moduli: Vec<M>,
    moduli_product: BigUint<Vec<T>>,
    punctured_product: Vec<T>,
    inv_punctured_product_mod_modulus: Vec<ShoupFactor<T>>,
}

// ===========================================================================
// Wrapping decompose dispatch trait – scalar / SIMD.
//
// Pattern: same as `UintModulus` slice traits — always provide a scalar
// blanket impl, then override with concrete SIMD impls under `cfg(simd)`.
// ===========================================================================

/// Dispatch trait dispatching wrapping-decompose inner loops to scalar or SIMD.
trait WrappingDecomposeDispatch: FheUint {
    /// `residues[i] = if small_values[i] < half { small_values[i] } else { temp + small_values[i] }`
    fn decompose_chunk(residues: &mut [Self], small_values: &[Self], half: Self, temp: Self);

    /// Fused: `dest[i] += factor * decomposed(small_values[i])  (mod modulus)`.
    fn decompose_chunk_scaled(
        dest: &mut [Self],
        small_values: &[Self],
        half: Self,
        temp: Self,
        modulus: Self,
        factor: ShoupFactor<Self>,
    );
}

// ---- scalar helpers ------------------------------------------------------

#[inline]
fn decompose_chunk_scalar<T: FheUint>(residues: &mut [T], small_values: &[T], half: T, temp: T) {
    for (residue, &value) in residues.iter_mut().zip(small_values) {
        *residue = if value < half { value } else { temp + value };
    }
}

#[inline]
fn decompose_chunk_scaled_scalar<T: FheUint>(
    dest: &mut [T],
    small_values: &[T],
    half: T,
    temp: T,
    modulus: T,
    factor: ShoupFactor<T>,
) {
    for (d, &value) in dest.iter_mut().zip(small_values) {
        let centered = if value < half { value } else { temp + value };
        UintModulus(modulus).reduce_add_assign(d, factor.factor_mul_modulo(centered, modulus));
    }
}

// ---- SIMD helpers --------------------------------------------------------

#[cfg(feature = "simd")]
use primus_factor::SimdShoupFactor;

#[cfg(feature = "simd")]
#[inline]
fn decompose_chunk_simd<T: FheUint, const N: usize>(
    residues: &mut [T],
    small_values: &[T],
    half: T,
    temp: T,
) where
    Simd<T, N>: SimdArray<T, N>,
{
    let half_simd = Simd::splat(half);
    let temp_simd = Simd::splat(temp);
    let (res_chunks, res_rem) = residues.as_chunks_mut::<N>();
    let (val_chunks, val_rem) = small_values.as_chunks::<N>();
    for (res, val) in res_chunks.iter_mut().zip(val_chunks) {
        let v = Simd::from_array(*val);
        let mask = v.simd_lt(half_simd);
        *res = mask.select(v, temp_simd + v).to_array();
    }
    decompose_chunk_scalar(res_rem, val_rem, half, temp);
}

#[cfg(feature = "simd")]
#[inline]
fn decompose_chunk_scaled_simd<T: FheUint, const N: usize>(
    dest: &mut [T],
    small_values: &[T],
    half: T,
    temp: T,
    modulus: T,
    factor: ShoupFactor<T>,
) where
    Simd<T, N>: SimdArray<T, N>,
{
    let half_simd = Simd::splat(half);
    let temp_simd = Simd::splat(temp);
    let modulus_simd = Simd::splat(modulus);
    let simd_factor = SimdShoupFactor::<T, N>::from(factor);

    let (dest_chunks, dest_rem) = dest.as_chunks_mut::<N>();
    let (val_chunks, val_rem) = small_values.as_chunks::<N>();
    for (dest_chunk, val_chunk) in dest_chunks.iter_mut().zip(val_chunks) {
        let v = Simd::from_array(*val_chunk);
        let mask = v.simd_lt(half_simd);
        let centered = mask.select(v, temp_simd + v);
        let product = simd_factor.factor_mul_modulo(centered, modulus_simd);
        let dest_val = Simd::from_array(*dest_chunk);
        let sum = dest_val + product;
        *dest_chunk = sum.simd_min(sum - modulus_simd).to_array();
    }
    decompose_chunk_scaled_scalar(dest_rem, val_rem, half, temp, modulus, factor);
}

// ---- macro-generated impls -----------------------------------------------
//
// Pattern: same as `primus_factor` — blanket impl marked `default` when
// `min_specialization` is available (nightly + simd), then concrete SIMD
// impls override it at monomorphisation time. Without simd, the blanket is
// the only impl and `default` is omitted.

macro_rules! impl_decompose_blanket {
    ($($default_kw:ident)?) => {
        impl<T: FheUint> WrappingDecomposeDispatch for T {
            $($default_kw)? fn decompose_chunk(residues: &mut [Self], small_values: &[Self], half: Self, temp: Self) {
                decompose_chunk_scalar(residues, small_values, half, temp);
            }
            $($default_kw)? fn decompose_chunk_scaled(
                dest: &mut [Self],
                small_values: &[Self],
                half: Self,
                temp: Self,
                modulus: Self,
                factor: ShoupFactor<Self>,
            ) {
                decompose_chunk_scaled_scalar(dest, small_values, half, temp, modulus, factor);
            }
        }
    };
}

#[cfg(not(feature = "simd"))]
impl_decompose_blanket!();

#[cfg(feature = "simd")]
impl_decompose_blanket!(default);

#[cfg(feature = "simd")]
macro_rules! impl_decompose_simd {
    ($t:ty, $lanes:expr) => {
        impl WrappingDecomposeDispatch for $t {
            #[inline]
            fn decompose_chunk(
                residues: &mut [Self],
                small_values: &[Self],
                half: Self,
                temp: Self,
            ) {
                decompose_chunk_simd::<$t, { $lanes }>(residues, small_values, half, temp);
            }
            #[inline]
            fn decompose_chunk_scaled(
                dest: &mut [Self],
                small_values: &[Self],
                half: Self,
                temp: Self,
                modulus: Self,
                factor: ShoupFactor<Self>,
            ) {
                decompose_chunk_scaled_simd::<$t, { $lanes }>(
                    dest,
                    small_values,
                    half,
                    temp,
                    modulus,
                    factor,
                );
            }
        }
    };
}

#[cfg(feature = "simd")]
impl_decompose_simd!(u16, u16::LANE_COUNT);
#[cfg(feature = "simd")]
impl_decompose_simd!(u32, u32::LANE_COUNT);
#[cfg(feature = "simd")]
impl_decompose_simd!(u64, u64::LANE_COUNT);

impl<T, M> RNSBase<T, M>
where
    T: FheUint,
    M: FieldContext<T>,
{
    /// Creates a new [`RNSBase<T, M>`].
    ///
    /// # Panics
    ///
    /// Panics if any inverse modulo operation panics.
    ///
    /// # Errors
    ///
    /// This function will return an error if moduli are not co-prime with each others.
    #[inline]
    pub fn new(moduli: &[M]) -> Result<Self, RNSError> {
        let moduli_values = moduli
            .iter()
            .map(|m| unsafe { m.value_unchecked() })
            .collect::<Vec<_>>();

        if moduli_values
            .iter()
            .tuple_combinations()
            .any(|(&a, &b)| !a.is_coprime(b))
        {
            return Err(RNSError::CoPrimeError);
        }

        let moduli_product = multiply_many_values(&moduli_values);

        let big_uint_len = moduli_product.len();
        let mut punctured_product = vec![T::ZERO; big_uint_len * moduli.len()];
        punctured_product
            .chunks_exact_mut(big_uint_len)
            .enumerate()
            .for_each(|(i, chunk)| {
                multiply_many_values_except_inplace(&moduli_values, i, chunk);
            });

        let inv_punctured_product_mod_modulus = punctured_product
            .chunks_exact(big_uint_len)
            .zip(moduli)
            .map(|(p, &modulus)| {
                let inv = p.modulo(modulus).try_inv_modulo(modulus).unwrap();
                ShoupFactor::new(inv, unsafe { modulus.value_unchecked() })
            })
            .collect::<Vec<ShoupFactor<T>>>();

        Ok(Self {
            moduli: moduli.to_vec(),
            moduli_product,
            punctured_product,
            inv_punctured_product_mod_modulus,
        })
    }

    /// Returns a reference to the moduli of this [`RNSBase<T, M>`].
    #[inline]
    pub fn moduli(&self) -> &[M] {
        &self.moduli
    }

    #[inline]
    pub fn moduli_count(&self) -> usize {
        self.moduli.len()
    }

    /// Returns a reference to the moduli product of this [`RNSBase<T, M>`].
    #[inline]
    pub fn moduli_product(&self) -> BigUint<&[T]> {
        self.moduli_product.view()
    }

    #[inline]
    pub fn big_uint_value_len(&self) -> usize {
        self.moduli_product.len()
    }

    /// Returns a reference to the punctured product of this [`RNSBase<T, M>`].
    #[inline]
    pub fn punctured_product(&self) -> &[T] {
        &self.punctured_product
    }

    /// Returns an iterator over the punctured product of this [`RNSBase<T, M>`].
    #[inline]
    pub fn iter_punctured_product(&self) -> std::slice::ChunksExact<'_, T> {
        self.punctured_product
            .chunks_exact(self.moduli_product.len())
    }

    /// Returns a reference to the inverse punctured product mod modulus of this [`RNSBase<T, M>`].
    #[inline]
    pub fn inv_punctured_product_mod_modulus(&self) -> &[ShoupFactor<T>] {
        &self.inv_punctured_product_mod_modulus
    }

    /// Decomposes a value into its RNS representation.
    #[inline]
    pub fn decompose(&self, BigUint(value): BigUint<&[T]>) -> Vec<T> {
        self.moduli
            .iter()
            .map(|&modulus| value.modulo(modulus))
            .collect()
    }

    #[inline]
    pub fn decompose_to_rns_factor(&self, BigUint(value): BigUint<&[T]>) -> Vec<ShoupFactor<T>> {
        self.moduli
            .iter()
            .map(|&modulus| {
                ShoupFactor::new(value.modulo(modulus), unsafe { modulus.value_unchecked() })
            })
            .collect()
    }

    pub fn wrapping_decompose(&self, value: T, small_value_modulus: T) -> Vec<T> {
        if small_value_modulus != T::TWO {
            let half = (small_value_modulus + T::ONE) / T::TWO;
            self.moduli
                .iter()
                .map(|m| unsafe { m.value_unchecked() })
                .map(|modulus| {
                    if value < half {
                        value
                    } else {
                        modulus - small_value_modulus + value
                    }
                })
                .collect()
        } else {
            vec![value; self.moduli_count()]
        }
    }

    /// Decomposes a value into its RNS representation, writing the result into the provided slice.
    #[inline]
    pub fn decompose_inplace(&self, BigUint(value): BigUint<&[T]>, residues: &mut [T]) {
        debug_assert_eq!(self.moduli_count(), residues.len());

        for (residue, &modulus) in residues.iter_mut().zip(self.moduli.iter()) {
            *residue = value.modulo(modulus);
        }
    }

    pub fn wrapping_decompose_inplace(&self, value: T, residues: &mut [T], small_value_modulus: T) {
        debug_assert_eq!(self.moduli_count(), residues.len());

        if small_value_modulus != T::TWO {
            let half = (small_value_modulus + T::ONE) / T::TWO;
            self.moduli
                .iter()
                .map(|m| unsafe { m.value_unchecked() })
                .zip(residues)
                .map(|(modulus, residue)| {
                    *residue = if value < half {
                        value
                    } else {
                        modulus - small_value_modulus + value
                    };
                })
                .collect()
        } else {
            residues.fill(value);
        }
    }

    pub fn wrapping_decompose_small_values_inplace(
        &self,
        small_values: &[T],
        multi_residues: &mut [T],
        value_count: usize,
        small_values_modulus: T,
    ) {
        debug_assert_eq!(multi_residues.len(), self.moduli_count() * value_count);
        debug_assert_eq!(small_values.len(), value_count);
        debug_assert!(
            self.moduli
                .iter()
                .all(|m| unsafe { m.value_unchecked() } > small_values_modulus)
        );
        if small_values_modulus != T::TWO {
            let half = (small_values_modulus + T::ONE) / T::TWO;
            for (residues, modulus) in multi_residues
                .chunks_exact_mut(value_count)
                .zip(self.moduli().iter().map(|m| unsafe { m.value_unchecked() }))
            {
                let temp = modulus - small_values_modulus;
                T::decompose_chunk(residues, small_values, half, temp);
            }
        } else {
            for residues in multi_residues.chunks_exact_mut(value_count) {
                residues.copy_from_slice(small_values);
            }
        }
    }

    /// Fused: wrapping-decompose a small polynomial into RNS, scale by `factor`, and
    /// accumulate into `destination` (which is in multi-residue layout).
    ///
    /// Equivalent to `wrapping_decompose_small_values_inplace` followed by
    /// `CrtPolynomial::add_mul_factor_assign`, but without the intermediate storage.
    pub fn add_wrapping_decompose_small_values_scaled(
        &self,
        small_values: &[T],
        destination: &mut [T],
        value_count: usize,
        small_values_modulus: T,
        factor: &[ShoupFactor<T>],
    ) {
        debug_assert_eq!(destination.len(), self.moduli_count() * value_count);
        debug_assert_eq!(small_values.len(), value_count);
        debug_assert_eq!(factor.len(), self.moduli_count());
        debug_assert!(
            self.moduli
                .iter()
                .all(|m| unsafe { m.value_unchecked() } > small_values_modulus)
        );

        if small_values_modulus != T::TWO {
            let half = (small_values_modulus + T::ONE) / T::TWO;
            izip!(
                destination.chunks_exact_mut(value_count),
                self.moduli().iter().map(|m| unsafe { m.value_unchecked() }),
                factor,
            )
            .for_each(|(dest_chunk, modulus, &factor)| {
                let temp = modulus - small_values_modulus;
                T::decompose_chunk_scaled(dest_chunk, small_values, half, temp, modulus, factor);
            });
        } else {
            izip!(
                destination.chunks_exact_mut(value_count),
                self.moduli().iter().map(|m| unsafe { m.value_unchecked() }),
                factor,
            )
            .for_each(|(dest_chunk, _modulus, &factor)| {
                factor.add_factor_mul_slice_assign(dest_chunk, small_values, _modulus);
            });
        }
    }

    pub fn decompose_big_uint_values_inplace(
        &self,
        big_uint_values: &[T],
        multi_residues: &mut [T],
        value_count: usize,
    ) {
        debug_assert_eq!(multi_residues.len(), self.moduli_count() * value_count);
        debug_assert_eq!(
            big_uint_values.len(),
            self.big_uint_value_len() * value_count
        );

        let value_len = self.big_uint_value_len();
        for (residues, &modulus) in multi_residues
            .chunks_exact_mut(value_count)
            .zip(self.moduli())
        {
            for (residue, value) in residues
                .iter_mut()
                .zip(big_uint_values.chunks_exact(value_len))
            {
                *residue = value.modulo(modulus);
            }
        }
    }

    #[inline]
    pub fn wrapping_decompose_small_polynomial_inplace<A, B>(
        &self,
        small_poly: &Polynomial<A>,
        crt_poly: &mut CrtPolynomial<B>,
        poly_length: usize,
        small_poly_modulus: T,
    ) where
        A: RawData<Elem = T> + Data,
        B: RawData<Elem = T> + DataMut,
    {
        self.wrapping_decompose_small_values_inplace(
            small_poly.as_ref(),
            crt_poly.as_mut(),
            poly_length,
            small_poly_modulus,
        );
    }

    #[inline]
    pub fn add_wrapping_decompose_small_polynomial_scaled<A, C>(
        &self,
        small_poly: &Polynomial<A>,
        destination: &mut CrtPolynomial<C>,
        poly_length: usize,
        small_poly_modulus: T,
        factor: &[ShoupFactor<T>],
    ) where
        A: RawData<Elem = T> + Data,
        C: RawData<Elem = T> + DataMut,
    {
        self.add_wrapping_decompose_small_values_scaled(
            small_poly.as_ref(),
            destination.as_mut(),
            poly_length,
            small_poly_modulus,
            factor,
        );
    }

    #[inline]
    pub fn decompose_polynomial_inplace<A, B>(
        &self,
        big_uint_poly: &BigUintPolynomial<A>,
        crt_poly: &mut CrtPolynomial<B>,
        poly_length: usize,
    ) where
        A: RawData<Elem = T> + Data,
        B: RawData<Elem = T> + DataMut,
    {
        self.decompose_big_uint_values_inplace(
            big_uint_poly.as_slice(),
            crt_poly.as_mut(),
            poly_length,
        );
    }

    /// Composes a value from its RNS representation.
    pub fn compose(&self, residues: &[T]) -> BigUint<Vec<T>> {
        debug_assert_eq!(self.moduli_count(), residues.len());

        let value_len = self.big_uint_value_len();
        let moduli_product = &self.moduli_product();

        let mut value = BigUint(vec![T::ZERO; value_len]);

        izip!(
            residues,
            &self.inv_punctured_product_mod_modulus,
            BigUintIter::new(&self.punctured_product, value_len),
            &self.moduli
        )
        .for_each(
            |(&ri, &inv_mi, mi, &modulus): (&T, &ShoupFactor<T>, BigUint<&[T]>, &M)| {
                let product = inv_mi.factor_mul_modulo(ri, unsafe { modulus.value_unchecked() });
                let carry = mi.mul_value_add_inplace(product, &mut value);
                if !carry.is_zero() || value.cmp(moduli_product).is_ge() {
                    let _ = value.sub_assign(moduli_product);
                }
            },
        );

        value
    }

    pub fn compose_inplace(&self, residues: &[T], value: &mut BigUint<&mut [T]>) {
        debug_assert_eq!(self.moduli_count(), residues.len());
        debug_assert_eq!(self.big_uint_value_len(), value.len());

        let value_len = self.moduli_product.len();
        let moduli_product = &self.moduli_product();

        value.set_zero();

        izip!(
            residues,
            &self.inv_punctured_product_mod_modulus,
            BigUintIter::new(&self.punctured_product, value_len),
            &self.moduli
        )
        .for_each(
            |(&ri, &inv_mi, mi, &modulus): (&T, &ShoupFactor<T>, BigUint<&[T]>, &M)| {
                let product = inv_mi.factor_mul_modulo(ri, unsafe { modulus.value_unchecked() });
                let carry = mi.mul_value_add_inplace(product, value);
                if !carry.is_zero() || value.cmp(moduli_product).is_ge() {
                    let _ = value.sub_assign(moduli_product);
                }
            },
        );
    }

    pub fn compose_multiple_values_inplace(
        &self,
        multi_residues: &[T],
        big_uint_values: &mut [T],
        value_count: usize,
        residues: &mut [T],
    ) {
        debug_assert_eq!(multi_residues.len(), self.moduli_count() * value_count);
        debug_assert_eq!(
            big_uint_values.len(),
            self.big_uint_value_len() * value_count
        );

        let big_uint_value_len = self.big_uint_value_len();
        // let mut residues = vec![T::ZERO; self.moduli_count()];

        let mut iters: Vec<Iter<'_, T>> = multi_residues
            .chunks_exact(value_count)
            .map(|s| s.iter())
            .collect();

        for ref mut value in BigUintIterMut::new(big_uint_values, big_uint_value_len) {
            for (iter, residue) in iters.iter_mut().zip(residues.iter_mut()) {
                *residue = *iter.next().unwrap();
            }
            self.compose_inplace(residues, value);
        }
    }

    #[inline]
    pub fn compose_polynomial_inplace<A, B>(
        &self,
        crt_poly: &CrtPolynomial<A>,
        big_uint_poly: &mut BigUintPolynomial<B>,
        poly_length: usize,
        compose_buffer: &mut [T],
    ) where
        A: RawData<Elem = T> + Data,
        B: RawData<Elem = T> + DataMut,
    {
        self.compose_multiple_values_inplace(
            crt_poly.as_ref(),
            big_uint_poly.as_mut_slice(),
            poly_length,
            compose_buffer,
        );
    }
}
