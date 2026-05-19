use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub(crate) fn impl_reduce_slice_ops(
    name: &Ident,
    modulus: &TokenStream,
    ty: &syn::Path,
    ratio: &[TokenStream; 2],
) -> TokenStream {
    let [r0, r1] = ratio;
    quote! {
        // -----------------------------------------------------------------
        // ReduceOnceSlice
        // -----------------------------------------------------------------

        #[cfg(not(all(feature = "nightly", feature = "simd")))]
        impl ::primus_modulus::reduce::ReduceOnceSlice<#ty> for #name {
            #[inline]
            fn reduce_once_slice_assign(self, values: &mut [#ty]) {
                use ::primus_modulus::reduce::ReduceOnceAssign;
                values.iter_mut().for_each(|v| self.reduce_once_assign(v));
            }
            #[inline]
            fn reduce_once_slice_to(self, input: &[#ty], output: &mut [#ty]) {
                use ::primus_modulus::reduce::ReduceOnce;
                debug_assert_eq!(input.len(), output.len());
                input.iter().zip(output).for_each(|(&v, o)| *o = self.reduce_once(v));
            }
        }

        #[cfg(all(feature = "nightly", feature = "simd"))]
        impl ::primus_modulus::reduce::ReduceOnceSlice<#ty> for #name {
            #[inline]
            fn reduce_once_slice_assign(self, values: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_once_slice_assign::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    values,
                )
            }
            #[inline]
            fn reduce_once_slice_to(self, input: &[#ty], output: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_once_slice_to::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    input,
                    output,
                )
            }
        }

        // -----------------------------------------------------------------
        // ReduceNegSlice
        // -----------------------------------------------------------------

        #[cfg(not(all(feature = "nightly", feature = "simd")))]
        impl ::primus_modulus::reduce::ReduceNegSlice<#ty> for #name {
            #[inline]
            fn reduce_neg_slice_assign(self, values: &mut [#ty]) {
                use ::primus_modulus::reduce::ReduceNegAssign;
                values.iter_mut().for_each(|v| self.reduce_neg_assign(v));
            }
            #[inline]
            fn reduce_neg_slice_to(self, input: &[#ty], output: &mut [#ty]) {
                use ::primus_modulus::reduce::ReduceNeg;
                debug_assert_eq!(input.len(), output.len());
                input.iter().zip(output).for_each(|(&v, o)| *o = self.reduce_neg(v));
            }
        }

        #[cfg(all(feature = "nightly", feature = "simd"))]
        impl ::primus_modulus::reduce::ReduceNegSlice<#ty> for #name {
            #[inline]
            fn reduce_neg_slice_assign(self, values: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_neg_slice_assign::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    values,
                )
            }
            #[inline]
            fn reduce_neg_slice_to(self, input: &[#ty], output: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_neg_slice_to::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    input,
                    output,
                )
            }
        }

        // -----------------------------------------------------------------
        // ReduceAddSlice
        // -----------------------------------------------------------------

        #[cfg(not(all(feature = "nightly", feature = "simd")))]
        impl ::primus_modulus::reduce::ReduceAddSlice<#ty> for #name {
            #[inline]
            fn reduce_add_slice_assign(self, a: &mut [#ty], b: &[#ty]) {
                use ::primus_modulus::reduce::ReduceAddAssign;
                debug_assert_eq!(a.len(), b.len());
                a.iter_mut().zip(b).for_each(|(a, &b)| self.reduce_add_assign(a, b));
            }
            #[inline]
            fn reduce_add_slice_to(self, a: &[#ty], b: &[#ty], output: &mut [#ty]) {
                use ::primus_modulus::reduce::ReduceAdd;
                debug_assert_eq!(a.len(), b.len());
                debug_assert_eq!(a.len(), output.len());
                a.iter().zip(b).zip(output).for_each(|((&a, &b), o)| *o = self.reduce_add(a, b));
            }
        }

        #[cfg(all(feature = "nightly", feature = "simd"))]
        impl ::primus_modulus::reduce::ReduceAddSlice<#ty> for #name {
            #[inline]
            fn reduce_add_slice_assign(self, a: &mut [#ty], b: &[#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_add_slice_assign::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    b,
                )
            }
            #[inline]
            fn reduce_add_slice_to(self, a: &[#ty], b: &[#ty], output: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_add_slice_to::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    b,
                    output,
                )
            }
        }

        // -----------------------------------------------------------------
        // ReduceSubSlice
        // -----------------------------------------------------------------

        #[cfg(not(all(feature = "nightly", feature = "simd")))]
        impl ::primus_modulus::reduce::ReduceSubSlice<#ty> for #name {
            #[inline]
            fn reduce_sub_slice_assign(self, a: &mut [#ty], b: &[#ty]) {
                use ::primus_modulus::reduce::ReduceSubAssign;
                debug_assert_eq!(a.len(), b.len());
                a.iter_mut().zip(b).for_each(|(a, &b)| self.reduce_sub_assign(a, b));
            }
            #[inline]
            fn reduce_sub_slice_to(self, a: &[#ty], b: &[#ty], output: &mut [#ty]) {
                use ::primus_modulus::reduce::ReduceSub;
                debug_assert_eq!(a.len(), b.len());
                debug_assert_eq!(a.len(), output.len());
                a.iter().zip(b).zip(output).for_each(|((&a, &b), o)| *o = self.reduce_sub(a, b));
            }
        }

        #[cfg(all(feature = "nightly", feature = "simd"))]
        impl ::primus_modulus::reduce::ReduceSubSlice<#ty> for #name {
            #[inline]
            fn reduce_sub_slice_assign(self, a: &mut [#ty], b: &[#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_sub_slice_assign::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    b,
                )
            }
            #[inline]
            fn reduce_sub_slice_to(self, a: &[#ty], b: &[#ty], output: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_sub_slice_to::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    b,
                    output,
                )
            }
        }

        // -----------------------------------------------------------------
        // ReduceMulSlice
        // -----------------------------------------------------------------

        #[cfg(not(all(feature = "nightly", feature = "simd")))]
        impl ::primus_modulus::reduce::ReduceMulSlice<#ty> for #name {
            #[inline]
            fn reduce_mul_slice_assign(self, a: &mut [#ty], b: &[#ty]) {
                use ::primus_modulus::reduce::ReduceMulAssign;
                debug_assert_eq!(a.len(), b.len());
                a.iter_mut().zip(b).for_each(|(a, &b)| self.reduce_mul_assign(a, b));
            }
            #[inline]
            fn reduce_mul_slice_to(self, a: &[#ty], b: &[#ty], output: &mut [#ty]) {
                use ::primus_modulus::reduce::ReduceMul;
                debug_assert_eq!(a.len(), b.len());
                debug_assert_eq!(a.len(), output.len());
                a.iter().zip(b).zip(output).for_each(|((&a, &b), o)| *o = self.reduce_mul(a, b));
            }
            #[inline]
            fn reduce_scalar_mul_slice_assign(self, a: &mut [#ty], scalar: #ty) {
                use ::primus_modulus::reduce::ReduceMulAssign;
                a.iter_mut().for_each(|a| self.reduce_mul_assign(a, scalar));
            }
            #[inline]
            fn reduce_scalar_mul_slice_to(self, a: &[#ty], scalar: #ty, output: &mut [#ty]) {
                use ::primus_modulus::reduce::ReduceMul;
                debug_assert_eq!(a.len(), output.len());
                a.iter().zip(output).for_each(|(&a, o)| *o = self.reduce_mul(a, scalar));
            }
        }

        #[cfg(all(feature = "nightly", feature = "simd"))]
        impl ::primus_modulus::reduce::ReduceMulSlice<#ty> for #name {
            #[inline]
            fn reduce_mul_slice_assign(self, a: &mut [#ty], b: &[#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_mul_slice_assign::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    b,
                )
            }
            #[inline]
            fn reduce_mul_slice_to(self, a: &[#ty], b: &[#ty], output: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_mul_slice_to::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    b,
                    output,
                )
            }
            #[inline]
            fn reduce_scalar_mul_slice_assign(self, a: &mut [#ty], scalar: #ty) {
                ::primus_modulus::barrett_simd_kernel::reduce_scalar_mul_slice_assign::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    scalar,
                )
            }
            #[inline]
            fn reduce_scalar_mul_slice_to(self, a: &[#ty], scalar: #ty, output: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_scalar_mul_slice_to::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    scalar,
                    output,
                )
            }
        }

        // -----------------------------------------------------------------
        // ReduceMulAddSlice
        // -----------------------------------------------------------------

        #[cfg(not(all(feature = "nightly", feature = "simd")))]
        impl ::primus_modulus::reduce::ReduceMulAddSlice<#ty> for #name {
            #[inline]
            fn reduce_add_mul_slice_assign(self, acc: &mut [#ty], a: &[#ty], b: &[#ty]) {
                use ::primus_modulus::reduce::ReduceMulAdd;
                debug_assert_eq!(acc.len(), a.len());
                debug_assert_eq!(acc.len(), b.len());
                acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
                    *acc = self.reduce_mul_add(a, b, *acc);
                });
            }

            #[inline]
            fn reduce_sub_mul_slice_assign(self, acc: &mut [#ty], a: &[#ty], b: &[#ty]) {
                use ::primus_modulus::reduce::{ReduceMul, ReduceSubAssign};
                debug_assert_eq!(acc.len(), a.len());
                debug_assert_eq!(acc.len(), b.len());
                acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
                    let prod = self.reduce_mul(a, b);
                    self.reduce_sub_assign(acc, prod);
                });
            }

            #[inline]
            fn reduce_mul_add_slice_to(
                self,
                a: &[#ty],
                b: &[#ty],
                c: &[#ty],
                output: &mut [#ty],
            ) {
                use ::primus_modulus::reduce::ReduceMulAdd;
                debug_assert_eq!(a.len(), b.len());
                debug_assert_eq!(a.len(), c.len());
                debug_assert_eq!(a.len(), output.len());
                a.iter().zip(b).zip(c).zip(output).for_each(|(((&a, &b), &c), o)| {
                    *o = self.reduce_mul_add(a, b, c);
                });
            }

            #[inline]
            fn reduce_scalar_mul_add_slice_to(
                self,
                scalar: #ty,
                b: &[#ty],
                c: &[#ty],
                output: &mut [#ty],
            ) {
                use ::primus_modulus::reduce::ReduceMulAdd;
                debug_assert_eq!(b.len(), c.len());
                debug_assert_eq!(b.len(), output.len());
                b.iter().zip(c).zip(output).for_each(|((&b, &c), o)| {
                    *o = self.reduce_mul_add(scalar, b, c);
                });
            }
        }

        #[cfg(all(feature = "nightly", feature = "simd"))]
        impl ::primus_modulus::reduce::ReduceMulAddSlice<#ty> for #name {
            #[inline]
            fn reduce_add_mul_slice_assign(self, acc: &mut [#ty], a: &[#ty], b: &[#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_add_mul_slice_assign::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    acc,
                    a,
                    b,
                )
            }

            #[inline]
            fn reduce_sub_mul_slice_assign(self, acc: &mut [#ty], a: &[#ty], b: &[#ty]) {
                ::primus_modulus::barrett_simd_kernel::reduce_sub_mul_slice_assign::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    acc,
                    a,
                    b,
                )
            }

            #[inline]
            fn reduce_mul_add_slice_to(
                self,
                a: &[#ty],
                b: &[#ty],
                c: &[#ty],
                output: &mut [#ty],
            ) {
                ::primus_modulus::barrett_simd_kernel::reduce_mul_add_slice_to::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    a,
                    b,
                    c,
                    output,
                )
            }

            #[inline]
            fn reduce_scalar_mul_add_slice_to(
                self,
                scalar: #ty,
                b: &[#ty],
                c: &[#ty],
                output: &mut [#ty],
            ) {
                ::primus_modulus::barrett_simd_kernel::reduce_scalar_mul_add_slice_to::<#ty, {
                    ::primus_integer::lanes::VECTOR_BITS
                        / (<#ty>::BITS as usize)
                }>(
                    ::primus_modulus::BarrettModulus::<#ty>::from_parts(
                        #modulus,
                        [#r0, #r1],
                    ),
                    scalar,
                    b,
                    c,
                    output,
                )
            }
        }
    }
}
