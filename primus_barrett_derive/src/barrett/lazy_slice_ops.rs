use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub(crate) fn impl_lazy_reduce_slice_ops(
    name: &Ident,
    modulus: &TokenStream,
    ty: &syn::Path,
    ratio: &[TokenStream; 2],
) -> TokenStream {
    let [r0, r1] = ratio;
    quote! {
        // -----------------------------------------------------------------
        // LazyReduceMulSlice
        // -----------------------------------------------------------------

        #[cfg(not(all(feature = "nightly", feature = "simd")))]
        impl ::primus_modulus::reduce::LazyReduceMulSlice<#ty> for #name {
            #[inline]
            fn lazy_reduce_mul_slice_assign(self, a: &mut [#ty], b: &[#ty]) {
                use ::primus_modulus::reduce::LazyReduceMulAssign;
                debug_assert_eq!(a.len(), b.len());
                a.iter_mut().zip(b).for_each(|(a, &b)| self.lazy_reduce_mul_assign(a, b));
            }
            #[inline]
            fn lazy_reduce_mul_slice_to(self, a: &[#ty], b: &[#ty], output: &mut [#ty]) {
                use ::primus_modulus::reduce::LazyReduceMul;
                debug_assert_eq!(a.len(), b.len());
                debug_assert_eq!(a.len(), output.len());
                a.iter().zip(b).zip(output).for_each(|((&a, &b), o)| *o = self.lazy_reduce_mul(a, b));
            }
            #[inline]
            fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [#ty], scalar: #ty) {
                use ::primus_modulus::reduce::LazyReduceMulAssign;
                a.iter_mut().for_each(|a| self.lazy_reduce_mul_assign(a, scalar));
            }
            #[inline]
            fn lazy_reduce_scalar_mul_slice_to(self, a: &[#ty], scalar: #ty, output: &mut [#ty]) {
                use ::primus_modulus::reduce::LazyReduceMul;
                debug_assert_eq!(a.len(), output.len());
                a.iter().zip(output).for_each(|(&a, o)| *o = self.lazy_reduce_mul(a, scalar));
            }
        }

        #[cfg(all(feature = "nightly", feature = "simd"))]
        impl ::primus_modulus::reduce::LazyReduceMulSlice<#ty> for #name {
            #[inline]
            fn lazy_reduce_mul_slice_assign(self, a: &mut [#ty], b: &[#ty]) {
                ::primus_modulus::barrett_simd_kernel::lazy_reduce_mul_slice_assign::<#ty, {
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
            fn lazy_reduce_mul_slice_to(self, a: &[#ty], b: &[#ty], output: &mut [#ty]) {
                ::primus_modulus::barrett_simd_kernel::lazy_reduce_mul_slice_to::<#ty, {
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
            fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [#ty], scalar: #ty) {
                ::primus_modulus::barrett_simd_kernel::lazy_reduce_scalar_mul_slice_assign::<#ty, {
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
            fn lazy_reduce_scalar_mul_slice_to(
                self, a: &[#ty], scalar: #ty, output: &mut [#ty],
            ) {
                ::primus_modulus::barrett_simd_kernel::lazy_reduce_scalar_mul_slice_to::<#ty, {
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
        // LazyReduceMulAddSlice
        // -----------------------------------------------------------------

        #[cfg(not(all(feature = "nightly", feature = "simd")))]
        impl ::primus_modulus::reduce::LazyReduceMulAddSlice<#ty> for #name {
            #[inline]
            fn lazy_reduce_add_mul_slice_assign(self, acc: &mut [#ty], a: &[#ty], b: &[#ty]) {
                use ::primus_modulus::reduce::LazyReduceMulAdd;
                debug_assert_eq!(acc.len(), a.len());
                debug_assert_eq!(acc.len(), b.len());
                acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
                    *acc = self.lazy_reduce_mul_add(a, b, *acc);
                });
            }

            #[inline]
            fn lazy_reduce_sub_mul_slice_assign(self, acc: &mut [#ty], a: &[#ty], b: &[#ty]) {
                use ::primus_modulus::reduce::LazyReduceMul;
                debug_assert_eq!(acc.len(), a.len());
                debug_assert_eq!(acc.len(), b.len());
                let two_m = #modulus << 1u32;
                acc.iter_mut().zip(a).zip(b).for_each(|((acc, &a), &b)| {
                    let prod_lazy = self.lazy_reduce_mul(a, b);
                    let diff = acc.wrapping_sub(prod_lazy);
                    *acc = if *acc < prod_lazy { diff.wrapping_add(two_m) } else { diff };
                });
            }

            #[inline]
            fn lazy_reduce_mul_add_slice_to(
                self,
                a: &[#ty],
                b: &[#ty],
                c: &[#ty],
                output: &mut [#ty],
            ) {
                use ::primus_modulus::reduce::LazyReduceMulAdd;
                debug_assert_eq!(a.len(), b.len());
                debug_assert_eq!(a.len(), c.len());
                debug_assert_eq!(a.len(), output.len());
                a.iter().zip(b).zip(c).zip(output).for_each(|(((&a, &b), &c), o)| {
                    *o = self.lazy_reduce_mul_add(a, b, c);
                });
            }

            #[inline]
            fn lazy_reduce_scalar_mul_add_slice_to(
                self,
                scalar: #ty,
                b: &[#ty],
                c: &[#ty],
                output: &mut [#ty],
            ) {
                use ::primus_modulus::reduce::LazyReduceMulAdd;
                debug_assert_eq!(b.len(), c.len());
                debug_assert_eq!(b.len(), output.len());
                b.iter().zip(c).zip(output).for_each(|((&b, &c), o)| {
                    *o = self.lazy_reduce_mul_add(scalar, b, c);
                });
            }
        }

        #[cfg(all(feature = "nightly", feature = "simd"))]
        impl ::primus_modulus::reduce::LazyReduceMulAddSlice<#ty> for #name {
            #[inline]
            fn lazy_reduce_add_mul_slice_assign(self, acc: &mut [#ty], a: &[#ty], b: &[#ty]) {
                ::primus_modulus::barrett_simd_kernel::lazy_reduce_add_mul_slice_assign::<#ty, {
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
            fn lazy_reduce_sub_mul_slice_assign(self, acc: &mut [#ty], a: &[#ty], b: &[#ty]) {
                ::primus_modulus::barrett_simd_kernel::lazy_reduce_sub_mul_slice_assign::<#ty, {
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
            fn lazy_reduce_mul_add_slice_to(
                self,
                a: &[#ty],
                b: &[#ty],
                c: &[#ty],
                output: &mut [#ty],
            ) {
                ::primus_modulus::barrett_simd_kernel::lazy_reduce_mul_add_slice_to::<#ty, {
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
            fn lazy_reduce_scalar_mul_add_slice_to(
                self,
                scalar: #ty,
                b: &[#ty],
                c: &[#ty],
                output: &mut [#ty],
            ) {
                ::primus_modulus::barrett_simd_kernel::lazy_reduce_scalar_mul_add_slice_to::<#ty, {
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
