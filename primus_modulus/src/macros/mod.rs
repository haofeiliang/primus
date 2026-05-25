macro_rules! impl_reduce_once_slice_scalar {
    (impl ReduceOnceSlice<$t:ty> for $ModType:ty) => {
        impl ReduceOnceSlice<$t> for $ModType {
            #[inline]
            fn reduce_once_slice_assign(self, values: &mut [$t]) {
                values.iter_mut().for_each(|v| self.reduce_once_assign(v));
            }
            #[inline]
            fn reduce_once_slice_to(self, input: &[$t], output: &mut [$t]) {
                debug_assert_eq!(input.len(), output.len());
                output
                    .iter_mut()
                    .zip(input)
                    .for_each(|(x, &y)| *x = self.reduce_once(y));
            }
        }
    };
}

macro_rules! impl_reduce_neg_slice_scalar {
    (impl ReduceNegSlice<$t:ty> for $ModType:ty) => {
        impl ReduceNegSlice<$t> for $ModType {
            #[inline]
            fn reduce_neg_slice_assign(self, values: &mut [$t]) {
                values.iter_mut().for_each(|v| self.reduce_neg_assign(v));
            }
            #[inline]
            fn reduce_neg_slice_to(self, input: &[$t], output: &mut [$t]) {
                debug_assert_eq!(input.len(), output.len());
                output
                    .iter_mut()
                    .zip(input)
                    .for_each(|(x, &y)| *x = self.reduce_neg(y));
            }
        }
    };
}

macro_rules! impl_reduce_add_slice_scalar {
    (impl ReduceAddSlice<$t:ty> for $ModType:ty) => {
        impl ReduceAddSlice<$t> for $ModType {
            #[inline]
            fn reduce_add_slice_assign(self, a: &mut [$t], b: &[$t]) {
                debug_assert_eq!(a.len(), b.len());
                a.iter_mut()
                    .zip(b)
                    .for_each(|(x, &y)| self.reduce_add_assign(x, y));
            }
            #[inline]
            fn reduce_add_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                debug_assert_eq!(output.len(), a.len());
                debug_assert_eq!(output.len(), b.len());
                output.iter_mut().zip(a).zip(b).for_each(|((out, &x), &y)| {
                    *out = self.reduce_add(x, y);
                });
            }
        }
    };
}

macro_rules! impl_reduce_sub_slice_scalar {
    (impl ReduceSubSlice<$t:ty> for $ModType:ty) => {
        impl ReduceSubSlice<$t> for $ModType {
            #[inline]
            fn reduce_sub_slice_assign(self, a: &mut [$t], b: &[$t]) {
                debug_assert_eq!(a.len(), b.len());
                a.iter_mut()
                    .zip(b)
                    .for_each(|(x, &y)| self.reduce_sub_assign(x, y));
            }
            #[inline]
            fn reduce_sub_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                debug_assert_eq!(output.len(), a.len());
                debug_assert_eq!(output.len(), b.len());
                output.iter_mut().zip(a).zip(b).for_each(|((out, &x), &y)| {
                    *out = self.reduce_sub(x, y);
                });
            }
            #[inline]
            fn reduce_sub_slice_rev_assign(self, a: &[$t], b: &mut [$t]) {
                debug_assert_eq!(a.len(), b.len());
                a.iter()
                    .zip(b.iter_mut())
                    .for_each(|(&x, y)| *y = self.reduce_sub(x, *y));
            }
        }
    };
}

#[cfg(feature = "simd")]
macro_rules! impl_reduce_once_slice_simd_with {
    (impl ReduceOnceSlice<$t:ty> for $ModType:ty; $lanes:expr, $fn_mod:path, access = $($access:tt)+) => {
        impl ReduceOnceSlice<$t> for $ModType {
            #[inline]
            fn reduce_once_slice_assign(self, values: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_once_slice_assign::<$t, { $lanes }>(
                    self.$($access)+ , values,
                )
            }
            #[inline]
            fn reduce_once_slice_to(self, input: &[$t], output: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_once_slice_to::<$t, { $lanes }>(
                    self.$($access)+ , input, output,
                )
            }
        }
    };

    (impl ReduceOnceSlice<$t:ty> for $ModType:ty; $lanes:expr, $fn_mod:path, method = $method:ident) => {
        impl ReduceOnceSlice<$t> for $ModType {
            #[inline]
            fn reduce_once_slice_assign(self, values: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_once_slice_assign::<$t, { $lanes }>(
                    self.$method() , values,
                )
            }
            #[inline]
            fn reduce_once_slice_to(self, input: &[$t], output: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_once_slice_to::<$t, { $lanes }>(
                    self.$method() , input, output,
                )
            }
        }
    };
}

#[cfg(feature = "simd")]
macro_rules! impl_reduce_neg_slice_simd_with {
    (impl ReduceNegSlice<$t:ty> for $ModType:ty; $lanes:expr, $fn_mod:path, access = $($access:tt)+) => {
        impl ReduceNegSlice<$t> for $ModType {
            #[inline]
            fn reduce_neg_slice_assign(self, values: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_neg_slice_assign::<$t, { $lanes }>(
                   self.$($access)+ , values,
                )
            }
            #[inline]
            fn reduce_neg_slice_to(self, input: &[$t], output: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_neg_slice_to::<$t, { $lanes }>(
                    self.$($access)+ , input, output,
                )
            }
        }
    };
    (impl ReduceNegSlice<$t:ty> for $ModType:ty; $lanes:expr, $fn_mod:path, method = $method:ident) => {
        impl ReduceNegSlice<$t> for $ModType {
            #[inline]
            fn reduce_neg_slice_assign(self, values: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_neg_slice_assign::<$t, { $lanes }>(
                   self.$method() , values,
                )
            }
            #[inline]
            fn reduce_neg_slice_to(self, input: &[$t], output: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_neg_slice_to::<$t, { $lanes }>(
                    self.$method() , input, output,
                )
            }
        }
    };
}

#[cfg(feature = "simd")]
macro_rules! impl_reduce_add_slice_simd_with {
    (impl ReduceAddSlice<$t:ty> for $ModType:ty; $lanes:expr, $fn_mod:path, access = $($access:tt)+) => {
        impl ReduceAddSlice<$t> for $ModType {
            #[inline]
            fn reduce_add_slice_assign(self, a: &mut [$t], b: &[$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_add_slice_assign::<$t, { $lanes }>(
                    self.$($access)+ , a, b,
                )
            }
            #[inline]
            fn reduce_add_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_add_slice_to::<$t, { $lanes }>(
                    self.$($access)+ , a, b, output,
                )
            }
        }
    };
    (impl ReduceAddSlice<$t:ty> for $ModType:ty; $lanes:expr, $fn_mod:path, method = $method:ident) => {
        impl ReduceAddSlice<$t> for $ModType {
            #[inline]
            fn reduce_add_slice_assign(self, a: &mut [$t], b: &[$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_add_slice_assign::<$t, { $lanes }>(
                    self.$method() , a, b,
                )
            }
            #[inline]
            fn reduce_add_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_add_slice_to::<$t, { $lanes }>(
                    self.$method() , a, b, output,
                )
            }
        }
    };
}

#[cfg(feature = "simd")]
macro_rules! impl_reduce_sub_slice_simd_with {
    (impl ReduceSubSlice<$t:ty> for $ModType:ty; $lanes:expr, $fn_mod:path, access = $($access:tt)+) => {
        impl ReduceSubSlice<$t> for $ModType {
            #[inline]
            fn reduce_sub_slice_assign(self, a: &mut [$t], b: &[$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_sub_slice_assign::<$t, { $lanes }>(
                    self.$($access)+ , a, b,
                )
            }
            #[inline]
            fn reduce_sub_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_sub_slice_to::<$t, { $lanes }>(
                    self.$($access)+ , a, b, output,
                )
            }
            #[inline]
            fn reduce_sub_slice_rev_assign(self, a: &[$t], b: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_sub_slice_rev_assign::<$t, { $lanes }>(
                    self.$($access)+ , a, b,
                )
            }
        }
    };
    (impl ReduceSubSlice<$t:ty> for $ModType:ty; $lanes:expr, $fn_mod:path, method = $method:ident) => {
        impl ReduceSubSlice<$t> for $ModType {
            #[inline]
            fn reduce_sub_slice_assign(self, a: &mut [$t], b: &[$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_sub_slice_assign::<$t, { $lanes }>(
                    self.$method() , a, b,
                )
            }
            #[inline]
            fn reduce_sub_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_sub_slice_to::<$t, { $lanes }>(
                    self.$method() , a, b, output,
                )
            }
            #[inline]
            fn reduce_sub_slice_rev_assign(self, a: &[$t], b: &mut [$t]) {
                use $fn_mod as __fn_mod;

                __fn_mod::reduce_sub_slice_rev_assign::<$t, { $lanes }>(
                    self.$method() , a, b,
                )
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Pure-delegate `LazyReduce*Slice` impls.
//
// For modulus types whose lazy multiplication is identical to the canonical
// reduction (e.g. `NativeModulus`, `PowOf2Modulus` — wrapping or mask
// arithmetic where "lazy" gives no headroom), the lazy slice traits are 1:1
// forwards to the non-lazy ones. This macro emits those forwards.
//
// Each method is `#[inline]` so monomorphisation collapses the call.
// ---------------------------------------------------------------------------

macro_rules! impl_lazy_mul_slice_delegates {
    (impl LazyReduceMulSlice<$t:ty> for $ModType:ty) => {
        impl LazyReduceMulSlice<$t> for $ModType {
            #[inline]
            fn lazy_reduce_mul_slice_assign(self, a: &mut [$t], b: &[$t]) {
                self.reduce_mul_slice_assign(a, b);
            }
            #[inline]
            fn lazy_reduce_mul_slice_to(self, a: &[$t], b: &[$t], output: &mut [$t]) {
                self.reduce_mul_slice_to(a, b, output);
            }
            #[inline]
            fn lazy_reduce_scalar_mul_slice_assign(self, a: &mut [$t], scalar: $t) {
                self.reduce_scalar_mul_slice_assign(a, scalar);
            }
            #[inline]
            fn lazy_reduce_scalar_mul_slice_to(self, a: &[$t], scalar: $t, output: &mut [$t]) {
                self.reduce_scalar_mul_slice_to(a, scalar, output);
            }
        }
    };
    (impl LazyReduceMulAddSlice<$t:ty> for $ModType:ty) => {
        impl LazyReduceMulAddSlice<$t> for $ModType {
            #[inline]
            fn lazy_reduce_add_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) {
                self.reduce_add_mul_slice_assign(acc, a, b);
            }
            #[inline]
            fn lazy_reduce_sub_mul_slice_assign(self, acc: &mut [$t], a: &[$t], b: &[$t]) {
                self.reduce_sub_mul_slice_assign(acc, a, b);
            }
            #[inline]
            fn lazy_reduce_mul_add_slice_to(self, a: &[$t], b: &[$t], c: &[$t], output: &mut [$t]) {
                self.reduce_mul_add_slice_to(a, b, c, output);
            }
            #[inline]
            fn lazy_reduce_scalar_mul_add_slice_to(
                self,
                scalar: $t,
                b: &[$t],
                c: &[$t],
                output: &mut [$t],
            ) {
                self.reduce_scalar_mul_add_slice_to(scalar, b, c, output);
            }
        }
    };
}
