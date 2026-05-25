use primus_reduce::prelude::*;

use super::UintModulus;

macro_rules! uint_scalar {
    ($($t:ty),*) => {
        $(
            impl_reduce_once_slice_scalar!(impl ReduceOnceSlice<$t> for UintModulus<$t>);
            impl_reduce_neg_slice_scalar!(impl ReduceNegSlice<$t> for UintModulus<$t>);
            impl_reduce_add_slice_scalar!(impl ReduceAddSlice<$t> for UintModulus<$t>);
            impl_reduce_sub_slice_scalar!(impl ReduceSubSlice<$t> for UintModulus<$t>);
        )*
    };
}

#[cfg(feature = "simd")]
macro_rules! uint_simd {
    ($($t:ty),*) => {
        $(
            impl_reduce_once_slice_simd_with!(impl ReduceOnceSlice<$t> for UintModulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), crate::common::uint::simd, access = 0);
            impl_reduce_neg_slice_simd_with!(impl ReduceNegSlice<$t> for UintModulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), crate::common::uint::simd, access = 0);
            impl_reduce_add_slice_simd_with!(impl ReduceAddSlice<$t> for UintModulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), crate::common::uint::simd, access = 0);
            impl_reduce_sub_slice_simd_with!(impl ReduceSubSlice<$t> for UintModulus<$t>; primus_integer::lanes::VECTOR_BITS / (<$t>::BITS as usize), crate::common::uint::simd, access = 0);
        )*
    };
}

uint_scalar!(u128);

#[cfg(not(feature = "simd"))]
uint_scalar!(u8, u16, u32, u64, usize);

#[cfg(feature = "simd")]
uint_simd!(u8, u16, u32, u64, usize);
