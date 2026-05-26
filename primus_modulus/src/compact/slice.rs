use primus_reduce::prelude::*;

#[cfg(feature = "simd")]
use primus_integer::SimdUnsignedInteger;

use crate::CompactModulus;

macro_rules! compact_scalar {
    ($($t:ty),*) => {
        $(
            impl_reduce_once_slice_scalar!(impl ReduceOnceSlice<$t> for CompactModulus<$t>);
            impl_reduce_neg_slice_scalar!(impl ReduceNegSlice<$t> for CompactModulus<$t>);
            impl_reduce_add_slice_scalar!(impl ReduceAddSlice<$t> for CompactModulus<$t>);
            impl_reduce_sub_slice_scalar!(impl ReduceSubSlice<$t> for CompactModulus<$t>);
        )*
    };
}

#[cfg(feature = "simd")]
macro_rules! compact_simd {
    ($($t:ty),*) => {
        $(
            impl_reduce_once_slice_simd_with!(impl ReduceOnceSlice<$t> for CompactModulus<$t>; <$t>::LANE_COUNT, crate::common::compact::simd, access = 0);
            impl_reduce_neg_slice_simd_with!(impl ReduceNegSlice<$t> for CompactModulus<$t>; <$t>::LANE_COUNT, crate::common::compact::simd, access = 0);
            impl_reduce_add_slice_simd_with!(impl ReduceAddSlice<$t> for CompactModulus<$t>; <$t>::LANE_COUNT, crate::common::compact::simd, access = 0);
            impl_reduce_sub_slice_simd_with!(impl ReduceSubSlice<$t> for CompactModulus<$t>; <$t>::LANE_COUNT, crate::common::compact::simd, access = 0);
        )*
    };
}

compact_scalar!(u128);

#[cfg(not(feature = "simd"))]
compact_scalar!(u8, u16, u32, u64, usize);

#[cfg(feature = "simd")]
compact_simd!(u8, u16, u32, u64, usize);
