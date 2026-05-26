macro_rules! impl_iters {
    ($poly:ident, $short_name:ident) => {
        paste::paste! {
            pub struct [<$poly Iter>]<'a, T>
            where
                T: FheUint,
            {
                pub iter: core::slice::ChunksExact<'a, T>
            }

            impl<'a, T: FheUint> [<$poly Iter>]<'a, T> {
                #[inline]
                pub fn new(data:&'a [T], [<$short_name _len>]:usize) -> Self{
                    Self {
                        iter: data.chunks_exact([<$short_name _len>])
                    }
                }
            }

            impl<'a, T: FheUint> Iterator for [<$poly Iter>]<'a, T> {
                type Item = $poly<&'a [T]>;

                #[inline]
                fn next(&mut self) -> Option<Self::Item> {
                    self.iter.next().map(|slice| $poly(slice))
                }
            }

            impl<'a, T: FheUint> core::iter::FusedIterator for [<$poly Iter>]<'a, T> {}
        }

        paste::paste! {
            pub struct [<$poly IterMut>]<'a, T>
            where
                T: FheUint,
            {
                pub iter: core::slice::ChunksExactMut<'a, T>
            }

            impl<'a, T: FheUint> [<$poly IterMut>]<'a, T> {
                #[inline]
                pub fn new(data:&'a mut [T], [<$short_name _len>]:usize) -> Self{
                    Self {
                        iter: data.chunks_exact_mut([<$short_name _len>])
                    }
                }
            }

            impl<'a, T: FheUint> Iterator for [<$poly IterMut>]<'a, T> {
                type Item = $poly<&'a mut [T]>;

                #[inline]
                fn next(&mut self) -> Option<Self::Item> {
                    self.iter.next().map(|slice| $poly(slice))
                }
            }

            impl<'a, T: FheUint> core::iter::FusedIterator for [<$poly IterMut>]<'a, T> {}
        }
    };
}
