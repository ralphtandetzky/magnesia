/// Trait providing a function for complex conjugation.
///
/// This trait can also be implemented for real numbers.
/// In this case the `conj()` function is a no-op.
pub trait Conj {
    /// Computes the complex conjugate of the input.
    fn conj(self) -> Self;
}

#[rustfmt::skip] impl Conj for i8   { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for i16  { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for i32  { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for i64  { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for i128 { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for u8   { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for u16  { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for u32  { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for u64  { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for u128 { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for f32  { fn conj(self) -> Self { self } }
#[rustfmt::skip] impl Conj for f64  { fn conj(self) -> Self { self } }
