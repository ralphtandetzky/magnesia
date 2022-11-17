/// Trait providing a function for complex conjugation.
///
/// This trait can also be implemented for real numbers.
/// In this case the `conj()` function is a no-op.
pub trait Conj {
    /// Computes the complex conjugate of the input.
    fn conj(self) -> Self;
}

/// Trait providing a function for in-place complex conjugation.
///
/// This trait can also be implemented for real numbers.
/// In this case the `conj_assign()` function is a no-op.
pub trait ConjAssign {
    /// Computes the complex conjugate of the input in-place.
    fn conj_assign(&mut self);
}

impl<T: ConjAssign> Conj for T {
    fn conj(mut self) -> Self {
        self.conj_assign();
        self
    }
}

#[rustfmt::skip] impl ConjAssign for i8   { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for i16  { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for i32  { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for i64  { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for i128 { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for u8   { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for u16  { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for u32  { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for u64  { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for u128 { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for f32  { fn conj_assign(&mut self) {} }
#[rustfmt::skip] impl ConjAssign for f64  { fn conj_assign(&mut self) {} }
