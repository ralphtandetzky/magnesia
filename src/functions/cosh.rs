/// Provides the hyperbolic cosine function $x \mapsto \cosh x$.
pub trait Cosh {
    /// Applies the hyperbolic cosine function to the argument and returns the
    /// result.
    fn cosh(self) -> Self;
}

impl Cosh for f32 {
    fn cosh(self) -> Self {
        f32::cosh(self)
    }
}

#[test]
fn test_cosh_f32() {
    assert_eq!(<f32 as Cosh>::cosh(0f32), 1f32);
    assert!(
        (<f32 as Cosh>::cosh(1f32).abs() - (1f32.exp() + (-1f32).exp()) / 2f32) <= f32::EPSILON
    );
}

impl Cosh for f64 {
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
}

#[test]
fn test_cosh_f64() {
    assert_eq!(<f64 as Cosh>::cosh(0f64), 1f64);
    assert!(
        (<f64 as Cosh>::cosh(1f64).abs() - (1f64.exp() + (-1f64).exp()) / 2f64) <= f64::EPSILON
    );
}
