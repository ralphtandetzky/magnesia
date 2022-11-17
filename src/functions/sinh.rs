/// Provides the hyperbolic sine function $x \mapsto \sinh x$.
pub trait Sinh {
    /// Applies the hyperbolic sine function to the argument and returns the
    /// result.
    fn sinh(self) -> Self;
}

impl Sinh for f32 {
    fn sinh(self) -> Self {
        f32::sinh(self)
    }
}

#[test]
fn test_sinh_f32() {
    assert_eq!(<f32 as Sinh>::sinh(0f32), 0f32);
    assert!(
        (<f32 as Sinh>::sinh(1f32).abs() - (1f32.exp() - (-1f32).exp()) / 2f32) <= f32::EPSILON
    );
}

impl Sinh for f64 {
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
}

#[test]
fn test_sinh_f64() {
    assert_eq!(<f64 as Sinh>::sinh(0f64), 0f64);
    assert!(
        (<f64 as Sinh>::sinh(1f64).abs() - (1f64.exp() - (-1f64).exp()) / 2f64) <= f64::EPSILON
    );
}
