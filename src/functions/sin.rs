/// Provides the sine function $x \mapsto \sin x$.
pub trait Sin {
    /// Applies the sine function to the argument and returns the
    /// result.
    fn sin(self) -> Self;
}

impl Sin for f32 {
    fn sin(self) -> Self {
        f32::sin(self)
    }
}

#[test]
fn test_sin_f32() {
    assert_eq!(<f32 as Sin>::sin(0f32), 0f32);
    assert!(<f32 as Sin>::sin(std::f32::consts::PI).abs() <= f32::EPSILON * std::f32::consts::PI);
    assert!(<f32 as Sin>::sin(-std::f32::consts::PI).abs() <= f32::EPSILON * std::f32::consts::PI);
    assert!(
        <f32 as Sin>::sin(2f32 * std::f32::consts::PI).abs()
            <= 2f32 * f32::EPSILON * std::f32::consts::PI
    );
    assert!((<f32 as Sin>::sin(0.5f32 * std::f32::consts::PI) - 1.0) <= f32::EPSILON);
}

impl Sin for f64 {
    fn sin(self) -> Self {
        f64::sin(self)
    }
}

#[test]
fn test_sin_f64() {
    assert_eq!(<f64 as Sin>::sin(0f64), 0f64);
    assert!(<f64 as Sin>::sin(std::f64::consts::PI).abs() <= f64::EPSILON * std::f64::consts::PI);
    assert!(<f64 as Sin>::sin(-std::f64::consts::PI).abs() <= f64::EPSILON * std::f64::consts::PI);
    assert!(
        <f64 as Sin>::sin(2f64 * std::f64::consts::PI).abs()
            <= 2f64 * f64::EPSILON * std::f64::consts::PI
    );
    assert!((<f64 as Sin>::sin(0.5f64 * std::f64::consts::PI) - 1.0) <= f64::EPSILON);
}
