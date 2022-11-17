/// Provides the cosine function $x \mapsto \cos x$.
pub trait Cos {
    /// Applies the cosine function to the argument and returns the
    /// result.
    fn cos(self) -> Self;
}

impl Cos for f32 {
    fn cos(self) -> Self {
        f32::cos(self)
    }
}

#[test]
fn test_cos_f32() {
    assert_eq!(<f32 as Cos>::cos(0f32), 1f32);
    assert!(
        (<f32 as Cos>::cos(std::f32::consts::PI) + 1.0).abs()
            <= f32::EPSILON * std::f32::consts::PI
    );
    assert!(
        (<f32 as Cos>::cos(-std::f32::consts::PI) + 1.0).abs()
            <= f32::EPSILON * std::f32::consts::PI
    );
    assert!(
        (<f32 as Cos>::cos(2f32 * std::f32::consts::PI) - 1.0).abs()
            <= 2f32 * f32::EPSILON * std::f32::consts::PI
    );
    assert!(<f32 as Cos>::cos(0.5f32 * std::f32::consts::PI).abs() <= f32::EPSILON);
}

impl Cos for f64 {
    fn cos(self) -> Self {
        f64::cos(self)
    }
}

#[test]
fn test_cos_f64() {
    assert_eq!(<f64 as Cos>::cos(0f64), 1f64);
    assert!(
        (<f64 as Cos>::cos(std::f64::consts::PI) + 1.0).abs()
            <= f64::EPSILON * std::f64::consts::PI
    );
    assert!(
        (<f64 as Cos>::cos(-std::f64::consts::PI) + 1.0).abs()
            <= f64::EPSILON * std::f64::consts::PI
    );
    assert!(
        (<f64 as Cos>::cos(2f64 * std::f64::consts::PI) - 1.0).abs()
            <= 2f64 * f64::EPSILON * std::f64::consts::PI
    );
    assert!(<f64 as Cos>::cos(0.5f64 * std::f64::consts::PI).abs() <= f64::EPSILON);
}
