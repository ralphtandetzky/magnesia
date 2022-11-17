/// Provides the exponential function $x \mapsto e^x$.
pub trait Exp {
    /// Applies the exponential function to the argument and returns the
    /// result.
    fn exp(self) -> Self;
}

impl Exp for f32 {
    fn exp(self) -> Self {
        f32::exp(self)
    }
}

#[test]
fn test_exp_f32() {
    assert!((<f32 as Exp>::exp(0f32) - 1f32).abs() <= f32::EPSILON);
    assert!((<f32 as Exp>::exp(1f32) - std::f32::consts::E) <= f32::EPSILON * std::f32::consts::E);
    assert!(
        (<f32 as Exp>::exp(-1f32) - 1f32 / std::f32::consts::E)
            <= f32::EPSILON / std::f32::consts::E
    );
    assert!(
        (<f32 as Exp>::exp(2f32) - std::f32::consts::E * std::f32::consts::E)
            <= f32::EPSILON * std::f32::consts::E * std::f32::consts::E
    );
    assert!(
        (<f32 as Exp>::exp(0.5f32) - std::f32::consts::E.sqrt())
            <= f32::EPSILON * std::f32::consts::E.sqrt()
    );
}

impl Exp for f64 {
    fn exp(self) -> Self {
        f64::exp(self)
    }
}

#[test]
fn test_exp_f64() {
    assert!((<f64 as Exp>::exp(0f64) - 1f64).abs() <= f64::EPSILON);
    assert!((<f64 as Exp>::exp(1f64) - std::f64::consts::E) <= f64::EPSILON * std::f64::consts::E);
    assert!(
        (<f64 as Exp>::exp(-1f64) - 1f64 / std::f64::consts::E)
            <= f64::EPSILON / std::f64::consts::E
    );
    assert!(
        (<f64 as Exp>::exp(2f64) - std::f64::consts::E * std::f64::consts::E)
            <= f64::EPSILON * std::f64::consts::E * std::f64::consts::E
    );
    assert!(
        (<f64 as Exp>::exp(0.5f64) - std::f64::consts::E.sqrt())
            <= f64::EPSILON * std::f64::consts::E.sqrt()
    );
}
