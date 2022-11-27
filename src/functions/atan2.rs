/// Provides a function computing the quadrant arctangent of `self` and `other` in radians.
pub trait Atan2 {
    /// Computes the quadrant arctangent of `self` and `other` in radians.
    fn atan2(self, other: Self) -> Self;
}

impl Atan2 for f32 {
    fn atan2(self, other: Self) -> Self {
        f32::atan2(self, other)
    }
}

#[test]
fn test_atan2_f32() {
    assert!((<f32 as Atan2>::atan2(0f32, 1f32) - 0f32) <= 2f32 * f32::EPSILON);
    assert!(
        (<f32 as Atan2>::atan2(1f32, 3f32.sqrt()) - std::f32::consts::PI / 6f32)
            <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(1f32, 1f32) - std::f32::consts::PI / 4f32) <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(3f32.sqrt(), 1f32) - std::f32::consts::PI / 3f32)
            <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(1f32, 0f32) - std::f32::consts::PI / 2f32) <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(3f32.sqrt(), -1f32) - 2f32 * std::f32::consts::PI / 3f32)
            <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(1f32, 1f32) - 3f32 * std::f32::consts::PI / 4f32)
            <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(1f32, -3f32.sqrt()) - 5f32 * std::f32::consts::PI / 6f32)
            <= 2f32 * f32::EPSILON
    );
    assert!((<f32 as Atan2>::atan2(0f32, -1f32) - std::f32::consts::PI) <= 2f32 * f32::EPSILON);
    assert!(
        (<f32 as Atan2>::atan2(-1f32, -3f32.sqrt()) + 5f32 * std::f32::consts::PI / 6f32)
            <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(-1f32, -1f32) + 3f32 * std::f32::consts::PI / 4f32)
            <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(-3f32.sqrt(), -1f32) + 2f32 * std::f32::consts::PI / 3f32)
            <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(-1f32, 0f32) + std::f32::consts::PI / 2f32) <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(-3f32.sqrt(), 1f32) + std::f32::consts::PI / 3f32)
            <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(-1f32, 1f32) + std::f32::consts::PI / 4f32) <= 2f32 * f32::EPSILON
    );
    assert!(
        (<f32 as Atan2>::atan2(-1f32, 3f32.sqrt()) + std::f32::consts::PI / 6f32)
            <= 2f32 * f32::EPSILON
    );
}

impl Atan2 for f64 {
    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }
}

#[test]
fn test_atan2_f64() {
    assert!((<f64 as Atan2>::atan2(0f64, 1f64) - 0f64) <= 2f64 * f64::EPSILON);
    assert!(
        (<f64 as Atan2>::atan2(1f64, 3f64.sqrt()) - std::f64::consts::PI / 6f64)
            <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(1f64, 1f64) - std::f64::consts::PI / 4f64) <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(3f64.sqrt(), 1f64) - std::f64::consts::PI / 3f64)
            <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(1f64, 0f64) - std::f64::consts::PI / 2f64) <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(3f64.sqrt(), -1f64) - 2f64 * std::f64::consts::PI / 3f64)
            <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(1f64, 1f64) - 3f64 * std::f64::consts::PI / 4f64)
            <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(1f64, -3f64.sqrt()) - 5f64 * std::f64::consts::PI / 6f64)
            <= 2f64 * f64::EPSILON
    );
    assert!((<f64 as Atan2>::atan2(0f64, -1f64) - std::f64::consts::PI) <= 2f64 * f64::EPSILON);
    assert!(
        (<f64 as Atan2>::atan2(-1f64, -3f64.sqrt()) + 5f64 * std::f64::consts::PI / 6f64)
            <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(-1f64, -1f64) + 3f64 * std::f64::consts::PI / 4f64)
            <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(-3f64.sqrt(), -1f64) + 2f64 * std::f64::consts::PI / 3f64)
            <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(-1f64, 0f64) + std::f64::consts::PI / 2f64) <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(-3f64.sqrt(), 1f64) + std::f64::consts::PI / 3f64)
            <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(-1f64, 1f64) + std::f64::consts::PI / 4f64) <= 2f64 * f64::EPSILON
    );
    assert!(
        (<f64 as Atan2>::atan2(-1f64, 3f64.sqrt()) + std::f64::consts::PI / 6f64)
            <= 2f64 * f64::EPSILON
    );
}
