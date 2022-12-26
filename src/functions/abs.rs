/// Provides the absolute value function $x \mapsto \lvert x\rvert$.
pub trait Abs {
    /// The resulting type when taking the absolute value.
    type Output;

    /// Computes the absolute value.
    fn abs(self) -> Self::Output;
}

impl Abs for f32 {
    type Output = Self;

    fn abs(self) -> Self {
        f32::abs(self)
    }
}

#[test]
fn test_abs_f32() {
    assert_eq!(<f32 as Abs>::abs(0f32), 0f32);
    assert_eq!(<f32 as Abs>::abs(1f32), 1f32);
    assert_eq!(<f32 as Abs>::abs(5f32), 5f32);
    assert_eq!(<f32 as Abs>::abs(f32::INFINITY), f32::INFINITY);
    assert_eq!(<f32 as Abs>::abs(-2f32), 2f32);
    assert_eq!(<f32 as Abs>::abs(-f32::INFINITY), f32::INFINITY);
}

impl Abs for &f32 {
    type Output = f32;

    fn abs(self) -> f32 {
        f32::abs(*self)
    }
}

#[test]
fn test_abs_ref_f32() {
    assert_eq!(<&f32 as Abs>::abs(&0f32), 0f32);
    assert_eq!(<&f32 as Abs>::abs(&1f32), 1f32);
    assert_eq!(<&f32 as Abs>::abs(&5f32), 5f32);
    assert_eq!(<&f32 as Abs>::abs(&f32::INFINITY), f32::INFINITY);
    assert_eq!(<&f32 as Abs>::abs(&-2f32), 2f32);
    assert_eq!(<&f32 as Abs>::abs(&-f32::INFINITY), f32::INFINITY);
}

impl Abs for f64 {
    type Output = Self;

    fn abs(self) -> Self {
        f64::abs(self)
    }
}

#[test]
fn test_abs_f64() {
    assert_eq!(<f64 as Abs>::abs(0f64), 0f64);
    assert_eq!(<f64 as Abs>::abs(1f64), 1f64);
    assert_eq!(<f64 as Abs>::abs(5f64), 5f64);
    assert_eq!(<f64 as Abs>::abs(f64::INFINITY), f64::INFINITY);
    assert_eq!(<f64 as Abs>::abs(-2f64), 2f64);
    assert_eq!(<f64 as Abs>::abs(-f64::INFINITY), f64::INFINITY);
}

impl Abs for &f64 {
    type Output = f64;

    fn abs(self) -> f64 {
        f64::abs(*self)
    }
}

#[test]
fn test_abs_ref_f64() {
    assert_eq!(<&f64 as Abs>::abs(&0f64), 0f64);
    assert_eq!(<&f64 as Abs>::abs(&1f64), 1f64);
    assert_eq!(<&f64 as Abs>::abs(&5f64), 5f64);
    assert_eq!(<&f64 as Abs>::abs(&f64::INFINITY), f64::INFINITY);
    assert_eq!(<&f64 as Abs>::abs(&-2f64), 2f64);
    assert_eq!(<&f64 as Abs>::abs(&-f64::INFINITY), f64::INFINITY);
}
