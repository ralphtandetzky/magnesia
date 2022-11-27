/// Provides the square root function $x \mapsto \sqrt{x}$.
pub trait Sqrt {
    /// Returns the square root of a given value.
    ///
    /// In case of negative inputs NaN shall be returned.
    fn sqrt(&self) -> Self;
}

impl Sqrt for f32 {
    fn sqrt(&self) -> Self {
        Self::sqrt(*self)
    }
}

#[test]
fn test_sqrt_f32_on_positive_inputs() {
    assert_eq!(<f32 as Sqrt>::sqrt(&1f32), 1f32);
    assert_eq!(<f32 as Sqrt>::sqrt(&4f32), 2f32);
    assert_eq!(<f32 as Sqrt>::sqrt(&9f32), 3f32);
    assert_eq!(<f32 as Sqrt>::sqrt(&2f32), std::f32::consts::SQRT_2);
}

#[test]
fn test_sqrt_f32_on_zero_inputs() {
    assert_eq!(<f32 as Sqrt>::sqrt(&0f32), 0f32);
    assert_eq!(<f32 as Sqrt>::sqrt(&-0f32), 0f32);
}

#[test]
fn test_sqrt_f32_on_negative_inputs() {
    assert!(<f32 as Sqrt>::sqrt(&-1f32).is_nan());
    assert!(<f32 as Sqrt>::sqrt(&-50f32).is_nan());
}

#[test]
fn test_sqrt_f32_on_infinite_inputs() {
    assert_eq!(<f32 as Sqrt>::sqrt(&std::f32::INFINITY), std::f32::INFINITY);
    assert!(<f32 as Sqrt>::sqrt(&-std::f32::INFINITY).is_nan());
}

#[test]
fn test_sqrt_f32_on_nan_inputs() {
    assert!(<f32 as Sqrt>::sqrt(&std::f32::NAN).is_nan());
    assert!(<f32 as Sqrt>::sqrt(&-std::f32::NAN).is_nan());
}

impl Sqrt for f64 {
    fn sqrt(&self) -> Self {
        Self::sqrt(*self)
    }
}

#[test]
fn test_sqrt_f64_on_positive_inputs() {
    assert_eq!(<f64 as Sqrt>::sqrt(&1f64), 1f64);
    assert_eq!(<f64 as Sqrt>::sqrt(&4f64), 2f64);
    assert_eq!(<f64 as Sqrt>::sqrt(&9f64), 3f64);
    assert_eq!(<f64 as Sqrt>::sqrt(&2f64), std::f64::consts::SQRT_2);
}

#[test]
fn test_sqrt_f64_on_zero_inputs() {
    assert_eq!(<f64 as Sqrt>::sqrt(&0f64), 0f64);
    assert_eq!(<f64 as Sqrt>::sqrt(&-0f64), 0f64);
}

#[test]
fn test_sqrt_f64_on_negative_inputs() {
    assert!(<f64 as Sqrt>::sqrt(&-1f64).is_nan());
    assert!(<f64 as Sqrt>::sqrt(&-50f64).is_nan());
}

#[test]
fn test_sqrt_f64_on_infinite_inputs() {
    assert_eq!(<f64 as Sqrt>::sqrt(&std::f64::INFINITY), std::f64::INFINITY);
    assert!(<f64 as Sqrt>::sqrt(&-std::f64::INFINITY).is_nan());
}

#[test]
fn test_sqrt_f64_on_nan_inputs() {
    assert!(<f64 as Sqrt>::sqrt(&std::f64::NAN).is_nan());
    assert!(<f64 as Sqrt>::sqrt(&-std::f64::NAN).is_nan());
}
