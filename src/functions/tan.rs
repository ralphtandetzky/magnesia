/// Provides the tangent function $x \mapsto \tan x$.
pub trait Tan {
    /// Applies the tangent function to the argument and returns the
    /// result.
    fn tan(self) -> Self;
}

impl Tan for f32 {
    fn tan(self) -> Self {
        f32::tan(self)
    }
}

#[test]
fn test_tan_f32() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let x = rng.gen_range(-10.0..10.0);
        assert_eq!(f32::tan(x), Tan::tan(x));
    }
}

impl Tan for f64 {
    fn tan(self) -> Self {
        f64::tan(self)
    }
}

#[test]
fn test_tan_f64() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let x = rng.gen_range(-10.0..10.0);
        assert_eq!(f64::tan(x), Tan::tan(x));
    }
}
