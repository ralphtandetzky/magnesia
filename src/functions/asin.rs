/// Provides the arcsine function $x \mapsto \arcsin x$.
pub trait Asin {
    /// Applies the arcsine function to the argument and returns the
    /// result.
    fn asin(self) -> Self;
}

impl Asin for f32 {
    fn asin(self) -> Self {
        f32::asin(self)
    }
}

#[test]
fn test_asin_f32() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let x = rng.gen_range(-1.0..1.0);
        assert_eq!(f32::asin(x), Asin::asin(x));
    }
}

impl Asin for f64 {
    fn asin(self) -> Self {
        f64::asin(self)
    }
}

#[test]
fn test_asin_f64() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let x = rng.gen_range(-1.0..1.0);
        assert_eq!(f64::asin(x), Asin::asin(x));
    }
}
