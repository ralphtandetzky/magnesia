/// Provides the natural logarithm function $x \mapsto \ln x$.
pub trait Ln {
    /// Computes the natural logarithm (base $e$) of the argument and returns the
    /// result.
    fn ln(self) -> Self;
}

impl Ln for f32 {
    fn ln(self) -> Self {
        f32::ln(self)
    }
}

#[test]
fn test_ln_f32() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let x = rng.gen_range(0f32..2f32);
        let y = Ln::ln(x);
        let expected = f32::ln(x);
        assert_eq!(y, expected)
    }
}

impl Ln for f64 {
    fn ln(self) -> Self {
        f64::ln(self)
    }
}

#[test]
fn test_ln_f64() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let x = rng.gen_range(0f64..2f64);
        let y = Ln::ln(x);
        let expected = f64::ln(x);
        assert_eq!(y, expected)
    }
}
