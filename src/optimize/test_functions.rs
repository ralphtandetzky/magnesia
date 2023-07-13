use std::f32::consts::{E, PI};

/// The [Ackley function](https://en.wikipedia.org/wiki/Ackley_function),
/// a test function for non-convex optimization.
///
/// The domain that should be used for testing is `-5 <= x, y <= 5`.
pub fn ackley(x: f32, y: f32) -> f32 {
    -20.0 * (-0.2 * (0.5 * (x * x + y * y)).sqrt()).exp_m1()
        - ((0.5 * ((2.0 * PI * x).cos() + (2.0 * PI * y).cos())).exp() - E)
}
