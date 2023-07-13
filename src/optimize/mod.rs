/// This module provides an implementation of the differential evolution
/// algorithm.
mod differential_evolution;

/// This module provides common test functions for optimization algorithms.
///
/// See [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
/// for a large list of test functions.
pub mod test_functions;

pub use differential_evolution::optimize as differential_evolution;
