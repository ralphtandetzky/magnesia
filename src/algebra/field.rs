use super::{ops_with_ref::DivWithRef, Ring};

/// A trait for data structures modeling an algebraic
/// [field](https://en.wikipedia.org/wiki/Field_(mathematics)).
///
/// In mathematics a field is a commutative ring with division.
/// See [this Wikipedia page on
/// fields](https://en.wikipedia.org/wiki/Field_(mathematics)) for more
/// details.
///
/// Please refer to the documentation of [`Ring`] for the design rationale
/// of this trait.
/// The comments there also apply here.
///
/// See also: [`Ring`], [`Complex`](super::Complex)
pub trait Field: Ring + DivWithRef {}

impl Field for f32 {}
impl Field for f64 {}
