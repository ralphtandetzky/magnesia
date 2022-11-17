use std::ops::{Div, Mul};

/// This trait allows zero overhead implementation of the `*` operator by
/// taking operands by reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait MulRefs {
    /// Multiplies `*self` by `*rhs`.
    fn mul_refs(&self, rhs: &Self) -> Self;
}

impl<T> MulRefs for T
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    fn mul_refs(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

/// This trait allows zero overhead implementation of the `/` operator by
/// taking operands by reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait DivRefs {
    /// Divides `*self` by `*rhs`.
    fn div_refs(&self, rhs: &Self) -> Self;
}

impl<T> DivRefs for T
where
    for<'a> &'a T: Div<&'a T, Output = T>,
{
    fn div_refs(&self, rhs: &Self) -> Self {
        self / rhs
    }
}
