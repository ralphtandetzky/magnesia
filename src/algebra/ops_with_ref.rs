use std::ops::{AddAssign, Mul, Neg, SubAssign};

/// This trait allows zero overhead implementation of the `+=` operator by
/// taking operands by reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait AddAssignWithRef {
    /// Adds `*rhs` to `self` in-place.
    fn add_assign_with_ref(&mut self, rhs: &Self);
}

impl<T: AddAssign<T> + Copy> AddAssignWithRef for T {
    fn add_assign_with_ref(&mut self, rhs: &Self) {
        *self += *rhs;
    }
}

/// This trait allows zero overhead implementation of the `-=` operator by
/// taking operands by reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait SubAssignWithRef {
    /// Subtracts `*rhs` from `self` in-place.
    fn sub_assign_with_ref(&mut self, rhs: &Self);
}

impl<T: SubAssign<T> + Clone> SubAssignWithRef for T
where
    T: Copy,
{
    fn sub_assign_with_ref(&mut self, rhs: &Self) {
        *self -= *rhs;
    }
}

/// This trait allows zero overhead implementation of the `+=` operator by
/// taking operands by reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait MulWithRef {
    /// Multiplies `*self` and `*rhs`.
    fn mul_with_ref(&self, rhs: &Self) -> Self;
}

impl<T: Mul<T, Output = Self> + Copy> MulWithRef for T {
    fn mul_with_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}

/// This trait allows zero overhead implementation of the unary `-` operator
/// by taking its operand by mutable reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait NegAssign {
    /// Applies the unary `-` operator in-place.
    fn neg_assign(&mut self);
}

impl<T: Neg<Output = T> + Clone> NegAssign for T
where
    T: Copy,
{
    fn neg_assign(&mut self) {
        *self = -*self;
    }
}
