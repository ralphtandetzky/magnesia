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

/// This trait allows zero overhead implementation of the `-=` operator by
/// taking operands by reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait SubAssignWithRef {
    /// Subtracts `*rhs` from `self` in-place.
    fn sub_assign_with_ref(&mut self, rhs: &Self);
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

/// This trait allows zero overhead implementation of the unary `-` operator
/// by taking its operand by mutable reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait NegAssign {
    /// Applies the unary `-` operator in-place.
    fn neg_assign(&mut self);
}

trait AutoImplementRingOpsWithRef:
    Copy + AddAssign<Self> + SubAssign<Self> + Mul<Self, Output = Self> + Neg<Output = Self>
{
}

impl<T: AutoImplementRingOpsWithRef> AddAssignWithRef for T {
    fn add_assign_with_ref(&mut self, rhs: &Self) {
        *self += *rhs;
    }
}

impl<T: AutoImplementRingOpsWithRef> SubAssignWithRef for T {
    fn sub_assign_with_ref(&mut self, rhs: &Self) {
        *self -= *rhs;
    }
}

impl<T: AutoImplementRingOpsWithRef> MulWithRef for T {
    fn mul_with_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}

impl<T: AutoImplementRingOpsWithRef> NegAssign for T {
    fn neg_assign(&mut self) {
        *self = -*self;
    }
}

impl AutoImplementRingOpsWithRef for i8 {}
impl AutoImplementRingOpsWithRef for i16 {}
impl AutoImplementRingOpsWithRef for i32 {}
impl AutoImplementRingOpsWithRef for i64 {}
impl AutoImplementRingOpsWithRef for i128 {}
impl AutoImplementRingOpsWithRef for isize {}
impl AutoImplementRingOpsWithRef for f32 {}
impl AutoImplementRingOpsWithRef for f64 {}
