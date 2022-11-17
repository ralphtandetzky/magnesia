use std::ops::{AddAssign, Div, Mul, Neg, SubAssign};

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

/// This trait allows zero overhead implementation of the `*` operator by
/// taking operands by reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait MulWithRef {
    /// Multiplies `*self` by `*rhs`.
    fn mul_with_ref(&self, rhs: &Self) -> Self;
}

/// This trait allows zero overhead implementation of the `/` operator by
/// taking operands by reference.
///
/// See the documentation of [`Ring`](crate::algebra::Ring) for a detailed
/// explanation on the design rationale.
pub trait DivWithRef {
    /// Divides `*self` by `*rhs`.
    fn div_with_ref(&self, rhs: &Self) -> Self;
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

/// Marker trait which causes the reference operator traits such as
/// [`AddAssignWithRef`] to be implemented automatically.
///
/// Please mark your types for which you would like the reference operator
/// traits to be implemented automatically with this trait.
/// For some types this is undesired, so the automatic implementation is
/// predicated on this trait.
pub trait AutoImplementOpsWithRef {}

impl<T> AddAssignWithRef for T
where
    T: AutoImplementOpsWithRef + AddAssign<T> + Copy,
{
    fn add_assign_with_ref(&mut self, rhs: &Self) {
        *self += *rhs;
    }
}

impl<T> SubAssignWithRef for T
where
    T: AutoImplementOpsWithRef + SubAssign<T> + Copy,
{
    fn sub_assign_with_ref(&mut self, rhs: &Self) {
        *self -= *rhs;
    }
}

impl<T> MulWithRef for T
where
    T: AutoImplementOpsWithRef + Mul<T, Output = T> + Copy,
{
    fn mul_with_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}

impl<T> DivWithRef for T
where
    T: AutoImplementOpsWithRef + Div<T, Output = T> + Copy,
{
    fn div_with_ref(&self, rhs: &Self) -> Self {
        *self / *rhs
    }
}

impl<T> NegAssign for T
where
    T: AutoImplementOpsWithRef + Neg<Output = T> + Copy,
{
    fn neg_assign(&mut self) {
        *self = -*self;
    }
}

impl AutoImplementOpsWithRef for i8 {}
impl AutoImplementOpsWithRef for i16 {}
impl AutoImplementOpsWithRef for i32 {}
impl AutoImplementOpsWithRef for i64 {}
impl AutoImplementOpsWithRef for i128 {}
impl AutoImplementOpsWithRef for isize {}
impl AutoImplementOpsWithRef for u8 {}
impl AutoImplementOpsWithRef for u16 {}
impl AutoImplementOpsWithRef for u32 {}
impl AutoImplementOpsWithRef for u64 {}
impl AutoImplementOpsWithRef for u128 {}
impl AutoImplementOpsWithRef for usize {}
impl AutoImplementOpsWithRef for f32 {}
impl AutoImplementOpsWithRef for f64 {}
