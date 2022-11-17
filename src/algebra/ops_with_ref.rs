use std::ops::{AddAssign, Mul, Neg, SubAssign};

pub trait AddAssignWithRef {
    fn add_assign_with_ref(&mut self, rhs: &Self);
}

impl<T: AddAssign<T> + Copy> AddAssignWithRef for T {
    fn add_assign_with_ref(&mut self, rhs: &Self) {
        *self += *rhs;
    }
}

pub trait SubAssignWithRef {
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

pub trait MulWithRef {
    fn mul_with_ref(&self, rhs: &Self) -> Self;
}

impl<T: Mul<T, Output = Self> + Copy> MulWithRef for T {
    fn mul_with_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}

pub trait NegAssign {
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
