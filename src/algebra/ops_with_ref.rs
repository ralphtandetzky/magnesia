pub trait AddAssignWithRef {
    fn add_assign_with_ref(&mut self, rhs: &Self);
}

impl<T: std::ops::AddAssign<T> + Copy> AddAssignWithRef for T
{
    fn add_assign_with_ref(&mut self, rhs: &Self) {
        *self += *rhs;
    }
}

pub trait SubAssignWithRef: Clone + std::ops::Sub<Self, Output = Self> + std::ops::SubAssign<Self> {
    fn sub_assign_with_ref(&mut self, rhs: &Self);
}

impl<T: std::ops::Sub<T, Output = Self> + std::ops::SubAssign<T> + Clone> SubAssignWithRef for T
where
    T: Copy,
{
    fn sub_assign_with_ref(&mut self, rhs: &Self) {
        *self -= *rhs;
    }
}

pub trait MulWithRef
{
    fn mul_with_ref(&self, rhs :&Self) -> Self;
}

impl<T: std::ops::Mul<T, Output = Self> + Copy> MulWithRef for T
{
    fn mul_with_ref(&self, rhs :&Self) -> Self
    {
        *self * *rhs
    }
}

pub trait NegAssign: Clone + std::ops::Neg<Output = Self> {
    fn neg_assign(&mut self);
}

impl<T: std::ops::Neg<Output = T> + Clone> NegAssign for T
where
    T: Copy,
{
    fn neg_assign(&mut self) {
        *self = -*self;
    }
}
