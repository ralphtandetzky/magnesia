use super::ops_with_ref::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Zero {
    fn zero() -> Self;
}

impl<T: From<i8>> Zero for T {
    fn zero() -> Self {
        Self::from(0)
    }
}

pub trait One {
    fn one() -> Self;
}

impl<T: From<i8>> One for T {
    fn one() -> Self {
        Self::from(1)
    }
}

pub trait Ring:
    Clone
    + Zero
    + One
    + Add
    + AddAssign
    + AddAssignWithRef
    + Sub
    + SubAssign
    + SubAssignWithRef
    + Mul
    + MulAssign
    + MulWithRef
    + Neg
    + NegAssign
{
}

impl Ring for i8 {}
impl Ring for i16 {}
impl Ring for i32 {}
impl Ring for i64 {}
impl Ring for i128 {}
impl Ring for isize {}
impl Ring for f32 {}
impl Ring for f64 {}
