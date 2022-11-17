use std::ops::{Div, Mul};

/// This trait tells that the implementing type `T` provides a multiplication
/// of `&T` by `&T`.
///
/// Taking arguments by reference (not by value) can allow more efficient
/// client code compared to the trait `Mul<T, Output=T>`.
/// This is because the client code never needs to make an extra copy
/// of the arguments.
pub trait MulRefs {
    /// Multiplies `self` by `rhs`.
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

#[test]
fn test_mul_refs_for_u16() {
    let a = 19u16;
    let b = 13u16;
    let c = a.mul_refs(&b);
    assert_eq!(c, a * b)
}

/// This trait tells that the implementing type `T` provides a division
/// operator taking `&T` and `&T` as arguments.
///
/// Taking arguments by reference (not by value) can allow more efficient
/// client code compared to the trait `Div<T, Output=T>`.
/// This is because the client code never needs to make an extra copy
/// of the arguments.
pub trait DivRefs {
    /// Divides `self` by `rhs`.
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

#[test]
fn test_div_refs_for_f64() {
    let a = 19.12f64;
    let b = -5.234f64;
    let c = a.div_refs(&b);
    assert_eq!(c, a / b)
}
