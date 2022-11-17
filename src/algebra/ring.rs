use std::{
    cmp::PartialEq,
    ops::{AddAssign, Neg, SubAssign},
};

use super::ops_with_ref::MulRefs;

/// Trait providing a zero element of an additive algebraic structure
/// (like a group, a ring or a vector space).
pub trait Zero {
    /// Returns the zero element of an additive algebraic structure.
    fn zero() -> Self;
}

impl<T: From<i8>> Zero for T {
    fn zero() -> Self {
        Self::from(0)
    }
}

#[test]
fn test_zero_for_i64() {
    let z = i64::zero();
    assert_eq!(z, 0);
}

/// Trait providing a one element of a multiplicative algebraic structure
/// (like a group or a ring or a field).
pub trait One {
    /// Returns the one element of a multiplicative algebraic structure.
    fn one() -> Self;
}

impl<T: From<i8>> One for T {
    fn one() -> Self {
        Self::from(1)
    }
}

#[test]
fn test_one_for_f32() {
    let z = f32::one();
    assert_eq!(z, 1.0);
}

/// Trait for data structures which model an
/// [algebraic ring](https://en.wikipedia.org/wiki/Ring_(mathematics)).
///
/// # Implementing and Using the `Ring` Trait
///
/// In short, a ring in mathematics is an algebraic structure that is
///
///   * an abelian group under addition,
///   * a monoid under multiplication,
///   * with multiplication being distributive with respect to addition.
///
/// Please read [this Wikipedia page on
/// rings](https://en.wikipedia.org/wiki/Ring_(mathematics)) for more details.
/// Hence, when implementing the ring trait, the respective mathematical laws
/// (like the commutative law, etc.) should be guaranteed, so the code which
/// uses the `Ring` trait will work as expected.
///
/// Seeing this, one might expect the traits [`Add`](std::ops::Add),
/// [`Mul`](std::ops::Mul), etc. to be required for a `Ring`.
/// However, these may prohibit to implement data structures with zero overhead
/// as will be explained in *Design Rational of the Ring Trait* below.
/// Therefore, traits taking references like `AddAssign<&Self>` are used
/// instead.
///
/// The `Ring` trait expects a mathematically correct implementation of an
/// algebraic ring.
/// This being said, computations need not always to be exact.
/// For example, IEEE-754 floating point types like `f32` or `f64` only have
/// limited precision and suffer from rounding.
/// Therefore mathematical laws are compromised, but that's okay.
/// For example, given three floating point values `x`, `y` and `z`, the
/// expressions `(x + y) + z` and `x + (y + z)` may not result in exactly the
/// same floating point number due to rounding, even if all values are finite.
/// Additionally, floating point numbers allow infinities and `NaN` values.
/// This uncertainty should be dealt with properly by code which uses the
/// `Ring` trait.
///
/// # Design Rationale of the Ring Trait
///
/// The traits [`Zero`] and [`One`] provide the neutral elements of addition
/// and multiplication to the ring.
/// The traits `AddAssign<&Self>`, `SubAssign<&Self>` and [`MulRefs`] may
/// appear a bit strange at first.
/// However, they are important to implement some structures with zero
/// overhead as we'll explain now.
///
/// If operands need to be passed into an operator function by value, then
/// clients may be forced to clone borrowed instances unnecessarily.
/// But it should be possible to implement abstract mathematical structures
/// efficiently even for types which are expensive to copy.
/// Therefore, operators should take their arguments by reference.
///
/// You may notice that we provide in-place operators for addition and
/// subtraction ([`AddAssign<&Self>`] and [`SubAssign<&Self>`]), but the
/// multiplication operator [`MulRefs`] is not in-place, but returns `Self`.
/// This distinction is made on purpose, because for most data structures
/// addition and subtraction can be performed in-place very efficiently,
/// but multiplication cannot be performed in-place efficiently.
/// (Please convince yourself by looking considering matrices or polynomials
/// as examples.)
///
/// See also: [`Field`](super::Field), [`Polynomial`](super::Polynomial),
/// [`SMatrix`](crate::linalg::SMatrix)
pub trait Ring
where
    for<'a> Self: Clone
        + Zero
        + One
        + AddAssign<&'a Self>
        + SubAssign<&'a Self>
        + MulRefs
        + Neg<Output = Self>
        + PartialEq,
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
