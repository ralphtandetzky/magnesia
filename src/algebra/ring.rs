use super::ops_with_ref::*;
use std::cmp::PartialEq;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Trait providing a zero element of an additive algebraic structure (like a group, a ring or a vector space).
pub trait Zero {
    /// Returns the zero element of an additive algebraic structure.
    fn zero() -> Self;
}

impl<T: From<i8>> Zero for T {
    fn zero() -> Self {
        Self::from(0)
    }
}

/// Trait providing a one element of a multiplicative algebraic structure (like a group or a ring or a field).
pub trait One {
    /// Returns the one element of a multiplicative algebraic structure.
    fn one() -> Self;
}

impl<T: From<i8>> One for T {
    fn one() -> Self {
        Self::from(1)
    }
}

/// Trait for data structures which model an
/// [algebraic ring](https://en.wikipedia.org/wiki/Ring_(mathematics)).
///
/// In short, a ring in mathematics is an algebraic structure that is
/// * an abelian group under addition,
/// * a monoid under multiplication,
/// * with multiplication being distributive with respect to addition.
/// Please see https://en.wikipedia.org/wiki/Ring_(mathematics) for more
/// details.
/// Hence, when implementing the ring trait, the respective mathematical laws
/// (like the commutative law, etc.) should be guaranteed, so the code which
/// uses the `Ring` trait will work as expected.
/// This being said does not mean that computations always need to be exact.
/// For example, IEEE 754 floating point types like `f32` or `f64` only have
/// limited precision and suffer from rounding.
/// Therefore mathematical laws are compromised.
/// For example, given three floating point values `x`, `y` and `z`, the
/// expressions `(x + y) + z` and `x + (y + z)` may not result in exactly the
/// same floating point number due to rounding, even if all values are finite.
/// Additionally, floating point numbers allow infinities and `NaN` values.
/// This uncertainty should be dealt with properly by code which uses the
/// `Ring` trait.
///
/// The `Ring` trait requires a number of standard traits to be implemented
/// which allow easy handling and operator overloading.
/// In addition, there are some non-standard traits.
/// The traits `Zero` and `One` provide the neutral elements of addition and
/// multiplication to the ring.
/// The traits `AddAssignWithRef`, `SubAssignWithRef` and `MulWithRef` may
/// appear a bit strange at first.
/// However, they are important to implement some structures with zero
/// overhead.
/// For example, one may implement a ring of arbitrarily large integers which
/// may be expensive to copy.
/// The normal addition and multiplication operators would always *move* both
/// operands into the called operator function, even though a reference on
/// both sides would be completely sufficient.
/// This would force the user of the large integer data structure to clone
/// borrowed instances unnecessarily.
/// This becomes even more problematic, if you want to deal with polynomials
/// of such integers.
/// These polynomials would always have to clone and move coefficients and
/// values which can cause a lot of overhead.
/// To allow efficient implementations for types that may be expensive to
/// clone these traits are required.
/// Fortunately, for copyable types for which the standard operator traits are
/// available, the traits will be implemented automatically.
///
/// To fully support rings with floating point number structures, we only
/// require `PartialEq`, not `Eq` to be implemented.
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
    + PartialEq
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
