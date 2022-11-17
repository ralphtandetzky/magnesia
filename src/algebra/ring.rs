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
/// # Implementing and Using the `Ring` Trait
///
/// In short, a ring in mathematics is an algebraic structure that is
///
///   * an abelian group under addition,
///   * a monoid under multiplication,
///   * with multiplication being distributive with respect to addition.
///
/// Please read [this Wikipedia page on rings](https://en.wikipedia.org/wiki/Ring_(mathematics))
/// for more details.
/// Hence, when implementing the ring trait, the respective mathematical laws
/// (like the commutative law, etc.) should be guaranteed, so the code which
/// uses the `Ring` trait will work as expected.
/// This being said it does not mean that computations always need to be exact.
/// For example, IEEE-754 floating point types like `f32` or `f64` only have
/// limited precision and suffer from rounding.
/// Therefore mathematical laws are compromised.
/// For example, given three floating point values `x`, `y` and `z`, the
/// expressions `(x + y) + z` and `x + (y + z)` may not result in exactly the
/// same floating point number due to rounding, even if all values are finite.
/// Additionally, floating point numbers allow infinities and `NaN` values.
/// This uncertainty should be dealt with properly by code which uses the
/// `Ring` trait.
///
/// # Design Rationale of the Ring Trait
///
/// The `Ring` trait requires a number of standard traits to be implemented
/// which allow easy handling and operator overloading.
/// In addition, there are some non-standard traits.
/// The traits `Zero` and `One` provide the neutral elements of addition and
/// multiplication to the ring.
/// The traits `AddAssignWithRef`, `SubAssignWithRef` and `MulWithRef` may
/// appear a bit strange at first.
/// However, they are important to implement some structures with zero
/// overhead as we'll explain now.
///
/// Arithmetic operator traits from the standard library usually take at least
/// one of their arguments by value.
/// If the data type for which the operator is implemented is expensive to
/// clone, then it may be preferable to pass operands by reference.
/// Otherwise, clients may be forced to clone borrowed instances
/// unnecessarily.
/// Unfortunately, it is impossible to implement an `Add` operator for `&i32`
/// for example.
/// Hence, in type-generic code, requiring an `Add` implementation for `&T`
/// precludes built-in types like `i32` to be used.
/// But it should be possible to implement abstract mathematical structures
/// (like polynomials or matrices) for both built-in element types and element
/// types which are potentially expensive to copy (like large integers)
/// efficiently.
/// This is what the operator traits [`AddAssignWithRef`], [`SubAssignWithRef`]
/// and [`MulWithRef`] are designed for.
///
/// You may notice that we provide in-place operators for addition and
/// subtraction ([`AddAssignWithRef`] and [`SubAssignWithRef`]), but the
/// multiplication operator [`MulWithRef`] is not in-place, but returns `Self`.
/// This distinction is made on purpose, because for most data structures
/// addition and subtraction can be performed in-place very efficiently,
/// but multiplication cannot be performed in-place efficiently.
/// (Please convince yourself by looking considering matrices or polynomials
/// as examples.)
pub trait Ring:
    Clone
    + Zero
    + One
    + Add<Output = Self>
    + AddAssign
    + AddAssignWithRef
    + Sub<Output = Self>
    + SubAssign
    + SubAssignWithRef
    + Mul<Output = Self>
    + MulAssign
    + MulWithRef
    + Neg<Output = Self>
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
