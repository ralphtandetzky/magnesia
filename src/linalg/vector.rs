use crate::algebra::{AddAssignWithRef, Zero};
use std::ops::{Add, AddAssign};

/// Statically sized mathematical vector.
#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub struct SVector<T, const DIM: usize>([T; DIM]);

impl<T: Zero, const DIM: usize> Zero for SVector<T, DIM> {
    /// Initializes a zero vector.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// # use magnesia::algebra::Zero;
    /// let v = SVector::zero();
    /// assert_eq!(v, SVector::from([0, 0, 0]))
    /// ```
    fn zero() -> Self {
        Self([(); DIM].map(|_| T::zero()))
    }
}

impl<T, const DIM: usize> From<[T; DIM]> for SVector<T, DIM> {
    /// Turns a statically sized array into a vector.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let v = SVector::from([1,2]);
    /// ```
    fn from(s: [T; DIM]) -> Self {
        Self(s)
    }
}

impl<T, const DIM: usize> Add for SVector<T, DIM>
where
    Self: AddAssignWithRef,
{
    type Output = Self;

    /// Implements the `+` operator for `SVector`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let u = SVector::from([1,2]);
    /// let v = SVector::from([3,4]);
    /// let w = u + v;
    /// assert_eq!(w, SVector::from([4,6]));
    /// ```
    fn add(mut self, other: Self) -> Self {
        self.add_assign_with_ref(&other);
        self
    }
}

impl<T, const DIM: usize> Add<&Self> for SVector<T, DIM>
where
    Self: AddAssignWithRef,
{
    type Output = Self;

    /// Implements the `+` operator for `SVector`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let u = SVector::from([1,2]);
    /// let v = SVector::from([3,4]);
    /// let w = u + &v;
    /// assert_eq!(w, SVector::from([4,6]));
    /// ```
    fn add(mut self, other: &Self) -> Self {
        self.add_assign_with_ref(other);
        self
    }
}

impl<T, const DIM: usize> AddAssign<Self> for SVector<T, DIM>
where
    Self: AddAssignWithRef,
{
    /// Implements the `+=` operator for `SVector`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let mut u = SVector::from([1,2]);
    /// let v = SVector::from([3,4]);
    /// u += v;
    /// assert_eq!(u, SVector::from([4,6]));
    /// ```
    fn add_assign(&mut self, other: Self) {
        self.add_assign_with_ref(&other);
    }
}

impl<T, const DIM: usize> AddAssign<&Self> for SVector<T, DIM>
where
    Self: AddAssignWithRef,
{
    /// Implements the `+=` operator for `&SVector`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let mut u = SVector::from([1,2]);
    /// let v = SVector::from([3,4]);
    /// u += &v;
    /// assert_eq!(u, SVector::from([4,6]));
    /// ```
    fn add_assign(&mut self, other: &Self) {
        self.add_assign_with_ref(other);
    }
}

impl<T: AddAssignWithRef, const DIM: usize> AddAssignWithRef for SVector<T, DIM> {
    /// Implements the [`AddAssignWithRef`] operator for `SVector`.
    ///
    /// It is recommended to instead use the `+=` operator with references:
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let mut u = SVector::from([1,2]);
    /// let v = SVector::from([3,4]);
    /// u += &v;
    /// assert_eq!(u, SVector::from([4,6]));
    /// ```
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// # use magnesia::algebra::AddAssignWithRef;
    /// let mut u = SVector::from([1,2]);
    /// let v = SVector::from([3,4]);
    /// u.add_assign_with_ref(&v);
    /// assert_eq!(u, SVector::from([4,6]));
    /// ```
    fn add_assign_with_ref(&mut self, other: &Self) {
        for (s, o) in self.0.iter_mut().zip(other.0.iter()) {
            s.add_assign_with_ref(o);
        }
    }
}
