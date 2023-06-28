use crate::algebra::{MulRefs, Zero};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// Statically sized mathematical vector.
#[derive(Debug, PartialEq, Clone, Copy, Eq)]
pub struct SVector<T, const DIM: usize>(pub [T; DIM]);

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
    for<'a> T: AddAssign<&'a T>,
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
        self += &other;
        self
    }
}

impl<T, const DIM: usize> Add<&Self> for SVector<T, DIM>
where
    for<'a> T: AddAssign<&'a T>,
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
        self += other;
        self
    }
}

impl<T, const DIM: usize> AddAssign<Self> for SVector<T, DIM>
where
    for<'a> T: AddAssign<&'a T>,
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
        *self += &other;
    }
}

impl<T, const DIM: usize> AddAssign<&Self> for SVector<T, DIM>
where
    for<'a> T: AddAssign<&'a T>,
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
        for (s, o) in self.0.iter_mut().zip(other.0.iter()) {
            *s += o;
        }
    }
}

impl<T, const DIM: usize> Sub for SVector<T, DIM>
where
    for<'a> T: SubAssign<&'a T>,
{
    type Output = Self;

    /// Implements the `-` operator for `SVector`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let u = SVector::from([4,6]);
    /// let v = SVector::from([3,4]);
    /// let w = u - v;
    /// assert_eq!(w, SVector::from([1,2]));
    /// ```
    fn sub(mut self, other: Self) -> Self {
        self -= &other;
        self
    }
}

impl<T, const DIM: usize> Sub<&Self> for SVector<T, DIM>
where
    for<'a> T: SubAssign<&'a T>,
{
    type Output = Self;

    /// Implements the `+` operator for `SVector`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let u = SVector::from([4,6]);
    /// let v = SVector::from([3,4]);
    /// let w = u - &v;
    /// assert_eq!(w, SVector::from([1,2]));
    /// ```
    fn sub(mut self, other: &Self) -> Self {
        self -= other;
        self
    }
}

impl<T, const DIM: usize> SubAssign<Self> for SVector<T, DIM>
where
    for<'a> T: SubAssign<&'a T>,
{
    /// Implements the `-=` operator for `SVector`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let mut u = SVector::from([4,6]);
    /// let v = SVector::from([3,4]);
    /// u -= v;
    /// assert_eq!(u, SVector::from([1,2]));
    /// ```
    fn sub_assign(&mut self, other: Self) {
        *self -= &other;
    }
}

impl<T, const DIM: usize> SubAssign<&Self> for SVector<T, DIM>
where
    for<'a> T: SubAssign<&'a T>,
{
    /// Implements the `-=` operator for `&SVector`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let mut u = SVector::from([4,6]);
    /// let v = SVector::from([3,4]);
    /// u -= &v;
    /// assert_eq!(u, SVector::from([1,2]));
    /// ```
    fn sub_assign(&mut self, other: &Self) {
        for (s, o) in self.0.iter_mut().zip(other.0.iter()) {
            *s -= o;
        }
    }
}

impl<T, const DIM: usize> Mul<Self> for SVector<T, DIM>
where
    for<'a> Self: Mul<&'a Self, Output = T>,
{
    type Output = T;

    /// Implements the dot product of two vectors.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let u = SVector::from([1,2]);
    /// let v = SVector::from([3,4]);
    /// let x = u * v;
    /// assert_eq!(x, 1 * 3 + 2 * 4);
    /// ```
    fn mul(self, other: Self) -> T {
        self * &other
    }
}

impl<T, const DIM: usize> Mul<&Self> for SVector<T, DIM>
where
    for<'a> T: MulRefs + AddAssign<&'a T> + Zero,
{
    type Output = T;

    /// Implements the dot product of two vectors.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let u = SVector::from([1,2]);
    /// let v = SVector::from([3,4]);
    /// let x = u * &v;
    /// assert_eq!(x, 1 * 3 + 2 * 4);
    /// ```
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: &Self) -> Self::Output {
        let mut sum = T::zero();
        for (l, r) in self.0.iter().zip(other.0.iter()) {
            let prod = l.mul_refs(r);
            sum += &prod;
        }
        sum
    }
}

impl<T, const DIM: usize> Mul<T> for SVector<T, DIM>
where
    T: MulRefs,
{
    type Output = Self;

    /// Multiplies a vector with a scalar.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let u = SVector::from([1,2]);
    /// let v = u * 2;
    /// assert_eq!(v, SVector::from([2,4]));
    /// ```
    fn mul(self, other: T) -> Self::Output {
        self * &other
    }
}

impl<T, const DIM: usize> Mul<&T> for SVector<T, DIM>
where
    T: MulRefs,
{
    type Output = Self;

    /// Multiplies a vector with a scalar.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let u = SVector::from([1,2]);
    /// let v = u * &2;
    /// assert_eq!(v, SVector::from([2,4]));
    /// ```
    fn mul(mut self, other: &T) -> Self::Output {
        self *= other;
        self
    }
}

impl<T, const DIM: usize> MulAssign<T> for SVector<T, DIM>
where
    T: MulRefs,
{
    /// Multiplies a vector with a scalar in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let mut u = SVector::from([1,2]);
    /// u *= 2;
    /// assert_eq!(u, SVector::from([2,4]));
    /// ```
    fn mul_assign(&mut self, other: T) {
        *self *= &other;
    }
}

impl<T, const DIM: usize> MulAssign<&T> for SVector<T, DIM>
where
    T: MulRefs,
{
    /// Multiplies a vector with a scalar in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SVector;
    /// let mut u = SVector::from([1,2]);
    /// u *= &2;
    /// assert_eq!(u, SVector::from([2,4]));
    /// ```
    fn mul_assign(&mut self, other: &T) {
        for x in self {
            let a = x.mul_refs(other);
            *x = a;
        }
    }
}

impl<T, const DIM: usize> IntoIterator for SVector<T, DIM> {
    type Item = <[T; DIM] as IntoIterator>::Item;
    type IntoIter = <[T; DIM] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIterator::into_iter(self.0)
    }
}

impl<'a, T, const DIM: usize> IntoIterator for &'a SVector<T, DIM> {
    type Item = <&'a [T; DIM] as IntoIterator>::Item;
    type IntoIter = <&'a [T; DIM] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T, const DIM: usize> IntoIterator for &'a mut SVector<T, DIM> {
    type Item = <&'a mut [T; DIM] as IntoIterator>::Item;
    type IntoIter = <&'a mut [T; DIM] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}
