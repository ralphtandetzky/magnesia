use crate::algebra::{AddAssignWithRef, One, SubAssignWithRef, Zero};
use std::ops::{Add, AddAssign, Sub, SubAssign};

trait Dimension {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SMatrix<T, const NUM_ROWS: usize, const NUM_COLS: usize>([[T; NUM_COLS]; NUM_ROWS]);

impl<T: Zero, const NUM_ROWS: usize, const NUM_COLS: usize> Zero
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Returns a matrix filled with zeros.
    ///
    /// # Example
    ///
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// # use magnesia::algebra::Zero;
    /// let m = SMatrix::zero();
    /// assert_eq!(m, SMatrix::from([[0,0],[0,0],[0,0]]));
    /// ```
    fn zero() -> Self {
        Self {
            0: [(); NUM_ROWS].map(|_| [(); NUM_COLS].map(|_| T::zero())),
        }
    }
}

impl<T: Zero + One, const NUM_ROWS: usize, const NUM_COLS: usize> One
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Returns a matrix filled with ones on the diagonal and zeros everywhere else.
    ///
    /// # Example
    ///
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// # use magnesia::algebra::One;
    /// let m = SMatrix::one();
    /// assert_eq!(m, SMatrix::from([[1,0],[0,1],[0,0]]));
    /// ```
    fn one() -> Self {
        let mut m: usize = 0;
        Self {
            0: [(); NUM_ROWS].map(|_| {
                let mut n: usize = 0;
                let row = [(); NUM_COLS].map(|_| {
                    let x = if m == n { T::one() } else { T::zero() };
                    n += 1;
                    x
                });
                m += 1;
                row
            }),
        }
    }
}

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> From<[[T; NUM_COLS]; NUM_ROWS]>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Creates a statically sized matrix from an array.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let a = SMatrix::from([[1,2],[3,4]]);
    /// ```
    fn from(coefficients: [[T; NUM_COLS]; NUM_ROWS]) -> Self {
        Self { 0: coefficients }
    }
}

impl<T: AddAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> Add
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    type Output = Self;

    /// Adds two matrices.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let a = SMatrix::from([[0,1,2],[3,4,5]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// let c = a + b;
    /// assert_eq!(c, SMatrix::from([[1,2,3],[4,5,6]]));
    /// ```
    fn add(mut self, other: Self) -> Self {
        self += &other;
        self
    }
}

impl<T: AddAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> Add<&Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    type Output = Self;

    /// Adds two matrices.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let a = SMatrix::from([[0,1,2],[3,4,5]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// let c = a + &b;
    /// assert_eq!(c, SMatrix::from([[1,2,3],[4,5,6]]));
    /// ```
    fn add(mut self, other: &Self) -> Self {
        self += other;
        self
    }
}

impl<T: AddAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> AddAssign<Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Adds two matrices in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut a = SMatrix::from([[0,1,2],[3,4,5]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// a += b;
    /// assert_eq!(a, SMatrix::from([[1,2,3],[4,5,6]]));
    /// ```
    fn add_assign(&mut self, other: Self) {
        *self += &other;
    }
}

impl<T: AddAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> AddAssign<&Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Adds two matrices in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut a = SMatrix::from([[0,1,2],[3,4,5]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// a += &b;
    /// assert_eq!(a, SMatrix::from([[1,2,3],[4,5,6]]));
    /// ```
    fn add_assign(&mut self, other: &Self) {
        for (lo, ro) in self.0.iter_mut().zip(other.0.iter()) {
            for (li, ri) in lo.iter_mut().zip(ro.iter()) {
                li.add_assign_with_ref(ri);
            }
        }
    }
}

impl<T: SubAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> Sub
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    type Output = Self;

    /// Subtracts two matrices.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let a = SMatrix::from([[1,2,3],[4,5,6]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// let c = a - b;
    /// assert_eq!(c, SMatrix::from([[0,1,2],[3,4,5]]));
    /// ```
    fn sub(mut self, other: Self) -> Self {
        self -= &other;
        self
    }
}

impl<T: SubAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> Sub<&Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    type Output = Self;

    /// Subtracts two matrices.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let a = SMatrix::from([[1,2,3],[4,5,6]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// let c = a - &b;
    /// assert_eq!(c, SMatrix::from([[0,1,2],[3,4,5]]));
    /// ```
    fn sub(mut self, other: &Self) -> Self {
        self -= other;
        self
    }
}

impl<T: SubAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> SubAssign<Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Subtracts two matrices in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut a = SMatrix::from([[1,2,3],[4,5,6]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// a -= b;
    /// assert_eq!(a, SMatrix::from([[0,1,2],[3,4,5]]));
    /// ```
    fn sub_assign(&mut self, other: Self) {
        *self -= &other;
    }
}

impl<T: SubAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> SubAssign<&Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Subtracts two matrices in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut a = SMatrix::from([[1,2,3],[4,5,6]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// a -= &b;
    /// assert_eq!(a, SMatrix::from([[0,1,2],[3,4,5]]));
    /// ```
    fn sub_assign(&mut self, other: &Self) {
        for (lo, ro) in self.0.iter_mut().zip(other.0.iter()) {
            for (li, ri) in lo.iter_mut().zip(ro.iter()) {
                li.sub_assign_with_ref(ri);
            }
        }
    }
}
