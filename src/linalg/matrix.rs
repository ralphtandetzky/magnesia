use crate::algebra::{AddAssignWithRef, MulWithRef, NegAssign, One, Ring, SubAssignWithRef, Zero};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

trait Dimension {}

/// A matrix type with static number of rows and columns.
///
/// # Example
/// ```
/// # use magnesia::linalg::SMatrix;
/// let a = SMatrix::from([[0,1,2],[3,4,5]]);
/// let b = SMatrix::from([[1,1,1],[1,1,1]]);
/// let c = a + b;
/// assert_eq!(c, SMatrix::from([[1,2,3],[4,5,6]]));
/// ```
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
        Self([(); NUM_ROWS].map(|_| [(); NUM_COLS].map(|_| T::zero())))
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
        Self([(); NUM_ROWS].map(|_| {
            let mut n: usize = 0;
            let row = [(); NUM_COLS].map(|_| {
                let x = if m == n { T::one() } else { T::zero() };
                n += 1;
                x
            });
            m += 1;
            row
        }))
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
        Self(coefficients)
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

impl<T: AddAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> AddAssignWithRef
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Adds two matrices in-place.
    ///
    /// It is recommended to use the operator syntax instead:
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut a = SMatrix::from([[0,1,2],[3,4,5]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// a += &b;
    /// assert_eq!(a, SMatrix::from([[1,2,3],[4,5,6]]));
    /// ```
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// # use magnesia::algebra::AddAssignWithRef;
    /// let mut a = SMatrix::from([[0,1,2],[3,4,5]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// a.add_assign_with_ref(&b);
    /// assert_eq!(a, SMatrix::from([[1,2,3],[4,5,6]]));
    /// ```
    fn add_assign_with_ref(&mut self, other: &Self) {
        *self += other;
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

impl<T: SubAssignWithRef, const NUM_ROWS: usize, const NUM_COLS: usize> SubAssignWithRef
    for SMatrix<T, NUM_ROWS, NUM_COLS>
{
    /// Subtracts two matrices in-place.
    ///
    /// It is recommended to use the operator syntax instead:
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut a = SMatrix::from([[1,2,3],[4,5,6]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// a -= &b;
    /// assert_eq!(a, SMatrix::from([[0,1,2],[3,4,5]]));
    /// ```
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// # use magnesia::algebra::SubAssignWithRef;
    /// let mut a = SMatrix::from([[1,2,3],[4,5,6]]);
    /// let b = SMatrix::from([[1,1,1],[1,1,1]]);
    /// a.sub_assign_with_ref(&b);
    /// assert_eq!(a, SMatrix::from([[0,1,2],[3,4,5]]));
    /// ```
    fn sub_assign_with_ref(&mut self, other: &Self) {
        *self -= other;
    }
}

impl<'a, T: AddAssign + MulWithRef + Zero, const L: usize, const M: usize, const N: usize>
    Mul<&'a SMatrix<T, M, N>> for &'a SMatrix<T, L, M>
{
    type Output = SMatrix<T, L, N>;

    /// Multiplies two matrices
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let a = SMatrix::from([[0,1,2],[3,4,5]]);
    /// let b = SMatrix::from([[0,1],[2,3],[4,5]]);
    /// let c = &a * &b;
    /// assert_eq!(c, SMatrix::from([[10,13],[28,40]]));
    /// ```
    fn mul(self, other: &'a SMatrix<T, M, N>) -> Self::Output {
        let mut l: usize = 0;
        SMatrix([(); L].map(|_| {
            let mut n: usize = 0;
            let row = [(); N].map(|_| {
                let mut sum = T::zero();
                for m in 0..M {
                    sum += self.0[l][m].mul_with_ref(&other.0[m][n]);
                }
                n += 1;
                sum
            });
            l += 1;
            row
        }))
    }
}

impl<T: AddAssign + MulWithRef + Zero, const L: usize, const M: usize, const N: usize>
    Mul<SMatrix<T, M, N>> for SMatrix<T, L, M>
{
    type Output = SMatrix<T, L, N>;

    /// Multiplies two matrices
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let a = SMatrix::from([[0,1,2],[3,4,5]]);
    /// let b = SMatrix::from([[0,1],[2,3],[4,5]]);
    /// let c = a * b;
    /// assert_eq!(c, SMatrix::from([[10,13],[28,40]]));
    /// ```
    fn mul(self, other: SMatrix<T, M, N>) -> Self::Output {
        &self * &other
    }
}

impl<T, const M: usize, const N: usize> MulAssign<&SMatrix<T, N, N>> for SMatrix<T, M, N>
where
    for<'a> &'a Self: Mul<&'a SMatrix<T, N, N>, Output = Self>,
{
    /// Multiplies two matrices
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut a = SMatrix::from([[0,1],[2,3],[4,5]]);
    /// let b = SMatrix::from([[4,5],[6,7]]);
    /// a *= &b;
    /// assert_eq!(a, SMatrix::from([[6,7],[26,31],[46,55]]));
    /// ```
    fn mul_assign(&mut self, other: &SMatrix<T, N, N>) {
        let r: &Self = self;
        let result = r * other;
        *self = result;
    }
}

impl<T, const M: usize, const N: usize> MulAssign<SMatrix<T, N, N>> for SMatrix<T, M, N>
where
    for<'a> &'a Self: Mul<&'a SMatrix<T, N, N>, Output = Self>,
{
    /// Multiplies two matrices and assigns the result to the first operand.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut a = SMatrix::from([[0,1],[2,3],[4,5]]);
    /// let b = SMatrix::from([[4,5],[6,7]]);
    /// a *= b;
    /// assert_eq!(a, SMatrix::from([[6,7],[26,31],[46,55]]));
    /// ```
    fn mul_assign(&mut self, other: SMatrix<T, N, N>) {
        let r: &Self = self;
        let result = r * &other;
        *self = result;
    }
}

impl<T: AddAssign + MulWithRef + Zero, const N: usize> MulWithRef for SMatrix<T, N, N> {
    /// Multiplies two matrices
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// # use magnesia::algebra::MulWithRef;
    /// let a = SMatrix::from([[0,1],[2,3]]);
    /// let b = SMatrix::from([[1,2],[3,4]]);
    /// let c = a.mul_with_ref(&b);
    /// assert_eq!(c, SMatrix::from([[3,4],[11,16]]));
    /// ```
    fn mul_with_ref(&self, other: &Self) -> Self {
        self * other
    }
}

impl<T: NegAssign, const M: usize, const N: usize> NegAssign for SMatrix<T, M, N> {
    /// Negates a matrix in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// # use magnesia::algebra::NegAssign;
    /// let mut m = SMatrix::from([[1,2,3],[4,5,6]]);
    /// m.neg_assign();
    /// assert_eq!(m, SMatrix::from([[-1,-2,-3],[-4,-5,-6]]));
    fn neg_assign(&mut self) {
        for row in self.0.iter_mut() {
            for val in row.iter_mut() {
                val.neg_assign();
            }
        }
    }
}

impl<T: NegAssign, const M: usize, const N: usize> Neg for SMatrix<T, M, N> {
    type Output = Self;

    /// Implements the unary `-` operator.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let a = SMatrix::from([[1,2,3],[4,5,6]]);
    /// let b = -a;
    /// assert_eq!(b, SMatrix::from([[-1,-2,-3],[-4,-5,-6]]));
    fn neg(mut self) -> Self {
        self.neg_assign();
        self
    }
}

impl<T: Ring, const N: usize> Ring for SMatrix<T, N, N> {}
