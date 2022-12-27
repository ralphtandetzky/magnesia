use super::SVector;
use crate::algebra::{Conj, MulRefs, One, Ring, Zero};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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
        let mut m = 0_usize;
        Self([(); NUM_ROWS].map(|_| {
            let mut n = 0_usize;
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

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> Add for SMatrix<T, NUM_ROWS, NUM_COLS>
where
    for<'a> T: AddAssign<&'a T>,
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

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> Add<&Self> for SMatrix<T, NUM_ROWS, NUM_COLS>
where
    for<'a> T: AddAssign<&'a T>,
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

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> AddAssign<Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
where
    for<'a> T: AddAssign<&'a T>,
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

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> AddAssign<&Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
where
    for<'a> T: AddAssign<&'a T>,
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
                *li += ri;
            }
        }
    }
}

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> Sub for SMatrix<T, NUM_ROWS, NUM_COLS>
where
    for<'a> T: SubAssign<&'a T>,
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

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> Sub<&Self> for SMatrix<T, NUM_ROWS, NUM_COLS>
where
    for<'a> T: SubAssign<&'a T>,
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

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> SubAssign<Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
where
    for<'a> T: SubAssign<&'a T>,
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

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> SubAssign<&Self>
    for SMatrix<T, NUM_ROWS, NUM_COLS>
where
    for<'a> T: SubAssign<&'a T>,
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
                *li -= ri;
            }
        }
    }
}

impl<'a, T, const L: usize, const M: usize, const N: usize> Mul<&'a SMatrix<T, M, N>>
    for &'a SMatrix<T, L, M>
where
    for<'b> T: AddAssign<&'b T> + MulRefs + Zero,
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
        let mut l = 0_usize;
        SMatrix([(); L].map(|_| {
            let mut n = 0_usize;
            let row = [(); N].map(|_| {
                let mut sum = T::zero();
                for m in 0..M {
                    let prod = self.0[l][m].mul_refs(&other.0[m][n]);
                    sum += &prod;
                }
                n += 1;
                sum
            });
            l += 1;
            row
        }))
    }
}

impl<T, const L: usize, const M: usize, const N: usize> Mul<SMatrix<T, M, N>> for SMatrix<T, L, M>
where
    for<'b> T: AddAssign<&'b T> + MulRefs + Zero,
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

impl<'a, T, const M: usize, const N: usize> Mul<&'a SVector<T, N>> for &'a SMatrix<T, M, N>
where
    for<'b> T: MulRefs + AddAssign<&'b T> + Zero,
{
    type Output = SVector<T, M>;

    /// Implements multiplication of matrix by vector.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// # use magnesia::linalg::SVector;
    /// let m = SMatrix::from([[1,2], [3,4]]);
    /// let u = SVector::from([1,2]);
    /// let v = &m * &u;
    /// assert_eq!(v, SVector::from([1*1 + 2*2, 1*3 + 2*4]));
    /// ```
    fn mul(self, vec: &'a SVector<T, N>) -> Self::Output {
        let mut i = 0_usize;
        SVector::from([(); M].map(|_| {
            let mut sum = T::zero();
            for (m, v) in self.0[i].iter().zip(vec) {
                let prod = m.mul_refs(v);
                sum += &prod;
            }
            i += 1;
            sum
        }))
    }
}

impl<T: Neg<Output = T>, const M: usize, const N: usize> Neg for SMatrix<T, M, N> {
    type Output = Self;
    /// Negates a matrix in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::SMatrix;
    /// let mut m = SMatrix::from([[1,2,3],[4,5,6]]);
    /// assert_eq!(-m, SMatrix::from([[-1,-2,-3],[-4,-5,-6]]));
    fn neg(self) -> Self {
        Self(self.0.map(|row| row.map(|x| -x)))
    }
}

impl<T: Ring, const N: usize> Ring for SMatrix<T, N, N> {}

impl<T: Conj, const M: usize, const N: usize> Conj for SMatrix<T, M, N> {
    fn conj(self) -> Self {
        Self(self.0.map(|row| row.map(|x| x.conj())))
    }
}

#[test]
fn test_conj_assign_on_i8_smatrix() {
    let a = SMatrix::from([[1i8, 2i8], [3i8, 4i8]]);
    let b = a.conj();
    assert_eq!(a, b);
}

#[test]
fn test_conj_assign_on_complex_i8_smatrix() {
    use crate::algebra::Complex;
    let a = SMatrix::from([
        [Complex::new(1i8, 1i8), Complex::new(2i8, 2i8)],
        [Complex::new(3i8, 3i8), Complex::new(4i8, 4i8)],
    ]);
    let b = SMatrix::from([
        [Complex::new(1i8, -1i8), Complex::new(2i8, -2i8)],
        [Complex::new(3i8, -3i8), Complex::new(4i8, -4i8)],
    ]);
    assert_eq!(a.conj(), b);
}
