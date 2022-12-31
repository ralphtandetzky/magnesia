use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub, SubAssign};

use crate::algebra::{Conj, One, Zero};

/// A matrix-like interface.
pub trait MatrixExpr: Sized {
    /// The element type of the matrix.
    type Entry;

    /// Returns an entry of the matrix.
    fn entry(&self, row: usize, col: usize) -> Self::Entry;

    /// Returns the number of rows of the matrix.
    fn num_rows(&self) -> usize;

    /// Returns the number of columns of the matrix.
    fn num_cols(&self) -> usize;

    /// Evaluates all entries of the matrix and stores them in a [`DMatrix`].
    fn eval(self) -> DMatrix<Self::Entry> {
        let data = (0..self.num_rows())
            .flat_map(|r| (0..self.num_cols()).map(move |c| (r, c)))
            .map(|(r, c)| self.entry(r, c))
            .collect();
        DMatrix {
            data,
            num_rows: self.num_rows(),
            num_cols: self.num_cols(),
        }
    }

    /// Wraps the matrix expression into an [`ExprWrapper`].
    fn wrap(self) -> ExprWrapper<Self> {
        ExprWrapper(self)
    }
}

pub struct ExprWrapper<T: MatrixExpr>(T);

impl<T: MatrixExpr> MatrixExpr for ExprWrapper<T> {
    type Entry = T::Entry;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        self.0.entry(row, col)
    }

    fn num_rows(&self) -> usize {
        self.0.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.0.num_cols()
    }

    fn eval(self) -> DMatrix<Self::Entry> {
        self.0.eval()
    }
}

pub fn make_matrix_expr<F, Out>(
    num_rows: usize,
    num_cols: usize,
    f: F,
) -> ExprWrapper<impl MatrixExpr<Entry = Out>>
where
    F: Fn(usize, usize) -> Out,
{
    struct FnMatrixExpr<F_, Out_>(F_, usize, usize)
    where
        F_: Fn(usize, usize) -> Out_;

    impl<F_, Out_> MatrixExpr for FnMatrixExpr<F_, Out_>
    where
        F_: Fn(usize, usize) -> Out_,
    {
        type Entry = Out_;

        fn entry(&self, row: usize, col: usize) -> Self::Entry {
            (self.0)(row, col)
        }

        fn num_rows(&self) -> usize {
            self.1
        }

        fn num_cols(&self) -> usize {
            self.2
        }
    }

    FnMatrixExpr(f, num_rows, num_cols).wrap()
}

#[test]
fn test_make_matrix_expr() {
    let a = make_matrix_expr(2, 3, |x, y| x + y).eval();
    let b = [[0, 1, 2], [1, 2, 3]].eval();
    assert_eq!(a, b);
}

impl<Lhs: MatrixExpr> ExprWrapper<Lhs> {
    pub fn map<F, Out>(self, f: F) -> ExprWrapper<impl MatrixExpr<Entry = Out>>
    where
        F: Fn(Lhs::Entry) -> Out,
    {
        make_matrix_expr(self.0.num_rows(), self.0.num_cols(), move |row, col| {
            f(self.0.entry(row, col))
        })
    }
}

#[test]
fn test_map_matrix_expr() {
    let a = [[1, 2, 3], [4, 5, 6]].wrap().map(|a| 2 * a + 1).eval();
    let b = [[3, 5, 7], [9, 11, 13]].eval();
    assert_eq!(a, b);
}

impl<Lhs: MatrixExpr> ExprWrapper<Lhs> {
    pub fn zip<Rhs>(
        self,
        rhs: Rhs,
    ) -> ExprWrapper<impl MatrixExpr<Entry = (Lhs::Entry, Rhs::Entry)>>
    where
        Rhs: MatrixExpr,
    {
        assert_eq!(self.num_rows(), rhs.num_rows());
        assert_eq!(self.num_cols(), rhs.num_cols());
        make_matrix_expr(self.0.num_rows(), self.0.num_cols(), move |row, col| {
            (self.0.entry(row, col), rhs.entry(row, col))
        })
    }
}

#[test]
fn test_zip_matrix_expr() {
    let a = [[1, 2, 3], [4, 5, 6]]
        .eval()
        .wrap()
        .zip([[7, 8, 9], [10, 11, 12]].eval())
        .eval();
    let b = [[(1, 7), (2, 8), (3, 9)], [(4, 10), (5, 11), (6, 12)]].eval();
    assert_eq!(a, b);
}

impl<Lhs: MatrixExpr> ExprWrapper<Lhs> {
    pub fn mul_elemwise<Rhs: MatrixExpr>(
        self,
        rhs: Rhs,
    ) -> ExprWrapper<impl MatrixExpr<Entry = <Lhs::Entry as Mul<Rhs::Entry>>::Output>>
    where
        Lhs::Entry: Mul<Rhs::Entry>,
    {
        self.zip(rhs).map(|(lhs, rhs)| lhs * rhs)
    }
}

#[test]
fn test_mul_elemwise_matrix_expr() {
    let a = [[1, 2, 3], [4, 5, 6]].wrap();
    let b = [[0, 1, 2], [3, 4, 5]].wrap();
    let c = a.mul_elemwise(b).eval();
    let d = [[0, 2, 6], [12, 20, 30]].eval();
    assert_eq!(c, d);
}

impl<Lhs: MatrixExpr> ExprWrapper<Lhs> {
    pub fn div_elemwise<Rhs: MatrixExpr>(
        self,
        rhs: Rhs,
    ) -> ExprWrapper<impl MatrixExpr<Entry = <Lhs::Entry as Div<Rhs::Entry>>::Output>>
    where
        Lhs::Entry: Div<Rhs::Entry>,
    {
        self.zip(rhs).map(|(lhs, rhs)| lhs / rhs)
    }
}

#[test]
fn test_div_elemwise_matrix_expr() {
    let a = [[0, 2, 6], [12, 20, 30]].wrap();
    let b = [[1, 2, 3], [4, 5, 6]].wrap();
    let c = a.div_elemwise(b).eval();
    let d = [[0, 1, 2], [3, 4, 5]].eval();
    assert_eq!(c, d);
}

impl<Rhs, Lhs> Add<Rhs> for ExprWrapper<Lhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Lhs::Entry: Add<Rhs::Entry>,
{
    type Output = ExprWrapper<AddExpr<Lhs, Rhs>>;

    fn add(self, rhs: Rhs) -> Self::Output {
        assert_eq!(self.num_rows(), rhs.num_rows());
        assert_eq!(self.num_cols(), rhs.num_cols());
        AddExpr(self.0, rhs).wrap()
    }
}

#[test]
fn test_add_expr_wrapper() {
    let a = [[1, 2, 3], [4, 5, 6]].wrap();
    let b = [[2, 2, 2], [3, 3, 3]].wrap();
    let c = a + b;
    let d = [[3, 4, 5], [7, 8, 9]].wrap();
    assert_eq!(c.eval(), d.eval());
}

pub struct AddExpr<Lhs, Rhs>(Lhs, Rhs);

impl<Lhs, Rhs> MatrixExpr for AddExpr<Lhs, Rhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Lhs::Entry: Add<Rhs::Entry>,
{
    type Entry = <Lhs::Entry as Add<Rhs::Entry>>::Output;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        self.0.entry(row, col) + self.1.entry(row, col)
    }

    fn num_rows(&self) -> usize {
        self.0.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.0.num_cols()
    }
}

impl<Rhs, Lhs> Sub<Rhs> for ExprWrapper<Lhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Lhs::Entry: Sub<Rhs::Entry>,
{
    type Output = ExprWrapper<SubExpr<Lhs, Rhs>>;

    fn sub(self, rhs: Rhs) -> Self::Output {
        assert_eq!(self.num_rows(), rhs.num_rows());
        assert_eq!(self.num_cols(), rhs.num_cols());
        SubExpr(self.0, rhs).wrap()
    }
}

#[test]
fn test_sub_expr_wrapper() {
    let a = [[1, 2, 3], [4, 5, 6]].wrap();
    let b = [[2, 2, 2], [3, 3, 3]].wrap();
    let c = a - b;
    let d = [[-1, 0, 1], [1, 2, 3]].wrap();
    assert_eq!(c.eval(), d.eval());
}

pub struct SubExpr<Lhs, Rhs>(Lhs, Rhs);

impl<Lhs, Rhs> MatrixExpr for SubExpr<Lhs, Rhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Lhs::Entry: Sub<Rhs::Entry>,
{
    type Entry = <Lhs::Entry as Sub<Rhs::Entry>>::Output;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        self.0.entry(row, col) - self.1.entry(row, col)
    }

    fn num_rows(&self) -> usize {
        self.0.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.0.num_cols()
    }
}

impl<Expr> ExprWrapper<Expr>
where
    Expr: MatrixExpr,
{
    pub fn t(self) -> ExprWrapper<impl MatrixExpr<Entry = Expr::Entry>> {
        make_matrix_expr(self.0.num_cols(), self.0.num_rows(), move |r, c| {
            self.entry(c, r)
        })
    }
}

#[test]
fn test_expr_wrapper_transpose() {
    let a = [[1, 2, 3], [4, 5, 6]].wrap().t();
    let b = [[1, 4], [2, 5], [3, 6]].wrap();
    assert_eq!(a.eval(), b.eval());
}

impl<Expr> ExprWrapper<Expr>
where
    Expr: MatrixExpr,
    Expr::Entry: Conj,
{
    pub fn h(self) -> ExprWrapper<impl MatrixExpr<Entry = Expr::Entry>> {
        make_matrix_expr(self.0.num_cols(), self.0.num_rows(), move |r, c| {
            self.entry(c, r).conj()
        })
    }
}

#[test]
fn test_expr_wrapper_conjugate_transpose() {
    use crate::algebra::Complex;
    let a = [
        [Complex::new(0, 1), Complex::new(2, 3), Complex::new(4, 5)],
        [Complex::new(6, 7), Complex::new(8, 9), Complex::new(10, 11)],
    ]
    .wrap()
    .h();
    let b = [
        [Complex::new(0, -1), Complex::new(6, -7)],
        [Complex::new(2, -3), Complex::new(8, -9)],
        [Complex::new(4, -5), Complex::new(10, -11)],
    ]
    .wrap();
    assert_eq!(a.eval(), b.eval());
}

/// A matrix type with dynamic number of rows and columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DMatrix<T> {
    data: Box<[T]>,
    num_rows: usize,
    num_cols: usize,
}

impl<T> MatrixExpr for DMatrix<T>
where
    T: Clone,
{
    type Entry = T;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        self.data[row * self.num_cols + col].clone()
    }

    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_cols(&self) -> usize {
        self.num_cols
    }

    fn eval(self) -> DMatrix<Self::Entry> {
        self
    }
}

impl<T> MatrixExpr for &DMatrix<T>
where
    T: Clone,
{
    type Entry = T;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        (*self).data[row * (*self).num_cols + col].clone()
    }

    fn num_rows(&self) -> usize {
        (*self).num_rows
    }

    fn num_cols(&self) -> usize {
        (*self).num_cols
    }

    fn eval(self) -> DMatrix<Self::Entry> {
        self.clone()
    }
}

impl<T> DMatrix<T>
where
    T: One + Zero,
{
    /// Creates a unit matrix expression of size $n\times n$.
    ///
    /// This is a matrix which is $1$ on the diagonal and $0$ everywhere else.
    pub fn eye(n: usize) -> ExprWrapper<impl MatrixExpr<Entry = T>> {
        make_matrix_expr(n, n, |r, c| if r == c { T::one() } else { T::zero() })
    }
}

#[test]
fn test_dmatrix_eye() {
    assert_eq!(DMatrix::<i32>::eye(1).eval(), [[1]].eval());
    assert_eq!(DMatrix::<i32>::eye(2).eval(), [[1, 0], [0, 1]].eval());
    assert_eq!(
        DMatrix::<i32>::eye(3).eval(),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].eval()
    );
}

impl<T> DMatrix<T>
where
    T: Zero,
{
    /// Returns a matrix expression filled with zeros.
    pub fn zeros(num_rows: usize, num_cols: usize) -> ExprWrapper<impl MatrixExpr<Entry = T>> {
        make_matrix_expr(num_rows, num_cols, |_, _| T::zero())
    }
}

#[test]
fn test_dmatrix_zeros() {
    assert_eq!(
        DMatrix::<i32>::zeros(2, 3).eval(),
        [[0, 0, 0], [0, 0, 0]].eval()
    );
}

impl<T> Index<[usize; 2]> for DMatrix<T> {
    type Output = T;

    fn index(&self, row_and_col: [usize; 2]) -> &Self::Output {
        &self.data[row_and_col[0] * self.num_cols + row_and_col[1]]
    }
}

impl<T> DMatrix<T>
where
    T: One,
{
    /// Returns a matrix expression filled with ones.
    pub fn ones(num_rows: usize, num_cols: usize) -> ExprWrapper<impl MatrixExpr<Entry = T>> {
        make_matrix_expr(num_rows, num_cols, |_, _| T::one())
    }
}

#[test]
fn test_dmatrix_ones() {
    assert_eq!(
        DMatrix::<i32>::ones(2, 3).eval(),
        [[1, 1, 1], [1, 1, 1]].eval()
    );
}

impl<T> DMatrix<T>
where
    T: Clone,
{
    /// Returns a matrix expression filled with the passed value `val`.
    pub fn same(
        num_rows: usize,
        num_cols: usize,
        val: T,
    ) -> ExprWrapper<impl MatrixExpr<Entry = T>> {
        make_matrix_expr(num_rows, num_cols, move |_, _| val.clone())
    }
}

#[test]
fn test_dmatrix_same() {
    assert_eq!(
        DMatrix::<i32>::same(2, 3, 42).eval(),
        [[42, 42, 42], [42, 42, 42]].eval()
    );
}

#[test]
fn test_index_dmatrix() {
    let a = [[1, 2], [3, 4]].eval();
    assert_eq!(a[[0, 0]], 1);
    assert_eq!(a[[0, 1]], 2);
    assert_eq!(a[[1, 0]], 3);
    assert_eq!(a[[1, 1]], 4);
}

impl<T> IndexMut<[usize; 2]> for DMatrix<T> {
    fn index_mut(&mut self, row_and_col: [usize; 2]) -> &mut <Self as Index<[usize; 2]>>::Output {
        &mut self.data[row_and_col[0] * self.num_cols + row_and_col[1]]
    }
}

#[test]
fn test_index_mut_dmatrix() {
    let mut a = [[0, 0], [0, 0]].eval();
    a[[0, 0]] = 1;
    a[[0, 1]] = 2;
    a[[1, 0]] = 3;
    a[[1, 1]] = 4;
    assert_eq!(a[[0, 0]], 1);
    assert_eq!(a[[0, 1]], 2);
    assert_eq!(a[[1, 0]], 3);
    assert_eq!(a[[1, 1]], 4);
}

impl<T, Rhs> Add<Rhs> for &DMatrix<T>
where
    Rhs: MatrixExpr,
    ExprWrapper<Self>: Add<Rhs>,
    T: Clone,
{
    type Output = <ExprWrapper<Self> as Add<Rhs>>::Output;

    fn add(self, rhs: Rhs) -> Self::Output {
        self.wrap() + rhs
    }
}

#[test]
fn test_add_dmatrix() {
    let a = [[1, 2], [3, 4], [5, 6]].eval();
    let b = [[3, 3], [3, 3], [3, 3]].eval();
    let c = (&a + &b).eval();
    let d = [[4, 5], [6, 7], [8, 9]].eval();
    assert_eq!(c, d);
}

impl<T, Rhs> AddAssign<Rhs> for DMatrix<T>
where
    T: AddAssign<Rhs::Entry> + Clone,
    Rhs: MatrixExpr,
{
    fn add_assign(&mut self, rhs: Rhs) {
        let num_cols = self.num_cols();
        for row in 0..self.num_rows() {
            for col in 0..num_cols {
                self.data[row * num_cols + col] += rhs.entry(row, col);
            }
        }
    }
}

#[test]
fn test_add_assign_dmatrix() {
    let mut a = [[1, 2], [3, 4]].eval();
    a += [[2, 2], [2, 2]];
    assert_eq!(a, [[3, 4], [5, 6]].eval());
}

impl<T, Rhs> Sub<Rhs> for &DMatrix<T>
where
    T: Clone,
    ExprWrapper<Self>: Sub<Rhs>,
{
    type Output = <ExprWrapper<Self> as Sub<Rhs>>::Output;

    fn sub(self, rhs: Rhs) -> Self::Output {
        self.wrap() - rhs
    }
}

#[test]
fn test_sub_dmatrix() {
    let a = [[1, 2], [3, 4], [5, 6]].eval();
    let b = [[3, 3], [3, 3], [3, 3]].eval();
    let c = (&a - &b).eval();
    let d = [[-2, -1], [0, 1], [2, 3]].eval();
    assert_eq!(c, d);
}

impl<T, Rhs> SubAssign<Rhs> for DMatrix<T>
where
    T: SubAssign<Rhs::Entry> + Clone,
    Rhs: MatrixExpr,
{
    fn sub_assign(&mut self, rhs: Rhs) {
        let num_cols = self.num_cols();
        for row in 0..self.num_rows() {
            for col in 0..num_cols {
                self.data[row * num_cols + col] -= rhs.entry(row, col);
            }
        }
    }
}

#[test]
fn test_sub_assign_dmatrix() {
    let mut a = [[1, 2], [3, 4]].eval();
    a -= [[2, 2], [2, 2]];
    assert_eq!(a, [[-1, 0], [1, 2]].eval());
}

pub struct MulDMatrix<'a, T> {
    lhs: &'a DMatrix<T>,
    rhs: &'a DMatrix<T>,
}

impl<'a, T> MulDMatrix<'a, T> {
    fn new(lhs: &'a DMatrix<T>, rhs: &'a DMatrix<T>) -> Self {
        Self { lhs, rhs }
    }
}

impl<'a, T> MatrixExpr for MulDMatrix<'a, T>
where
    T: Mul<T> + Clone,
    <T as Mul<T>>::Output: AddAssign,
{
    type Entry = <T as Mul<T>>::Output;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        let mut sum = self.lhs.entry(row, 0) * self.rhs.entry(0, col);
        for i in 1..self.lhs.num_cols() {
            sum += self.lhs.entry(row, i) * self.rhs.entry(i, col);
        }
        sum
    }

    fn num_rows(&self) -> usize {
        self.lhs.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.rhs.num_cols()
    }
}

impl<'a, T> Mul<Self> for &'a DMatrix<T>
where
    T: Mul<T> + Clone,
    <T as Mul<T>>::Output: AddAssign,
{
    type Output = MulDMatrix<'a, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.num_cols(), rhs.num_rows(),
            "The number of columns of the left hand side matrix should be equal to the number of rows of the right hand side matrix.");
        MulDMatrix::new(self, rhs)
    }
}

#[test]
fn test_mul_dmatrix() {
    let a = [[-1, 0, 1], [2, 3, 4]].eval();
    let b = [[0, 1], [2, 3], [4, 5]].eval();
    let c = (&a * &b).eval();
    let d = [[4, 4], [22, 31]].eval();
    assert_eq!(c, d);
}

impl<T> DMatrix<T> {
    /// Multiplies a `DMatrix` with another matrix expression element-wise.
    ///
    /// # Example
    ///
    /// ```
    /// # use magnesia::linalg::MatrixExpr;
    /// let a = [[1,2,3],[4,5,6]].eval();
    /// let b = a.mul_elemwise([[0,1,2],[3,4,5]]).eval();
    /// let c = [[0,2,6],[12,20,30]].eval();
    /// assert_eq!(b, c);
    /// ```
    pub fn mul_elemwise<'a, Lhs: MatrixExpr>(
        &'a self,
        lhs: Lhs,
    ) -> ExprWrapper<impl MatrixExpr<Entry = T::Output> + 'a>
    where
        T: Mul<Lhs::Entry> + Clone,
        Lhs: 'a,
    {
        self.wrap().mul_elemwise(lhs)
    }
}

impl<T> DMatrix<T> {
    /// Divides a `DMatrix` by another matrix expression element-wise.
    ///
    /// # Example
    ///
    /// ```
    /// # use magnesia::linalg::MatrixExpr;
    /// let a = [[1, 4, 9],[16, 25, 36]].eval();
    /// let b = a.div_elemwise([[1, 2, 3],[4, 5, 6]]).eval();
    /// let c = [[1, 2, 3],[4, 5, 6]].eval();
    /// assert_eq!(b, c);
    /// ```
    pub fn div_elemwise<'a, Lhs: MatrixExpr>(
        &'a self,
        lhs: Lhs,
    ) -> ExprWrapper<impl MatrixExpr<Entry = T::Output> + 'a>
    where
        T: Div<Lhs::Entry> + Clone,
        Lhs: 'a,
    {
        self.wrap().div_elemwise(lhs)
    }
}

impl<T> DMatrix<T>
where
    T: Clone,
{
    /// Returns the transposed matrix as a matrix expression.
    pub fn t<'a>(&'a self) -> ExprWrapper<impl MatrixExpr<Entry = T> + 'a> {
        self.wrap().t()
    }
}

#[test]
fn test_dmatrix_transpose() {
    let a = [[1, 2, 3], [4, 5, 6]].eval().t().eval();
    let b = [[1, 4], [2, 5], [3, 6]].eval();
    assert_eq!(a, b);
}

impl<T> DMatrix<T>
where
    T: Clone + Conj,
{
    /// Returns the transposed matrix as a matrix expression.
    pub fn h<'a>(&'a self) -> ExprWrapper<impl MatrixExpr<Entry = T> + 'a> {
        self.wrap().h()
    }
}

#[test]
fn test_dmatrix_conjugate_transpose() {
    use crate::algebra::Complex;
    let a = [
        [Complex::new(0, 1), Complex::new(2, 3), Complex::new(4, 5)],
        [Complex::new(6, 7), Complex::new(8, 9), Complex::new(10, 11)],
    ]
    .wrap()
    .h()
    .eval();
    let b = [
        [Complex::new(0, -1), Complex::new(6, -7)],
        [Complex::new(2, -3), Complex::new(8, -9)],
        [Complex::new(4, -5), Complex::new(10, -11)],
    ]
    .eval();
    assert_eq!(a, b);
}

impl<T: Clone, const NUM_ROWS: usize, const NUM_COLS: usize> MatrixExpr
    for [[T; NUM_COLS]; NUM_ROWS]
{
    type Entry = T;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        self[row][col].clone()
    }

    fn num_rows(&self) -> usize {
        NUM_ROWS
    }

    fn num_cols(&self) -> usize {
        NUM_COLS
    }

    fn eval(self) -> DMatrix<Self::Entry> {
        DMatrix {
            data: self.into_iter().flatten().collect(),
            num_rows: NUM_ROWS,
            num_cols: NUM_COLS,
        }
    }
}
