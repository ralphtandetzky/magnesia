use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub, SubAssign},
};

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

    /// Returns the transposed of the `self` matrix.
    fn t(self) -> ExprWrapper<TransposedExpr<Self>> {
        TransposedExpr(self).wrap()
    }

    /// Returns the conjugate transpose (also called Hermetian transpose).
    fn h(self) -> ExprWrapper<ConjugateTransposedExpr<Self>>
    where
        Self::Entry: Conj,
    {
        ConjugateTransposedExpr(self).wrap()
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

pub fn make_unary_matrix_expr<Expr, F, Out>(
    expr: Expr,
    f: F,
) -> ExprWrapper<impl MatrixExpr<Entry = Out>>
where
    Expr: MatrixExpr,
    F: Fn(Expr::Entry) -> Out,
{
    make_matrix_expr(expr.num_rows(), expr.num_cols(), move |row, col| {
        f(expr.entry(row, col))
    })
}

#[test]
fn test_make_unary_matrix_expr() {
    let a = [[1, 2, 3], [4, 5, 6]].eval();
    let c = make_unary_matrix_expr(a, |a| 2 * a + 1).eval();
    let d = [[3, 5, 7], [9, 11, 13]].eval();
    assert_eq!(c, d);
}

pub fn make_binary_matrix_expr<Lhs, Rhs, F, Out>(
    lhs: Lhs,
    rhs: Rhs,
    f: F,
) -> ExprWrapper<impl MatrixExpr<Entry = Out>>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    F: Fn(Lhs::Entry, Rhs::Entry) -> Out,
{
    assert_eq!(lhs.num_rows(), rhs.num_rows());
    assert_eq!(lhs.num_cols(), rhs.num_cols());
    make_matrix_expr(lhs.num_rows(), lhs.num_cols(), move |row, col| {
        f(lhs.entry(row, col), rhs.entry(row, col))
    })
}

#[test]
fn test_make_binary_matrix_expr() {
    let a = [[1, 2, 3], [4, 5, 6]].eval();
    let b = [[3, 1, 6], [2, 5, 4]].eval();
    let c = make_binary_matrix_expr(a, b, |a, b| a ^ b).eval();
    let d = [[1 ^ 3, 2 ^ 1, 3 ^ 6], [4 ^ 2, 5 ^ 5, 6 ^ 4]].eval();
    assert_eq!(c, d);
}

impl<Lhs: MatrixExpr> ExprWrapper<Lhs> {
    pub fn apply_bin_op_elemwise<Op: BinOp<Lhs::Entry, Rhs::Entry>, Rhs: MatrixExpr>(
        self,
        op: Op,
        rhs: Rhs,
    ) -> ExprWrapper<BinOpExpr<Op, Lhs, Rhs>> {
        assert_eq!(self.num_rows(), rhs.num_rows(),
            "Number of rows on the left hand side should be equal to the number of rows on the right hand side");
        assert_eq!(self.num_cols(), rhs.num_cols(),
            "Number of columns on the left hand side should be equal to the number of columns on the right hand side");
        BinOpExpr::new(op, self.0, rhs).wrap()
    }

    pub fn apply_bin_fn_elemwise<F: Fn(Lhs::Entry, Rhs::Entry) -> Out, Rhs: MatrixExpr, Out>(
        self,
        rhs: Rhs,
        f: F,
    ) -> ExprWrapper<impl MatrixExpr<Entry = Out>> {
        self.apply_bin_op_elemwise(BinFnOp::new(f), rhs)
    }

    pub fn mul_elemwise<Rhs: MatrixExpr>(
        self,
        rhs: Rhs,
    ) -> ExprWrapper<impl MatrixExpr<Entry = <Lhs::Entry as Mul<Rhs::Entry>>::Output>>
    where
        Lhs::Entry: Mul<Rhs::Entry>,
    {
        self.apply_bin_fn_elemwise(rhs, |x, y| x * y)
    }

    pub fn div_elemwise<Rhs: MatrixExpr>(
        self,
        rhs: Rhs,
    ) -> ExprWrapper<impl MatrixExpr<Entry = <Lhs::Entry as Div<Rhs::Entry>>::Output>>
    where
        Lhs::Entry: Div<Rhs::Entry>,
    {
        self.apply_bin_fn_elemwise(rhs, |x, y| x / y)
    }
}

pub trait BinOp<Lhs, Rhs> {
    type Output;

    fn apply(&self, lhs: Lhs, rhs: Rhs) -> Self::Output;
}

pub struct BinOpExpr<Op: BinOp<Lhs::Entry, Rhs::Entry>, Lhs: MatrixExpr, Rhs: MatrixExpr> {
    op: Op,
    lhs: Lhs,
    rhs: Rhs,
}

impl<Op: BinOp<Lhs::Entry, Rhs::Entry>, Lhs: MatrixExpr, Rhs: MatrixExpr> BinOpExpr<Op, Lhs, Rhs> {
    pub fn new(op: Op, lhs: Lhs, rhs: Rhs) -> Self {
        Self { op, lhs, rhs }
    }
}

impl<Op, Lhs, Rhs> MatrixExpr for BinOpExpr<Op, Lhs, Rhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Op: BinOp<Lhs::Entry, Rhs::Entry>,
{
    type Entry = Op::Output;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        self.op
            .apply(self.lhs.entry(row, col), self.rhs.entry(row, col))
    }

    fn num_rows(&self) -> usize {
        self.lhs.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.lhs.num_cols()
    }
}

pub struct AddOp<Lhs: Add<Rhs>, Rhs>(PhantomData<(Lhs, Rhs)>);

impl<Lhs: Add<Rhs>, Rhs> AddOp<Lhs, Rhs> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Lhs: Add<Rhs>, Rhs> BinOp<Lhs, Rhs> for AddOp<Lhs, Rhs> {
    type Output = Lhs::Output;

    fn apply(&self, lhs: Lhs, rhs: Rhs) -> Self::Output {
        lhs + rhs
    }
}

impl<Rhs, Lhs: MatrixExpr> Add<Rhs> for ExprWrapper<Lhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Lhs::Entry: Add<Rhs::Entry>,
{
    type Output = ExprWrapper<BinOpExpr<AddOp<Lhs::Entry, Rhs::Entry>, Lhs, Rhs>>;

    fn add(self, rhs: Rhs) -> Self::Output {
        self.apply_bin_op_elemwise(AddOp::new(), rhs)
    }
}

pub struct SubOp<Lhs: Sub<Rhs>, Rhs>(PhantomData<(Lhs, Rhs)>);

impl<Lhs: Sub<Rhs>, Rhs> SubOp<Lhs, Rhs> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Lhs: Sub<Rhs>, Rhs> BinOp<Lhs, Rhs> for SubOp<Lhs, Rhs> {
    type Output = Lhs::Output;

    fn apply(&self, lhs: Lhs, rhs: Rhs) -> Self::Output {
        lhs - rhs
    }
}

impl<Rhs, Lhs: MatrixExpr> Sub<Rhs> for ExprWrapper<Lhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Lhs::Entry: Sub<Rhs::Entry>,
{
    type Output = ExprWrapper<BinOpExpr<SubOp<Lhs::Entry, Rhs::Entry>, Lhs, Rhs>>;

    fn sub(self, rhs: Rhs) -> Self::Output {
        self.apply_bin_op_elemwise(SubOp::new(), rhs)
    }
}

pub struct BinFnOp<F: Fn(Lhs, Rhs) -> Out, Lhs, Rhs, Out> {
    f: F,
    _phantom: PhantomData<(Lhs, Rhs, Out)>,
}

impl<F: Fn(Lhs, Rhs) -> Out, Lhs, Rhs, Out> BinFnOp<F, Lhs, Rhs, Out> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<F: Fn(Lhs, Rhs) -> Out, Lhs, Rhs, Out> BinOp<Lhs, Rhs> for BinFnOp<F, Lhs, Rhs, Out> {
    type Output = Out;

    fn apply(&self, lhs: Lhs, rhs: Rhs) -> Self::Output {
        (self.f)(lhs, rhs)
    }
}

pub struct TransposedExpr<Expr: MatrixExpr>(Expr);

impl<Expr: MatrixExpr> MatrixExpr for TransposedExpr<Expr> {
    type Entry = Expr::Entry;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        self.0.entry(col, row)
    }

    fn num_rows(&self) -> usize {
        self.0.num_cols()
    }

    fn num_cols(&self) -> usize {
        self.0.num_rows()
    }
}

#[test]
fn test_transpose() {
    let a = [[1, 2, 3], [4, 5, 6]].t().eval();
    let b = [[1, 4], [2, 5], [3, 6]].eval();
    assert_eq!(a, b);
}

pub struct ConjugateTransposedExpr<Expr: MatrixExpr>(Expr);

impl<Expr: MatrixExpr> MatrixExpr for ConjugateTransposedExpr<Expr>
where
    Expr::Entry: Conj,
{
    type Entry = Expr::Entry;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        self.0.entry(col, row).conj()
    }

    fn num_rows(&self) -> usize {
        self.0.num_cols()
    }

    fn num_cols(&self) -> usize {
        self.0.num_rows()
    }
}

#[test]
fn test_conjugate_transpose() {
    use crate::algebra::Complex;

    let a = [[Complex::new(1, 2)], [Complex::new(3, 4)]].h().eval();
    let b = [[Complex::new(1, -2), Complex::new(3, -4)]].eval();
    assert_eq!(a, b);
}

/// A matrix type with dynamic number of rows and columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DMatrix<T> {
    data: Box<[T]>,
    num_rows: usize,
    num_cols: usize,
}

struct EyeExpr<T: One + Zero> {
    n: usize,
    _phantom: PhantomData<T>,
}

impl<T: One + Zero> EyeExpr<T> {
    fn new(n: usize) -> Self {
        Self {
            n,
            _phantom: PhantomData,
        }
    }
}

impl<T: One + Zero> MatrixExpr for EyeExpr<T> {
    type Entry = T;

    fn entry(&self, row: usize, col: usize) -> Self::Entry {
        if row == col {
            T::one()
        } else {
            T::zero()
        }
    }

    fn num_rows(&self) -> usize {
        self.n
    }

    fn num_cols(&self) -> usize {
        self.n
    }
}

impl<T: One + Zero> DMatrix<T> {
    /// Creates a unit matrix expression of size $n\times n$.
    ///
    /// This is a matrix which is $1$ on the diagonal and $0$ everywhere else.
    pub fn eye(n: usize) -> ExprWrapper<impl MatrixExpr<Entry = T>> {
        EyeExpr::new(n).wrap()
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

impl<T> Index<[usize; 2]> for DMatrix<T> {
    type Output = T;

    fn index(&self, row_and_col: [usize; 2]) -> &Self::Output {
        &self.data[row_and_col[0] * self.num_cols + row_and_col[1]]
    }
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
    T: Clone,
    ExprWrapper<Self>: Add<Rhs>,
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
    T: Clone + Mul<T>,
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
    T: Clone + Mul<T>,
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
        T: Clone + Mul<Lhs::Entry>,
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
        T: Clone + Div<Lhs::Entry>,
        Lhs: 'a,
    {
        self.wrap().div_elemwise(lhs)
    }
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
