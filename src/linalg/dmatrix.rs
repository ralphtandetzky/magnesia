use std::{marker::PhantomData, ops::{Add, Sub, Mul, AddAssign}};

pub trait MatrixExpr {
    type Entry;
    fn entry(&self, row: usize, col: usize) -> Self::Entry;
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;

    fn eval(&self) -> DMatrix<Self::Entry> {
        let data = (0..self.num_rows())
            .map(|r| (0..self.num_cols()).map(move |c| (r, c)))
            .flatten()
            .map(|(r, c)| self.entry(r, c))
            .collect();
        DMatrix {
            data,
            num_rows: self.num_rows(),
            num_cols: self.num_cols(),
        }
    }

    fn wrap(self) -> ExprWrapper<Self>
    where
        Self: Sized,
    {
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
}

impl<Lhs: MatrixExpr> ExprWrapper<Lhs> {
    fn apply_bin_op_elemwise<Op: BinOp<Lhs::Entry, Rhs::Entry>, Rhs: MatrixExpr>(
        self,
        op: Op,
        rhs: Rhs,
    ) -> ExprWrapper<BinOpExpr<Op, Lhs, Rhs>> {
        assert_eq!(self.num_rows(), rhs.num_rows(),
            "Number of rows on the left hand side should be equal to the number of rows on the right hand side");
        assert_eq!(self.num_cols(), rhs.num_cols(),
            "Number of columns on the left hand side should be equal to the number of columns on the right hand side");
        ExprWrapper(BinOpExpr::new(op, self.0, rhs))
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

pub struct AddOp<Lhs: Add<Rhs>, Rhs>(PhantomData<Lhs>, PhantomData<Rhs>);

impl<Lhs: Add<Rhs>, Rhs> AddOp<Lhs, Rhs> {
    fn new() -> Self {
        Self(PhantomData, PhantomData)
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

pub struct SubOp<Lhs: Sub<Rhs>, Rhs>(PhantomData<Lhs>, PhantomData<Rhs>);

impl<Lhs: Sub<Rhs>, Rhs> SubOp<Lhs, Rhs> {
    fn new() -> Self {
        Self(PhantomData, PhantomData)
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

/// A matrix type with dynamic number of rows and columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DMatrix<T> {
    data: Box<[T]>,
    num_rows: usize,
    num_cols: usize,
}

impl<T, Expr: MatrixExpr<Entry=T>> From<Expr>
    for DMatrix<T>
{
    /// Creates a dynamically sized matrix from an array.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::DMatrix;
    /// let a = DMatrix::from([[1,2],[3,4]]);
    /// ```
    fn from(expr: Expr) -> Self {
        expr.eval()
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
    let a = DMatrix::from([[1, 2], [3, 4], [5, 6]]);
    let b = DMatrix::from([[3, 3], [3, 3], [3, 3]]);
    let c = (&a + &b).eval();
    let d = DMatrix::from([[4, 5], [6, 7], [8, 9]]);
    assert_eq!(c, d);
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
    let a = DMatrix::from([[1, 2], [3, 4], [5, 6]]);
    let b = DMatrix::from([[3, 3], [3, 3], [3, 3]]);
    let c = (&a - &b).eval();
    let d = DMatrix::from([[-2, -1], [0, 1], [2, 3]]);
    assert_eq!(c, d);
}

pub struct MulDMatrix<'a, T> {
    lhs : &'a DMatrix<T>,
    rhs : &'a DMatrix<T>,
}

impl<'a, T> MulDMatrix<'a, T> {
    fn new(lhs: &'a DMatrix<T>, rhs: &'a DMatrix<T>) -> Self { Self { lhs, rhs } }
}

impl<'a, T> MatrixExpr for MulDMatrix<'a, T>
where
    T:Clone + Mul<T>,
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
    T:Clone + Mul<T>,
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
    let a = DMatrix::from([[-1, 0, 1], [2, 3, 4]]);
    let b = DMatrix::from([[0, 1], [2, 3], [4, 5]]);
    let c = (&a * &b).eval();
    let d = DMatrix::from([[4, 4], [22, 31]]);
    assert_eq!(c, d);
}

impl<T: Clone, const NUM_ROWS: usize, const NUM_COLS: usize> MatrixExpr for [[T; NUM_COLS]; NUM_ROWS]
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
}
