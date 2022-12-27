use std::{marker::PhantomData, ops::Add};

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
        lhs.add(rhs)
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

/// A matrix type with dynamic number of rows and columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DMatrix<T> {
    data: Box<[T]>,
    num_rows: usize,
    num_cols: usize,
}

impl<T, const NUM_ROWS: usize, const NUM_COLS: usize> From<[[T; NUM_COLS]; NUM_ROWS]>
    for DMatrix<T>
{
    /// Creates a dynamically sized matrix from an array.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::DMatrix;
    /// let a = DMatrix::from([[1,2],[3,4]]);
    /// ```
    fn from(coefficients: [[T; NUM_COLS]; NUM_ROWS]) -> Self {
        let data = coefficients.into_iter().flatten().collect();
        Self {
            data,
            num_rows: NUM_ROWS,
            num_cols: NUM_COLS,
        }
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
    let a = DMatrix::from([[1, 2], [3, 4]]);
    let b = DMatrix::from([[3, 3], [3, 3]]);
    let c = (&a + &b).eval();
    let d = DMatrix::from([[4, 5], [6, 7]]);
    assert_eq!(c, d);
}
