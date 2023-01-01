use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub, SubAssign};

use crate::algebra::{Conj, One, Zero};

use super::{dvector::VectorExprWrapper, make_vector_expr, VectorExpr};

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

    /// Wraps the matrix expression into an [`MatrixExprWrapper`].
    fn wrap(self) -> MatrixExprWrapper<Self> {
        MatrixExprWrapper(self)
    }
}

/// A wrapper type for matrix expressions.
///
/// This `struct` wraps anything that implements the trait [`MatrixExpr`]
/// and forwards any function calls.
/// Additionally, it implements a large number of functions and operators and
/// thus extends the interface of the wrapped object.
///
/// # Design Rationale
///
/// Unfortunately, this cannot be done in the [`MatrixExpr`] trait directly,
/// because traits like [`std::ops::Add`] cannot be implemented for all types
/// satisfying the [`MatrixExpr`] trait, since it is a foreign trait.
/// Also it is currently not possible to specify return types as
/// `impl MatrixExpr<Entry=T>` in default implementations of a trait,
/// but it is possible to do so for the `struct MatrixExpressionWrapped`.
/// For some implementations the return type cannot be spelled out, because
/// they involve lambdas defined inside the function implementation.
/// This is particularly true for the function [`make_matrix_expr()`] which
/// is widely used to implement other functions conveniently.
/// It would certainly be possible to avoid all lambdas and encapsulate them
/// into extra types, but this would be much more verbose.
/// Thus some functions are implemented with `impl MatrixExpr` return types
/// and those cannot be used inside trait default function implementations.
/// This is why the library's implementation relies on the `MatrixExprWrapper`
/// type.
pub struct MatrixExprWrapper<T: MatrixExpr>(T);

impl<T: MatrixExpr> MatrixExpr for MatrixExprWrapper<T> {
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

/// Creates a matrix expression from a function which returns the entries of
/// the matrix.
///
/// The matrix will take the form
///
/// $$
///     \begin{pmatrix}
///         f( 0 , 0 ) & \cdots & f( 0 ,c-1) \\
///         \vdots     & \ddots & \vdots     \\
///         f(r-1, 0 ) & \cdots & f(r-1,c-1)
///     \end{pmatrix}
/// $$
///
/// where $r$ is the number of rows and $c$ is the number of columns.
// TODO: Check is the formula above renders correctly.
pub fn make_matrix_expr<F, Out>(
    num_rows: usize,
    num_cols: usize,
    f: F,
) -> MatrixExprWrapper<impl MatrixExpr<Entry = Out>>
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

impl<Lhs: MatrixExpr> MatrixExprWrapper<Lhs> {
    /// Applies a functor to each element of a matrix and returns the obtained
    /// matrix.
    ///
    /// Given the matrix
    /// $$
    ///     \begin{pmatrix}
    ///         a_{11} & \cdots & a_{1n} \\
    ///         \vdots & \ddots & \vdots \\
    ///         a_{m1} & \cdots & a_{mn}
    ///     \end{pmatrix}
    /// $$
    /// this function will return the matrix
    /// $$
    ///     \begin{pmatrix}
    ///         f(a_{11}) & \cdots & f(a_{1n}) \\
    ///          \vdots   & \ddots &  \vdots   \\
    ///         f(a_{m1}) & \cdots & f(a_{mn})
    ///     \end{pmatrix}
    /// $$
    pub fn map<F, Out>(self, f: F) -> MatrixExprWrapper<impl MatrixExpr<Entry = Out>>
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

impl<Lhs: MatrixExpr> MatrixExprWrapper<Lhs> {
    /// Given a two matrices of the same dimensions, this function will
    /// return the matrix of pairs of matching elements.
    ///
    /// In other words, the matrices
    /// $$
    ///     \begin{pmatrix}
    ///         a_{11} & \cdots & a_{1n} \\
    ///         \vdots & \ddots & \vdots \\
    ///         a_{m1} & \cdots & a_{mn}
    ///     \end{pmatrix}, \qquad
    ///     \begin{pmatrix}
    ///         b_{11} & \cdots & b_{1n} \\
    ///         \vdots & \ddots & \vdots \\
    ///         b_{m1} & \cdots & b_{mn}
    ///     \end{pmatrix}
    /// $$
    /// will be mapped to
    /// $$
    ///     \begin{pmatrix}
    ///         (a_{11}, b_{11}) & \cdots & (a_{1n}, b_{1n}) \\
    ///              \vdots      & \ddots &      \vdots      \\
    ///         (a_{m1}, b_{m1}) & \cdots & (a_{mn}, b_{mn})
    ///     \end{pmatrix}.
    /// $$
    // TODO: Check if the above formulas render correctly.
    pub fn zip<Rhs>(
        self,
        rhs: Rhs,
    ) -> MatrixExprWrapper<impl MatrixExpr<Entry = (Lhs::Entry, Rhs::Entry)>>
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

impl<Rhs, Lhs> Add<Rhs> for MatrixExprWrapper<Lhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Lhs::Entry: Add<Rhs::Entry>,
{
    type Output = MatrixExprWrapper<AddExpr<Lhs, Rhs>>;

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

impl<Rhs, Lhs> Sub<Rhs> for MatrixExprWrapper<Lhs>
where
    Lhs: MatrixExpr,
    Rhs: MatrixExpr,
    Lhs::Entry: Sub<Rhs::Entry>,
{
    type Output = MatrixExprWrapper<SubExpr<Lhs, Rhs>>;

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

impl<Lhs: MatrixExpr> MatrixExprWrapper<Lhs> {
    /// Multiplies two matrices element-wise.
    ///
    /// In other words, the matrices
    /// $$
    ///     \begin{pmatrix}
    ///         a_{11} & \cdots & a_{1n} \\
    ///         \vdots & \ddots & \vdots \\
    ///         a_{m1} & \cdots & a_{mn}
    ///     \end{pmatrix}, \qquad
    ///     \begin{pmatrix}
    ///         b_{11} & \cdots & b_{1n} \\
    ///         \vdots & \ddots & \vdots \\
    ///         b_{m1} & \cdots & b_{mn}
    ///     \end{pmatrix}
    /// $$
    /// will be mapped to
    /// $$
    ///     \begin{pmatrix}
    ///         a_{11}\cdot b_{11}) & \cdots & a_{1n}\cdot b_{1n}) \\
    ///               \vdots        & \ddots &       \vdots        \\
    ///         a_{m1}\cdot b_{m1}) & \cdots & a_{mn}\cdot b_{mn})
    ///     \end{pmatrix}.
    /// $$
    // TODO: Check if the above formulas render correctly.
    pub fn mul_elemwise<Rhs: MatrixExpr>(
        self,
        rhs: Rhs,
    ) -> MatrixExprWrapper<impl MatrixExpr<Entry = <Lhs::Entry as Mul<Rhs::Entry>>::Output>>
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

impl<Lhs: MatrixExpr> MatrixExprWrapper<Lhs> {
    /// Divides two matrices element-wise.
    ///
    /// In other words, the matrices
    /// $$
    ///     \begin{pmatrix}
    ///         a_{11} & \cdots & a_{1n} \\
    ///         \vdots & \ddots & \vdots \\
    ///         a_{m1} & \cdots & a_{mn}
    ///     \end{pmatrix}, \qquad
    ///     \begin{pmatrix}
    ///         b_{11} & \cdots & b_{1n} \\
    ///         \vdots & \ddots & \vdots \\
    ///         b_{m1} & \cdots & b_{mn}
    ///     \end{pmatrix}
    /// $$
    /// will be mapped to
    /// $$
    ///     \begin{pmatrix}
    ///         a_{11}/b_{11}) & \cdots & a_{1n}/b_{1n}) \\
    ///             \vdots     & \ddots &     \vdots     \\
    ///         a_{m1}/b_{m1}) & \cdots & a_{mn}/b_{mn})
    ///     \end{pmatrix}.
    /// $$
    // TODO: Check if the above formulas render correctly.
    pub fn div_elemwise<Rhs: MatrixExpr>(
        self,
        rhs: Rhs,
    ) -> MatrixExprWrapper<impl MatrixExpr<Entry = <Lhs::Entry as Div<Rhs::Entry>>::Output>>
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

impl<Expr> MatrixExprWrapper<Expr>
where
    Expr: MatrixExpr,
{
    /// Returns the transposed matrix of a given matrix.
    ///
    /// If you are dealing with complex valued matrices, then the Hermitian
    /// transpose [`MatrixExprWrapper::h()`] may be what you need instead,
    /// because it also performs complex conjugation on the matrix elements.
    pub fn t(self) -> MatrixExprWrapper<impl MatrixExpr<Entry = Expr::Entry>> {
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

impl<Expr> MatrixExprWrapper<Expr>
where
    Expr: MatrixExpr,
    Expr::Entry: Conj,
{
    /// Returns the Hermitian transposed matrix of a given matrix.
    ///
    /// The matrix will be mirrored at the diagonal and the entries will be
    /// conjugated. If you do not want complex conjugation, then the plain
    /// transpose of a matrix can be obtained by the method
    /// [`MatrixExprWrapper<Expr>::t()`].
    /// For non-complex numbers conjugation is a no-op and does not hurt.
    pub fn h(self) -> MatrixExprWrapper<impl MatrixExpr<Entry = Expr::Entry>> {
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

impl<Expr> MatrixExprWrapper<Expr>
where
    Expr: MatrixExpr,
{
    /// Returns the vector containing the diagonal elements of a matrix.
    ///
    /// The matrix does not need to be square for this function to work.
    ///
    /// Given the matrix
    /// $$
    ///     \begin{pmatrix}
    ///         a_{11} & \cdots & a_{1n} \\
    ///         \vdots & \ddots & \vdots \\
    ///         a_{m1} & \cdots & a_{mn}
    ///     \end{pmatrix}
    /// $$
    /// the vector
    /// $$
    ///     \begin{pmatrix}
    ///         a_{11}  \\
    ///         a_{22}  \\
    ///         \vdots  \\
    ///         a_{kk}
    ///     \end{pmatrix}
    /// $$
    /// where $k$ is the minimum of $m$ and $n$.
    pub fn diag(self) -> VectorExprWrapper<impl VectorExpr<Entry = Expr::Entry>> {
        make_vector_expr(self.0.num_rows().min(self.0.num_cols()), move |index| {
            self.0.entry(index, index)
        })
    }
}

#[test]
fn test_diag_expr_wrapper() {
    let v = [[1, 2, 3], [4, 5, 6]].wrap().diag();
    assert_eq!([1, 5].eval(), v.eval());
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
    pub fn eye(n: usize) -> MatrixExprWrapper<impl MatrixExpr<Entry = T>> {
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
    pub fn zeros(
        num_rows: usize,
        num_cols: usize,
    ) -> MatrixExprWrapper<impl MatrixExpr<Entry = T>> {
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

impl<T> DMatrix<T>
where
    T: One,
{
    /// Returns a matrix expression filled with ones.
    pub fn ones(num_rows: usize, num_cols: usize) -> MatrixExprWrapper<impl MatrixExpr<Entry = T>> {
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
    ) -> MatrixExprWrapper<impl MatrixExpr<Entry = T>> {
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
    Rhs: MatrixExpr,
    MatrixExprWrapper<Self>: Add<Rhs>,
    T: Clone,
{
    type Output = <MatrixExprWrapper<Self> as Add<Rhs>>::Output;

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
    MatrixExprWrapper<Self>: Sub<Rhs>,
{
    type Output = <MatrixExprWrapper<Self> as Sub<Rhs>>::Output;

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
    ) -> MatrixExprWrapper<impl MatrixExpr<Entry = T::Output> + 'a>
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
    ) -> MatrixExprWrapper<impl MatrixExpr<Entry = T::Output> + 'a>
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
    #[allow(clippy::needless_lifetimes)] // false positive
    pub fn t<'a>(&'a self) -> MatrixExprWrapper<impl MatrixExpr<Entry = T> + 'a> {
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
    #[allow(clippy::needless_lifetimes)] // false positive
    pub fn h<'a>(&'a self) -> MatrixExprWrapper<impl MatrixExpr<Entry = T> + 'a> {
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

impl<T> DMatrix<T>
where
    T: Clone,
{
    /// Returns the vector expression consisting of the diagonal elements of
    /// the matrix.
    ///
    /// # Example
    /// ```
    /// # use magnesia::linalg::MatrixExpr;
    /// # use magnesia::linalg::VectorExpr;
    /// let v = [[1, 2, 3],
    ///          [4, 5, 6]].eval().diag().eval();
    /// assert_eq!([1, 5].eval(), v);
    /// ```
    #[allow(clippy::needless_lifetimes)] // False positive warning
    pub fn diag<'a>(&'a self) -> VectorExprWrapper<impl VectorExpr<Entry = T> + 'a> {
        self.wrap().diag()
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
