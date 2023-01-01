use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub, SubAssign};

use crate::algebra::{Conj, One, Zero};

/// A vector-like interface.
pub trait VectorExpr: Sized {
    /// The element type of the vector.
    type Entry;

    /// Returns an entry of the vector.
    fn entry(&self, index: usize) -> Self::Entry;

    /// Returns the number of elements of the vector.
    fn len(&self) -> usize;

    /// Returns whether the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluates all entries of the vector and stores them in a [`DVector`].
    fn eval(self) -> DVector<Self::Entry> {
        DVector((0..self.len()).map(|index| self.entry(index)).collect())
    }

    /// Wraps the vector expression into an [`ExprWrapper`].
    fn wrap(self) -> VectorExprWrapper<Self> {
        VectorExprWrapper(self)
    }
}

pub struct VectorExprWrapper<Expr: VectorExpr>(Expr);

impl<Expr: VectorExpr> VectorExpr for VectorExprWrapper<Expr> {
    type Entry = Expr::Entry;

    fn entry(&self, index: usize) -> Self::Entry {
        self.0.entry(index)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn eval(self) -> DVector<Self::Entry> {
        self.0.eval()
    }
}

pub fn make_vector_expr<F, Out>(len: usize, f: F) -> VectorExprWrapper<impl VectorExpr<Entry = Out>>
where
    F: Fn(usize) -> Out,
{
    struct FnVectorExpr<F_, Out_>(F_, usize)
    where
        F_: Fn(usize) -> Out_;

    impl<F_, Out_> VectorExpr for FnVectorExpr<F_, Out_>
    where
        F_: Fn(usize) -> Out_,
    {
        type Entry = Out_;

        fn entry(&self, index: usize) -> Self::Entry {
            (self.0)(index)
        }

        fn len(&self) -> usize {
            self.1
        }
    }

    FnVectorExpr(f, len).wrap()
}

#[test]
fn test_make_vector_expr() {
    let v = make_vector_expr(5, |index| index * index).eval();
    let u = [0, 1, 4, 9, 16].eval();
    assert_eq!(u, v);
}

impl<Lhs: VectorExpr> VectorExprWrapper<Lhs> {
    pub fn map<F, Out>(self, f: F) -> VectorExprWrapper<impl VectorExpr<Entry = Out>>
    where
        F: Fn(Lhs::Entry) -> Out,
    {
        make_vector_expr(self.0.len(), move |index| f(self.0.entry(index)))
    }
}

#[test]
fn test_map_vector_expr() {
    let a = [1, 5, 3, 2, 4].wrap().map(|x| x * x).eval();
    let b = [1, 25, 9, 4, 16].eval();
    assert_eq!(a, b);
}

impl<Lhs: VectorExpr> VectorExprWrapper<Lhs> {
    pub fn zip<Rhs>(
        self,
        rhs: Rhs,
    ) -> VectorExprWrapper<impl VectorExpr<Entry = (Lhs::Entry, Rhs::Entry)>>
    where
        Rhs: VectorExpr,
    {
        assert_eq!(self.len(), rhs.len());
        make_vector_expr(self.0.len(), move |index| {
            (self.0.entry(index), rhs.entry(index))
        })
    }
}

#[test]
fn test_zip_vector_expr() {
    let a = [1, 2, 3, 4, 5, 6]
        .eval()
        .wrap()
        .zip([7, 8, 9, 10, 11, 12].eval())
        .eval();
    let b = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)].eval();
    assert_eq!(a, b);
}

impl<Rhs, Lhs> Add<Rhs> for VectorExprWrapper<Lhs>
where
    Lhs: VectorExpr,
    Rhs: VectorExpr,
    Lhs::Entry: Add<Rhs::Entry>,
{
    type Output = VectorExprWrapper<AddExpr<Lhs, Rhs>>;

    fn add(self, rhs: Rhs) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        AddExpr(self.0, rhs).wrap()
    }
}

#[test]
fn test_add_expr_wrapper() {
    let a = [1, 2, 3, 4, 5, 6].wrap();
    let b = [2, 2, 2, 3, 3, 3].wrap();
    let c = a + b;
    let d = [3, 4, 5, 7, 8, 9].wrap();
    assert_eq!(c.eval(), d.eval());
}

pub struct AddExpr<Lhs, Rhs>(Lhs, Rhs);

impl<Lhs, Rhs> VectorExpr for AddExpr<Lhs, Rhs>
where
    Lhs: VectorExpr,
    Rhs: VectorExpr,
    Lhs::Entry: Add<Rhs::Entry>,
{
    type Entry = <Lhs::Entry as Add<Rhs::Entry>>::Output;

    fn entry(&self, index: usize) -> Self::Entry {
        self.0.entry(index) + self.1.entry(index)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<Rhs, Lhs> Sub<Rhs> for VectorExprWrapper<Lhs>
where
    Lhs: VectorExpr,
    Rhs: VectorExpr,
    Lhs::Entry: Sub<Rhs::Entry>,
{
    type Output = VectorExprWrapper<SubExpr<Lhs, Rhs>>;

    fn sub(self, rhs: Rhs) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        SubExpr(self.0, rhs).wrap()
    }
}

#[test]
fn test_sub_expr_wrapper() {
    let a = [1, 2, 3, 4, 5, 6].wrap();
    let b = [2, 2, 2, 3, 3, 3].wrap();
    let c = a - b;
    let d = [-1, 0, 1, 1, 2, 3].wrap();
    assert_eq!(c.eval(), d.eval());
}

pub struct SubExpr<Lhs, Rhs>(Lhs, Rhs);

impl<Lhs, Rhs> VectorExpr for SubExpr<Lhs, Rhs>
where
    Lhs: VectorExpr,
    Rhs: VectorExpr,
    Lhs::Entry: Sub<Rhs::Entry>,
{
    type Entry = <Lhs::Entry as Sub<Rhs::Entry>>::Output;

    fn entry(&self, index: usize) -> Self::Entry {
        self.0.entry(index) - self.1.entry(index)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<Lhs: VectorExpr> VectorExprWrapper<Lhs> {
    pub fn mul_elemwise<Rhs: VectorExpr>(
        self,
        rhs: Rhs,
    ) -> VectorExprWrapper<impl VectorExpr<Entry = <Lhs::Entry as Mul<Rhs::Entry>>::Output>>
    where
        Lhs::Entry: Mul<Rhs::Entry>,
    {
        self.zip(rhs).map(|(x, y)| x * y)
    }
}

#[test]
fn test_mul_elemwise_vector_expr_wrapper() {
    let u = [1, 2, 3, 4, 5].wrap();
    let v = [0, 1, 2, 3, 4].wrap();
    let uv = u.mul_elemwise(v);
    assert_eq!([0, 2, 6, 12, 20].eval(), uv.eval());
}

impl<Lhs: VectorExpr> VectorExprWrapper<Lhs> {
    pub fn div_elemwise<Rhs: VectorExpr>(
        self,
        rhs: Rhs,
    ) -> VectorExprWrapper<impl VectorExpr<Entry = <Lhs::Entry as Div<Rhs::Entry>>::Output>>
    where
        Lhs::Entry: Div<Rhs::Entry>,
    {
        self.zip(rhs).map(|(x, y)| x / y)
    }
}

#[test]
fn test_div_elemwise_vector_expr_wrapper() {
    let uv = [0, 2, 6, 12, 20].wrap();
    let u = [1, 2, 3, 4, 5].wrap();
    let v = uv.div_elemwise(u);
    assert_eq!([0, 1, 2, 3, 4].eval(), v.eval());
}

impl<Lhs: VectorExpr> VectorExprWrapper<Lhs> {
    pub fn conj(self) -> VectorExprWrapper<impl VectorExpr<Entry = Lhs::Entry>>
    where
        Lhs::Entry: Conj,
    {
        self.map(|x| x.conj())
    }
}

#[test]
fn test_conjugate_vector_expr_wrapper() {
    use crate::algebra::Complex;
    let v = [Complex::new(1, 2), Complex::new(3, 4), Complex::new(5, 6)]
        .wrap()
        .conj()
        .eval();
    assert_eq!(
        [
            Complex::new(1, -2),
            Complex::new(3, -4),
            Complex::new(5, -6)
        ]
        .eval(),
        v
    );
}

/// A vector type with dynamic number of entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DVector<T>(Box<[T]>);

impl<T: Clone> VectorExpr for DVector<T> {
    type Entry = T;

    fn entry(&self, index: usize) -> Self::Entry {
        self.0[index].clone()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn eval(self) -> DVector<Self::Entry> {
        self
    }
}

impl<T: Clone> VectorExpr for &DVector<T> {
    type Entry = T;

    fn entry(&self, index: usize) -> Self::Entry {
        (*self).0[index].clone()
    }

    fn len(&self) -> usize {
        (*self).0.len()
    }

    fn eval(self) -> DVector<Self::Entry> {
        self.clone()
    }
}

#[test]
fn test_dvector_entry() {
    let v = [1, 2, 4].eval();
    assert_eq!(v.entry(0), 1);
    assert_eq!(v.entry(1), 2);
    assert_eq!(v.entry(2), 4);
}

#[test]
fn test_dvector_len() {
    let v = [1, 2, 4].eval();
    assert_eq!(v.len(), 3);
}

#[test]
fn test_dvector_eval() {
    let arr = [1, 2, 4];
    let u = arr.eval();
    let v = arr.eval().eval();
    assert_eq!(u, v);
}

impl<T> DVector<T>
where
    T: Zero,
{
    /// Returns a vector expression filled with zeros.
    pub fn zeros(len: usize) -> VectorExprWrapper<impl VectorExpr<Entry = T>> {
        make_vector_expr(len, |_| T::zero())
    }
}

#[test]
fn test_dvector_zeros() {
    assert_eq!(DVector::<i32>::zeros(5).eval(), [0, 0, 0, 0, 0].eval());
}

impl<T> DVector<T>
where
    T: One,
{
    /// Returns a vector expression filled with ones.
    pub fn ones(len: usize) -> VectorExprWrapper<impl VectorExpr<Entry = T>> {
        make_vector_expr(len, |_| T::one())
    }
}

#[test]
fn test_vector_ones() {
    assert_eq!(DVector::<i32>::ones(5).eval(), [1, 1, 1, 1, 1,].eval());
}

impl<T> DVector<T>
where
    T: Clone,
{
    /// Returns a vector expression filled with the passed value `val`.
    pub fn same(len: usize, val: T) -> VectorExprWrapper<impl VectorExpr<Entry = T>> {
        make_vector_expr(len, move |_| val.clone())
    }
}

#[test]
fn test_vector_same() {
    assert_eq!(
        DVector::<i32>::same(5, 42).eval(),
        [42, 42, 42, 42, 42].eval()
    );
}

impl<T> Index<usize> for DVector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[test]
fn test_index_dvector() {
    let a = [1, 2, 3, 7, 8].eval();
    assert_eq!(a[0], 1);
    assert_eq!(a[1], 2);
    assert_eq!(a[2], 3);
    assert_eq!(a[3], 7);
    assert_eq!(a[4], 8);
}

impl<T> IndexMut<usize> for DVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut <Self as Index<usize>>::Output {
        &mut self.0[index]
    }
}

#[test]
fn test_index_mut_dvector() {
    let mut a = [0, 0, 0, 0, 0].eval();
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 7;
    a[4] = 8;
    assert_eq!(a, [1, 2, 3, 7, 8].eval());
    a[2] = 42;
    assert_eq!(a, [1, 2, 42, 7, 8].eval());
}

impl<T, Rhs> Add<Rhs> for &DVector<T>
where
    T: Clone,
    VectorExprWrapper<Self>: Add<Rhs>,
{
    type Output = <VectorExprWrapper<Self> as Add<Rhs>>::Output;

    fn add(self, rhs: Rhs) -> Self::Output {
        self.wrap() + rhs
    }
}

#[test]
fn test_dvec_add() {
    let a = [1, 2, 3].eval();
    let b = (&a + [2, 2, 2].wrap()).eval();
    assert_eq!(b, [3, 4, 5].eval());
}

impl<T, Rhs> AddAssign<Rhs> for DVector<T>
where
    T: AddAssign<Rhs::Entry> + Clone,
    Rhs: VectorExpr,
{
    fn add_assign(&mut self, rhs: Rhs) {
        for index in 0..self.len() {
            self.0[index] += rhs.entry(index);
        }
    }
}

#[test]
fn test_add_assign_dvector() {
    let mut a = [1, 2, 3, 4, 5].eval();
    a += [2, 2, 2, 2, 2].wrap();
    assert_eq!(a, [3, 4, 5, 6, 7].eval());
}

impl<T, Rhs> Sub<Rhs> for &DVector<T>
where
    T: Clone,
    VectorExprWrapper<Self>: Sub<Rhs>,
{
    type Output = <VectorExprWrapper<Self> as Sub<Rhs>>::Output;

    fn sub(self, rhs: Rhs) -> Self::Output {
        self.wrap() - rhs
    }
}

#[test]
fn test_dvec_sub() {
    let a = [1, 2, 3].eval();
    let b = (&a + [2, 2, 2].wrap()).eval();
    assert_eq!(b, [3, 4, 5].eval());
}

impl<T, Rhs> SubAssign<Rhs> for DVector<T>
where
    T: SubAssign<Rhs::Entry> + Clone,
    Rhs: VectorExpr,
{
    fn sub_assign(&mut self, rhs: Rhs) {
        for index in 0..self.len() {
            self.0[index] -= rhs.entry(index);
        }
    }
}

#[test]
fn test_sub_assign_dvector() {
    let mut a = [1, 2, 3, 4, 5].eval();
    a += [2, 2, 2, 2, 2].wrap();
    assert_eq!(a, [3, 4, 5, 6, 7].eval());
}

impl<T> Mul for DVector<T>
where
    T: Add<Output = T> + Clone + Conj + Mul<Output = T> + Zero,
{
    type Output = T;

    fn mul(self, rhs: Self) -> T {
        &self * &rhs
    }
}

#[test]
fn test_dvector_mul() {
    let u = [1, 2, 3].eval();
    let v = [2, 3, 4].eval();
    assert_eq!(u * v, 1 * 2 + 2 * 3 + 3 * 4);
}

impl<T> Mul for &DVector<T>
where
    T: Add<Output = T> + Clone + Conj + Mul<Output = T> + Zero,
{
    type Output = T;

    fn mul(self, rhs: Self) -> T {
        self.0
            .iter()
            .zip(rhs.0.iter())
            .map(|(lhs, rhs)| lhs.clone().conj() * rhs.clone())
            .fold(<T as Mul>::Output::zero(), |lhs, rhs| lhs + rhs)
    }
}

#[test]
fn test_ref_dvector_mul() {
    let u = [1, 2, 3].eval();
    let v = [2, 3, 4].eval();
    assert_eq!(&u * &v, 1 * 2 + 2 * 3 + 3 * 4);
}

impl<T> DVector<T> {
    /// Multiplies a `DVector` with another vector expression element-wise.
    ///
    /// # Example
    ///
    /// ```
    /// # use magnesia::linalg::VectorExpr;
    /// let a = [1,2,3].eval();
    /// let b = a.mul_elemwise([0,1,2].wrap()).eval();
    /// let c = [0,2,6].eval();
    /// assert_eq!(b, c);
    /// ```
    pub fn mul_elemwise<'a, Lhs: VectorExpr>(
        &'a self,
        lhs: Lhs,
    ) -> VectorExprWrapper<impl VectorExpr<Entry = T::Output> + 'a>
    where
        T: Clone + Mul<Lhs::Entry>,
        Lhs: 'a,
    {
        self.wrap().mul_elemwise(lhs)
    }
}

impl<T> DVector<T> {
    /// Divides a `DVector` by another vector expression element-wise.
    ///
    /// # Example
    ///
    /// ```
    /// # use magnesia::linalg::VectorExpr;
    /// let a = [1,4,15].eval();
    /// let b = a.div_elemwise([1,2,3].wrap()).eval();
    /// let c = [1,2,5].eval();
    /// assert_eq!(b, c);
    /// ```
    pub fn div_elemwise<'a, Lhs: VectorExpr>(
        &'a self,
        lhs: Lhs,
    ) -> VectorExprWrapper<impl VectorExpr<Entry = T::Output> + 'a>
    where
        T: Clone + Div<Lhs::Entry>,
        Lhs: 'a,
    {
        self.wrap().div_elemwise(lhs)
    }
}

impl<T> DVector<T>
where
    T: Clone + Conj,
{
    /// Returns the conjugated vector as a vector expression.
    #[allow(clippy::needless_lifetimes)] // false positive
    pub fn conj<'a>(&'a self) -> VectorExprWrapper<impl VectorExpr<Entry = T> + 'a> {
        self.wrap().conj()
    }
}

#[test]
fn test_dvector_conjugate() {
    use crate::algebra::Complex;
    let a = [Complex::new(1, 2), Complex::new(3, 4), Complex::new(5, 6)]
        .eval()
        .conj()
        .eval();
    let b = [
        Complex::new(1, -2),
        Complex::new(3, -4),
        Complex::new(5, -6),
    ]
    .eval();
    assert_eq!(a, b);
}

impl<T: Clone> VectorExpr for &[T] {
    type Entry = T;

    fn entry(&self, index: usize) -> Self::Entry {
        self[index].clone()
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn eval(self) -> DVector<Self::Entry> {
        DVector(self.iter().cloned().collect())
    }
}
