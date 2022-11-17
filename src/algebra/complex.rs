use super::{conj::Conj, Field, MulRefs, One, Ring, Sqrt, Zero};
use crate::functions::{Cos, Exp, Sin};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Complex numbers consisting of real and imaginary part.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Complex<T> {
    /// The real part of the complex number.
    pub re: T,
    /// The imaginary part of the complex number.
    pub im: T,
}

impl<T> Complex<T> {
    /// Creates a new [`Complex<T>`] from the real part and the imaginary part.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// let z = Complex::new(1.0, 2.0);
    /// assert_eq!(z.re, 1.0);
    /// assert_eq!(z.im, 2.0);
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T> Conj for Complex<T>
where
    T: Neg<Output = T>,
{
    /// Computes the complex conjugate of a complex number.
    ///
    /// # Example
    /// ```
    /// # use::magnesia::algebra::Complex;
    /// # use::magnesia::algebra::Conj;
    /// let z = Complex::new(1.0, 2.0);
    /// assert_eq!(z.conj(), Complex::new(1.0, -2.0));
    /// ```
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl<T> Complex<T>
where
    for<'a> T: AddAssign<&'a T> + MulRefs,
{
    /// Computes the square of the absolute value of a complex number.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// let z = Complex::new(1, 2);
    /// let a = z.sqr_norm();
    /// assert_eq!(a, 5);
    /// ```
    pub fn sqr_norm(&self) -> T {
        let mut p1 = self.re.mul_refs(&self.re);
        let p2 = self.im.mul_refs(&self.im);
        p1 += &p2;
        p1
    }
}

impl<T> Complex<T>
where
    for<'a> T: AddAssign<&'a T> + MulRefs + Sqrt,
{
    /// Computes the absolute value of a complex number.
    ///
    /// # Note on Accuracy
    ///
    /// If the result squared evaluates to infinity or zero in the range of
    /// the type `T`, then the result may overflow to infinity or underflow
    /// to zero respectively.
    /// Concretely, for `f32` values above `1.844674353e19` may be rounded up
    /// to infinity and values below `1.084202172e-19` may be rounded down to
    /// zero.
    /// For `f64` values above `1.340780793e152` may be rounded up to infinity
    /// and values below `1.491668146e-152` may be rounded down to zero.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// let z = Complex::new(3f32, 4f32);
    /// let a = z.abs();
    /// assert_eq!(a, 5f32);
    /// ```
    pub fn abs(&self) -> T {
        self.sqr_norm().sqrt()
    }
}

impl<T: Zero> Zero for Complex<T> {
    fn zero() -> Self {
        Complex::new(T::zero(), T::zero())
    }
}

#[test]
fn test_complex_zero_for_f32() {
    let z = Complex::<f32>::zero();
    assert_eq!(z.re, 0.0);
    assert_eq!(z.im, 0.0);
}

impl<T: Zero + One> One for Complex<T> {
    fn one() -> Self {
        Complex::new(T::one(), T::zero())
    }
}

#[test]
fn test_complex_one_for_i32() {
    let z = Complex::<i32>::one();
    assert_eq!(z.re, 1);
    assert_eq!(z.im, 0);
}

impl<U, T: Add<U>> Add<Complex<U>> for Complex<T> {
    type Output = Complex<<T as Add<U>>::Output>;

    fn add(self, other: Complex<U>) -> Self::Output {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

#[test]
fn test_complex_add_ints() {
    let a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    let c = a + b;
    assert_eq!(c, Complex::new(4, 6));
}

impl<T: AddAssign<U>, U> AddAssign<Complex<U>> for Complex<T> {
    fn add_assign(&mut self, other: Complex<U>) {
        self.re += other.re;
        self.im += other.im;
    }
}

#[test]
fn test_complex_add_assign_ints() {
    let mut a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    a += b;
    assert_eq!(a, Complex::new(4, 6));
}

impl<T> AddAssign<&Complex<T>> for Complex<T>
where
    for<'a> T: AddAssign<&'a T>,
{
    fn add_assign(&mut self, other: &Self) {
        self.re += &other.re;
        self.im += &other.im;
    }
}

#[test]
fn test_add_assign_ref_complex_ints() {
    let mut a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    a += &b;
    assert_eq!(a, Complex::new(4, 6));
}

impl<U, T: Sub<U>> Sub<Complex<U>> for Complex<T> {
    type Output = Complex<<T as Sub<U>>::Output>;

    fn sub(self, other: Complex<U>) -> Self::Output {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

#[test]
fn test_sub_for_complex_ints() {
    let a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    let c = a - b;
    assert_eq!(c, Complex::new(-2, -2));
}

impl<T: SubAssign<U>, U> SubAssign<Complex<U>> for Complex<T> {
    fn sub_assign(&mut self, other: Complex<U>) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

#[test]
fn test_sub_assign_for_complex_ints() {
    let mut a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    a -= b;
    assert_eq!(a, Complex::new(-2, -2));
}

impl<T> SubAssign<&Self> for Complex<T>
where
    for<'a> T: SubAssign<&'a T>,
{
    fn sub_assign(&mut self, other: &Self) {
        self.re -= &other.re;
        self.im -= &other.im;
    }
}

#[test]
fn test_sub_assign_for_ref_complex_ints() {
    let mut a = Complex::new(1, 2);
    let b = Complex::new(3, 4);
    a -= &b;
    assert_eq!(a, Complex::new(-2, -2));
}

impl<T> Mul for Complex<T>
where
    T: Ring,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        &self * &other
    }
}

#[test]
fn test_mul_for_complex_ints() {
    let a = Complex::new(1, 2);
    let b = a * a;
    assert_eq!(b, Complex::new(-3, 4));
}

impl<T: Ring> MulAssign for Complex<T> {
    fn mul_assign(&mut self, rhs: Self) {
        let s: &_ = self;
        let prod = s * &rhs;
        *self = prod;
    }
}

#[test]
fn test_mul_assign_for_comples_ints() {
    let mut a = Complex::new(1, 2);
    a *= a;
    assert_eq!(a, Complex::new(-3, 4));
}

impl<'a, T> Mul<&'a Complex<T>> for &'a Complex<T>
where
    T: Ring,
{
    type Output = Complex<T>;

    fn mul(self, other: &'a Complex<T>) -> Complex<T> {
        let mut p1 = self.re.mul_refs(&other.re);
        let mut p2 = self.re.mul_refs(&other.im);
        let p3 = self.im.mul_refs(&other.re);
        let p4 = self.im.mul_refs(&other.im);
        p1 -= &p4;
        p2 += &p3;
        Complex::new(p1, p2)
    }
}

#[test]
fn test_mul_for_ref_complex_int() {
    let a = Complex::new(1, 2);
    let b = &a * &a;
    assert_eq!(b, Complex::new(-3, 4));
}

impl<T> Div<Self> for &Complex<T>
where
    T: Field,
{
    type Output = Complex<T>;

    fn div(self, other: Self) -> Self::Output {
        let rcp_sqr_norm = T::one().div_refs(&other.sqr_norm());
        let mut p1 = self.re.mul_refs(&other.re);
        let mut p2 = self.im.mul_refs(&other.re);
        let p3 = self.re.mul_refs(&other.im);
        let p4 = self.im.mul_refs(&other.im);
        p1 += &p4;
        p2 -= &p3;
        Complex::new(p1.mul_refs(&rcp_sqr_norm), p2.mul_refs(&rcp_sqr_norm))
    }
}

#[test]
fn test_div_for_ref_complex_f32() {
    let a = Complex::new(1.0f32, 2.0f32);
    let b = Complex::new(1.0f32, 1.0f32);
    let c = a * b;
    let d = &c / &b;
    assert_eq!(d, a);
}

impl<T> Div for Complex<T>
where
    T: Field,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        &self / &other
    }
}

#[test]
fn test_div_complex_f32() {
    let a = Complex::new(1.0f32, 2.0f32);
    let b = Complex::new(1.0f32, 1.0f32);
    let c = a * b / b;
    assert_eq!(c, a);
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}

#[test]
fn test_neg_for_complex_ints() {
    let a = Complex::new(1, 2);
    let b = -a;
    assert_eq!(b, Complex::new(-1, -2));
}

impl<T: Ring> Ring for Complex<T> {}

// Actually, this is a lie. `Complex<T>` will only be a `Field` in the
// mathematical sense, if there is no `x` of type `T` with `x * x + 1 == 0`.
// Otherwise, the ring `T` will not be free from zero divisors and thus not
// be a field.
impl<T: Field> Field for Complex<T> {}

impl<T> Exp for Complex<T>
where
    T: Exp + Sin + Cos + MulRefs + Clone,
{
    fn exp(self) -> Self {
        let e = self.re.exp();
        let c = self.im.clone().cos();
        let s = self.im.sin();
        Self::new(e.mul_refs(&c), e.mul_refs(&s))
    }
}

#[test]
fn test_exp_complex_f32() {
    let z = Complex::new(1f32, std::f32::consts::PI / 6f32);
    let e = z.exp();
    assert!((e.re - std::f32::consts::E * 0.75.sqrt()).abs() <= f32::EPSILON * std::f32::consts::E);
    assert!((e.im - std::f32::consts::E / 2f32).abs() <= f32::EPSILON * std::f32::consts::E);
}
