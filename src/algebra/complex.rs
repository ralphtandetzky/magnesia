use super::{conj::Conj, Field, MulRefs, One, Ring, Zero};
use crate::functions::{Abs, Atan2, Cos, Cosh, Exp, Ln, Sin, Sinh, Sqrt, Tan};
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

impl<T> Abs for Complex<T>
where
    for<'a> T: AddAssign<&'a T> + MulRefs + Sqrt,
{
    type Output = T;

    fn abs(self) -> Self::Output {
        self.sqr_norm().sqrt()
    }
}

#[test]
fn test_abs_complex_f32() {
    let z = Complex::new(3f32, 4f32);
    let a = z.abs();
    assert_eq!(a, 5f32);
    let z = Complex::new(1f32, -1f32);
    let a = z.abs();
    assert_eq!(a, 2f32.sqrt());
}

impl<'a, T> Abs for &'a Complex<T>
where
    for<'b> T: AddAssign<&'b T> + MulRefs + Sqrt,
{
    type Output = T;

    fn abs(self) -> Self::Output {
        self.sqr_norm().sqrt()
    }
}

#[test]
fn test_abs_ref_complex_f32() {
    let z = Complex::new(3f32, 4f32);
    let a = (&z).abs();
    assert_eq!(a, 5f32);
    let z = Complex::new(1f32, -1f32);
    let a = (&z).abs();
    assert_eq!(a, 2f32.sqrt());
}

impl<T> Complex<T>
where
    T: Atan2,
{
    /// Computes the argument (the angle) of a complex number in radians.
    ///
    /// The result will lie in the range $[-\pi, \pi]$.
    ///
    /// # Example
    ///
    /// ```
    /// # use magnesia::algebra::Complex;
    /// let z = Complex::new(1f32, 3f32.sqrt());
    /// let phi = z.arg();
    /// assert!((phi - std::f32::consts::PI / 3f32).abs() <= f32::EPSILON);
    /// ```
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
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

impl<T> Mul<T> for Complex<T>
where
    T: MulRefs,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        Complex::new(self.re.mul_refs(&other), self.im.mul_refs(&other))
    }
}

impl<T: Ring> MulAssign for Complex<T> {
    fn mul_assign(&mut self, rhs: Self) {
        let s: &_ = self;
        let prod = s * &rhs;
        *self = prod;
    }
}

#[test]
fn test_mul_assign_for_complex_ints() {
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
#[allow(clippy::op_ref)]
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
#[allow(clippy::op_ref)]
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

impl<'a, 'b, T> Div<&'a T> for &'b Complex<T>
where
    for<'c, 'd> &'c T: Div<&'d T, Output = T>,
{
    type Output = Complex<T>;

    fn div(self, other: &'a T) -> Complex<T> {
        Complex::new(&self.re / other, &self.im / other)
    }
}

#[test]
#[allow(clippy::op_ref)]
fn test_div_ref_complex_f32_ref_f32() {
    let a = Complex::new(3f32, 5f32);
    let b = 4f32;
    let c = &a / &b;
    assert_eq!(c, Complex::new(0.75f32, 1.25f32));
}

impl<'a, T> Div<&'a T> for Complex<T>
where
    for<'b> T: Div<&'b T, Output = T>,
{
    type Output = Complex<T>;

    fn div(self, other: &'a T) -> Complex<T> {
        Complex::new(self.re / other, self.im / other)
    }
}

#[test]
#[allow(clippy::op_ref)]
fn test_div_complex_f32_ref_f32() {
    let a = Complex::new(3f32, 5f32);
    let b = 4f32;
    let c = a / &b;
    assert_eq!(c, Complex::new(0.75f32, 1.25f32));
}

impl<T> Div<T> for Complex<T>
where
    for<'a> T: Div<T, Output = T> + Div<&'a T, Output = T>,
{
    type Output = Complex<T>;

    fn div(self, other: T) -> Complex<T> {
        Complex::new(self.re / &other, self.im / other)
    }
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

impl<T> Sin for Complex<T>
where
    T: Sin + Cos + Sinh + Cosh + MulRefs + Clone,
{
    fn sin(self) -> Self {
        let re = self.re.clone().sin().mul_refs(&self.im.clone().cosh());
        let im = self.re.cos().mul_refs(&self.im.sinh());
        Self::new(re, im)
    }
}

#[test]
fn test_sin_complex_f32() {
    let i = Complex::new(0f32, 1f32);
    let z = Complex::new(0.3f32, -1.2f32);
    let a = z.sin();
    let b = ((i * z).exp() - (-i * z).exp()) / (i * 2f32);
    assert!((a - b).abs() <= z.abs().exp() * 2f32 * f32::EPSILON);
}

impl<T> Cos for Complex<T>
where
    T: Sin + Cos + Sinh + Cosh + MulRefs + Neg<Output = T> + Clone,
{
    fn cos(self) -> Self {
        let re = self.re.clone().cos().mul_refs(&self.im.clone().cosh());
        let im = -self.re.sin().mul_refs(&self.im.sinh());
        Self::new(re, im)
    }
}

#[test]
fn test_cos_complex_f64() {
    let i = Complex::new(0f64, 1f64);
    let z = Complex::new(0.3f64, -1.2f64);
    let a = z.cos();
    let b = ((i * z).exp() + (-i * z).exp()) / Complex::new(2f64, 0f64);
    assert!((a - b).abs() <= z.abs().exp() * 2f64 * f64::EPSILON);
}

impl<T> Sinh for Complex<T>
where
    T: Sin + Cos + Sinh + Cosh + MulRefs + Clone,
{
    fn sinh(self) -> Complex<T> {
        let re = self.re.clone().sinh().mul_refs(&self.im.clone().cos());
        let im = self.re.cosh().mul_refs(&self.im.sin());
        Self::new(re, im)
    }
}

#[test]
fn test_sinh_complex_f32() {
    let z = Complex::new(0.3f64, -1.2f64);
    let a = z.sinh();
    let b = (z.exp() - (-z).exp()) / Complex::new(2f64, 0f64);
    assert!((a - b).abs() <= z.abs().exp() * 2f64 * f64::EPSILON);
}

impl<T> Cosh for Complex<T>
where
    T: Sin + Cos + Sinh + Cosh + MulRefs + Clone,
{
    fn cosh(self) -> Complex<T> {
        let re = self.re.clone().cosh().mul_refs(&self.im.clone().cos());
        let im = self.re.sinh().mul_refs(&self.im.sin());
        Self::new(re, im)
    }
}

#[test]
fn test_cosh_complex_f32() {
    let z = Complex::new(0.3f64, -1.2f64);
    let a = z.cosh();
    let b = (z.exp() + (-z).exp()) / Complex::new(2f64, 0f64);
    assert!((a - b).abs() <= z.abs().exp() * 2f64 * f64::EPSILON);
}

impl<T> Tan for Complex<T>
where
    Complex<T>: Sin + Cos + Div<Output = Complex<T>> + Clone,
{
    fn tan(self) -> Complex<T> {
        self.clone().sin() / self.cos()
    }
}

#[test]
fn test_tan_complex_f32() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let a: Complex<f32> = Complex::new(rng.gen_range(-4.0..4.0), rng.gen_range(-4.0..4.0));
        let b = Complex::new(rng.gen_range(-4.0..4.0), rng.gen_range(-4.0..4.0));
        let c = a + b;
        let ta = a.tan();
        let tb = b.tan();
        let tc = c.tan();
        let one = Complex::new(1.0, 0.0);
        let expected = (ta + tb) / (one - ta * tb);
        let diff_abs = (tc - expected).abs();
        let tolerance =
            (tc.sqr_norm() + 1.0).max(1.0 / (one - ta * tb).sqr_norm()) * (8.0 * f32::EPSILON);
        assert!(diff_abs <= tolerance);
    }
}

impl<T> Ln for Complex<T>
where
    for<'a> T: AddAssign<&'a T> + Atan2 + Clone + Div<Output = T> + Ln + MulRefs + One,
{
    fn ln(self) -> Self {
        let two = {
            let mut x = T::one();
            x += &T::one();
            x
        };
        Self::new(self.sqr_norm().ln() / two, self.im.atan2(self.re))
    }
}

#[test]
fn test_ln_complex_f32() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let re_a = rng.gen_range(-5f32..5f32);
        let im_a = rng.gen_range(-5f32..5f32);
        let re_b = rng.gen_range(-5f32..5f32);
        let im_b = rng.gen_range(-5f32..5f32);
        let a = Complex::new(re_a, im_a);
        let b = Complex::new(re_b, im_b);
        let c = a * b;
        let ln_a = a.ln();
        let ln_b = b.ln();
        let ln_c = c.ln();
        let tolerance = (a.abs().max(1f32 / a.abs()) + b.abs().max(1f32 / b.abs())) * f32::EPSILON;
        let diff = ln_c - ln_a - ln_b;
        assert!(diff.re.abs() <= 10f32 * tolerance);
        if diff.im > 1f32 {
            assert!((diff.im - 2.0 * std::f32::consts::PI).abs() <= 8.0 * f32::EPSILON);
        } else if diff.im < -1f32 {
            assert!((diff.im + 2.0 * std::f32::consts::PI).abs() <= 8.0 * f32::EPSILON);
        } else {
            assert!(diff.im.abs() <= 8.0 * std::f32::consts::PI);
        }
    }
}

impl<T> Sqrt for Complex<T>
where
    for<'a> T: AddAssign<&'a T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + MulRefs
        + Neg<Output = T>
        + One
        + PartialOrd
        + Sqrt
        + Sub<&'a T, Output = T>
        + Zero,
    for<'a, 'b> &'a T: Abs<Output = T> + Add<&'b T, Output = T> + Sub<&'b T, Output = T>,
{
    fn sqrt(&self) -> Self {
        let two = T::one() + &T::one();
        let one_half = T::one() / &two;
        let abs = self.abs();
        let re;
        let abs_im;
        if self.re.abs() <= self.im.abs().mul_refs(&two) {
            re = (&abs + &self.re).mul_refs(&one_half).sqrt();
            abs_im = (&abs - &self.re).mul_refs(&one_half).sqrt();
        } else {
            // This branch is for numerical stability. The add or subtract
            // operation above can lead to large relative errors.
            // To avoid this we need another branch and division which is more
            // expensive, but numerically more accurate.
            let denominator = (&abs + &self.re.abs()).mul_refs(&two).sqrt();
            let alpha = self.im.abs() / &denominator;
            let beta = (&abs + &self.re.abs()).mul_refs(&one_half).sqrt();
            if self.re >= T::zero() {
                re = beta;
                abs_im = alpha;
            } else {
                re = alpha;
                abs_im = beta
            };
        }
        Self::new(re, if self.im < T::zero() { -abs_im } else { abs_im })
    }
}

#[test]
fn test_sqrt_complex_f32() {
    use rand::prelude::*;
    let mut rng = thread_rng();
    for _ in 0..100 {
        let re = rng.gen_range(-5f32..5f32);
        let im = rng.gen_range(-5f32..5f32);
        let z = Complex::new(re, im);
        let s = z.sqrt();
        let ss = s * s;
        let error = (z - ss).abs();
        let tolerance = 4f32 * f32::EPSILON * z.abs();
        assert!(s.re >= 0f32);
        assert!(error <= tolerance);
    }
}
