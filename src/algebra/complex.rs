use super::{AddAssignWithRef, MulWithRef, NegAssign, One, Ring, SubAssignWithRef, Zero};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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

impl<T: NegAssign> Complex<T> {
    /// Computes the complex conjugate of a complex number.
    ///
    /// # Example
    /// ```
    /// # use::magnesia::algebra::Complex;
    /// let z = Complex::new(1.0, 2.0);
    /// assert_eq!(z.conj(), Complex::new(1.0, -2.0));
    /// ```
    pub fn conj(mut self) -> Self {
        self.im.neg_assign();
        self
    }
}

impl<T: Zero> Zero for Complex<T> {
    /// Constructs a zero complex number.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// # use magnesia::algebra::Zero;
    /// let z = Complex::<f32>::zero();
    /// assert_eq!(z.re, 0.0);
    /// assert_eq!(z.im, 0.0);
    /// ```
    fn zero() -> Self {
        Complex::new(T::zero(), T::zero())
    }
}

impl<T: Zero + One> One for Complex<T> {
    /// Constructs the complex number `'1'`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// # use magnesia::algebra::One;
    /// let z = Complex::<i32>::one();
    /// assert_eq!(z.re, 1);
    /// assert_eq!(z.im, 0);
    /// ```
    fn one() -> Self {
        Complex::new(T::one(), T::zero())
    }
}

impl<U, T: Add<U>> Add<Complex<U>> for Complex<T> {
    type Output = Complex<<T as Add<U>>::Output>;

    /// Implements the `+` operator for complex numbers.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// let a = Complex::new(1, 2);
    /// let b = Complex::new(3, 4);
    /// let c = a + b;
    /// assert_eq!(c, Complex::new(4,6));
    /// ```
    fn add(self, other: Complex<U>) -> Self::Output {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

impl<T: AddAssign<U>, U> AddAssign<Complex<U>> for Complex<T> {
    /// Implements the `+=` operator for complex numbers.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// let mut a = Complex::new(1, 2);
    /// let b = Complex::new(3, 4);
    /// a += b;
    /// assert_eq!(a, Complex::new(4,6));
    /// ```
    fn add_assign(&mut self, other: Complex<U>) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl<T: AddAssignWithRef> AddAssignWithRef for Complex<T> {
    /// Adds two complex numbers in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// # use crate::magnesia::algebra::AddAssignWithRef;
    /// let mut a = Complex::new(1, 2);
    /// let b = Complex::new(3, 4);
    /// a.add_assign_with_ref(&b);
    /// assert_eq!(a, Complex::new(4,6));
    /// ```
    fn add_assign_with_ref(&mut self, other: &Self) {
        self.re.add_assign_with_ref(&other.re);
        self.im.add_assign_with_ref(&other.im);
    }
}

impl<U, T: Sub<U>> Sub<Complex<U>> for Complex<T> {
    type Output = Complex<<T as Sub<U>>::Output>;

    /// Implements the `-` operator for complex numbers.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// let a = Complex::new(1, 2);
    /// let b = Complex::new(3, 4);
    /// let c = a - b;
    /// assert_eq!(c, Complex::new(-2,-2));
    /// ```
    fn sub(self, other: Complex<U>) -> Self::Output {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

impl<T: SubAssign<U>, U> SubAssign<Complex<U>> for Complex<T> {
    /// Implements the `-=` operator for complex numbers.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// let mut a = Complex::new(1, 2);
    /// let b = Complex::new(3, 4);
    /// a -= b;
    /// assert_eq!(a, Complex::new(-2,-2));
    /// ```
    fn sub_assign(&mut self, other: Complex<U>) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl<T: SubAssignWithRef> SubAssignWithRef for Complex<T> {
    /// Subtracts two complex numbers in-place.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Complex;
    /// # use crate::magnesia::algebra::SubAssignWithRef;
    /// let mut a = Complex::new(1, 2);
    /// let b = Complex::new(3, 4);
    /// a.sub_assign_with_ref(&b);
    /// assert_eq!(a, Complex::new(-2,-2));
    /// ```
    fn sub_assign_with_ref(&mut self, other: &Self) {
        self.re.sub_assign_with_ref(&other.re);
        self.im.sub_assign_with_ref(&other.im);
    }
}

impl<T: MulWithRef + AddAssignWithRef + SubAssignWithRef> Mul for Complex<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        self.mul_with_ref(&other)
    }
}

impl<T: MulWithRef + AddAssignWithRef + SubAssignWithRef> MulAssign for Complex<T> {
    fn mul_assign(&mut self, rhs: Self) {
        let prod = self.mul_with_ref(&rhs);
        *self = prod;
    }
}

impl<T: MulWithRef + AddAssignWithRef + SubAssignWithRef> MulWithRef for Complex<T> {
    /// Implements multiplication of complex numbers by refernce.
    ///
    /// # Example
    /// ```
    /// # use::magnesia::algebra::Complex;
    /// # use::magnesia::algebra::MulWithRef;
    /// let a = Complex::new(1, 2);
    /// let b = a.mul_with_ref(&a);
    /// assert_eq!(b, Complex::new(-3, 4));
    /// ```
    fn mul_with_ref(&self, other: &Self) -> Self {
        let mut p1 = self.re.mul_with_ref(&other.re);
        let mut p2 = self.re.mul_with_ref(&other.im);
        let p3 = self.im.mul_with_ref(&other.re);
        let p4 = self.im.mul_with_ref(&other.im);
        p1.sub_assign_with_ref(&p4);
        p2.add_assign_with_ref(&p3);
        Complex::new(p1, p2)
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl<T: NegAssign> NegAssign for Complex<T> {
    fn neg_assign(&mut self) {
        self.re.neg_assign();
        self.im.neg_assign();
    }
}

impl<T: Ring> Ring for Complex<T> {}
