use super::ops_with_ref::*;
use super::ring::{One, Ring, Zero};
use std::ops::{AddAssign, SubAssign, Mul};

/**
A polynomial of arbitrary order.

# Example
```
# use magnesia::algebra::Polynomial;
let p = Polynomial::from_coefficients(vec![1, 2, 3]); // p(x) = 1 + 2*x + 3*x^2
assert_eq!(p.eval(2), 1 + 2*2 + 3*4);
```
*/
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Polynomial<T: Ring> {
    a: Vec<T>,
}

impl<T: Ring> Polynomial<T> {
    /// Creates a new `Polynomial` from a vector of coefficients.
    ///
    /// Note that the array must start with the coefficients that are
    /// multiplied with the lower powers of `x` and ends with the coefficients
    /// that are multiplied with the highest powers of `x`.
    /// This may be somewhat unintuitive, since in maths polynomials are
    /// usually written in reverse direction.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// // p(x) = 1 + 2*x + 3*x^2
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// ```
    pub fn from_coefficients(coefficients: Vec<T>) -> Polynomial<T> {
        Polynomial { a: coefficients }
    }

    /// Evaluates a polynomial given the reference to a value.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// // p(x) = 1 + 2*x + 3*x^2
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let two = 2;
    /// assert_eq!(p.eval_ref(&two), 1 + 2*two + 3*two*two);
    /// ```
    pub fn eval_ref(&self, x: &T) -> T {
        let mut y = T::zero();
        for a in self.a.iter().rev() {
            y = y.mul_with_ref(x);
            y.add_assign_with_ref(a);
        }
        y
    }

    /// Evaluates a polynomial given a value.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// // p(x) = 1 + 2*x + 3*x^2
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// assert_eq!(p.eval(2), 1 + 2*2 + 3*4);
    /// ```
    pub fn eval(&self, x: T) -> T {
        self.eval_ref(&x)
    }
}

impl<T: Ring> Zero for Polynomial<T> {
    /// Creates a zero polynomial.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// # use magnesia::algebra::Zero;
    /// let p = Polynomial::zero();
    /// assert_eq!(p.eval(5), 0);
    /// ```
    fn zero() -> Self {
        Polynomial { a: Vec::new() }
    }
}

impl<T: Ring> One for Polynomial<T> {
    /// Creates a zero polynomial.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// # use magnesia::algebra::One;
    /// let p = Polynomial::one();
    /// assert_eq!(p.eval(5), 1);
    /// ```
    fn one() -> Self {
        Polynomial { a: vec![T::one()] }
    }
}

impl<T: Ring> std::ops::Add<Polynomial<T>> for Polynomial<T> {
    type Output = Self;

    /// Adds two polynomials.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    /// let r = p + q;
    /// assert_eq!(r, Polynomial::from_coefficients(vec![3, 5, 7]));
    /// ```
    fn add(mut self, rhs: Polynomial<T>) -> Self {
        if self.a.len() < rhs.a.len() {
            rhs.add(self)
        } else {
            for (i, x) in rhs.a.into_iter().enumerate() {
                self.a[i] += x;
            }
            self
        }
    }
}

impl<T: Ring> std::ops::AddAssign for Polynomial<T> {
    /// The addition assignment operator `+=`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    /// p += q;
    /// assert_eq!(p, Polynomial::from_coefficients(vec![3, 5, 7]));
    /// ```
    fn add_assign(&mut self, rhs: Self) {
        let my_len = self.a.len();
        if self.a.len() < rhs.a.len() {
            self.a.reserve(rhs.a.len() - self.a.len());
        }
        for (i, x) in rhs.a.into_iter().enumerate() {
            if i < my_len {
                self.a[i] += x;
            } else {
                self.a.push(x);
            }
        }
    }
}

impl<T: Ring> AddAssignWithRef for Polynomial<T> {
    /// An add assignment operator with references on both sides.
    ///
    /// If you want to add two polynomials by reference you can use the
    /// following syntax
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    /// p += &q;
    /// ```
    /// instead. This is the recommended way to write it.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// # use magnesia::algebra::AddAssignWithRef;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    /// p.add_assign_with_ref(&q);
    /// assert_eq!(p, Polynomial::from_coefficients(vec![3, 5, 7]));
    /// ```
    fn add_assign_with_ref(&mut self, rhs: &Self) {
        if self.a.len() < rhs.a.len() {
            self.a.resize_with(rhs.a.len(), || T::zero());
        }
        for i in 0..rhs.a.len() {
            self.a[i].add_assign_with_ref(&rhs.a[i]);
        }
    }
}

impl<T: Ring> std::ops::Sub for Polynomial<T> {
    type Output = Self;

    /// Subtracts two polynomials.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 5]);
    /// let r = p - q;
    /// assert_eq!(r, Polynomial::from_coefficients(vec![-1, -1, -2]));
    /// ```
    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl<T: Ring> std::ops::SubAssign for Polynomial<T> {
    /// Implements the `-=` operator.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 5]);
    /// p -= q;
    /// assert_eq!(p, Polynomial::from_coefficients(vec![-1, -1, -2]));
    /// ```
    fn sub_assign(&mut self, rhs: Self) {
        *self += -rhs
    }
}

impl<T: Ring> SubAssignWithRef for Polynomial<T> {
    /// A subtract assignment operator with references on both sides.
    ///
    /// If you want to subtract two polynomials by reference you can use the
    /// following syntax
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    /// p -= &q;
    /// ```
    /// instead. This is the recommended way to write it.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// # use magnesia::algebra::SubAssignWithRef;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 5]);
    /// p.sub_assign_with_ref(&q);
    /// assert_eq!(p, Polynomial::from_coefficients(vec![-1, -1, -2]));
    /// ```
    fn sub_assign_with_ref(&mut self, rhs: &Self) {
        if self.a.len() < rhs.a.len() {
            self.a.resize_with(rhs.a.len(), || T::zero());
        }
        for i in 0..rhs.a.len() {
            self.a[i].sub_assign_with_ref(&rhs.a[i]);
        }
    }
}

impl<T: Ring> std::ops::Mul for Polynomial<T> {
    type Output = Self;

    /// Implements the `*` operator for multiplying two polynomials.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// // p(x) = 1 + 2*x + 3*x^2
    /// // q(x) = x
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![0, 1]);
    /// let r = p * q;
    /// assert_eq!(r, Polynomial::from_coefficients(vec![0, 1, 2, 3]));
    /// ```
    fn mul(self, rhs: Self) -> Self {
        self.mul_with_ref(&rhs)
    }
}

impl<T: Ring> std::ops::MulAssign for Polynomial<T> {
    /// Implements the `*=` operator for multiplying two polynomials.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// // p(x) = 1 + 2*x + 3*x^2
    /// // q(x) = x
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![0, 1]);
    /// p *= q;
    /// assert_eq!(p, Polynomial::from_coefficients(vec![0, 1, 2, 3]));
    /// ```
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_with_ref(&rhs);
    }
}

impl<T: Ring> MulWithRef for Polynomial<T> {
    /// A multiply operator with references on both sides.
    ///
    /// If you want to multiply two polynomials by reference you can use the
    /// following syntax
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// // p(x) = 1 + 2*x + 3*x^2
    /// // q(x) = x
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![0, 1]);
    /// let r = &p * &q;
    /// ```
    /// instead. This is the recommended way to write it.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// # use magnesia::algebra::MulWithRef;
    /// // p(x) = 1 + 2*x + 3*x^2
    /// // q(x) = x
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![0, 1]);
    /// let r = p.mul_with_ref(&q);
    /// assert_eq!(r, Polynomial::from_coefficients(vec![0, 1, 2, 3]));
    /// ```
    fn mul_with_ref(&self, rhs: &Self) -> Self {
        if self.a.is_empty() || rhs.a.is_empty() {
            return Self::zero();
        }
        let mut a = vec![T::zero(); self.a.len() + rhs.a.len() - 1];
        for i in 0..self.a.len() {
            for j in 0..rhs.a.len() {
                a[i + j] += self.a[i].mul_with_ref(&rhs.a[j]);
            }
        }
        Self::from_coefficients(a)
    }
}

impl<T: Ring> std::ops::Neg for Polynomial<T> {
    type Output = Self;

    /// Implements the unary `-` operator.
    ///
    /// # Example
    /// ```
    /// use magnesia::algebra::Polynomial;
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = -p;
    /// assert_eq!(q, Polynomial::from_coefficients(vec![-1, -2, -3]));
    /// ```
    fn neg(mut self) -> Self {
        self.neg_assign();
        self
    }
}

impl<T: Ring> NegAssign for Polynomial<T> {
    /// Implements in-place negation for a polynomial.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// # use magnesia::algebra::NegAssign;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// p.neg_assign();
    /// assert_eq!(p, Polynomial::from_coefficients(vec![-1, -2, -3]));
    /// ```
    fn neg_assign(&mut self) {
        for x in self.a.iter_mut() {
            x.neg_assign();
        }
    }
}

impl<T: Ring> Ring for Polynomial<T> {}

impl<T: Ring> AddAssign<&Polynomial<T>> for Polynomial<T> {
    /// Implements the `+=` operator for `&Polynomial<T>`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    /// p += &q;
    /// assert_eq!(p, Polynomial::from_coefficients(vec![3, 5, 7]));
    /// ```
    fn add_assign(&mut self, rhs: &Polynomial<T>) {
        self.add_assign_with_ref(rhs);
    }
}

impl<T: Ring> SubAssign<&Polynomial<T>> for Polynomial<T> {
    /// Implements the `-=` operator for `&Polynomial<T>`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![2, 3, 5]);
    /// p -= &q;
    /// assert_eq!(p, Polynomial::from_coefficients(vec![-1, -1, -2]));
    /// ```
    fn sub_assign(&mut self, rhs: &Polynomial<T>) {
        self.sub_assign_with_ref(rhs);
    }
}

impl<T: Ring> Mul for &Polynomial<T> {
    type Output = Polynomial<T>;

    /// Implements the `*` operator for `&Polynomial<T>`.
    ///
    /// # Example
    /// ```
    /// # use magnesia::algebra::Polynomial;
    /// // p(x) = 1 + 2*x + 3*x^2
    /// // q(x) = x
    /// let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    /// let q = Polynomial::from_coefficients(vec![0, 1]);
    /// let r = &p * &q;
    /// assert_eq!(r, Polynomial::from_coefficients(vec![0, 1, 2, 3]));
    /// ```
    fn mul(self, rhs: &Polynomial<T>) -> Polynomial<T> {
        self.mul_with_ref(rhs)
    }
}
