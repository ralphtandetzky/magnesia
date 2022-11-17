use super::ops_with_ref::*;
use super::ring::{One, Ring, Zero};
use std::ops::{AddAssign, Mul, SubAssign};

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
            y = y.mul_refs(x);
            y += a;
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
    fn zero() -> Self {
        Polynomial { a: Vec::new() }
    }
}

#[test]
fn test_zero_polynomial_for_ints() {
    let p = Polynomial::zero();
    assert_eq!(p.eval(5), 0);
}

impl<T: Ring> One for Polynomial<T> {
    fn one() -> Self {
        Polynomial { a: vec![T::one()] }
    }
}

#[test]
fn test_one_polynomial_for_ints() {
    let p = Polynomial::one();
    assert_eq!(p.eval(5), 1);
}

impl<T: Ring> std::ops::Add<Polynomial<T>> for Polynomial<T> {
    type Output = Self;

    fn add(mut self, rhs: Polynomial<T>) -> Self {
        if self.a.len() < rhs.a.len() {
            rhs.add(self)
        } else {
            for (i, x) in rhs.a.into_iter().enumerate() {
                self.a[i] += &x;
            }
            self
        }
    }
}

#[test]
fn test_add_polynomials_for_ints() {
    let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    let r = p + q;
    assert_eq!(r, Polynomial::from_coefficients(vec![3, 5, 7]));
}

impl<T: Ring> std::ops::AddAssign for Polynomial<T> {
    fn add_assign(&mut self, rhs: Self) {
        let my_len = self.a.len();
        if self.a.len() < rhs.a.len() {
            self.a.reserve(rhs.a.len() - self.a.len());
        }
        for (i, x) in rhs.a.into_iter().enumerate() {
            if i < my_len {
                self.a[i] += &x;
            } else {
                self.a.push(x);
            }
        }
    }
}

#[test]
fn test_add_assign_for_polynomials_of_ints() {
    let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    p += q;
    assert_eq!(p, Polynomial::from_coefficients(vec![3, 5, 7]));
}

impl<T: Ring> AddAssign<&Polynomial<T>> for Polynomial<T> {
    fn add_assign(&mut self, rhs: &Self) {
        if self.a.len() < rhs.a.len() {
            self.a.resize_with(rhs.a.len(), || T::zero());
        }
        for i in 0..rhs.a.len() {
            self.a[i] += &rhs.a[i];
        }
    }
}

#[test]
fn test_add_assign_for_ref_polynomials_of_ints() {
    let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![2, 3, 4]);
    p += &q;
    assert_eq!(p, Polynomial::from_coefficients(vec![3, 5, 7]));
}

impl<T: Ring> std::ops::Sub for Polynomial<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

#[test]
fn test_sub_for_polynomials_of_int() {
    let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![2, 3, 5]);
    let r = p - q;
    assert_eq!(r, Polynomial::from_coefficients(vec![-1, -1, -2]));
}

impl<T: Ring> std::ops::SubAssign for Polynomial<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self += -rhs
    }
}

#[test]
fn test_sub_assign_for_polynomials_of_ints() {
    let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![2, 3, 5]);
    p -= q;
    assert_eq!(p, Polynomial::from_coefficients(vec![-1, -1, -2]));
}

impl<T: Ring> SubAssign<&Polynomial<T>> for Polynomial<T> {
    fn sub_assign(&mut self, rhs: &Self) {
        if self.a.len() < rhs.a.len() {
            self.a.resize_with(rhs.a.len(), || T::zero());
        }
        for i in 0..rhs.a.len() {
            self.a[i] -= &rhs.a[i];
        }
    }
}

#[test]
fn test_sub_assign_for_ref_polynomials_of_ints() {
    let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![2, 3, 5]);
    p -= &q;
    assert_eq!(p, Polynomial::from_coefficients(vec![-1, -1, -2]));
}

impl<T: Ring> std::ops::Mul for Polynomial<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.mul_refs(&rhs)
    }
}

#[test]
fn test_mul_for_polynomials_of_ints() {
    // p(x) = 1 + 2*x + 3*x^2
    // q(x) = x
    let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![0, 1]);
    let r = p * q;
    assert_eq!(r, Polynomial::from_coefficients(vec![0, 1, 2, 3]));
}

impl<T: Ring> std::ops::MulAssign for Polynomial<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_refs(&rhs);
    }
}

#[test]
fn test_mul_assign_for_polynomials_of_ints() {
    // p(x) = 1 + 2*x + 3*x^2
    // q(x) = x
    let mut p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![0, 1]);
    p *= q;
    assert_eq!(p, Polynomial::from_coefficients(vec![0, 1, 2, 3]));
}

impl<T: Ring> Mul for &Polynomial<T> {
    type Output = Polynomial<T>;

    fn mul(self, rhs: Self) -> Polynomial<T> {
        if self.a.is_empty() || rhs.a.is_empty() {
            return <Polynomial<T>>::zero();
        }
        let mut a = vec![T::zero(); self.a.len() + rhs.a.len() - 1];
        for i in 0..self.a.len() {
            for j in 0..rhs.a.len() {
                let prod = self.a[i].mul_refs(&rhs.a[j]);
                a[i + j] += &prod;
            }
        }
        <Polynomial<T>>::from_coefficients(a)
    }
}

#[test]
fn test_mul_for_ref_polynomials_of_ints() {
    // p(x) = 1 + 2*x + 3*x^2
    // q(x) = x
    let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = Polynomial::from_coefficients(vec![0, 1]);
    let r = &p * &q;
    assert_eq!(r, Polynomial::from_coefficients(vec![0, 1, 2, 3]));
}

impl<T: Ring> std::ops::Neg for Polynomial<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            a: self.a.into_iter().map(|x| -x).collect(),
        }
    }
}

#[test]
fn test_neg_for_polynomials_of_ints() {
    let p = Polynomial::from_coefficients(vec![1, 2, 3]);
    let q = -p;
    assert_eq!(q, Polynomial::from_coefficients(vec![-1, -2, -3]));
}

impl<T: Ring> Ring for Polynomial<T> {}
