use super::ring::{Ring, Zero, One};
use super::ops_with_ref::*;

#[derive(Debug, Clone)]
pub struct Polynomial<T: Ring> {
    a: Vec<T>,
}

impl<T: Ring> Polynomial<T> {
    pub fn from_coefficients(coefficients: Vec<T>) -> Polynomial<T> {
        Polynomial { a: coefficients }
    }
}

impl<T: Ring> Zero for Polynomial<T> {
    fn zero() -> Self {
        Polynomial { a: Vec::new() }
    }
}

impl<T: Ring> One for Polynomial<T> {
    fn one() -> Self {
        Polynomial { a: vec!{T::one()} }
    }
}

impl<T: Ring> std::ops::Add<Polynomial<T>> for Polynomial<T> {
    type Output = Self;

    fn add(mut self, rhs: Polynomial<T>) -> Self {
        if self.a.len() < rhs.a.len() {
            rhs.add(self)
        } else {
            let i: usize = 0;
            for x in rhs.a.into_iter() {
                self.a[i] += x;
            }
            self
        }
    }
}

impl<T: Ring> std::ops::AddAssign for Polynomial<T> {
    fn add_assign(&mut self, rhs: Self) {
        let my_len = self.a.len();
        let mut i: usize = 0;
        if self.a.len() < rhs.a.len() {
            self.a.reserve(rhs.a.len() - self.a.len());
        }
        for x in rhs.a.into_iter() {
            if i < my_len {
                self.a[i] += x;
            } else {
                self.a.push(x);
            }
            i += 1;
        }
    }
}

impl<T: Ring> AddAssignWithRef for Polynomial<T> {
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

    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl<T: Ring> std::ops::SubAssign for Polynomial<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self += -rhs
    }
}

impl<T: Ring> SubAssignWithRef for Polynomial<T> {
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

    fn mul(self, rhs: Self) -> Self {
        self.mul_with_ref(&rhs)
    }
}

impl<T: Ring> std::ops::MulAssign for Polynomial<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_with_ref(&rhs);
    }
}

impl<T: Ring> MulWithRef for Polynomial<T> {
    fn mul_with_ref(&self, rhs: &Self) -> Self {
        if self.a.len() == 0 || rhs.a.len() == 0 {
            return Self::zero()
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

    fn neg(mut self) -> Self {
        self.neg_assign();
        self
    }
}

impl<T: Ring> NegAssign for Polynomial<T> {
    fn neg_assign(&mut self) {
        for x in self.a.iter_mut() {
            x.neg_assign();
        }
    }
}

impl<T: Ring> Ring for Polynomial<T> {}
