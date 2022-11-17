//! A math library for Rust featuring
//!
//!   * Polynomials
//!   * Vectors and Matrices
//!   * Complex Numbers
//!
//! # Goal and Scope
//!
//! The goal of Magnesia is to provide an easy to use yet efficient API for
//! mathematical applications.
//! Magnesia may not be feature rich and optimized yet, but its interfaces are
//! designed such that extensions and zero overhead implementations are
//! possible.
//!
//! In the future the package shall contain the following features as well:
//!
//!   * Quaternions
//!   * Big Integers
//!   * Fast Fourier Transform
//!   * Symbolic and Numerical Differentiation
//!   * Numerical Optimization
//!   * Root Rinding
//!   * Interpolation
//!   * Integration and Solving Differential Equations
//!   * Geometrical Primitives
//!   * etc.
//!
//! Contributions for this cause are highly welcome!

#![deny(missing_docs)]

/// This module provides facilities from abstract algebra.
///
/// This includes the following:
///
///   * Polynomials
///   * Complex Numbers
///   * Rings (as trait)
///   * Some helper traits
///
/// In the future, this module may also include the following:
///
///   * Quaternions
///   * Big Integers
///   * Fractions
///   * Finite Fields
///
/// Structures belonging to linear algebra can be found in the module
/// [`algebra`](crate::linalg).
pub mod algebra;

/// This module provides facilities from linear algebra.
///
/// This includes the following:
///   * Vectors
///   * Matrices
///
/// In the future, this module may also include the following:
///
///   * Matrix Decompositions
///   * Affine Transformations
///   * Tensors
pub mod linalg;
