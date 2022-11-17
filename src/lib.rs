//! A math library for Rust featuring
//!
//!   * Polynomials
//!   * Vectors and Matrices
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
//!   * Complex numbers and Quaternions
//!   * Big Integers
//!   * Fast Fourier Transform
//!   * Symbolic and numerical differentiation
//!   * Numerical Optimization
//!   * Root finding
//!   * Interpolation
//!   * Integration and solving differential equations
//!   * Geometrical primitives
//!   * etc.
//!
//! Contributions for this cause are highly welcome!

#![deny(missing_docs)]

/// This module provides facilities from abstract algebra.
///
/// This includes the following:
///
///   * Polynomials
///   * Rings (as trait)
///   * Some helper traits
///
/// In the future, this module may also include the following:
///
///   * Complex Numbers
///   * Big Integers
///   * Quaternions
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
