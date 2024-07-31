//! Eigen decomposition of Hermitian matrices using Lanczos algorithm
//!
//! Using [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) to estimate the extremal
//! Eigen values and Eigen vectors of an Symmetrics Hermitian matrix.
//!
//! Supports both dense and sparse matrices via [nalgebra_sparse].
//!
//! Works well for large sparse matrices
//!
//! # Examples
//!
//! ```
//! # use nalgebra::DMatrix;
//! use lanczos::{Hermitian, Order};
//!
//! # let matrix = DMatrix::<f64>::new_random(100, 100);
//! let eigen = matrix.eigsh(50, Order::Smallest);
//!
//! // Sorted by eigenvalue in ascending order
//! eprintln!("{}", eigen.eigenvalues);
//! // Columns sorted according to eigenvalues
//! eprintln!("{}", eigen.eigenvectors);
//!
//! // Second smallest eigen value
//! eprintln!("{}", eigen.eigenvalues[1]);
//! // Eigen vector corresponding to the second smallest eigen value
//! eprintln!("{}", eigen.eigenvectors.column(1));
//! ```
mod hermitian;

pub use hermitian::Hermitian;
use nalgebra::{DMatrix, DVector, Dyn, SymmetricEigen};

#[derive(Copy, Clone)]
pub enum Order {
    Smallest,
    Largest,
}

/// Eigen decomposition of an Hermitian matrix
#[derive(Clone, Debug)]
pub struct HermitianEigen {
    /// Eigen values in order
    pub eigenvalues: DVector<f64>,
    /// Eigen vectors corresponding to the eigen values
    pub eigenvectors: DMatrix<f64>,
}

impl HermitianEigen {
    pub fn new<H>(hermitian: &H, iterations: usize, order: Order, tolerance: f64) -> Self
    where
        H: Hermitian + Sized,
    {
        assert!(
            hermitian.is_square(),
            "hermitian matrix must be square ({}x{})",
            hermitian.nrows(),
            hermitian.ncols()
        );

        // iterations must not be larger from the matrix's dimension
        let iterations = std::cmp::min(iterations, hermitian.ncols());

        let mut alpha = DVector::<f64>::zeros(iterations);
        let mut beta = DVector::<f64>::zeros(iterations - 1);

        let mut vs = DMatrix::<f64>::zeros(hermitian.nrows(), iterations);
        let v0 = DVector::<f64>::new_random(hermitian.nrows()).normalize();

        vs.set_column(0, &v0);

        let mut w_prime = hermitian.vector_product(vs.column(0));
        // alpha[0] = w_prime.conjugate().dot(&v0); // TODO: complex
        alpha[0] = w_prime.dot(&v0);
        let mut w = w_prime - alpha[0] * &v0;

        for i in 1..iterations {
            beta[i - 1] = w.norm();
            if beta[i - 1] > tolerance {
                vs.set_column(i, &w.normalize());
            } else {
                // find a random orthogonal vector
                for j in 0..i {
                    w = DVector::<f64>::new_random(hermitian.nrows());
                    let projection = w.dot(&vs.column(j));
                    w -= projection * vs.column(j)
                }

                if w.norm() > tolerance {
                    vs.set_column(i, &w.normalize());
                } else {
                    vs.set_column(i, &w);
                }
            }

            w_prime = hermitian.vector_product(vs.column(i));
            // alpha[i] = w_prime.conjugate().dot(&vs.column(i)); // TODO: complex
            alpha[i] = w_prime.dot(&vs.column(i));
            w = w_prime - alpha[i] * vs.column(i) - beta[i - 1] * vs.column(i - 1);

            // orthogonalize to previous vectors
            for j in 0..i {
                let projection = w.dot(&vs.column(j));
                w -= projection * vs.column(j)
            }
        }

        let t = construct_tridiagonal(alpha, beta);
        let eig = t.symmetric_eigen();
        let (eigenvalues, eigenvectors) = sort_eigenpairs(&eig, order);

        // Convert the triagonal matrix T's eigenvectors
        // to the Hermitian input matrix's eigenvectors
        let eigenvectors = vs * eigenvectors;

        Self {
            eigenvalues,
            eigenvectors,
        }
    }
}

fn construct_tridiagonal(alpha: DVector<f64>, beta: DVector<f64>) -> DMatrix<f64> {
    // construct tridiagonal
    let dim = alpha.len();
    DMatrix::<f64>::from_fn(dim, dim, |i, j| {
        if i == j {
            alpha[i]
        } else if i == j + 1 {
            beta[j]
        } else if j == i + 1 {
            beta[i]
        } else {
            0.0
        }
    })
}

fn sort_eigenpairs(eig: &SymmetricEigen<f64, Dyn>, order: Order) -> (DVector<f64>, DMatrix<f64>) {
    let dim = eig.eigenvalues.len();
    let mut eigvalues_index: Vec<(usize, f64)> =
        eig.eigenvalues.iter().copied().enumerate().collect();

    match order {
        Order::Smallest => eigvalues_index.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
        Order::Largest => eigvalues_index.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()),
    }

    let mut sorted_eigenvectors = DMatrix::<f64>::zeros(dim, dim);

    eigvalues_index
        .iter()
        .enumerate()
        .for_each(|(i, (j, _))| sorted_eigenvectors.set_column(i, &eig.eigenvectors.column(*j)));

    let sorted_eigenvalues =
        DVector::<f64>::from_iterator(dim, eigvalues_index.iter().map(|(_, v)| v).copied());

    (sorted_eigenvalues, sorted_eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_close(a: f64, b: f64, tolerance: f64) -> bool {
        eprintln!("{} ~= {}", a, b);
        (a - b).abs() < tolerance
    }

    fn all_close(v_a: DVector<f64>, v_b: DVector<f64>, tolerance: f64) -> bool {
        std::iter::zip(v_a.iter(), v_b.iter()).all(|(a, b)| is_close(*a, *b, tolerance))
    }

    fn compare_algos(
        dim: usize,
        neigenpairs: usize,
        iterations: usize,
        order: Order,
        tolerance: f64,
    ) {
        let tmp = DMatrix::<f64>::new_random(dim, dim);

        // hermitian matrix
        let matrix =
            DMatrix::<f64>::from_fn(
                dim,
                dim,
                |i, j| if i <= j { tmp[(i, j)] } else { tmp[(j, i)] },
            );

        let eigen_lanczos = matrix.eigsh(iterations, order);
        let eigen_symmetric = matrix.symmetric_eigen();

        let (eigenvalues, eigenvectors) = sort_eigenpairs(&eigen_symmetric, order);

        std::iter::zip(
            eigen_lanczos.eigenvalues.iter().take(neigenpairs),
            eigenvalues.iter(),
        )
        .for_each(|(lambda_a, lambda_b)| assert!(is_close(*lambda_a, *lambda_b, tolerance)));

        std::iter::zip(
            eigen_lanczos
                .eigenvectors
                .columns(0, neigenpairs)
                .column_iter(),
            eigenvectors.columns(0, neigenpairs).column_iter(),
        )
        .for_each(|(v_a, v_b)| {
            assert!(
                all_close(v_a.into(), v_b.into(), tolerance)
                    || all_close(-1.0 * v_a, v_b.into(), tolerance)
            )
        });
    }

    #[test]
    fn test_eigh_50x50_10_smallest_tol_1e6() {
        let dim = 50;
        let neigenpairs = 10;
        let order = Order::Smallest;
        let tolerance = 1e-6;
        let iterations = 50;

        compare_algos(dim, neigenpairs, iterations, order, tolerance)
    }

    #[test]
    fn test_eigh_50x50_10_largest_tol_1e6() {
        let dim = 50;
        let neigenpairs = 10;
        let order = Order::Largest;
        let tolerance = 1e-6;
        let iterations = 50;

        compare_algos(dim, neigenpairs, iterations, order, tolerance)
    }

    #[test]
    fn test_eigh_50x50_3_smallest_tol_5e1_20_iterations() {
        let dim = 50;
        let neigenpairs = 3;
        let order = Order::Smallest;
        let tolerance = 5e-1;
        let iterations = 20;

        compare_algos(dim, neigenpairs, iterations, order, tolerance)
    }

    #[test]
    fn test_eigh_50x50_3_largest_tol_5e1_20_iterations() {
        let dim = 50;
        let neigenpairs = 3;
        let order = Order::Largest;
        let tolerance = 5e-1;
        let iterations = 20;

        compare_algos(dim, neigenpairs, iterations, order, tolerance)
    }
}