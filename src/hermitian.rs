use nalgebra::{DMatrix, DVector, DVectorView};
use nalgebra_sparse::{CscMatrix, CsrMatrix};

use crate::{HermitianEigen, Order};

pub trait Hermitian: Sized {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn vector_product(&self, v: DVectorView<f64>) -> DVector<f64>;

    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    /// Computes the Eigen decomposition of an Hermitian matrix
    ///
    /// # Arguments
    ///
    /// * `iterations` - Number of iterations of the Lanczos algorithm
    /// * `order` - Sort in ascending (Smallest) or Descending (Largest) order
    ///   of the Eigen values
    ///
    ///  # Example
    ///
    ///  ```
    /// # use nalgebra::DMatrix;
    /// # use lanczos::{Hermitian, Order};
    /// # let matrix = DMatrix::<f64>::new_random(100, 100);
    /// let eigen = matrix.eigsh(50, Order::Smallest);
    ///
    /// let eigenval = eigen.eigenvalues[0];
    /// let eigenvec = eigen.eigenvectors.column(0);
    ///  ```
    fn eigsh(&self, iterations: usize, order: Order) -> HermitianEigen {
        HermitianEigen::new(self, iterations, order, f64::EPSILON)
    }
}

impl Hermitian for DMatrix<f64> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn vector_product(&self, v: DVectorView<f64>) -> DVector<f64> {
        self * v
    }
}

impl Hermitian for CscMatrix<f64> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn vector_product(&self, v: DVectorView<f64>) -> DVector<f64> {
        self * v
    }
}

impl Hermitian for CsrMatrix<f64> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn vector_product(&self, v: DVectorView<f64>) -> DVector<f64> {
        self * v
    }
}
