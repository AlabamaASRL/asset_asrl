#ifndef EIGEN_ACCELERATESUPPORT_H
#define EIGEN_ACCELERATESUPPORT_H

#include "AccelerateUtils.h"

#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <Accelerate/Accelerate.h>

#include <Eigen/Sparse>

#include <cmath>
#include <numeric>

/*
The classes in this file are directly based on the AccelerateSupport module from Eigen 3.4 and 
are subject to Eigen's MPL2 license, which can be found in the notices folder of the GitHub repository.
Changes include the addition of several member variables and functions to provide more fine grained 
control of the Accelerate Sparse solvers, to avoid making unnecessary copies/allocations, to add an 
iterative refinement implementation (to bring solution accuracy more in line with PARDISO), as well as 
to more closely align with the methods of the PARDISO interface in PardisoInterface.h (particularly 
for methods that ASSET employs extensively). Note that the MPL2 license is only applied to this particular 
file and not the rest of the project, as per the MPL2 license.
 
*/

namespace Eigen {

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
class AccelerateImpl;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLLT
 * \brief A direct Cholesky (LLT) factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLLT
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLLT = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationCholesky, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLDLT
 * \brief The default Cholesky (LDLT) factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLT
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLT = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLT, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLDLTUnpivoted
 * \brief A direct Cholesky-like LDL^T factorization and solver based on Accelerate with only 1x1 pivots and no pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTUnpivoted
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTUnpivoted = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTUnpivoted, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLDLTSBK
 * \brief A direct Cholesky (LDLT) factorization and solver based on Accelerate with Supernode Bunch-Kaufman and static
 * pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTSBK
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTSBK = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTSBK, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateLDLTTPP
 * \brief A direct Cholesky (LDLT) factorization and solver based on Accelerate with full threshold partial pivoting
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam UpLo_ additional information about the matrix structure. Default is Lower.
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateLDLTTPP
 */
template <typename MatrixType, int UpLo = Lower>
using AccelerateLDLTTPP = AccelerateImpl<MatrixType, UpLo | Symmetric, SparseFactorizationLDLTTPP, true>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateQR
 * \brief A QR factorization and solver based on Accelerate
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateQR
 */
template <typename MatrixType>
using AccelerateQR = AccelerateImpl<MatrixType, 0, SparseFactorizationQR, false>;

/** \ingroup AccelerateSupport_Module
 * \typedef AccelerateCholeskyAtA
 * \brief A QR factorization and solver based on Accelerate without storing Q (equivalent to A^TA = R^T R)
 *
 * \warning Only single and double precision real scalar types are supported by Accelerate
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a SparseMatrix<>
 *
 * \sa \ref TutorialSparseSolverConcept, class AccelerateCholeskyAtA
 */
template <typename MatrixType>
using AccelerateCholeskyAtA = AccelerateImpl<MatrixType, 0, SparseFactorizationCholeskyAtA, false>;

namespace internal {
template <typename T>
struct AccelFactorizationDeleter {
  void operator()(T* sym) {
    if (sym) {
      SparseCleanup(*sym);
      delete sym;
      sym = nullptr;
    }
  }
};

template <typename DenseVecT, typename DenseMatT, typename SparseMatT, typename NumFactT>
struct SparseTypesTraitBase {
  typedef DenseVecT AccelDenseVector;
  typedef DenseMatT AccelDenseMatrix;
  typedef SparseMatT AccelSparseMatrix;

  typedef SparseOpaqueSymbolicFactorization SymbolicFactorization;
  typedef NumFactT NumericFactorization;

  typedef AccelFactorizationDeleter<SymbolicFactorization> SymbolicFactorizationDeleter;
  typedef AccelFactorizationDeleter<NumericFactorization> NumericFactorizationDeleter;
};

template <typename Scalar>
struct SparseTypesTrait {};

template <>
struct SparseTypesTrait<double> : SparseTypesTraitBase<DenseVector_Double, DenseMatrix_Double, SparseMatrix_Double,
                                                       SparseOpaqueFactorization_Double> {};

template <>
struct SparseTypesTrait<float>
    : SparseTypesTraitBase<DenseVector_Float, DenseMatrix_Float, SparseMatrix_Float, SparseOpaqueFactorization_Float> {
};

// Taken from https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/accelerate_sparse.cc
inline void* resizeForAccelerateAlignment(const size_t required_size, 
                                         std::vector<uint8_t>* mem_vec) 
{
  // As per the Accelerate documentation, all workspace memory passed to the
  // sparse solver functions must be 16-byte aligned.
  constexpr int kAccelerateRequiredAlignment = 16; 
  // Although malloc() on macOS should always be 16-byte aligned, it is unclear
  // if this holds for new(), or on other Apple OSs (phoneOS, watchOS etc).
  // As such we assume it is not and use std::align() to create a (potentially
  // offset) 16-byte aligned sub-buffer of the specified size within workspace.
  mem_vec->resize(required_size + kAccelerateRequiredAlignment);
  size_t size_from_aligned_start = mem_vec->size();
  void* aligned_solve_workspace_start =
      reinterpret_cast<void*>(mem_vec->data());
  aligned_solve_workspace_start = std::align(kAccelerateRequiredAlignment,
                                             required_size,
                                             aligned_solve_workspace_start,
                                             size_from_aligned_start);
  return aligned_solve_workspace_start;
}

}  // end namespace internal

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
class AccelerateImpl : public SparseSolverBase<AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_> > {
 protected:
  using Base = SparseSolverBase<AccelerateImpl>;
  using Base::derived;
  using Base::m_isInitialized;

 public:
  using Base::_solve_impl;

  typedef MatrixType_ MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  enum { ColsAtCompileTime = Dynamic, MaxColsAtCompileTime = Dynamic };
  enum { UpLo = UpLo_ };

  using AccelDenseVector = typename internal::SparseTypesTrait<Scalar>::AccelDenseVector;
  using AccelDenseMatrix = typename internal::SparseTypesTrait<Scalar>::AccelDenseMatrix;
  using AccelSparseMatrix = typename internal::SparseTypesTrait<Scalar>::AccelSparseMatrix;
  using SymbolicFactorization = typename internal::SparseTypesTrait<Scalar>::SymbolicFactorization;
  using NumericFactorization = typename internal::SparseTypesTrait<Scalar>::NumericFactorization;
  using SymbolicFactorizationDeleter = typename internal::SparseTypesTrait<Scalar>::SymbolicFactorizationDeleter;
  using NumericFactorizationDeleter = typename internal::SparseTypesTrait<Scalar>::NumericFactorizationDeleter;

  AccelerateImpl() {
    m_isInitialized = false;

    auto check_flag_set = [](int value, int flag) { return ((value & flag) == flag); };

    if (check_flag_set(UpLo_, Symmetric)) {
      m_sparseKind = SparseSymmetric;
      m_triType = (UpLo_ & Lower) ? SparseLowerTriangle : SparseUpperTriangle;
    } else if (check_flag_set(UpLo_, UnitLower)) {
      m_sparseKind = SparseUnitTriangular;
      m_triType = SparseLowerTriangle;
    } else if (check_flag_set(UpLo_, UnitUpper)) {
      m_sparseKind = SparseUnitTriangular;
      m_triType = SparseUpperTriangle;
    } else if (check_flag_set(UpLo_, StrictlyLower)) {
      m_sparseKind = SparseTriangular;
      m_triType = SparseLowerTriangle;
    } else if (check_flag_set(UpLo_, StrictlyUpper)) {
      m_sparseKind = SparseTriangular;
      m_triType = SparseUpperTriangle;
    } else if (check_flag_set(UpLo_, Lower)) {
      m_sparseKind = SparseTriangular;
      m_triType = SparseLowerTriangle;
    } else if (check_flag_set(UpLo_, Upper)) {
      m_sparseKind = SparseTriangular;
      m_triType = SparseUpperTriangle;
    } else {
      m_sparseKind = SparseOrdinary;
      m_triType = (UpLo_ & Lower) ? SparseLowerTriangle : SparseUpperTriangle;
    }

    m_order = SparseOrderMetis;  // Use METIS ordering by default for better performance
    m_doIterativeRefinement = false;
    m_iterativeRefinementIterations = 2;
  }

  explicit AccelerateImpl(const MatrixType& matrix) : AccelerateImpl() { compute(matrix); }

  ~AccelerateImpl() {}

  inline Index cols() const { return m_nCols; }
  inline Index rows() const { return m_nRows; }

  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "Decomposition is not initialized.");
    return m_info;
  }

  void setMatrix(const MatrixType& matrix);

  void analyzePattern(const MatrixType& matrix);

  void factorize(const MatrixType& matrix);

  void compute(const MatrixType& matrix);

  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const;

  /** Sets the ordering algorithm to use. */
  void setOrder(SparseOrder_t order) { m_order = order; }

  /** Sets the number of threads for accelerate */
  inline void setNumThreads(int num_threads) {
    accelerate_set_num_threads(num_threads);
  }

  void setIterativeRefinement(bool iterativeRefinement) { 
    m_doIterativeRefinement = iterativeRefinement; 
  }

  void setIterativeRefinementIterations(int iterations) {
    eigen_assert(iterations >= 0 && "Number of iterations must be non-negative.");
    m_iterativeRefinementIterations = iterations;
  }

  MatrixType& getMatrix() { return m_matrix; }

  template <int U = UpLo>
  typename std::enable_if<!bool(U & Symmetric), void>::type
  getMatrix(const MatrixType& matrix) {
    m_matrix = matrix;
    m_matrix.makeCompressed();
  }

  template <int U = UpLo>
  typename std::enable_if<bool(U & Symmetric), void>::type
  getMatrix(const MatrixType& matrix) {
    // Similar to Pardiso, use selfadjointView to ensure symmetry
    PermutationMatrix<Dynamic, Dynamic, StorageIndex> p_null;
    m_matrix.resize(matrix.rows(), matrix.cols());
    
    constexpr int TriangleType = (UpLo & Lower) ? Lower : Upper;
    m_matrix.template selfadjointView<TriangleType>() = 
        matrix.template selfadjointView<TriangleType>().twistedBy(p_null);
    m_matrix.makeCompressed();
  }

  template <int U = UpLo>
  typename std::enable_if<bool(U & Symmetric), MatrixType>::type
  getMatrixTwisted(const MatrixType& matrix) {
    eigen_assert(!m_permutation.empty() && "Permutation not available. Call compute() or analyzePattern() first.");
    eigen_assert(matrix.rows() == matrix.cols() && "Matrix must be square for twisted operation.");
    eigen_assert(static_cast<size_t>(matrix.rows()) == m_permutation.size() && "Matrix size must match permutation size.");
    
    // Create permutation matrix from the stored permutation vector
    PermutationMatrix<Dynamic, Dynamic, StorageIndex> p_perm;
    p_perm.indices() = Map<const Matrix<StorageIndex, Dynamic, 1>>(m_permutation.data(), m_permutation.size());
    
    MatrixType result;
    result.resize(matrix.rows(), matrix.cols());
    
    constexpr int TriangleType = (U & Lower) ? Lower : Upper;
    result.template selfadjointView<TriangleType>() = 
        matrix.template selfadjointView<TriangleType>().twistedBy(p_perm);
    
    result.makeCompressed();
    return result;
  }

  template <SparseFactorization_t S = Solver_>
  typename std::enable_if<S == SparseFactorizationLDLTTPP, Index>::type 
  peigs() const {
    eigen_assert(m_numericFactorization && "Numerical factorization must be computed first.");
    
    int num_positive = 0, num_zero = 0, num_negative = 0;
    int status = SparseGetInertia(*m_numericFactorization, &num_positive, &num_zero, &num_negative);
    
    if (status != 0) {
      // If SparseGetInertia fails, return -1 to indicate error
      return -1;
    }
    
    return static_cast<Index>(num_positive);
  }

  template <SparseFactorization_t S = Solver_>
  typename std::enable_if<S == SparseFactorizationLDLTTPP, Index>::type 
  neigs() const {
    eigen_assert(m_numericFactorization && "Numerical factorization must be computed first.");
    
    int num_positive = 0, num_zero = 0, num_negative = 0;
    int status = SparseGetInertia(*m_numericFactorization, &num_positive, &num_zero, &num_negative);
    
    if (status != 0) {
      // If SparseGetInertia fails, return -1 to indicate error
      return -1;
    }
    
    return static_cast<Index>(num_negative);
  }

  template <SparseFactorization_t S = Solver_>
  typename std::enable_if<S == SparseFactorizationLDLTTPP, Index>::type 
  zeigs() const {
    eigen_assert(m_numericFactorization && "Numerical factorization must be computed first.");
    
    int num_positive = 0, num_zero = 0, num_negative = 0;
    int status = SparseGetInertia(*m_numericFactorization, &num_positive, &num_zero, &num_negative);
    
    if (status != 0) {
      // If SparseGetInertia fails, return -1 to indicate error
      return -1;
    }
    
    return static_cast<Index>(num_zero);
  }

  inline int ppivs() const {
    // Accelerate doesn't provide direct pivot perturbation count
    // Return 0 as a safe default for now
    return 0;
  }

  // This method initializes the internal AccelSparseMatrix. This must be called 
  // after changing the sparsity pattern of m_matrix via the reference returned 
  // from MatrixType& getMatrix()
  void reinitializeInternalMatrixRepresentation();

  // Internal factorization methods 
  AccelerateImpl& analyzePattern_internal();
  AccelerateImpl& factorize_internal();
  AccelerateImpl& compute_internal();

  // Cleanup method
  void release();

  // Performance metrics 
  mutable int m_flops = 0;
  mutable int m_mem = 0;

 private:
  void buildAccelSparseMatrix() {
    SparseAttributes_t attributes{};
    attributes.kind = m_sparseKind;

    SparseMatrixStructure structure{};
    structure.blockSize = 1;

    if ((MatrixType::Flags & Eigen::ColMajor)) { // CSC format
      const Index nColumnsStarts = m_matrix.cols() + 1;
      m_columnStarts.resize(nColumnsStarts);
      std::copy_n(m_matrix.outerIndexPtr(), nColumnsStarts, m_columnStarts.data());

      structure.rowCount = static_cast<int>(m_matrix.rows());
      structure.columnCount = static_cast<int>(m_matrix.cols());
      structure.columnStarts = m_columnStarts.data();
      structure.rowIndices = const_cast<int*>(m_matrix.innerIndexPtr());
      attributes.transpose = false;
      attributes.triangle = m_triType;
    } else { // RowMajor (CSR) format
      // For CSR, Accelerate expects CSC. We use the 'transpose' attribute
      // to tell Accelerate to interpret the CSR matrix as a transposed CSC matrix.
      const Index nRowStarts = m_matrix.rows() + 1;
      m_columnStarts.resize(nRowStarts); // Reuse m_columnStarts for rowStarts
      std::copy_n(m_matrix.outerIndexPtr(), nRowStarts, m_columnStarts.data());

      structure.rowCount = static_cast<int>(m_matrix.cols()); // Swapped
      structure.columnCount = static_cast<int>(m_matrix.rows()); // Swapped
      structure.columnStarts = m_columnStarts.data(); // These are now rowStarts
      structure.rowIndices = const_cast<int*>(m_matrix.innerIndexPtr()); // These are now columnIndices
      attributes.transpose = true;
      
      // When transposing, we need to flip the triangle type for symmetric matrices
      if (m_sparseKind == SparseSymmetric) {
        attributes.triangle = (m_triType == SparseLowerTriangle) ? SparseUpperTriangle : SparseLowerTriangle;
      } else {
        attributes.triangle = m_triType;
      }
    }

    structure.attributes = attributes;
    m_accel_matrix.structure = structure;
    m_accel_matrix.data = const_cast<Scalar*>(m_matrix.valuePtr());
  }

  void doAnalysis() {
    m_numericFactorization.reset(nullptr);

    // Only resize permutation if necessary to avoid unnecessary allocations
    if (m_permutation.size() != static_cast<size_t>(m_nRows)) {
      m_permutation.resize(m_nRows);
    }
    std::iota(m_permutation.begin(), m_permutation.end(), 0);  // Initialize with identity

    SparseSymbolicFactorOptions fopts{};
    fopts.control = SparseDefaultControl;
    fopts.orderMethod = m_order;
    fopts.order = m_permutation.data();  // Provide storage for computed permutation
    fopts.ignoreRowsAndColumns = nullptr;
    fopts.malloc = malloc;
    fopts.free = free;

    fopts.reportError = [](const char* msg) {
      fmt::print(fmt::fg(fmt::color::red), "Accelerate Sparse Symbolic Factorization Error: {}\n", msg);
    };

    m_symbolicFactorization.reset(new SymbolicFactorization(SparseFactor(Solver_, m_accel_matrix.structure, fopts)));

    SparseStatus_t status = m_symbolicFactorization->status;

    updateInfoStatus(status);

    if (status != SparseStatusOK) {
      m_symbolicFactorization.reset(nullptr);
    }
  }

  void doFactorization() {
    SparseStatus_t status = SparseStatusReleased;

    if (m_symbolicFactorization) {

      SparseNumericFactorOptions nopts{};
      nopts.control = SparseDefaultControl;
      nopts.scalingMethod = SparseScalingDefault;
      nopts.scaling = nullptr;
      // Default values set by Apple
      nopts.pivotTolerance = 0.01;                   // Recommended value for difficult matrices in double
      nopts.zeroTolerance = 1e-4 * __DBL_EPSILON__;  // "A few" orders of magnitude below epsilon.

      // Get factor and workspace size
      const int factorSize = 
        std::is_same<Scalar, double>::value 
            ? m_symbolicFactorization->factorSize_Double 
            : m_symbolicFactorization->factorSize_Float;
      const int workspaceSize =
        std::is_same<Scalar, double>::value 
            ? m_symbolicFactorization->workspaceSize_Double 
            : m_symbolicFactorization->workspaceSize_Float;

      m_numericFactorization.reset(new NumericFactorization(SparseFactor(
        *m_symbolicFactorization, m_accel_matrix, nopts, 
        internal::resizeForAccelerateAlignment(factorSize, &m_factorStorage), 
        internal::resizeForAccelerateAlignment(workspaceSize, &m_workspace))));

      status = m_numericFactorization->status;

      if (status != SparseStatusOK) m_numericFactorization.reset(nullptr);
    }

    updateInfoStatus(status);
  }

 protected:
  void updateInfoStatus(SparseStatus_t status) const {
    switch (status) {
      case SparseStatusOK:
        m_info = Success;
        break;
      case SparseFactorizationFailed:
      case SparseMatrixIsSingular:
        m_info = NumericalIssue;
        break;
      case SparseInternalError:
      case SparseParameterError:
      case SparseStatusReleased:
      default:
        m_info = InvalidInput;
        break;
    }
  }

  std::vector<long> m_columnStarts;
  mutable MatrixType m_matrix;
  mutable AccelSparseMatrix m_accel_matrix;
  mutable ComputationInfo m_info;
  mutable std::vector<uint8_t> m_factorStorage;
  mutable std::vector<uint8_t> m_workspace;
  mutable std::vector<uint8_t> m_solve_workspace;  // Cache solve workspace
  mutable std::vector<Scalar> m_r_mem;
  mutable int m_cached_solve_workspace_size = 0;   // Track cached size
  Index m_nRows, m_nCols;
  std::unique_ptr<SymbolicFactorization, SymbolicFactorizationDeleter> m_symbolicFactorization;
  std::unique_ptr<NumericFactorization, NumericFactorizationDeleter> m_numericFactorization;
  SparseKind_t m_sparseKind;
  SparseTriangle_t m_triType;
  SparseOrder_t m_order;
  bool m_doIterativeRefinement;
  int m_iterativeRefinementIterations;
  mutable std::vector<StorageIndex> m_permutation;  // Store permutation from factorization
};

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::setMatrix(const MatrixType& matrix) {
  if (EnforceSquare_) {
    eigen_assert(matrix.rows() == matrix.cols());
  }

  //m_matrix = matrix;
  getMatrix(matrix);
  m_nRows = m_matrix.rows();
  m_nCols = m_matrix.cols();

  buildAccelSparseMatrix();

  m_isInitialized = false;
  m_symbolicFactorization.reset(nullptr);
  m_numericFactorization.reset(nullptr);
  m_cached_solve_workspace_size = 0;  // Clear cached workspace size
  m_info = Success;
}

/** Computes the symbolic and numeric decomposition of matrix \a a */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::compute(const MatrixType& a) {
  setMatrix(a);

  doAnalysis();

  if (m_symbolicFactorization) doFactorization();

  m_isInitialized = true;
}

/** Performs a symbolic decomposition on the sparsity pattern of matrix \a a.
 *
 * This function is particularly useful when solving for several problems having the same structure.
 *
 * \sa factorize()
 */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::analyzePattern(const MatrixType& a) {
  if (EnforceSquare_) {
    eigen_assert(a.rows() == a.cols());
  }
  setMatrix(a);

  doAnalysis();

  m_isInitialized = true;
}

/** Performs a numeric decomposition of matrix \a a.
 *
 * The given matrix must have the same sparsity pattern as the matrix on which the symbolic decomposition has been
 * performed.
 *
 * \sa analyzePattern()
 */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::factorize(const MatrixType& a) {
  eigen_assert(m_symbolicFactorization && "You must first call analyzePattern()");
  eigen_assert(m_nRows == a.rows() && m_nCols == a.cols());

  if (EnforceSquare_) {
    eigen_assert(a.rows() == a.cols());
  }

  m_matrix = a;
  buildAccelSparseMatrix();
  m_numericFactorization.reset(nullptr);

  doFactorization();
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
template <typename Rhs, typename Dest>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::_solve_impl(const MatrixBase<Rhs>& b,
                                                                              MatrixBase<Dest>& x) const {
  if (!m_numericFactorization) {
    m_info = InvalidInput;
    return;
  }

  eigen_assert(m_nRows == b.rows());
  eigen_assert(((b.cols() == 1) || b.outerStride() == b.rows()));

  SparseStatus_t status = SparseStatusOK;

  Scalar* b_ptr = const_cast<Scalar*>(b.derived().data());
  Scalar* x_ptr = const_cast<Scalar*>(x.derived().data());

  AccelDenseMatrix xmat{};
  xmat.attributes = SparseAttributes_t();
  xmat.columnCount = static_cast<int>(x.cols());
  xmat.rowCount = static_cast<int>(x.rows());
  xmat.columnStride = xmat.rowCount;
  xmat.data = x_ptr;

  AccelDenseMatrix bmat{};
  bmat.attributes = SparseAttributes_t();
  bmat.columnCount = static_cast<int>(b.cols());
  bmat.rowCount = static_cast<int>(b.rows());
  bmat.columnStride = bmat.rowCount;
  bmat.data = b_ptr;

  const int nrhs = (bmat.attributes.transpose) ? bmat.rowCount : bmat.columnCount;
  const int workspaceSize = m_numericFactorization->solveWorkspaceRequiredStatic + 
    nrhs*m_numericFactorization->solveWorkspaceRequiredPerRHS;

  // Use cached solve workspace to avoid repeated allocations
  void* ws;
  if (workspaceSize != m_cached_solve_workspace_size) {
    ws = internal::resizeForAccelerateAlignment(workspaceSize, &m_solve_workspace);
    m_cached_solve_workspace_size = workspaceSize;
  } else {
    // Reuse existing aligned workspace
    constexpr int kAccelerateRequiredAlignment = 16;
    ws = reinterpret_cast<void*>(m_solve_workspace.data() + 
         (kAccelerateRequiredAlignment - reinterpret_cast<uintptr_t>(m_solve_workspace.data()) % kAccelerateRequiredAlignment) % kAccelerateRequiredAlignment);
  }
  assert(ws != nullptr && "Accelerate workspace alignment failed");

  SparseSolve(*m_numericFactorization, bmat, xmat, ws);

  updateInfoStatus(status);

  if (m_doIterativeRefinement)
  {
    auto n = vDSP_Length(x.rows() * x.cols());
    if (m_r_mem.size() < n) {
        m_r_mem.resize(n);
    }
    AccelDenseMatrix ref_mat{};
    ref_mat.attributes = SparseAttributes_t();
    ref_mat.columnCount = static_cast<int>(x.cols());
    ref_mat.rowCount = static_cast<int>(x.rows());
    ref_mat.columnStride = ref_mat.rowCount;
    ref_mat.data = m_r_mem.data();

    for (int i = 0; i < m_iterativeRefinementIterations; ++i) {
        // Calculate residual and store in ref_mat
        vDSP_vnegD(
            bmat.data, 1,
            ref_mat.data, 1, n
        );
        SparseMultiplyAdd(m_accel_matrix, xmat, ref_mat);

        // Solve for correction and store in ref_mat
        SparseSolve(*m_numericFactorization, ref_mat, ws);

        // vDSP operation that calculates x -= correction
        vDSP_vsubD(
            ref_mat.data, 1,
            xmat.data, 1,
            xmat.data, 1,
            n
        );
    }
  }
}

/** Initializes the internal AccelSparseMatrix from the internal Eigen sparse matrix
 * 
 * This method must be called after changing the sparsity pattern of the m_matrix 
 * member via the reference returned from MatrixType& getMatrix().
 * 
 */
template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::reinitializeInternalMatrixRepresentation()
{
  // Update matrix dimensions in case they changed when the sparsity pattern was modified
  m_nRows = m_matrix.rows();
  m_nCols = m_matrix.cols();
  
  // Build/rebuild the internal AccelSparseMatrix from the current state of m_matrix
  buildAccelSparseMatrix();
  
  // Reset factorizations since the matrix structure has changed
  m_symbolicFactorization.reset(nullptr);
  m_numericFactorization.reset(nullptr);
  
  // Clear cached workspace size since matrix structure changed
  m_cached_solve_workspace_size = 0;
  
  // Reset initialization state - will need to recompute factorizations
  m_isInitialized = false;
  
  // Reset computation info
  m_info = Success;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>& 
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::analyzePattern_internal() {
  eigen_assert(m_matrix.rows() > 0 && m_matrix.cols() > 0 && "Matrix must be set before calling analyzePattern_internal()");
  
  m_symbolicFactorization.reset(nullptr);
  doAnalysis();
  
  m_isInitialized = true;
  return *this;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>& 
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::factorize_internal() {
  eigen_assert(m_symbolicFactorization && "You must first call analyzePattern_internal()");
  
  m_numericFactorization.reset(nullptr);
  doFactorization();
  
  // Update performance metrics after factorization
  if (m_numericFactorization) {
    // Estimate memory usage based on factorization storage
    m_mem = static_cast<int>(m_factorStorage.size() + m_workspace.size());
    
    // Estimate FLOP count based on matrix size and sparsity
    // This is a rough estimate since Accelerate doesn't provide exact FLOP counts
    Index nnz = m_matrix.nonZeros();
    m_flops = static_cast<int>(nnz * std::log(static_cast<double>(m_nRows)));
  }
  
  return *this;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>& 
AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::compute_internal() {
  eigen_assert(m_matrix.rows() > 0 && m_matrix.cols() > 0 && "Matrix must be set before calling compute_internal()");
  
  m_symbolicFactorization.reset(nullptr);
  m_numericFactorization.reset(nullptr);

  doAnalysis();
  
  if (m_symbolicFactorization) {
    doFactorization();
    
    // Update performance metrics after factorization
    if (m_numericFactorization) {
      // Estimate memory usage based on factorization storage
      m_mem = static_cast<int>(m_factorStorage.size() + m_workspace.size());
      
      // Estimate FLOP count based on matrix size and sparsity
      Index nnz = m_matrix.nonZeros();
      m_flops = static_cast<int>(nnz * std::log(static_cast<double>(m_nRows)));
    }
  }
  
  m_isInitialized = true;
  return *this;
}

template <typename MatrixType_, int UpLo_, SparseFactorization_t Solver_, bool EnforceSquare_>
void AccelerateImpl<MatrixType_, UpLo_, Solver_, EnforceSquare_>::release() {
  // Clean up factorizations
  m_numericFactorization.reset(nullptr);
  m_symbolicFactorization.reset(nullptr);
  
  // Clear storage vectors
  m_factorStorage.clear();
  m_workspace.clear();
  m_solve_workspace.clear();
  m_r_mem.clear();
  
  // Clear matrix data
  m_matrix.resize(0, 0);
  m_matrix.data().squeeze();
  
  // Clear permutation
  m_permutation.clear();
  
  // Reset cached sizes
  m_cached_solve_workspace_size = 0;
  
  // Reset performance metrics
  m_flops = 0;
  m_mem = 0;
  
  // Reset initialization state
  m_isInitialized = false;
  m_info = Success;
}

}  // end namespace Eigen

#endif  // EIGEN_ACCELERATESUPPORT_H
