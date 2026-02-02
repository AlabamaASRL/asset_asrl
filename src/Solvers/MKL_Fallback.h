#pragma once

// This header provides fallback implementations for MKL functions
// when compiling on platforms where MKL is not available (e.g., Windows ARM64)
// On Windows ARM64, we use OpenBLAS for BLAS/LAPACK but still need fallbacks
// for MKL-specific functions like PARDISO and threading controls

#ifdef ASSET_HAS_MKL
  // If MKL is available, use it directly
  #include "mkl.h"
  
#elif defined(ASSET_HAS_OPENBLAS)
  // Using OpenBLAS: provides BLAS/LAPACK but not MKL-specific functions
  #include <cblas.h>
  #include <chrono>
  
  // OpenBLAS doesn't have MKL's thread control functions, provide stubs
  inline void mkl_set_num_threads(int num_threads) {
    // OpenBLAS uses environment variables (OPENBLAS_NUM_THREADS) or
    // openblas_set_num_threads() which may not be available in all builds
    // For now, this is a no-op - users should set OPENBLAS_NUM_THREADS
    (void)num_threads;
  }
  
  inline void mkl_set_num_threads_local(int num_threads) {
    // No direct equivalent in OpenBLAS
    (void)num_threads;
  }
  
  // Fallback for dsecnd() - returns CPU/wall-clock time in seconds
  // Returns time since first call to match MKL behavior
  inline double dsecnd() {
    using namespace std::chrono;
    static auto first_call = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    auto elapsed = now - first_call;
    return duration_cast<duration<double>>(elapsed).count();
  }
  
#else
  // No BLAS library available - pure fallback implementations
  #include <chrono>
  
  // Fallback for mkl_set_num_threads - no-op when MKL is not available
  inline void mkl_set_num_threads(int num_threads) {
    // No-op: Thread control is not available without MKL
    // Users should rely on OpenMP or other threading mechanisms
    (void)num_threads;
  }
  
  // Fallback for mkl_set_num_threads_local - no-op when MKL is not available
  inline void mkl_set_num_threads_local(int num_threads) {
    // No-op: Thread control is not available without MKL
    (void)num_threads;
  }
  
  // Fallback for dsecnd() - returns CPU/wall-clock time in seconds
  // MKL's dsecnd returns time in seconds since an arbitrary point (often program start)
  // Returns time since first call to match MKL behavior
  inline double dsecnd() {
    using namespace std::chrono;
    static auto first_call = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    auto elapsed = now - first_call;
    return duration_cast<duration<double>>(elapsed).count();
  }

#endif // ASSET_HAS_MKL / ASSET_HAS_OPENBLAS

