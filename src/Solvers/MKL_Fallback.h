#pragma once

// This header provides fallback implementations for MKL functions
// when compiling on platforms where MKL is not available (e.g., Windows ARM64)

#ifdef ASSET_HAS_MKL
  // If MKL is available, use it directly
  #include "mkl.h"
#else
  // Fallback implementations for platforms without MKL
  
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
  
  // Fallback for dsecnd() - returns elapsed time in seconds
  inline double dsecnd() {
    static auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = current_time - start_time;
    return elapsed.count();
  }

#endif // ASSET_HAS_MKL
