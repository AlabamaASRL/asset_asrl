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
  
  // Fallback for dsecnd() - returns CPU/wall-clock time in seconds
  // MKL's dsecnd returns time in seconds since an arbitrary point (often program start)
  // This implementation returns seconds since epoch which is compatible for timing purposes
  inline double dsecnd() {
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return duration_cast<duration<double>>(duration).count();
  }

#endif // ASSET_HAS_MKL
