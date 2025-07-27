#pragma once

#include <cstdlib>
#include <string>

// Sets the number of threads for Accelerate via the environment variable VECLIB_MAXIMUM_THREADS.
inline void accelerate_set_num_threads(int num_threads) {
    setenv("VECLIB_MAXIMUM_THREADS", std::to_string(num_threads).c_str(), 1);
}
