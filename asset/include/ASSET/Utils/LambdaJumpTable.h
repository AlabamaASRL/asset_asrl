#pragma once

#include <type_traits>

namespace ASSET {

  template<int J1, int J2, int J3>
  struct LambdaJumpTable {
    template<class Ftype>
    static void run(Ftype& f, int crit_size) {
      if (crit_size <= J1) {
        f(std::integral_constant<int, J1>());
      } else if (crit_size <= J2) {
        f(std::integral_constant<int, J2>());
      } else if (crit_size <= J3) {
        f(std::integral_constant<int, J3>());
      } else {
        f(std::integral_constant<int, -1>());
      }
    }
  };

}  // namespace ASSET
