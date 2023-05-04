#pragma once

namespace ASSET {
  /*!
   * @brief Factorial function
   *
   * @param i
   * @return int
   */
  inline int factorial(int i) {
    return (i == 1 || i == 0) ? 1 : factorial(i - 1) * i;
  }

  /*!
   * @brief Factorial between two integers
   *
   * @param i
   * @param j
   * @return int
   */
  inline int factorialDiv(int i, int j) {
    if (i == 0) {
      return factorialDiv(i + 1, j);
    } else if (j == 0) {
      return factorialDiv(i, j + 1);
    } else if (j == i + 1) {
      return j;
    } else if (i == j + 1) {
      return i;
    } else if (i > j) {
      return i * factorialDiv(i - 1, j);
    } else if (j > i) {
      return j * factorialDiv(i, j - 1);
    } else if (i == j) {
      return 1;
    } else {
      return 0;
    }
  }
}  // namespace ASSET
