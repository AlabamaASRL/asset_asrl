#pragma once

#include "pch.h"

namespace ASSET {

  enum FDCoeffType {
    Backwards,
    Central,
    Forwards
  };

  template<int SZ>
  using arr = std::array<double, SZ>;

  template<int Order, int Accuracy, int Dir, int Shift>
  struct FDCoeffs;

  ////////////////////////////////////////////////////////////////////////////////
  //// Forward Coefficients
  /// No shift
  // 1st derivatives
  template<>
  struct FDCoeffs<1, 2, FDCoeffType::Forwards, 0> {
    static constexpr int N = 3;
    static constexpr arr<N> weights = {-1.5, 2, -0.5};
  };

  template<>
  struct FDCoeffs<1, 4, FDCoeffType::Forwards, 0> {
    static constexpr int N = 5;
    static constexpr arr<N> weights = {-25 / 12.0, 4, -3, 4 / 3.0, -1 / 4.0};
  };

  template<>
  struct FDCoeffs<1, 6, FDCoeffType::Forwards, 0> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {-49 / 20.0, 6, -15 / 2.0, 20 / 3.0, -15 / 4.0, 6 / 5.0, -1 / 6.0};
  };

  // 2nd derivatives
  template<>
  struct FDCoeffs<2, 2, FDCoeffType::Forwards, 0> {
    static constexpr int N = 4;
    static constexpr arr<N> weights = {2, -5, 4, -1};
  };

  template<>
  struct FDCoeffs<2, 4, FDCoeffType::Forwards, 0> {
    static constexpr int N = 6;
    static constexpr arr<N> weights = {15 / 4.0, -77 / 6.0, 107 / 6.0, -13, 61 / 12.0, -5 / 6.0};
  };

  template<>
  struct FDCoeffs<2, 6, FDCoeffType::Forwards, 0> {
    static constexpr int N = 8;
    static constexpr arr<N> weights = {
        469 / 90.0, -223 / 10.0, 879 / 20.0, -949 / 18.0, 41, -201 / 10.0, 1019 / 180.0, -7 / 10.0};
  };

  // 3rd derivatives
  template<>
  struct FDCoeffs<3, 2, FDCoeffType::Forwards, 0> {
    static constexpr int N = 5;
    static constexpr arr<N> weights = {-5 / 2.0, 9, -12, 7, -3 / 2.0};
  };

  template<>
  struct FDCoeffs<3, 4, FDCoeffType::Forwards, 0> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {-49 / 8.0, 29, -461 / 8.0, 62, -307 / 8.0, 13, -15 / 8.0};
  };

  template<>
  struct FDCoeffs<3, 6, FDCoeffType::Forwards, 0> {
    static constexpr int N = 9;
    static constexpr arr<N> weights = {-801 / 80.0,
                                       349 / 6.0,
                                       -18353 / 120.0,
                                       2391 / 10.0,
                                       -1457 / 30.0,
                                       -561 / 8.0,
                                       527 / 30.0,
                                       -469 / 240.0};
  };

  // 4th derivatives
  template<>
  struct FDCoeffs<4, 2, FDCoeffType::Forwards, 0> {
    static constexpr int N = 6;
    static constexpr arr<N> weights = {3, -14, 26, -24, 11, -2};
  };

  template<>
  struct FDCoeffs<4, 4, FDCoeffType::Forwards, 0> {
    static constexpr int N = 8;
    static constexpr arr<N> weights = {
        28 / 3.0, -111 / 2.0, 142, -1219 / 6.0, 176, -185 / 2.0, 82 / 3.0, -7 / 2.0};
  };

  //------------------------------------------------------------------------------
  /// 1 shift
  // 1st derivatives
  template<>
  struct FDCoeffs<1, 4, FDCoeffType::Forwards, 1> {
    static constexpr int N = 5;
    static constexpr arr<N> weights = {-1 / 4.0, -5 / 6.0, 3 / 2.0, -1 / 2.0, 1 / 12.0};
  };

  template<>
  struct FDCoeffs<1, 6, FDCoeffType::Forwards, 1> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {-1 / 6.0, -77 / 60.0, 5 / 2.0, -5 / 3.0, 5 / 6.0, -1 / 4.0, 1 / 30.0};
  };

  // 2nd derivatives
  template<>
  struct FDCoeffs<2, 4, FDCoeffType::Forwards, 1> {
    static constexpr int N = 6;
    static constexpr arr<N> weights = {5 / 6.0, -5 / 4.0, -1 / 3.0, 7 / 6.0, -1 / 2.0, 1 / 12.0};
  };

  template<>
  struct FDCoeffs<2, 6, FDCoeffType::Forwards, 1> {
    static constexpr int N = 8;
    static constexpr arr<N> weights = {
        7 / 10.0, -7 / 18.0, -27 / 10.0, 19 / 4.0, -67 / 18.0, 9 / 5.0, -1 / 2.0, 11 / 180.0};
  };

  // 3rd derivatives
  template<>
  struct FDCoeffs<3, 2, FDCoeffType::Forwards, 1> {
    static constexpr int N = 5;
    static constexpr arr<N> weights = {-3 / 2.0, 5, -6, 3, -1 / 2.0};
  };

  template<>
  struct FDCoeffs<3, 4, FDCoeffType::Forwards, 1> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {-15 / 8.0, 7, -83 / 8.0, 8, -29 / 8.0, 1, -1 / 8.0};
  };

  template<>
  struct FDCoeffs<3, 6, FDCoeffType::Forwards, 1> {
    static constexpr int N = 9;
    static constexpr arr<N> weights = {-469 / 240.0,
                                       303 / 40.0,
                                       -731 / 60.0,
                                       269 / 24.0,
                                       -57 / 8.0,
                                       407 / 120.0,
                                       -67 / 60.0,
                                       9 / 40.0,
                                       -1 / 48.0};
  };

  // 4th derivatives
  template<>
  struct FDCoeffs<4, 2, FDCoeffType::Forwards, 1> {
    static constexpr int N = 6;
    static constexpr arr<N> weights = {2, -9, 16, -14, 6, -1};
  };

  template<>
  struct FDCoeffs<4, 4, FDCoeffType::Forwards, 1> {
    static constexpr int N = 8;
    static constexpr arr<N> weights = {7 / 2.0, -56 / 3.0, 85 / 2.0, -54, 251 / 6.0, -20, 11 / 2.0, -2 / 3.0};
  };

  //------------------------------------------------------------------------------
  /// 2 shift
  // 1st derivatives
  template<>
  struct FDCoeffs<1, 6, FDCoeffType::Forwards, 2> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {1 / 30.0, -2 / 5.0, -7 / 12.0, 4 / 3.0, -1 / 2.0, 2 / 15.0, -1 / 60.0};
  };

  // 2nd derivatives
  template<>
  struct FDCoeffs<2, 6, FDCoeffType::Forwards, 2> {
    static constexpr int N = 8;
    static constexpr arr<N> weights = {
        -11 / 180.0, 107 / 90.0, -21 / 10.0, 13 / 18.0, 17 / 36.0, -3 / 10.0, 4 / 45.0, -1 / 90.0};
  };

  // 3rd derivatives
  template<>
  struct FDCoeffs<3, 4, FDCoeffType::Forwards, 2> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {-1 / 8.0, -1, 35 / 8.0, -6, 29 / 8.0, -1, 1 / 8.0};
  };

  template<>
  struct FDCoeffs<3, 6, FDCoeffType::Forwards, 2> {
    static constexpr int N = 9;
    static constexpr arr<N> weights = {-1 / 48.0,
                                       -53 / 30.0,
                                       273 / 40.0,
                                       -313 / 30.0,
                                       103 / 12.0,
                                       -9 / 2.0,
                                       197 / 120.0,
                                       -11 / 30.0,
                                       3 / 80.0};
  };

  // 4th derivatives
  template<>
  struct FDCoeffs<4, 4, FDCoeffType::Forwards, 2> {
    static constexpr int N = 8;
    static constexpr arr<N> weights = {
        2 / 3.0, -11 / 6.0, 0, 31 / 6.0, -22 / 3.0, 9 / 2.0, -4 / 3.0, 1 / 6.0};
  };

  ////////////////////////////////////////////////////////////////////////////////
  //// Backward Coefficients
  template<int Order, int Accuracy, int Shift>
  struct FDCoeffs<Order, Accuracy, FDCoeffType::Backwards, Shift>
      : FDCoeffs<Order, Accuracy, FDCoeffType::Forwards, Shift> {
    using Base = FDCoeffs<Order, Accuracy, FDCoeffType::Forwards, Shift>;
    using Base::N;

    static constexpr arr<N> flip(const arr<N> a) {
      arr<N> out;
      for (int i = 0; i < N; i++) {
        out[i] = C * a[N - i];
      }
      return out;
    }

    static constexpr int C = ((Order / 2) == ((Order + 1) / 2)) ? (1) : (-1);
    static constexpr arr<N> weights = flip(Base::weights);
  };

  ////////////////////////////////////////////////////////////////////////////////
  //// Centered Coefficients
  // 1st derivatives
  template<>
  struct FDCoeffs<1, 2, FDCoeffType::Central, 0> {
    static constexpr int N = 3;
    static constexpr arr<N> weights = {-0.5, 0, 0.5};
  };

  template<>
  struct FDCoeffs<1, 4, FDCoeffType::Central, 0> {
    static constexpr int N = 5;
    static constexpr arr<N> weights = {1 / 12.0, -2 / 3.0, 0, 2 / 3.0, -1 / 12.0};
  };

  template<>
  struct FDCoeffs<1, 6, FDCoeffType::Central, 0> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {-1 / 60.0, 3 / 20.0, -3 / 4.0, 0, 3 / 4.0, -3 / 20.0, 1 / 60.0};
  };

  template<>
  struct FDCoeffs<1, 8, FDCoeffType::Central, 0> {
    static constexpr int N = 9;
    static constexpr arr<N> weights = {
        1 / 280.0, -4 / 105.0, 1 / 5.0, -4 / 5.0, 0, 4 / 5.0, -1 / 5.0, 4 / 105.0, -1 / 280.0};
  };

  // 2nd derivatives
  template<>
  struct FDCoeffs<2, 2, FDCoeffType::Central, 0> {
    static constexpr int N = 3;
    static constexpr arr<N> weights = {1, -2, 1};
  };

  template<>
  struct FDCoeffs<2, 4, FDCoeffType::Central, 0> {
    static constexpr int N = 5;
    static constexpr arr<N> weights = {-1 / 12.0, 4 / 3.0, -5 / 2.0, 4 / 3.0, -1 / 12.0};
  };

  template<>
  struct FDCoeffs<2, 6, FDCoeffType::Central, 0> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {
        1 / 90.0, -3 / 20.0, 3 / 2.0, -49 / 18.0, 3 / 2.0, -3 / 20.0, 1 / 90.0};
  };

  template<>
  struct FDCoeffs<2, 8, FDCoeffType::Central, 0> {
    static constexpr int N = 9;
    static constexpr arr<N> weights = {
        -1 / 560.0, 8 / 315.0, -1 / 5.0, 8 / 5.0, -205 / 72.0, 8 / 5.0, -1 / 5.0, 8 / 315.0, -1 / 560.0};
  };

  // 3rd derivatives
  template<>
  struct FDCoeffs<3, 2, FDCoeffType::Central, 0> {
    static constexpr int N = 5;
    static constexpr arr<N> weights = {-0.5, 1, 0, -1, 0.5};
  };

  template<>
  struct FDCoeffs<3, 4, FDCoeffType::Central, 0> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {1 / 8.0, -1, 13 / 8.0, 0, -13 / 8.0, 1, -1 / 8.0};
  };

  template<>
  struct FDCoeffs<3, 6, FDCoeffType::Central, 0> {
    static constexpr int N = 9;
    static constexpr arr<N> weights = {
        -7 / 240.0, 3 / 10.0, -169 / 120.0, 61 / 30.0, 0, -61 / 30.0, 169 / 120.0, -3 / 10.0, 7 / 240.0};
  };

  // 4th derivatives
  template<>
  struct FDCoeffs<4, 2, FDCoeffType::Central, 0> {
    static constexpr int N = 5;
    static constexpr arr<N> weights = {1, -4, 6, -4, 1};
  };

  template<>
  struct FDCoeffs<4, 4, FDCoeffType::Central, 0> {
    static constexpr int N = 7;
    static constexpr arr<N> weights = {-1 / 6.0, 2, -13 / 2.0, 28 / 3.0, -13 / 2.0, 2, -1 / 6.0};
  };

  template<>
  struct FDCoeffs<4, 6, FDCoeffType::Central, 0> {
    static constexpr int N = 9;
    static constexpr arr<N> weights = {
        7 / 240.0, -2 / 5.0, 169 / 60.0, -122 / 15.0, 91 / 8.0, -122 / 15.0, 169 / 60.0, -2 / 5.0, 7 / 240.0};
  };

}  // namespace ASSET
