#pragma once

#include "pch.h"

namespace ASSET {

  enum RKOptions {
    RK4Classic,
    RK438,
    DOPRI54,
    DOPRI87,
    RK54,
    RK78,
    Ralston3,
    Ralston2,
    DOPRI5
  };

  static void RKFlagsBuild(py::module& m) {
    py::enum_<RKOptions>(m, "RKOptions")
        .value("RK4", RKOptions::RK4Classic)
        .value("DOPRI54", RKOptions::DOPRI54)
        .value("DOPRI87", RKOptions::DOPRI87);
  }

  template<RKOptions opt>
  struct RKCoeffs {};

  template<>
  struct RKCoeffs<RKOptions::RK4Classic> {
    static const int Stages = 4;
    static const bool isDiag = true;
    static const bool EmbeddedCorrector = false;

    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;

    static constexpr STDarray<STDarray<double, 3>, 3> ACoeffs = {STDarray<double, 3> {0.5, 0.0, 0.0},
                                                                 STDarray<double, 3> {0.0, 0.5, 0.0},
                                                                 STDarray<double, 3> {0.0, 0.0, 1.0}};

    static constexpr STDarray<double, 3> Times = {0.5, 0.5, 1.0};
    static constexpr STDarray<double, 4> BCoeffs = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    static constexpr STDarray<double, 4> CCoeffs = {0, 0, 0, 0};
  };


  template<>
  struct RKCoeffs<RKOptions::DOPRI54> {
    static const int Stages = 7;
    static const bool isDiag = false;
    static const bool EmbeddedCorrector = true;
    static const bool FSAL = true;

    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;

    static constexpr STDarray<STDarray<double, 6>, 6> ACoeffs = {
        STDarray<double, 6> {1 / 5.0, 0, 0, 0, 0, 0},
        STDarray<double, 6> {3.0 / 40.0, 9 / 40.0, 0, 0, 0, 0},
        STDarray<double, 6> {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0, 0, 0},
        STDarray<double, 6> {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0, 0},
        STDarray<double, 6> {
            9017.0 / 3168.0, -355.0 / 33.0, 46732 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0},
        STDarray<double, 6> {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0}};

    static constexpr STDarray<double, 6> Times = {1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0};
    static constexpr STDarray<double, 7> BCoeffs = {
        35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187 / 6784.0, 11.0 / 84.0, 0.0};
    static constexpr STDarray<double, 7> CCoeffs = {
        5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393 / 640.0, -92097 / 339200.0, 187 / 2100.0, 1.0 / 40.0};

    static constexpr STDarray<double, 7> MidCoeffs = {0.2002686376600479,
                                                      0.0000000000000000,
                                                      0.7836643588368518,
                                                      -0.0596492035318963,
                                                      0.1178653667448159,
                                                      -0.0899577761820872,
                                                      0.0478086164722679};
  };

  template<>
  struct RKCoeffs<RKOptions::DOPRI5> {
    static const int Stages = 6;
    static const bool isDiag = false;
    static const bool EmbeddedCorrector = false;

    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;

    static constexpr STDarray<STDarray<double, 5>, 5> ACoeffs = {
        STDarray<double, 5> {1 / 5.0, 0, 0, 0, 0},
        STDarray<double, 5> {3.0 / 40.0, 9 / 40.0, 0, 0, 0},
        STDarray<double, 5> {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0, 0},
        STDarray<double, 5> {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0},
        STDarray<double, 5> {
            9017.0 / 3168.0, -355.0 / 33.0, 46732 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0}};

    static constexpr STDarray<double, 5> Times = {1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0};
    static constexpr STDarray<double, 6> BCoeffs = {
        35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187 / 6784.0, 11.0 / 84.0};
    static constexpr STDarray<double, 6> CCoeffs = {
        5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393 / 640.0, -92097 / 339200.0, 187 / 2100.0};
  };

  template<>
  struct RKCoeffs<RKOptions::DOPRI87> {
    static const int Stages = 13;
    static const bool isDiag = false;
    static const bool EmbeddedCorrector = true;
    static const bool FSAL = false;
    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;

    static constexpr STDarray<STDarray<double, 12>, 12> ACoeffs = {
        STDarray<double, 12> {1.0 / 18.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        STDarray<double, 12> {1.0 / 48.0, 1.0 / 16.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        STDarray<double, 12> {1.0 / 32.0, 0, 3.0 / 32.0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        STDarray<double, 12> {5.0 / 16.0, 0, -75.0 / 64.0, 75.0 / 64.0, 0, 0, 0, 0, 0, 0, 0, 0},

        STDarray<double, 12> {3.0 / 80.0, 0, 0, 3.0 / 16.0, 3.0 / 20.0, 0, 0, 0, 0, 0, 0, 0},
        STDarray<double, 12> {29443841.0 / 614563906.0,
                              0,
                              0,
                              77736538.0 / 692538347.0,
                              -28693883.0 / 1125000000.0,
                              23124283.0 / 1800000000.0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0},
        STDarray<double, 12> {16016141.0 / 946692911.0,
                              0,
                              0,
                              61564180.0 / 158732637.0,
                              22789713.0 / 633445777.0,
                              545815736.0 / 2771057229.0,
                              -180193667.0 / 1043307555.0,
                              0,
                              0,
                              0,
                              0,
                              0},
        STDarray<double, 12> {39632708.0 / 573591083.0,
                              0,
                              0,
                              -433636366.0 / 683701615.0,
                              -421739975.0 / 2616292301.0,
                              100302831.0 / 723423059.0,
                              790204164.0 / 839813087.0,
                              800635310.0 / 3783071287.0,
                              0,
                              0,
                              0,
                              0},

        STDarray<double, 12> {246121993.0 / 1340847787.0,
                              0,
                              0,
                              -37695042795.0 / 15268766246.0,
                              -309121744.0 / 1061227803.0,
                              -12992083.0 / 490766935.0,
                              6005943493.0 / 2108947869.0,
                              393006217.0 / 1396673457.0,
                              123872331.0 / 1001029789.0,
                              0,
                              0,
                              0},
        STDarray<double, 12> {-1028468189.0 / 846180014.0,
                              0,
                              0,
                              8478235783.0 / 508512852.0,
                              1311729495.0 / 1432422823.0,
                              -10304129995.0 / 1701304382.0,
                              -48777925059.0 / 3047939560.0,
                              15336726248.0 / 1032824649,
                              -45442868181.0 / 3398467696,
                              3065993473.0 / 597172653.0,
                              0,
                              0},
        STDarray<double, 12> {185892177.0 / 718116043.0,
                              0,
                              0,
                              -3185094517.0 / 667107341.0,
                              -477755414.0 / 1098053517.0,
                              -703635378.0 / 230739211.0,
                              5731566787.0 / 1027545527,
                              5232866602.0 / 850066563.0,
                              -4093664535.0 / 808688257.0,
                              3962137247.0 / 1805957418,
                              65686358.0 / 487910083.0,
                              0},
        STDarray<double, 12> {403863854.0 / 491063109.0,
                              0,
                              0,
                              -5068492393.0 / 434740067.0,
                              -411421997.0 / 543043805.0,
                              652783627.0 / 914296604.0,
                              11173962825.0 / 925320556,
                              -13158990841.0 / 6184727034.0,
                              3936647629.0 / 1978049680.0,
                              -160528059.0 / 685178525,
                              248638103.0 / 1413531060.0,
                              0},

    };

    static constexpr STDarray<double, 12> Times = {1.0 / 18.0,
                                                   1.0 / 12.0,
                                                   1.0 / 8,
                                                   5.0 / 16.0,
                                                   3.0 / 8.0,
                                                   59.0 / 400.0,
                                                   93.0 / 200.0,
                                                   5490023248.0 / 9719169821.0,
                                                   13.0 / 20.0,
                                                   1201146811.0 / 1299019798.0,
                                                   1.0,
                                                   1.0};
    static constexpr STDarray<double, 13> BCoeffs = {14005451.0 / 335480064.0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     -59238493.0 / 1068277825.0,
                                                     181606767.0 / 758867731.0,
                                                     561292985.0 / 797845732.0,
                                                     -1041891430.0 / 1371343529.0,
                                                     760417239.0 / 1151165299.0,
                                                     118820643.0 / 751138087.0,
                                                     -528747749.0 / 2220607170.0,
                                                     1.0 / 4.0};
    static constexpr STDarray<double, 13> CCoeffs = {13451932.0 / 455176623.0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     -808719846.0 / 976000145.0,
                                                     1757004468.0 / 5645159321.0,
                                                     656045339.0 / 265891186.0,
                                                     -3867574721.0 / 1518517206.0,
                                                     465885868.0 / 322736535.0,
                                                     53011238.0 / 667516719.0,
                                                     2.0 / 45.0,
                                                     0};


    static constexpr STDarray<double, 14> MidCoeffs = {0.0820626072147879,
                                                       0.0000000000000000,
                                                       0.0000000000000000,
                                                       0.0000000000000000,
                                                       0.0000000000000000,
                                                       0.1020112560398276,
                                                       0.4777861354824404,
                                                       0.6193740287992207,
                                                       -0.4344650943510704,
                                                       0.1566681135866386,
                                                       -0.0037228739431160,
                                                       0.0141456884053577,
                                                       0.0138169474640115,
                                                       -0.0276768086980947};
  };

}  // namespace ASSET
