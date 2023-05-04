#pragma once

#include "pch.h"

namespace ASSET {

  template<int CS>
  struct LGLCoeffs {
    /// implement general algorithm for deteriming coefficients
  };

  //////////////////////////////////////////////////////////////
  ///////////////////// LGL-3///Cubic Hermite///////////////////
  template<>
  struct LGLCoeffs<2> {  /// Cubic Hermite-LGL3

    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;

    static constexpr STDarray<double, 2> CardinalSpacings = {0.0, 1.0};
    static constexpr STDarray<double, 1> InteriorSpacings = {0.5};

    //// Interpolate Center Point
    static constexpr STDarray<STDarray<double, 2>, 1> Cardinal_XInterp_Weights = {
        STDarray<double, 2> {0.5, 0.5}};
    static constexpr STDarray<STDarray<double, 2>, 1> Cardinal_UPoly_Weights = {
        STDarray<double, 2> {0.5, 0.5}};
    static constexpr STDarray<STDarray<double, 2>, 1> Cardinal_DXInterp_Weights = {
        STDarray<double, 2> {0.125, -0.125}};

    //// Defect Equations
    static constexpr STDarray<STDarray<double, 2>, 1> Cardinal_XDef_Weights = {
        STDarray<double, 2> {1.0, -1.0}};
    static constexpr STDarray<STDarray<double, 2>, 1> Cardinal_DXDef_Weights = {
        STDarray<double, 2> {1.0 / 6.0, 1.0 / 6.0}};
    static constexpr STDarray<double, 1> Interior_DXDef_Weights = {4.0 / 6.0};

    /// Weights for Integrals
    static constexpr STDarray<double, 2> Cardinal_Integral_Weights = {1.0 / 3.0, 1.0 / 3.0};

    static constexpr STDarray<double, 1> Interior_Integral_Weights = {4.0 / 3.0};
    static constexpr STDarray<double, 2> Reduced_Integral_Weights = {0.5, 0.5};
    /// Weights for Interpolation powers highest to lowest function t^3,t^2,t,1.0
    static constexpr STDarray<STDarray<double, 4>, 2> Cardinal_XPower_Weights = {
        STDarray<double, 4> {2.0, -3.0, 0.0, 1.0}, STDarray<double, 4> {-2.0, 3.0, 0.0, 0.0}};

    static constexpr STDarray<STDarray<double, 4>, 2> Cardinal_DXPower_Weights = {
        STDarray<double, 4> {1.0, -2.0, 1.0, 0.0}, STDarray<double, 4> {1.0, -1.0, 0.0, 0.0}};

    static constexpr STDarray<STDarray<double, 2>, 2> Cardinal_UPolyPower_Weights = {
        STDarray<double, 2> {-1.0, 1.0}, STDarray<double, 2> {1.0, 0.0}};


    static constexpr double Order = 3.0;
    static constexpr double ErrorWeight = 0.0026041666661458227;
  };
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////
  ///////////////////// LGL-5///////////////////////////////////
  template<>
  struct LGLCoeffs<3> {  /// Cubic Hermite-LGL3

    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;

    // Coefficients for first eqn
    static constexpr double SQRT21 = 4.58257569495584;

    static constexpr double a = (1.0 / 686.0);
    static constexpr double a1 = (39.0 * SQRT21 + 231.0) * a;
    static constexpr double a2 = 224.0 * a;
    static constexpr double a3 = (-39.0 * SQRT21 + 231.0) * a;
    static constexpr double a4 = (3.0 * SQRT21 + 21.0) * a;
    static constexpr double a5 = (-16.0 * SQRT21) * a;
    static constexpr double a6 = (3.0 * SQRT21 - 21.0) * a;

    // Coefficients for second eqn
    static constexpr double b = (1.0 / 686.0);
    static constexpr double b1 = (-39.0 * SQRT21 + 231.0) * b;
    static constexpr double b2 = 224.0 * b;
    static constexpr double b3 = (39.0 * SQRT21 + 231.0) * b;
    static constexpr double b4 = (-3.0 * SQRT21 + 21.0) * b;
    static constexpr double b5 = (16.0 * SQRT21) * b;
    static constexpr double b6 = (-3.0 * SQRT21 - 21.0) * b;

    // Coefficients for defect one
    static constexpr double c = (1.0 / 360);
    static constexpr double c1 = (32.0 * SQRT21 + 180.0) * c;
    static constexpr double c2 = -64.0 * SQRT21 * c;
    static constexpr double c3 = (32.0 * SQRT21 - 180.0) * c;
    static constexpr double c4 = (9.0 + SQRT21) * c;
    static constexpr double c5 = 98.0 * c;
    static constexpr double c6 = 64.0 * c;
    static constexpr double c7 = (9.0 - SQRT21) * c;

    // Coefficients for defect two
    static constexpr double d = (1.0 / 360.0);
    static constexpr double d1 = (-32.0 * SQRT21 + 180.0) * d;
    static constexpr double d2 = 64.0 * SQRT21 * d;
    static constexpr double d3 = (-32.0 * SQRT21 - 180.0) * d;
    static constexpr double d4 = (9.0 - SQRT21) * d;
    static constexpr double d5 = 98.0 * d;
    static constexpr double d6 = 64.0 * d;
    static constexpr double d7 = (9.0 + SQRT21) * d;

    static constexpr STDarray<double, 3> CardinalSpacings = {0.0, 0.5, 1.0};
    static constexpr STDarray<double, 2> InteriorSpacings = {0.172673164646011, 0.827326835353989};

    //// Interpolate Center Point
    static constexpr STDarray<STDarray<double, 3>, 2> Cardinal_XInterp_Weights = {
        STDarray<double, 3> {a1, a2, a3}, STDarray<double, 3> {b1, b2, b3}};

    static constexpr STDarray<STDarray<double, 3>, 2> Cardinal_UPoly_Weights = {
        STDarray<double, 3> {0.541612549639704, 0.571428571428571, -0.113041121068274},
        STDarray<double, 3> {-0.113041121068274, 0.571428571428571, 0.541612549639704},
    };

    static constexpr STDarray<STDarray<double, 3>, 2> Cardinal_DXInterp_Weights = {
        STDarray<double, 3> {a4, a5, a6}, STDarray<double, 3> {b4, b5, b6}};

    //// Defect Equations
    static constexpr STDarray<STDarray<double, 3>, 2> Cardinal_XDef_Weights = {
        STDarray<double, 3> {c1, c2, c3}, STDarray<double, 3> {d1, d2, d3}};
    static constexpr STDarray<STDarray<double, 3>, 2> Cardinal_DXDef_Weights = {
        STDarray<double, 3> {c4, c6, c7}, STDarray<double, 3> {d4, d6, d7}};

    static constexpr STDarray<double, 2> Interior_DXDef_Weights = {c5, d5};

    /// Weights for Integrals
    static constexpr STDarray<double, 3> Cardinal_Integral_Weights = {0.1, 0.7111111111111111111111, 0.1};
    static constexpr STDarray<double, 2> Interior_Integral_Weights = {0.544444444444444444444,
                                                                      0.544444444444444444444};

    static constexpr STDarray<double, 3> Reduced_Integral_Weights = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};

    /// Weights for Interpolation powers highest to lowest function
    /// t^5,t^4,t^3,t^2,t,1.0
    static constexpr STDarray<STDarray<double, 6>, 3> Cardinal_XPower_Weights = {
        STDarray<double, 6> {24.0, -68.0, 66.0, -23.0, 0.0, 1.0},
        STDarray<double, 6> {0.0, 16.0, -32.0, +16.0, 0.0, 0.0},
        STDarray<double, 6> {-24.0, 52.0, -34.0, +7.0, 0.0, 0.0}};

    static constexpr STDarray<STDarray<double, 6>, 3> Cardinal_DXPower_Weights = {
        STDarray<double, 6> {4.0, -12.0, +13.0, -6.0, 1.0, 0.0},
        STDarray<double, 6> {16.0, -40.0, +32.0, -8.0, 0.0, 0.0},
        STDarray<double, 6> {4.0, -8.0, +5.0, -1.0, 0.0, 0.0}};

    static constexpr STDarray<STDarray<double, 3>, 3> Cardinal_UPolyPower_Weights = {
        STDarray<double, 3> {2.0, -3.0, 1.0},
        STDarray<double, 3> {-4.0, 4.0, 0.0},
        STDarray<double, 3> {2.0, -1.0, 0.0},
    };

    static constexpr STDarray<STDarray<double, 3>, 1> UZeroSpline_Weights = {
        STDarray<double, 3> {-3.0, 4.0, -1.0}};
    static constexpr STDarray<STDarray<double, 3>, 1> UOneSpline_Weights = {
        STDarray<double, 3> {1.0, -4.0, 3.0}};

    static constexpr double Order = 5.0;
    static constexpr double ErrorWeight = 3.100198409908181e-06;
  };
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////
  ///////////////////// LGL-7///////////////////////////////////
  template<>
  struct LGLCoeffs<4> {
    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;

    static constexpr double T0 = +0.00000000000000;
    static constexpr double Ti1 = +8.48880518607166e-2;
    static constexpr double Ti2 = +2.65575603264643e-1;
    static constexpr double Tic = +0.50000000000000;
    static constexpr double Ti3 = +7.34424396735357e-1;
    static constexpr double Ti4 = +9.15111948139283e-1;
    static constexpr double TE = +1.00000000000000;
    ////First Defect done
    static constexpr double ai1 = +6.18612232711785e-1;
    static constexpr double ai21 = +3.34253095933642e-1;
    static constexpr double ai31 = +1.52679626438851e-2;
    static constexpr double ai_11 = +3.18667087106879e-2;

    static constexpr double bi1 = +8.84260109348311e-1;
    static constexpr double bi21 = -8.23622559094327e-1;
    static constexpr double bi31 = -2.35465327970606e-2;
    static constexpr double bi_11 = -3.70910174569208e-2;

    static constexpr double vi1 = +2.57387738427162e-2;
    static constexpr double vi21 = -5.50098654524528e-2;
    static constexpr double vi31 = -1.53026046503702e-2;
    static constexpr double v_11 = -2.38759243962924e-3;

    static constexpr double wi1 = +1.62213410652341e-2;
    static constexpr double wi11 = +1.38413023680783e-1;
    static constexpr double wi21 = +9.71662045547156e-2;
    static constexpr double wi31 = +1.85682012187242e-2;
    static constexpr double w_11 = +2.74945307600086e-3;
    ////////////////////////////////////
    ////Second Defect
    static constexpr double aic = +1.41445282326366e-1;
    static constexpr double ai2c = +3.58554717673634e-1;
    static constexpr double ai3c = +3.58554717673634e-1;
    static constexpr double ai_1c = +1.41445282326366e-1;

    static constexpr double bic = +7.86488731947674e-2;
    static constexpr double bi2c = +8.00076026297266e-1;
    static constexpr double bi3c = -8.00076026297266e-1;
    static constexpr double bi_1c = -7.86488731947674e-2;

    static constexpr double vic = +9.92317607754556e-3;
    static constexpr double vi2c = +9.62835932121973e-2;
    static constexpr double vi3c = -9.62835932121973e-2;
    static constexpr double v_1c = -9.92317607754556e-3;

    static constexpr double wic = +4.83872966828888e-3;
    static constexpr double wi2c = +1.00138284831491e-1;
    static constexpr double wicc = +2.43809523809524e-1;
    static constexpr double wi3c = +1.00138284831491e-1;
    static constexpr double w_1c = +4.83872966828888e-3;
    ///////////////////////////////////
    ////third defect
    static constexpr double ai4 = +3.18667087106879e-2;
    static constexpr double ai24 = +1.52679626438851e-2;
    static constexpr double ai34 = +3.34253095933642e-1;
    static constexpr double ai_14 = +6.18612232711785e-1;

    static constexpr double bi4 = +3.70910174569208e-2;
    static constexpr double bi24 = +2.35465327970606e-2;
    static constexpr double bi34 = +8.23622559094327e-1;
    static constexpr double bi_14 = -8.84260109348311e-1;

    static constexpr double vi4 = +2.38759243962924e-3;
    static constexpr double vi24 = +1.53026046503702e-2;
    static constexpr double vi34 = +5.50098654524528e-2;
    static constexpr double v_14 = -2.57387738427162e-2;

    static constexpr double wi4 = +2.74945307600086e-3;
    static constexpr double wi24 = +1.85682012187242e-2;
    static constexpr double wi34 = +9.71662045547156e-2;
    static constexpr double wi44 = +1.38413023680783e-1;
    static constexpr double w_14 = +1.62213410652341e-2;

    static constexpr double w0 = 0.04761904761904761904762;
    static constexpr double w1 = 0.276826047361565948011;
    static constexpr double w2 = 0.4317453812098626234179;
    static constexpr double w3 = 0.487619047619047619048;
    static constexpr double w4 = 0.431745381209862623418;
    static constexpr double w5 = 0.2768260473615659480107;
    static constexpr double w6 = 0.04761904761904761904762;

    static constexpr STDarray<double, 4> CardinalSpacings = {T0, Ti2, Ti3, TE};
    static constexpr STDarray<double, 3> InteriorSpacings = {Ti1, Tic, Ti4};

    //// Interpolate Center Point
    static constexpr STDarray<STDarray<double, 4>, 3> Cardinal_XInterp_Weights = {
        STDarray<double, 4> {ai1, ai21, ai31, ai_11},
        STDarray<double, 4> {aic, ai2c, ai3c, ai_1c},
        STDarray<double, 4> {ai4, ai24, ai34, ai_14}};

    static constexpr STDarray<STDarray<double, 4>, 3> Cardinal_DXInterp_Weights = {
        STDarray<double, 4> {vi1, vi21, vi31, v_11},
        STDarray<double, 4> {vic, vi2c, vi3c, v_1c},
        STDarray<double, 4> {vi4, vi24, vi34, v_14}};

    static constexpr STDarray<STDarray<double, 4>, 3> Cardinal_UPoly_Weights = {
        STDarray<double, 4> {0.550643660407289, 0.551767574740443, -0.153490305524281, 0.0510790703765507},
        STDarray<double, 4> {-0.140877081724073, 0.640877081724073, 0.640877081724073, -0.140877081724073},
        STDarray<double, 4> {0.0510790703765507, -0.153490305524281, 0.551767574740443, 0.550643660407289}};


    //// Defect Equations
    static constexpr STDarray<STDarray<double, 4>, 3> Cardinal_XDef_Weights = {
        STDarray<double, 4> {bi1, bi21, bi31, bi_11},
        STDarray<double, 4> {bic, bi2c, bi3c, bi_1c},
        STDarray<double, 4> {bi4, bi24, bi34, bi_14}};

    static constexpr STDarray<STDarray<double, 4>, 3> Cardinal_DXDef_Weights = {
        STDarray<double, 4> {wi1, wi21, wi31, w_11},
        STDarray<double, 4> {wic, wi2c, wi3c, w_1c},
        STDarray<double, 4> {wi4, wi24, wi34, w_14}};

    static constexpr STDarray<double, 3> Interior_DXDef_Weights = {wi11, wicc, wi44};

    /// Weights for Integrals
    static constexpr STDarray<double, 4> Cardinal_Integral_Weights = {w0, w2, w4, w6};
    static constexpr STDarray<double, 3> Interior_Integral_Weights = {w1, w3, w5};

    /// Weights for Interpolation powers highest to lowest function
    /// t^7,t^6,t^5,t^4,t^3,t^2,t,1.0
    static constexpr STDarray<STDarray<double, 8>, 4> Cardinal_XPower_Weights = {
        STDarray<double, 8> {322.113192893432,
                             -1262.16647180554,
                             +1953.18722397597,
                             -1497.44073672143,
                             +575.419724269949,
                             -92.112932612382,
                             0.0,
                             +1.0},
        STDarray<double, 8> {-64.79204848488,
                             +361.542466375583,
                             -764.578109564441,
                             +777.477964267544,
                             -383.431222919698,
                             73.7809503258911,
                             0.0,
                             0.0},
        STDarray<double, 8> {64.7920484849059,
                             -92.0018730186832,
                             -44.0436705064007,
                             +110.002715108135,
                             -43.8271690468644,
                             +5.07794897890751,
                             0.0,
                             0.0},
        STDarray<double, 8> {-322.11319289346,
                             +992.625878448649,
                             -1144.56544390514,
                             +609.960057345752,
                             -148.161332303388,
                             +13.2540333075835,
                             0.0,
                             0.0}};

    static constexpr STDarray<STDarray<double, 8>, 4> Cardinal_DXPower_Weights = {
        STDarray<double, 8> {26.2862997682608,
                             -105.145199073049,
                             +167.971831917166,
                             -135.907298995812,
                             +58.0483996910198,
                             -12.254033307585,
                             +1.0,
                             0.0},
        STDarray<double, 8> {119.581459799146,
                             -446.567920871157,
                             +645.538793826344,
                             -446.829224641783,
                             +145.40645229346,
                             -17.1295604060043,
                             0.0,
                             0.0},
        STDarray<double, 8> {119.581459799146,
                             -390.502297722867,
                             +477.341924381445,
                             -267.69702439259,
                             +67.4701675365851,
                             -6.19422960171859,
                             0.0,
                             0.0},
        STDarray<double, 8> {26.2862997682629,
                             -78.8588993047889,
                             +89.1129326123729,
                             -46.7943663834307,
                             +11.2540333075837,
                             -0.999999999999869,
                             0.0,
                             0.0}};
    static constexpr STDarray<double, 4> Reduced_Integral_Weights = {
        -5.12701665379258 / 4.0 + 10.2540333075852 / 3.0 - 6.12701665379258 / 2.0 + 1.0,
        10.9353308042859 / 4.0 - 18.9665045333251 / 3.0 + 8.03117372903925 / 2.0,
        -10.9353308042859 / 4.0 + 13.8394878795326 / 3.0 - 2.90415707524666 / 2.0,
        5.12701665379258 / 4.0 - 5.12701665379258 / 3.0 + 1.0 / 2.0};

    static constexpr STDarray<STDarray<double, 4>, 4> Cardinal_UPolyPower_Weights = {
        STDarray<double, 4> {-5.12701665379258, +10.2540333075852, -6.12701665379258, +1.0},
        STDarray<double, 4> {10.9353308042859, -18.9665045333251, +8.03117372903925, 0.0},
        STDarray<double, 4> {-10.9353308042859, +13.8394878795326, -2.90415707524666, 0.0},
        STDarray<double, 4> {5.12701665379258, -5.12701665379258, +1.0, 0.0}};

    static constexpr STDarray<STDarray<double, 4>, 2> UZeroSpline_Weights = {
        STDarray<double, 4> {-6.12701665379258, 8.03117372903925, -2.90415707524666, 1.0},
        STDarray<double, 4> {10.2540333075852 * 2.0,
                             -18.9665045333251 * 2.0,
                             +13.8394878795326 * 2.0,
                             -5.12701665379258 * 2.0}};

    static constexpr STDarray<STDarray<double, 4>, 2> UOneSpline_Weights = {
        STDarray<double, 4> {-5.12701665379258 * 3.0 + 10.2540333075852 * 2.0 - 6.12701665379258,
                             10.9353308042859 * 3.0 - 18.9665045333251 * 2.0 + 8.03117372903925,
                             -10.9353308042859 * 3.0 + 13.8394878795326 * 2.0 - 2.90415707524666,
                             5.12701665379258 * 3.0 - 5.12701665379258 * 2.0 + 1.0},

        STDarray<double, 4> {-5.12701665379258 * 6.0 + 10.2540333075852 * 2.0,
                             10.9353308042859 * 6.0 - 18.9665045333251 * 2.0,
                             -10.9353308042859 * 6.0 + 13.8394878795326 * 2.0,
                             5.12701665379258 * 6.0 - 5.12701665379258 * 2.0}};


    static constexpr double Order = 7.0;
    static constexpr double ErrorWeight = 2.9357939455472746e-09;
  };
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////
  ///////////////////// LGL-9///////////////////////////////////

  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
}  // namespace ASSET
