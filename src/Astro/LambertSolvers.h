#pragma once
#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {


  void LambertSolversBuild(FunctionRegistry& reg, py::module& m);


  template<class Scalar, class WayBool, class IntType, class BranchBool, class ExitInt>
  void lambert_izzo_impl(const Vector3<Scalar>& R1dim,
                         const Vector3<Scalar>& R2dim,
                         Scalar dtdim,
                         double mu,
                         WayBool longway,
                         IntType Nin,
                         BranchBool rightbranch,
                         Vector3<Scalar>& V1,
                         Vector3<Scalar>& V2,
                         ExitInt& exint) {


    /// <summary>
    /// An implementation of Dario Izzo's Lambert algorithm vectorized with Eigen::Array types.  2x increase
    /// in throughput with array size of 4 or 8.
    /// </summary>
    /// <returns></returns>

    constexpr bool RisScalar = std::is_floating_point<Scalar>::value;
    constexpr bool WayisScalar = std::is_same<WayBool, bool>::value;
    constexpr bool BranchisScalar = std::is_same<BranchBool, bool>::value;
    constexpr bool NisScalar = std::is_same<IntType, int>::value;

    constexpr double Pi = 3.14159265358979;
    constexpr double tol = 1.0e-11;
    constexpr int maxiters = 20;


    // Using Eigen's array types as scalars causes annoying issues w/ the cross and norm methods, rewriting
    // here
    auto cross = [](const Vector3<Scalar>& x1, const Vector3<Scalar>& x2, Vector3<Scalar>& out) {
      out[0] = (x1[1] * x2[2] - x1[2] * x2[1]);
      out[1] = (x2[0] * x1[2] - x2[2] * x1[0]);
      out[2] = (x1[0] * x2[1] - x1[1] * x2[0]);
    };

    auto norm = [](const Vector3<Scalar>& x1) { return Scalar(sqrt(x1.dot(x1))); };


    Scalar lstar = norm(R1dim);
    Scalar vstar = sqrt(Scalar(mu) / lstar);
    Scalar tstar = lstar / vstar;

    Vector3<Scalar> R1 = R1dim / lstar;
    Vector3<Scalar> R2 = R2dim / lstar;
    Scalar dt = dtdim / tstar;


    Scalar r2 = norm(R2);
    Scalar logdt = log(dt);
    Scalar theta = acos(Scalar(R1.dot(R2)) / r2);

    /////////////////////////////////////////////////////////////////////////////////////////

    Scalar lwsign(1.0);
    Scalar branchsign(1.0);
    bool NisSame = false;


    if constexpr (WayisScalar) {
      if (longway) {
        theta = Scalar(2 * Pi - theta);
        lwsign = Scalar(-1.0);
      }
    } else {
      for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
        if (longway[i]) {
          theta[i] = 2.0 * Pi - theta[i];
          lwsign[i] = -1.0;
        }
      }
    }
    if constexpr (BranchisScalar) {
      if (rightbranch)
        branchsign = Scalar(-1.0);
    } else {
      for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
        if (rightbranch[i])
          branchsign[i] = -1.0;
      }
    }

    Scalar c = sqrt(1.0 + r2 * (r2 - 2.0 * cos(theta)));
    Scalar s = (c + 1.0 + r2) / 2.0;
    Scalar am = s / 2.0;
    Scalar lambda = sqrt(r2) * cos(theta / 2.0) / s;
    Scalar Nmax = floor(sqrt(Scalar(2.0) / (s * s * s)) * dt / Pi);


    Scalar x1, x2, N;

    if constexpr (NisScalar) {
      N = Scalar(double(Nin));
      if (int(Nin) == 0) {
        x1 = -0.740867916771357;
        x2 = 0.4208790341605184;
      } else {
        // N = std::max(N, Nmax);
        if constexpr (BranchisScalar) {
          if (rightbranch) {
            x1 = Scalar(tan(.7234 * Pi / 2.0));
            x2 = Scalar(tan(.5234 * Pi / 2.0));
          } else {
            x1 = Scalar(tan(-.5234 * Pi / 2.0));
            x2 = Scalar(tan(-.2234 * Pi / 2.0));
          }
        } else {
          for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
            if (rightbranch[i]) {
              x1[i] = tan(.7234 * Pi / 2.0);
              x2[i] = tan(.5234 * Pi / 2.0);
            } else {
              x1[i] = tan(-.5234 * Pi / 2.0);
              x2[i] = tan(-.2234 * Pi / 2.0);
            }
          }
        }
      }
    } else {
      for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
        N[i] = Nin[i];
        if (Nin[i] == 0) {
          x1 = -0.740867916771357;
          x2 = 0.4208790341605184;
        } else {
          if constexpr (BranchisScalar) {
            if (rightbranch) {
              x1[i] = tan(.7234 * Pi / 2.0);
              x2[i] = tan(.5234 * Pi / 2.0);
            } else {
              x1[i] = tan(-.5234 * Pi / 2.0);
              x2[i] = tan(-.2234 * Pi / 2.0);
            }
          } else {
            if (rightbranch[i]) {
              x1[i] = tan(.7234 * Pi / 2.0);
              x2[i] = tan(.5234 * Pi / 2.0);
            } else {
              x1[i] = tan(-.5234 * Pi / 2.0);
              x2[i] = tan(-.2234 * Pi / 2.0);
            }
          }
        }
      }
    }


    auto tofcon = [&](Scalar xin) {
      Scalar x;
      if constexpr (NisScalar) {
        if (Nin == 0) {
          x = exp(xin) - 1.0;
        } else {
          x = atan(xin) * 2.0 / Pi;
        }
      } else {
        if (NisSame) {
          if (Nin[0] == 0) {
            x = exp(xin) - 1.0;
          } else {
            x = atan(xin) * 2.0 / Pi;
          }
        } else {
          for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
            if (Nin[i] == 0) {
              x[i] = exp(xin[i]);
            } else {
              x[i] = atan(xin[i]) * 2.0 / Pi;
            }
          }
        }
      }

      Scalar a = am / (1 - x * x);
      Scalar alpha, beta, tof;

      if constexpr (RisScalar) {
        if (x < 1) {
          alpha = 2 * acos(x);
          beta = lwsign * 2 * asin(sqrt((s - c) / (2 * a)));
        } else {
          alpha = 2 * acosh(x);
          beta = lwsign * 2 * asinh(sqrt((s - c) / (-2 * a)));
        }
      } else {
        if (x.maxCoeff() < 1.0) {
          alpha = 2 * acos(x);
          beta = lwsign * 2 * asin(sqrt((s - c) / (2 * a)));
        } else if (x.minCoeff() > 1.0) {
          alpha = 2 * acosh(x);
          beta = lwsign * 2 * asinh(sqrt((s - c) / (-2 * a)));
        } else {
          for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
            if (x[i] < 1) {
              alpha[i] = 2 * acos(x[i]);
              beta[i] = lwsign[i] * 2 * asin(sqrt((s[i] - c[i]) / (2 * a[i])));
            } else {
              alpha[i] = 2 * acosh(x[i]);
              beta[i] = lwsign[i] * 2 * asinh(sqrt((s[i] - c[i]) / (-2 * a[i])));
            }
          }
        }
      }


      if constexpr (RisScalar) {
        if (a > 0) {
          tof = a * sqrt(a) * ((alpha - sin(alpha)) - (beta - sin(beta)) + 2 * Pi * N);
        } else {
          tof = -a * sqrt(-a) * ((sinh(alpha) - alpha) - (sinh(beta) - beta));
        }
      } else {
        if (a.minCoeff() > 0.0) {
          tof = a * sqrt(a) * ((alpha - sin(alpha)) - (beta - sin(beta)) + 2 * Pi * N);
        } else if (a.maxCoeff() < 0.0) {
          tof = -a * sqrt(-a) * ((sinh(alpha) - alpha) - (sinh(beta) - beta));
        } else {
          for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
            if (a[i] > 0) {
              tof[i] =
                  a[i] * sqrt(a[i]) * ((alpha[i] - sin(alpha[i])) - (beta[i] - sin(beta[i])) + 2 * Pi * N[i]);
            } else {
              tof[i] = -a[i] * sqrt(-a[i]) * ((sinh(alpha[i]) - alpha[i]) - (sinh(beta[i]) - beta[i]));
            }
          }
        }
      }
      Scalar y;
      if constexpr (NisScalar) {
        if (Nin == 0) {
          y = log(tof) - logdt;
        } else {
          y = tof - dt;
        }
      } else {
        if (NisSame) {
          if (Nin[0] == 0) {
            y = log(tof) - logdt;
          } else {
            y = tof - dt;
          }
        } else {
          for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
            if (Nin[i] == 0) {
              y[i] = log(tof[i]) - logdt[i];
            } else {
              y[i] = tof[i] - dt[i];
            }
          }
        }
      }


      return y;
    };


    Scalar y1 = tofcon(x1);
    Scalar y2 = tofcon(x2);

    Scalar xn, yn, err;
    int it = 0;

    while (true) {


      xn = (x1 * y2 - y1 * x2) / (y2 - y1);
      yn = tofcon(xn);
      x1 = x2;
      x2 = xn;
      y1 = y2;
      y2 = yn;
      if constexpr (!RisScalar) {
        y1 += Scalar(1.0e-15);
      }

      err = abs(x1 - xn);
      if constexpr (RisScalar) {
        if (err < tol) {
          break;
        }
      } else {
        if (err.maxCoeff() < tol) {
          break;
        }
      }

      it++;
      if (it > maxiters)
        break;
    }


    if constexpr (RisScalar) {
      if (err < tol) {
        exint = 0;
      } else {
        exint = 1;
      }
    } else {
      for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
        if (err[i] < tol) {
          exint[i] = 0;
        } else {
          exint[i] = 1;
        }
      }
    }


    Scalar x;
    if constexpr (NisScalar) {
      if (Nin == 0) {
        x = exp(xn) - 1.0;
      } else {
        x = atan(xn) * 2.0 / Pi;
      }
    } else {
      if (NisSame) {
        if (Nin[0] == 0) {
          x = exp(xn) - 1.0;
        } else {
          x = atan(xn) * 2.0 / Pi;
        }
      } else {
        for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
          if (Nin[i] == 0) {
            x[i] = exp(xn[i]) - 1.0;
          } else {
            x[i] = atan(xn[i]) * 2.0 / Pi;
          }
        }
      }
    }


    Scalar a = am / (1.0 - x * x);
    Scalar alpha, beta, psi, eta, eta2;

    if constexpr (RisScalar) {
      if (x < 1) {
        alpha = 2.0 * acos(x);
        beta = lwsign * 2.0 * asin(sqrt((s - c) / (2.0 * a)));
        psi = (alpha - beta) / 2.0;
        eta2 = 2.0 * a * pow(sin(psi), 2) / s;
        eta = sqrt(eta2);
      } else {
        alpha = 2.0 * acosh(x);
        beta = lwsign * 2.0 * asinh(sqrt((s - c) / (-2.0 * a)));
        psi = (alpha - beta) / 2.0;
        eta2 = -2.0 * a * pow(sinh(psi), 2) / s;
        eta = sqrt(eta2);
      }
    } else {
      if (x.maxCoeff() < 1.0) {
        alpha = 2.0 * acos(x);
        beta = lwsign * 2.0 * asin(sqrt((s - c) / (2 * a)));
        psi = (alpha - beta) / 2.0;
        eta2 = 2.0 * a * pow(sin(psi), 2) / s;
        eta = sqrt(eta2);
      } else if (x.minCoeff() > 1.0) {
        alpha = 2.0 * acosh(x);
        beta = lwsign * 2.0 * asinh(sqrt((s - c) / (-2 * a)));
        psi = (alpha - beta) / 2.0;
        eta2 = -2.0 * a * pow(sinh(psi), 2) / s;
        eta = sqrt(eta2);
      } else {
        for (int i = 0; i < Scalar::SizeAtCompileTime; i++) {
          if (x[i] < 1) {
            alpha[i] = 2.0 * acos(x[i]);
            beta[i] = lwsign[i] * 2.0 * asin(sqrt((s[i] - c[i]) / (2.0 * a[i])));
            psi[i] = (alpha[i] - beta[i]) / 2.0;
            eta2[i] = 2 * a[i] * pow(sin(psi[i]), 2) / s[i];
            eta[i] = sqrt(eta2[i]);
          } else {
            alpha[i] = 2.0 * acosh(x[i]);
            beta[i] = lwsign[i] * 2.0 * asinh(sqrt((s[i] - c[i]) / (-2.0 * a[i])));
            psi[i] = (alpha[i] - beta[i]) / 2.0;
            eta2[i] = -2.0 * a[i] * pow(sinh(psi[i]), 2) / s[i];
            eta[i] = sqrt(eta2[i]);
          }
        }
      }
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    R2 = R2 / r2;

    Vector3<Scalar> nhat;
    Vector3<Scalar> that1;
    Vector3<Scalar> that2;

    cross(R1, R2, nhat);
    nhat = (nhat / norm(nhat)) * lwsign;

    cross(nhat, R1, that1);
    cross(nhat, R2, that2);

    Scalar Vr1 = (1.0 / (eta * sqrt(am))) * (2.0 * lambda * am - lambda - x * eta);
    Scalar Vt1 = sqrt((r2 / (am * eta2)) * pow(sin(theta / 2.0), 2));

    Scalar Vt2 = Vt1 / r2;
    Scalar Vr2 = (Vt1 - Vt2) / tan(theta / 2.0) - Vr1;

    Vr1 *= vstar;
    Vt1 *= vstar;
    Vr2 *= vstar;
    Vt2 *= vstar;


    V1 = (Vr1 * R1 + Vt1 * that1);
    V2 = (Vr2 * R2 + Vt2 * that2);
  }


  template<class Scalar>
  std::array<Vector3<Scalar>, 2> lambert_izzo(
      const Vector3<Scalar>& R1, const Vector3<Scalar>& R2, Scalar dt, double mu, bool longway) {

    Vector3<Scalar> V1;
    Vector3<Scalar> V2;
    int exitcode;
    lambert_izzo_impl(R1, R2, dt, mu, longway, 0, false, V1, V2, exitcode);

    return std::array<Vector3<Scalar>, 2> {V1, V2};
  }

  template<class Scalar>
  std::array<Vector3<Scalar>, 2> lambert_izzo(const Vector3<Scalar>& R1,
                                              const Vector3<Scalar>& R2,
                                              Scalar dt,
                                              double mu,
                                              bool longway,
                                              int Nrevs,
                                              bool rightbranch) {

    Vector3<Scalar> V1;
    Vector3<Scalar> V2;
    int exitcode;
    lambert_izzo_impl(R1, R2, dt, mu, longway, Nrevs, rightbranch, V1, V2, exitcode);

    return std::array<Vector3<Scalar>, 2> {V1, V2};
  }


}  // namespace ASSET