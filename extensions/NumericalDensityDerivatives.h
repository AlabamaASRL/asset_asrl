#pragma once

#include <Eigen/Core>

namespace ASSET {

struct DensityDerivatives {
  double value;
  Eigen::Vector4d gradient;
  Eigen::Matrix4d hessian;
};

/** Centered derivatives for opaque empirical-atmosphere reference routines.
 *
 * The underlying NRLMSISE-00 C and JB2008 Fortran routines accept only
 * doubles. This evaluates the direct model symmetrically with independent
 * physical steps for height, latitude, longitude, and time. The resulting
 * Hessian is symmetric by construction. Callers must keep optimization nodes
 * away from weather-bin and model-cutoff discontinuities, or use the smooth
 * outer blend/taper supplied by SmoothAtmosphereBlend.
 */
template<class Evaluator>
DensityDerivatives centered_density_derivatives(
    const Evaluator& evaluate, const Eigen::Vector4d& x,
    const Eigen::Vector4d& steps) {
  DensityDerivatives result;
  result.value = evaluate(x);
  result.gradient.setZero();
  result.hessian.setZero();
  Eigen::Vector4d xp = x, xm = x;
  for (int i = 0; i < 4; ++i) {
    xp[i] += steps[i];
    xm[i] -= steps[i];
    const double fp = evaluate(xp);
    const double fm = evaluate(xm);
    result.gradient[i] = (fp - fm) / (2.0 * steps[i]);
    result.hessian(i, i) =
        (fp - 2.0 * result.value + fm) / (steps[i] * steps[i]);
    xp[i] = x[i];
    xm[i] = x[i];
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      Eigen::Vector4d xpp = x, xpm = x, xmp = x, xmm = x;
      xpp[i] += steps[i]; xpp[j] += steps[j];
      xpm[i] += steps[i]; xpm[j] -= steps[j];
      xmp[i] -= steps[i]; xmp[j] += steps[j];
      xmm[i] -= steps[i]; xmm[j] -= steps[j];
      const double value =
          (evaluate(xpp) - evaluate(xpm) - evaluate(xmp) + evaluate(xmm)) /
          (4.0 * steps[i] * steps[j]);
      result.hessian(i, j) = value;
      result.hessian(j, i) = value;
    }
  }
  return result;
}

}  // namespace ASSET
