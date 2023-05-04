#pragma once
#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {

  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////

  struct IdealSolarSail_Impl {
    /// <summary>
    /// ASSET Function to compute thrust output of an Ideal Solar Sail
    /// </summary>
    /// <param name="mu"></param>
    /// <returns></returns>
    static auto Definition(double mu, double beta) {

      double scale = mu * beta;
      auto RN = Arguments<6>();
      auto r = RN.head<3>();
      auto n = RN.tail<3>();

      auto ndr = r.dot(n);
      auto ndr2 = ndr * ndr;
      auto acc = scale * (ndr2 * r.inverse_norm_power<4>() * n.normalized_power<3>());

      return acc;
    }
  };

  BUILD_FROM_EXPRESSION(IdealSolarSail, IdealSolarSail_Impl, double, double);


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////


  struct NonIdealSolarSail_Impl {
    /// <summary>
    /// ASSET Function to compute thrust output of an Ideal Solar Sail
    /// </summary>
    /// <param name="mu"></param>
    /// <returns></returns>
    static auto Definition(double mu, double beta, double n1, double n2, double t1) {
      double scale = mu * beta / 2.0;

      auto RN = Arguments<6>();
      auto r = RN.head<3>();
      auto n = RN.tail<3>();

      auto ndr = r.dot(n);
      auto rn = r.norm() * n.norm();
      auto ncrn = n.cross(r).cross(n);
      auto n3dr4 = n.normalized_power<3>().dot(r.normalized_power<4>());

      auto acc = n3dr4 * (((n1 * scale) * ndr + (n2 * scale) * rn) * n + (t1 * scale) * ncrn);
      return acc;
    }
  };

  BUILD_FROM_EXPRESSION(NonIdealSolarSail, NonIdealSolarSail_Impl, double, double, double, double, double);


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////


}  // namespace ASSET