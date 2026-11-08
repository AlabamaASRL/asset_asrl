#pragma once

// ============================================================
// FILE: ForceField3.h
//
// Runtime interface for a 3->3 position force field g(r), evaluated in a
// body-fixed frame. SpiceForceFrame holds any ForceField3 by clone, so a new
// kernel becomes fusable by deriving from this class and implementing the
// three virtual methods. The Jacobian is dg/dr.
// ============================================================

#include <Eigen/Dense>
#include <memory>

namespace ASSET {

struct ForceField3 {
  virtual ~ForceField3() = default;

  virtual Eigen::Vector3d accel(const Eigen::Vector3d& r) const = 0;

  virtual void accel_jacobian(const Eigen::Vector3d& r, Eigen::Vector3d& g,
                              Eigen::Matrix3d& G) const = 0;

  virtual std::unique_ptr<ForceField3> clone() const = 0;
};

}  // namespace ASSET
