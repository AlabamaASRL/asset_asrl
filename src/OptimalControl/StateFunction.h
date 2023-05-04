#pragma once

#include "OptimalControlFlags.h"
#include "pch.h"

namespace ASSET {

  template<class FuncType>
  struct StateFunction {
    FuncType Func;
    PhaseRegionFlags RegionFlag = PhaseRegionFlags::NotSet;
    Eigen::VectorXi XtUVars;
    Eigen::VectorXi OPVars;
    Eigen::VectorXi SPVars;

    Eigen::VectorXi EXTVars;  // dirty i know

    int StorageIndex = 0;
    int PhaseLocalIndex = 0;
    int GlobalIndex = 0;

    StateFunction(
        FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, Eigen::VectorXi opv, Eigen::VectorXi spv) {
      this->RegionFlag = Reg;
      this->Func = f;
      this->XtUVars = xtuv;
      this->OPVars = opv;
      this->SPVars = spv;
    }
    StateFunction(FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv) {
      this->Func = f;

      switch (Reg) {
        case PhaseRegionFlags::ODEParams: {
          this->RegionFlag = PhaseRegionFlags::Params;
          this->XtUVars.resize(0);
          this->OPVars = xtuv;
          this->SPVars.resize(0);
          break;
        }
        case PhaseRegionFlags::StaticParams: {
          this->RegionFlag = PhaseRegionFlags::Params;
          this->XtUVars.resize(0);
          this->OPVars.resize(0);
          this->SPVars = xtuv;
          break;
        }
        default: {
          this->RegionFlag = Reg;
          this->XtUVars = xtuv;
          this->OPVars.resize(0);
          this->SPVars.resize(0);
          break;
        }
      }
    }
    StateFunction(
        FuncType f, PhaseRegionFlags Reg, Eigen::VectorXi xtuv, PhaseRegionFlags ParReg, Eigen::VectorXi pv) {
      this->Func = f;
      this->RegionFlag = Reg;
      this->XtUVars = xtuv;
      switch (ParReg) {
        case PhaseRegionFlags::ODEParams: {
          this->OPVars = pv;
          this->SPVars.resize(0);
          break;
        }
        case PhaseRegionFlags::StaticParams: {
          this->RegionFlag = PhaseRegionFlags::Params;
          this->OPVars.resize(0);
          this->SPVars = pv;
          break;
        }
        default: {
          throw std::invalid_argument("Param region flag must be either StaticParams or ODEParams");
        }
      }
    }
    StateFunction() {
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<StateFunction<FuncType>>(m, name);
      obj.def(py::init<FuncType, PhaseRegionFlags, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>());
      obj.def(py::init<FuncType, PhaseRegionFlags, Eigen::VectorXi, PhaseRegionFlags, Eigen::VectorXi>());
      obj.def(py::init<FuncType, PhaseRegionFlags, Eigen::VectorXi>());
    }
  };

}  // namespace ASSET
