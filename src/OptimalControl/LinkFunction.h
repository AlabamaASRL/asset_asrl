#pragma once

#include "OptimalControlFlags.h"
#include "pch.h"

namespace ASSET {


  template<class FuncType>
  struct LinkFunction {
    FuncType Func;
    LinkFlags LinkFlag = LinkFlags::ReadRegions;

    Eigen::Matrix<PhaseRegionFlags, -1, 1> PhaseRegFlags;
    std::vector<Eigen::VectorXi> PhasesTolink;
    std::vector<Eigen::VectorXi> XtUVars;
    std::vector<Eigen::VectorXi> OPVars;
    std::vector<Eigen::VectorXi> SPVars;
    std::vector<Eigen::VectorXi> LinkParams;

    int StorageIndex = 0;
    int GlobalIndex = 0;

    void init(FuncType f,
              Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
              std::vector<Eigen::VectorXi> PTL,
              std::vector<Eigen::VectorXi> xtv,
              std::vector<Eigen::VectorXi> opv,
              std::vector<Eigen::VectorXi> spv,
              std::vector<Eigen::VectorXi> lv) {


      Eigen::VectorXi empty;
      empty.resize(0);

      int nappl = std::max(PTL.size(), lv.size());
      int nxs = xtv.size();
      int nos = opv.size();
      int nss = spv.size();

      int nmax = std::max({nxs, nos, nss});


      if (nappl == 0) {
        throw std::invalid_argument("PTL vector and link param vector cannot both have 0 size");
      }

      if (PTL.size() == 0) {
        if (RegFlags.size() != 0) {
          throw std::invalid_argument("PTL vector element and RegFlags vector must be same size");
        }
        for (int i = 0; i < nappl; i++) {
          PTL.push_back(empty);
        }
      }
      if (lv.size() == 0) {
        for (int i = 0; i < nappl; i++) {
          lv.push_back(empty);
        }
      }

      if (nmax > 0 && PTL.size() > 0) {
        if (nxs == 0) {
          for (int i = 0; i < nmax; i++) {
            xtv.push_back(empty);
          }
        }

        if (nos == 0) {
          for (int i = 0; i < nmax; i++) {
            opv.push_back(empty);
          }
        }

        if (nss == 0) {
          for (int i = 0; i < nmax; i++) {
            spv.push_back(empty);
          }
        }
      }


      this->PhasesTolink = PTL;
      this->PhaseRegFlags = RegFlags;
      this->Func = f;


      this->XtUVars = xtv;
      this->OPVars = opv;
      this->SPVars = spv;
      this->LinkParams = lv;
    }

    Eigen::Matrix<PhaseRegionFlags, -1, 1> makePhaseRegFlags(LinkFlags Flag) {
      Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags;
      this->LinkFlag = Flag;

      switch (Flag) {
        case LinkFlags::BackToFront: {
          RegFlags.resize(2);
          RegFlags << PhaseRegionFlags::Back, PhaseRegionFlags::Front;
          break;
        }
        case LinkFlags::BackToBack: {
          RegFlags.resize(2);
          RegFlags << PhaseRegionFlags::Back, PhaseRegionFlags::Back;

          break;
        }
        case LinkFlags::FrontToBack: {
          RegFlags.resize(2);
          RegFlags << PhaseRegionFlags::Front, PhaseRegionFlags::Back;

          break;
        }
        case LinkFlags::FrontToFront: {
          RegFlags.resize(2);
          RegFlags << PhaseRegionFlags::Front, PhaseRegionFlags::Front;
          break;
        }
        case LinkFlags::ParamsToParams: {
          RegFlags.resize(2);
          RegFlags << PhaseRegionFlags::Params, PhaseRegionFlags::Params;
          break;
        }
        case LinkFlags::PathToPath: {
          RegFlags.resize(2);
          RegFlags << PhaseRegionFlags::Path, PhaseRegionFlags::Path;
          break;
        }
        case LinkFlags::LinkParams: {
          RegFlags.resize(0);
          break;
        }
        default:
          break;
      }
      return RegFlags;
    }

    LinkFunction(FuncType f,
                 Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv,
                 std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv,
                 std::vector<Eigen::VectorXi> lv) {
      this->init(f, RegFlags, PTL, xtv, opv, spv, lv);
    }
    LinkFunction(FuncType f,
                 LinkFlags Flag,
                 std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv,
                 std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv,
                 std::vector<Eigen::VectorXi> lv) {
      this->init(f, makePhaseRegFlags(Flag), PTL, xtv, opv, spv, lv);
    }

    LinkFunction(FuncType f,
                 std::vector<std::string> RegFlags,
                 std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv,
                 std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv,
                 std::vector<Eigen::VectorXi> lv) {
      this->init(f, strto_PhaseRegionFlag(RegFlags), PTL, xtv, opv, spv, lv);
    }
    LinkFunction(FuncType f,
                 std::string Flag,
                 std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv,
                 std::vector<Eigen::VectorXi> opv,
                 std::vector<Eigen::VectorXi> spv,
                 std::vector<Eigen::VectorXi> lv) {
      this->init(f, makePhaseRegFlags(strto_LinkFlag(Flag)), PTL, xtv, opv, spv, lv);
    }


    LinkFunction(FuncType f,
                 Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv) {
      Eigen::VectorXi empty;
      empty.resize(0);
      std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
      std::vector<Eigen::VectorXi> emptyvecLV(PTL.size(), empty);

      this->init(f, RegFlags, PTL, xtv, emptyvec, emptyvec, emptyvecLV);
    }
    LinkFunction(FuncType f,
                 LinkFlags Flag,
                 std::vector<Eigen::VectorXi> PTL,
                 std::vector<Eigen::VectorXi> xtv) {
      Eigen::VectorXi empty;
      empty.resize(0);
      std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
      std::vector<Eigen::VectorXi> emptyvecLV(PTL.size(), empty);

      this->init(f, makePhaseRegFlags(Flag), PTL, xtv, emptyvec, emptyvec, emptyvecLV);
    }
    LinkFunction(FuncType f,
                 Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags,
                 std::vector<Eigen::VectorXi> PTL,
                 Eigen::VectorXi xtv) {
      Eigen::VectorXi empty;
      empty.resize(0);
      std::vector<Eigen::VectorXi> xtvvec(PTL[0].size(), xtv);
      std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
      std::vector<Eigen::VectorXi> emptyvecLV(PTL.size(), empty);

      this->init(f, RegFlags, PTL, xtvvec, emptyvec, emptyvec, emptyvecLV);
    }
    LinkFunction(FuncType f, LinkFlags Flag, std::vector<Eigen::VectorXi> PTL, Eigen::VectorXi xtv) {
      Eigen::VectorXi empty;
      empty.resize(0);
      std::vector<Eigen::VectorXi> xtvvec(PTL[0].size(), xtv);
      std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
      std::vector<Eigen::VectorXi> emptyvecLV(PTL.size(), empty);

      this->init(f, makePhaseRegFlags(Flag), PTL, xtvvec, emptyvec, emptyvec, emptyvecLV);
    }

    LinkFunction(FuncType f, std::string Flag, std::vector<Eigen::VectorXi> PTL, Eigen::VectorXi xtv) {
      Eigen::VectorXi empty;
      empty.resize(0);
      std::vector<Eigen::VectorXi> xtvvec(PTL[0].size(), xtv);
      std::vector<Eigen::VectorXi> emptyvec(PTL[0].size(), empty);
      std::vector<Eigen::VectorXi> emptyvecLV(PTL.size(), empty);

      this->init(f, makePhaseRegFlags(strto_LinkFlag(Flag)), PTL, xtvvec, emptyvec, emptyvec, emptyvecLV);
    }


    LinkFunction() {
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<LinkFunction<FuncType>>(m, name);

      obj.def(py::init<FuncType, LinkFlags, std::vector<Eigen::VectorXi>, Eigen::VectorXi>());
      obj.def(py::init<FuncType,
                       Eigen::Matrix<PhaseRegionFlags, -1, 1>,
                       std::vector<Eigen::VectorXi>,
                       std::vector<Eigen::VectorXi>,
                       std::vector<Eigen::VectorXi>,
                       std::vector<Eigen::VectorXi>,
                       std::vector<Eigen::VectorXi>>());
      obj.def(py::init<FuncType,
                       LinkFlags,
                       std::vector<Eigen::VectorXi>,
                       std::vector<Eigen::VectorXi>,
                       std::vector<Eigen::VectorXi>,
                       std::vector<Eigen::VectorXi>,
                       std::vector<Eigen::VectorXi>>());
    }
  };

}  // namespace ASSET
