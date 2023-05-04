#pragma once

#include "pch.h"

namespace ASSET {

  enum PhaseRegionFlags {
    NotSet,
    Front,
    Back,
    FrontandBack,
    BackandFront,
    Path,
    InnerPath,
    NodalPath,
    DefectPath,
    PairWisePath,
    DefectPairWisePath,
    FrontNodalBackPath,
    Params,
    ODEParams,
    StaticParams,
    Accumulate,
    BlockDefectPath,
  };

  enum LinkFlags {
    BackToFront,
    FrontToBack,
    FrontToFront,
    BackToBack,
    ParamsToParams,
    LinkParams,
    BackTwoToTwoFront,
    FrontTwoToTwoBack,
    PathToPath,
    ReadRegions,
  };

  enum ControlModes {
    HighestOrderSpline,
    FirstOrderSpline,
    NoSpline,
    BlockConstant,
  };

  enum TranscriptionModes {
    LGL3,
    LGL5,
    LGL7,
    Trapezoidal,
    CentralShooting,
  };

  enum IntegralModes {
    BaseIntegral,
    SimpsonIntegral,
    TrapIntegral,
  };


  static PhaseRegionFlags strto_PhaseRegionFlag(const std::string& str) {

    if (str == "Front" || str == "First")
      return PhaseRegionFlags::Front;
    else if (str == "Back" || str == "Last")
      return PhaseRegionFlags::Back;
    else if (str == "Path")
      return PhaseRegionFlags::Path;
    else if (str == "ODEParams")
      return PhaseRegionFlags::ODEParams;
    else if (str == "StaticParams")
      return PhaseRegionFlags::StaticParams;
    else if (str == "FrontandBack" || str == "FirstandLast")
      return PhaseRegionFlags::FrontandBack;
    else if (str == "BackandFront" || str == "LastandFirst")
      return PhaseRegionFlags::BackandFront;
    else if (str == "InnerPath")
      return PhaseRegionFlags::InnerPath;
    else if (str == "PairWisePath")
      return PhaseRegionFlags::PairWisePath;
    else {
      auto msg = fmt::format("Unrecognized PhaseRegionFlag: {0}\n", str);
      throw std::invalid_argument(msg);
      return PhaseRegionFlags::NotSet;
    }
  }

  static Eigen::Matrix<PhaseRegionFlags, -1, 1> strto_PhaseRegionFlag(const std::vector<std::string>& strs) {
    Eigen::Matrix<PhaseRegionFlags, -1, 1> regvec(strs.size());

    for (int i = 0; i < strs.size(); i++) {
      regvec[i] = strto_PhaseRegionFlag(strs[i]);
    }
    return regvec;
  }

  static TranscriptionModes strto_TranscriptionMode(const std::string& str) {

    if (str == "LGL3")
      return TranscriptionModes::LGL3;
    else if (str == "LGL5")
      return TranscriptionModes::LGL5;
    else if (str == "LGL7")
      return TranscriptionModes::LGL7;
    else if (str == "CentralShooting")
      return TranscriptionModes::CentralShooting;
    else if (str == "Trapezoidal")
      return TranscriptionModes::Trapezoidal;
    else {
      auto msg = fmt::format("Unrecognized TranscriptionModes: {0}\n", str);
      throw std::invalid_argument(msg);
      return TranscriptionModes::LGL3;
    }
  }

  static LinkFlags strto_LinkFlag(const std::string& str) {

    if (str == "BackToBack" || str == "LastToLast")
      return LinkFlags::BackToBack;
    else if (str == "BackToFront" || str == "LastToFirst")
      return LinkFlags::BackToFront;
    else if (str == "FrontToBack" || str == "FirstToLast")
      return LinkFlags::FrontToBack;
    else if (str == "FrontToFront" || str == "FirstToFirst")
      return LinkFlags::FrontToFront;
    else if (str == "LinkParams")
      return LinkFlags::LinkParams;
    else if (str == "PathToPath")
      return LinkFlags::PathToPath;
    else {
      auto msg = fmt::format("Unrecognized LinkFlag: {0}\n", str);
      throw std::invalid_argument(msg);
      return LinkFlags::ReadRegions;
    }
  }


  static ControlModes strto_ControlMode(const std::string& str) {

    if (str == "FirstOrderSpline")
      return ControlModes::FirstOrderSpline;
    else if (str == "BlockConstant")
      return ControlModes::BlockConstant;
    else if (str == "HighestOrderSpline")
      return ControlModes::HighestOrderSpline;
    else if (str == "NoSpline")
      return ControlModes::NoSpline;
    else {
      auto msg = fmt::format("Unrecognized ControlMode: {0}\n", str);
      throw std::invalid_argument(msg);
      return ControlModes::NoSpline;
    }
  }


  static void OCPFlagsBuild(py::module& m) {
    py::enum_<TranscriptionModes>(m, "TranscriptionModes")
        .value("LGL3", TranscriptionModes::LGL3)
        .value("LGL5", TranscriptionModes::LGL5)
        .value("LGL7", TranscriptionModes::LGL7)
        .value("Trapezoidal", TranscriptionModes::Trapezoidal)
        .value("CentralShooting", TranscriptionModes::CentralShooting);

    py::enum_<IntegralModes>(m, "IntegralModes")
        .value("BaseIntegral", IntegralModes::BaseIntegral)
        .value("TrapIntegral", IntegralModes::TrapIntegral);

    py::enum_<ControlModes>(m, "ControlModes")
        .value("HighestOrderSpline", ControlModes::HighestOrderSpline)
        .value("FirstOrderSpline", ControlModes::FirstOrderSpline)
        .value("NoSpline", ControlModes::NoSpline)
        .value("BlockConstant", ControlModes::BlockConstant);

    py::enum_<PhaseRegionFlags>(m, "PhaseRegionFlags")
        .value("Front", PhaseRegionFlags::Front)
        .value("Back", PhaseRegionFlags::Back)
        .value("Path", PhaseRegionFlags::Path)
        .value("NodalPath", PhaseRegionFlags::NodalPath)
        .value("FrontandBack", PhaseRegionFlags::FrontandBack)
        .value("BackandFront", PhaseRegionFlags::BackandFront)
        .value("Params", PhaseRegionFlags::Params)
        .value("InnerPath", PhaseRegionFlags::InnerPath)
        .value("ODEParams", PhaseRegionFlags::ODEParams)
        .value("StaticParams", PhaseRegionFlags::StaticParams)
        .value("PairWisePath", PhaseRegionFlags::PairWisePath);

    py::enum_<LinkFlags>(m, "LinkFlags")
        .value("BackToFront", LinkFlags::BackToFront)
        .value("BackToBack", LinkFlags::BackToBack)
        .value("FrontToBack", LinkFlags::FrontToBack)
        .value("ParamsToParams", LinkFlags::ParamsToParams)
        .value("LinkParams", LinkFlags::LinkParams)
        .value("FrontToFront", LinkFlags::FrontToFront)
        .value("PathToPath", LinkFlags::PathToPath)
        .value("BackTwoToTwoFront", LinkFlags::BackTwoToTwoFront)
        .value("FrontTwoToTwoBack", LinkFlags::FrontTwoToTwoBack);

    m.def("strto_PhaseRegionFlag", py::overload_cast<const std::string&>(&ASSET::strto_PhaseRegionFlag));
  }

}  // namespace ASSET
