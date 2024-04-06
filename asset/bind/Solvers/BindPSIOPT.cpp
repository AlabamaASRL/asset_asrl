#include <bind/Solvers/BindPSIOPT.h>

void ASSET::BindPSIOPT(py::module& m) {
  using namespace doc;
  auto obj = py::class_<PSIOPT, std::shared_ptr<PSIOPT>>(m, "PSIOPT");
  obj.def(py::init<std::shared_ptr<NonLinearProgram>>());
  obj.def(py::init<>());

  obj.def("optimize", &PSIOPT::optimize, PSIOPT_optimize);
  obj.def("solve_optimize", &PSIOPT::solve_optimize, PSIOPT_solve_optimize);
  obj.def("solve", &PSIOPT::solve, PSIOPT_solve);
  obj.def("setQPParams", &PSIOPT::setQPParams);

  obj.def_readwrite("MaxIters", &PSIOPT::MaxIters, PSIOPT_MaxIters);
  obj.def_readwrite("MaxAccIters", &PSIOPT::MaxAccIters, PSIOPT_MaxAccIters);
  obj.def_readwrite("MaxLSIters", &PSIOPT::MaxLSIters, PSIOPT_MaxLSIters);

  obj.def("set_MaxIters", &PSIOPT::set_MaxIters);
  obj.def("set_MaxAccIters", &PSIOPT::set_MaxAccIters);
  obj.def("set_MaxLSIters", &PSIOPT::set_MaxLSIters);

  obj.def_readwrite("alphaRed", &PSIOPT::alphaRed, PSIOPT_alphaRed);
  obj.def("set_alphaRed", &PSIOPT::set_alphaRed);

  obj.def_readwrite("WideConsole", &PSIOPT::WideConsole);

  obj.def_readwrite("FastFactorAlg", &PSIOPT::FastFactorAlg, PSIOPT_FastFactorAlg);

  obj.def_readwrite("LastTotalTime", &PSIOPT::LastTotalTime, PSIOPT_LastUserTime);
  obj.def_readwrite("LastPreTime", &PSIOPT::LastPreTime, PSIOPT_LastUserTime);
  obj.def_readwrite("LastFuncTime", &PSIOPT::LastFuncTime, PSIOPT_LastUserTime);
  obj.def_readwrite("LastKKTTime", &PSIOPT::LastKKTTime, PSIOPT_LastQPTime);
  obj.def_readwrite("LastMiscTime", &PSIOPT::LastMiscTime, PSIOPT_LastQPTime);
  obj.def_readwrite("LastIterNum", &PSIOPT::LastIterNum, PSIOPT_LastIterNum);
  obj.def_readwrite("LastObjVal", &PSIOPT::LastObjVal);

  obj.def_readwrite("ObjScale", &PSIOPT::ObjScale, PSIOPT_ObjScale);
  obj.def_readwrite("PrintLevel", &PSIOPT::PrintLevel, PSIOPT_PrintLevel);
  obj.def("set_PrintLevel", &PSIOPT::set_PrintLevel);

  obj.def_readwrite("ConvergeFlag", &PSIOPT::ConvergeFlag);

  obj.def("get_ConvergenceFlag", &PSIOPT::get_ConvergenceFlag);

  obj.def_readwrite("KKTtol", &PSIOPT::KKTtol, PSIOPT_KKTtol);
  obj.def_readwrite("Bartol", &PSIOPT::Bartol, PSIOPT_Bartol);
  obj.def_readwrite("EContol", &PSIOPT::EContol, PSIOPT_EContol);
  obj.def_readwrite("IContol", &PSIOPT::IContol, PSIOPT_IContol);

  obj.def("set_KKTtol", &PSIOPT::set_KKTtol);
  obj.def("set_Bartol", &PSIOPT::set_Bartol);
  obj.def("set_EContol", &PSIOPT::set_EContol);
  obj.def("set_IContol", &PSIOPT::set_IContol);

  obj.def("set_tols",
          &PSIOPT::set_tols,
          py::arg("KKTtol") = 1.0e-6,
          py::arg("EContol") = 1.0e-6,
          py::arg("IContol") = 1.0e-6,
          py::arg("Bartol") = 1.0e-6);

  obj.def_readwrite("AccKKTtol", &PSIOPT::AccKKTtol, PSIOPT_AccKKTtol);
  obj.def_readwrite("AccBartol", &PSIOPT::AccBartol, PSIOPT_AccBartol);
  obj.def_readwrite("AccEContol", &PSIOPT::AccEContol, PSIOPT_AccEContol);
  obj.def_readwrite("AccIContol", &PSIOPT::AccIContol, PSIOPT_AccIContol);

  obj.def("set_AccKKTtol", &PSIOPT::set_AccKKTtol);
  obj.def("set_AccBartol", &PSIOPT::set_AccBartol);
  obj.def("set_AccEContol", &PSIOPT::set_AccEContol);
  obj.def("set_AccIContol", &PSIOPT::set_AccIContol);

  obj.def("set_Acctols",
          &PSIOPT::set_Acctols,
          py::arg("AccKKTtol") = 1.0e-2,
          py::arg("AccEContol") = 1.0e-3,
          py::arg("AccIContol") = 1.0e-3,
          py::arg("AccBartol") = 1.0e-3);

  obj.def_readwrite("DivKKTtol", &PSIOPT::DivKKTtol, PSIOPT_DivKKTtol);
  obj.def_readwrite("DivBartol", &PSIOPT::DivBartol, PSIOPT_DivBartol);
  obj.def_readwrite("DivEContol", &PSIOPT::DivEContol, PSIOPT_DivEContol);
  obj.def_readwrite("DivIContol", &PSIOPT::DivIContol, PSIOPT_DivIContol);

  obj.def("set_DivKKTtol", &PSIOPT::set_DivKKTtol);
  obj.def("set_DivBartol", &PSIOPT::set_DivBartol);
  obj.def("set_DivEContol", &PSIOPT::set_DivEContol);
  obj.def("set_DivIContol", &PSIOPT::set_DivIContol);

  obj.def_readwrite("NegSlackReset", &PSIOPT::NegSlackReset, PSIOPT_NegSlackReset);

  obj.def_readwrite("BoundFraction", &PSIOPT::BoundFraction, PSIOPT_BoundFraction);
  obj.def("set_BoundFraction", &PSIOPT::set_BoundFraction);

  obj.def_readwrite("BoundPush", &PSIOPT::BoundPush, PSIOPT_BoundPush);

  /////////////////////////////////////////////////////////////

  obj.def_readwrite("deltaH", &PSIOPT::deltaH, PSIOPT_deltaH);
  obj.def_readwrite("incrH", &PSIOPT::incrH, PSIOPT_incrH);
  obj.def_readwrite("decrH", &PSIOPT::decrH, PSIOPT_decrH);

  obj.def("set_deltaH", &PSIOPT::set_deltaH);
  obj.def("set_incrH", &PSIOPT::set_incrH);
  obj.def("set_decrH", &PSIOPT::set_decrH);

  obj.def("set_HpertParams", &PSIOPT::set_HpertParams, py::arg("deltaH"), py::arg("incrH"), py::arg("decrH"));

  /////////////////////////////////////////////////////////////
  obj.def_readwrite("initMu", &PSIOPT::initMu, PSIOPT_initMu);
  obj.def_readwrite("MinMu", &PSIOPT::MinMu, PSIOPT_MinMu);
  obj.def_readwrite("MaxMu", &PSIOPT::MaxMu, PSIOPT_MaxMu);

  obj.def_readwrite("MaxSOC", &PSIOPT::MaxSOC, PSIOPT_MaxSOC);

  obj.def_readwrite("PDStepStrategy", &PSIOPT::PDStepStrategy, PSIOPT_PDStepStrategy);
  obj.def_readwrite("SOEboundRelax", &PSIOPT::SOEboundRelax, PSIOPT_SOEboundRelax);
  obj.def_readwrite("QPParSolve", &PSIOPT::QPParSolve, PSIOPT_QPParSolve);

  obj.def_readwrite("SoeMode", &PSIOPT::SoeMode, PSIOPT_SoeMode);

  //////////////////////////////////////////////////////////////////////////////////////////////////

  obj.def_readwrite("OptBarMode", &PSIOPT::OptBarMode, PSIOPT_OptBarMode);
  obj.def_readwrite("SoeBarMode", &PSIOPT::SoeBarMode, PSIOPT_SoeBarMode);

  obj.def("set_OptBarMode", py::overload_cast<PSIOPT::BarrierModes>(&PSIOPT::set_OptBarMode));
  obj.def("set_OptBarMode", py::overload_cast<const std::string&>(&PSIOPT::set_OptBarMode));
  obj.def("set_SoeBarMode", py::overload_cast<PSIOPT::BarrierModes>(&PSIOPT::set_SoeBarMode));
  obj.def("set_SoeBarMode", py::overload_cast<const std::string&>(&PSIOPT::set_SoeBarMode));

  //////////////////////////////////////////////////////////////////////////////////////////////////
  obj.def_readwrite("OptLSMode", &PSIOPT::OptLSMode, PSIOPT_OptLSMode);
  obj.def_readwrite("SoeLSMode", &PSIOPT::SoeLSMode, PSIOPT_SoeLSMode);

  obj.def("set_OptLSMode", py::overload_cast<PSIOPT::LineSearchModes>(&PSIOPT::set_OptLSMode));
  obj.def("set_OptLSMode", py::overload_cast<const std::string&>(&PSIOPT::set_OptLSMode));
  obj.def("set_SoeLSMode", py::overload_cast<PSIOPT::LineSearchModes>(&PSIOPT::set_SoeLSMode));
  obj.def("set_SoeLSMode", py::overload_cast<const std::string&>(&PSIOPT::set_SoeLSMode));

  //////////////////////////////////////////////////////////////////////////////////////////////////

  obj.def_readwrite("ForceQPanalysis", &PSIOPT::ForceQPanalysis, PSIOPT_ForceQPanalysis);
  obj.def_readwrite("QPRefSteps", &PSIOPT::QPRefSteps, PSIOPT_QPRefSteps);

  obj.def_readwrite("QPPivotPerturb", &PSIOPT::QPPivotPerturb, PSIOPT_QPPivotPerturb);
  obj.def_readwrite("QPThreads", &PSIOPT::QPThreads, PSIOPT_QPThreads);
  obj.def_readwrite("QPPivotStrategy", &PSIOPT::QPPivotStrategy, PSIOPT_QPPivotStrategy);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  obj.def_readwrite("QPOrderingMode", &PSIOPT::QPOrd, PSIOPT_QPOrd);

  obj.def("set_QPOrderingMode", py::overload_cast<PSIOPT::QPOrderingModes>(&PSIOPT::set_QPOrderingMode));
  obj.def("set_QPOrderingMode", py::overload_cast<const std::string&>(&PSIOPT::set_QPOrderingMode));

  //////////////////////////////////////////////////////////////////////////////////////////////////
  obj.def_readwrite("QPPrint", &PSIOPT::QPPrint);

  obj.def_readwrite("Diagnostic", &PSIOPT::Diagnostic);

  obj.def_readwrite("storespmat", &PSIOPT::storespmat, PSIOPT_storespmat);
  obj.def("getSPmat", &PSIOPT::getSPmat, PSIOPT_getSPmat);
  obj.def("getSPmat2", &PSIOPT::getSPmat2, PSIOPT_getSPmat2);

  obj.def_readwrite("CNRMode", &PSIOPT::CNRMode, PSIOPT_CNRMode);

  py::enum_<PSIOPT::BarrierModes>(m, "BarrierModes")
      .value("PROBE", PSIOPT::BarrierModes::PROBE)
      .value("LOQO", PSIOPT::BarrierModes::LOQO);
  py::enum_<PSIOPT::LineSearchModes>(m, "LineSearchModes")
      .value("AUGLANG", PSIOPT::LineSearchModes::AUGLANG)
      .value("LANG", PSIOPT::LineSearchModes::LANG)
      .value("L1", PSIOPT::LineSearchModes::L1)
      .value("NOLS", PSIOPT::LineSearchModes::NOLS);
  py::enum_<PSIOPT::QPPivotModes>(m, "QPPivotModes")
      .value("OneByOne", PSIOPT::QPPivotModes::OneByOne)
      .value("TwoByTwo", PSIOPT::QPPivotModes::TwoByTwo);
  py::enum_<PSIOPT::PDStepStrategies>(m, "PDStepStrategies")
      .value("PrimSlackEq_Iq", PSIOPT::PDStepStrategies::PrimSlackEq_Iq)
      .value("AllMinimum", PSIOPT::PDStepStrategies::AllMinimum)
      .value("PrimSlack_EqIq", PSIOPT::PDStepStrategies::PrimSlack_EqIq)
      .value("MaxEq", PSIOPT::PDStepStrategies::MaxEq);
  py::enum_<PSIOPT::ConvergenceFlags>(m, "ConvergenceFlags", py::arithmetic())
      .value("CONVERGED", PSIOPT::ConvergenceFlags::CONVERGED)
      .value("ACCEPTABLE", PSIOPT::ConvergenceFlags::ACCEPTABLE)
      .value("NOTCONVERGED", PSIOPT::ConvergenceFlags::NOTCONVERGED)
      .value("DIVERGING", PSIOPT::ConvergenceFlags::DIVERGING);
  py::enum_<PSIOPT::AlgorithmModes>(m, "AlgorithmModes")
      .value("OPT", PSIOPT::AlgorithmModes::OPT)
      .value("OPTNO", PSIOPT::AlgorithmModes::OPTNO)
      .value("SOE", PSIOPT::AlgorithmModes::SOE)
      .value("INIT", PSIOPT::AlgorithmModes::INIT);
  py::enum_<PSIOPT::QPOrderingModes>(m, "QPOrderingModes")
      .value("MINDEG", PSIOPT::QPOrderingModes::MINDEG)
      .value("METIS", PSIOPT::QPOrderingModes::METIS)
      .value("PARMETIS", PSIOPT::QPOrderingModes::PARMETIS);
}
