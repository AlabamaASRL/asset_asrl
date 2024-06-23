#include "ODEPhaseBase.h"
#include "PyDocString/OptimalControl/ODEPhaseBase_doc.h"

void ASSET::ODEPhaseBase::Build(py::module& m) {
    using namespace pybind11::literals;
    using namespace doc;
    auto obj =
        py::class_<ODEPhaseBase, std::shared_ptr<ODEPhaseBase>, OptimizationProblemBase>(m, "PhaseInterface");
    obj.doc() = "Base Class for All Optimal Control Phases";

    obj.def("enable_vectorization", &ODEPhaseBase::enable_vectorization);

    obj.def("setTraj",
        py::overload_cast<const std::vector<Eigen::VectorXd>&, Eigen::VectorXd, Eigen::VectorXi>(
            &ODEPhaseBase::setTraj),
        ODEPhaseBase_setTraj1);

    obj.def("setTraj",
        py::overload_cast<const std::vector<Eigen::VectorXd>&, Eigen::VectorXd, Eigen::VectorXi, bool>(
            &ODEPhaseBase::setTraj));

    obj.def("setTraj",
        py::overload_cast<const std::vector<Eigen::VectorXd>&, int>(&ODEPhaseBase::setTraj),
        ODEPhaseBase_setTraj2);

    obj.def("setTraj",
        py::overload_cast<const std::vector<Eigen::VectorXd>&, int, bool>(&ODEPhaseBase::setTraj));

    obj.def("setTraj",
        py::overload_cast<const std::vector<Eigen::VectorXd>&>(&ODEPhaseBase::setTraj));


    obj.def("switchTranscriptionMode",
        py::overload_cast<TranscriptionModes, VectorXd, VectorXi>(&ODEPhaseBase::switchTranscriptionMode),
        ODEPhaseBase_switchTranscriptionMode1);
    obj.def("switchTranscriptionMode",
        py::overload_cast<TranscriptionModes>(&ODEPhaseBase::switchTranscriptionMode),
        ODEPhaseBase_switchTranscriptionMode2);


    obj.def("switchTranscriptionMode",
        py::overload_cast<std::string, VectorXd, VectorXi>(&ODEPhaseBase::switchTranscriptionMode),
        ODEPhaseBase_switchTranscriptionMode1);
    obj.def("switchTranscriptionMode",
        py::overload_cast<std::string>(&ODEPhaseBase::switchTranscriptionMode),
        ODEPhaseBase_switchTranscriptionMode2);


    obj.def("transcribe", py::overload_cast<bool, bool>(&ODEPhaseBase::transcribe), ODEPhaseBase_transcribe);

    obj.def("refineTrajManual",
        py::overload_cast<int>(&ODEPhaseBase::refineTrajManual),
        ODEPhaseBase_refineTrajManual1);
    obj.def("refineTrajManual",
        py::overload_cast<VectorXd, VectorXi>(&ODEPhaseBase::refineTrajManual),
        ODEPhaseBase_refineTrajManual2);
    obj.def("refineTrajEqual", &ODEPhaseBase::refineTrajEqual, ODEPhaseBase_refineTrajEqual);

    obj.def("setStaticParams", py::overload_cast<VectorXd, VectorXd>(&ODEPhaseBase::setStaticParams), ODEPhaseBase_setStaticParams);
    obj.def("setStaticParams", py::overload_cast<VectorXd>(&ODEPhaseBase::setStaticParams), ODEPhaseBase_setStaticParams);


    obj.def("addStaticParams",
        py::overload_cast<VectorXd, VectorXd>(&ODEPhaseBase::addStaticParams));
    obj.def("addStaticParams",
        py::overload_cast<VectorXd>(&ODEPhaseBase::addStaticParams));
    obj.def("addStaticParamVgroups",
        py::overload_cast<std::map<std::string, Eigen::VectorXi>>(&ODEPhaseBase::addStaticParamVgroups));
    obj.def("setStaticParamVgroups",
        py::overload_cast<std::map<std::string, Eigen::VectorXi>>(&ODEPhaseBase::setStaticParamVgroups));
    obj.def("addStaticParamVgroup",
        py::overload_cast<Eigen::VectorXi, std::string>(&ODEPhaseBase::addStaticParamVgroup));
    obj.def("addStaticParamVgroup",
        py::overload_cast<int, std::string>(&ODEPhaseBase::addStaticParamVgroup));


    obj.def("setControlMode",
        py::overload_cast<ControlModes>(&ODEPhaseBase::setControlMode),
        ODEPhaseBase_setControlMode);
    obj.def("setControlMode",
        py::overload_cast<std::string>(&ODEPhaseBase::setControlMode),
        ODEPhaseBase_setControlMode);

    obj.def("setIntegralMode", &ODEPhaseBase::setIntegralMode, ODEPhaseBase_setIntegralMode);

    obj.def("subStaticParams", &ODEPhaseBase::subStaticParams, ODEPhaseBase_subStaticParams);

    obj.def("subVariables",
        py::overload_cast<PhaseRegionFlags, VectorXi, VectorXd>(&ODEPhaseBase::subVariables),
        ODEPhaseBase_subVariables);
    obj.def("subVariable",
        py::overload_cast<PhaseRegionFlags, int, double>(&ODEPhaseBase::subVariable),
        ODEPhaseBase_subVariable);

    obj.def("subVariables",
        py::overload_cast<std::string, VectorXi, VectorXd>(&ODEPhaseBase::subVariables),
        ODEPhaseBase_subVariables);
    obj.def("subVariable",
        py::overload_cast<std::string, int, double>(&ODEPhaseBase::subVariable),
        ODEPhaseBase_subVariable);

    obj.def("returnTraj", &ODEPhaseBase::returnTraj, ODEPhaseBase_returnTraj);
    obj.def("returnTrajRange", &ODEPhaseBase::returnTrajRange, ODEPhaseBase_returnTrajRange);
    obj.def("returnTrajRangeND", &ODEPhaseBase::returnTrajRangeND, ODEPhaseBase_returnTrajRangeND);
    obj.def("returnTrajTable", &ODEPhaseBase::returnTrajTable);

    obj.def("returnCostateTraj", &ODEPhaseBase::returnCostateTraj, ODEPhaseBase_returnCostateTraj);
    obj.def("returnTrajError", &ODEPhaseBase::returnTrajError);

    obj.def("returnUSplineConLmults", &ODEPhaseBase::returnUSplineConLmults);
    obj.def("returnUSplineConVals", &ODEPhaseBase::returnUSplineConVals);


    obj.def("returnEqualConLmults", &ODEPhaseBase::returnEqualConLmults, ODEPhaseBase_returnEqualConLmults);
    obj.def("returnEqualConVals", &ODEPhaseBase::returnEqualConVals);
    obj.def("returnEqualConScales", &ODEPhaseBase::returnEqualConScales);

    obj.def(
        "returnInequalConLmults", &ODEPhaseBase::returnInequalConLmults, ODEPhaseBase_returnInequalConLmults);
    obj.def("returnInequalConVals", &ODEPhaseBase::returnInequalConVals);
    obj.def("returnInequalConScales", &ODEPhaseBase::returnInequalConScales);

    obj.def("returnIntegralObjectiveScales", &ODEPhaseBase::returnIntegralObjectiveScales);
    obj.def("returnIntegralParamFunctionScales", &ODEPhaseBase::returnIntegralParamFunctionScales);
    obj.def("returnStateObjectiveScales", &ODEPhaseBase::returnStateObjectiveScales);
    obj.def("returnODEOutputScales", &ODEPhaseBase::returnODEOutputScales);


    obj.def("returnStaticParams", &ODEPhaseBase::returnStaticParams, ODEPhaseBase_returnStaticParam);




    obj.def("test_threads", &ODEPhaseBase::test_threads);

    obj.def("removeEqualCon", &ODEPhaseBase::removeEqualCon, ODEPhaseBase_removeEqualCon);
    obj.def("removeInequalCon", &ODEPhaseBase::removeInequalCon, ODEPhaseBase_removeInequalCon);
    obj.def("removeStateObjective", &ODEPhaseBase::removeStateObjective, ODEPhaseBase_removeStateObjective);
    obj.def("removeIntegralObjective",
        &ODEPhaseBase::removeIntegralObjective,
        ODEPhaseBase_removeIntegralObjective);
    obj.def("removeIntegralParamFunction",
        &ODEPhaseBase::removeIntegralParamFunction,
        ODEPhaseBase_removeIntegralParamFunction);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////
    ///// The New interface /////////////////////

    obj.def("addEqualCon",
        py::overload_cast<RegionType,
        VectorFunctionalX,
        VarIndexType,
        VarIndexType,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addEqualCon),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUVars"),
        py::arg("OPVars"),
        py::arg("SPVars"),
        py::arg("AutoScale") = std::string("auto"));

    obj.def("addEqualCon",
        py::overload_cast<RegionType,
        VectorFunctionalX,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addEqualCon),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("InputIndex"),
        py::arg("AutoScale") = std::string("auto"));

    obj.def("addBoundaryValue",
        py::overload_cast<RegionType,
        VarIndexType,
        const std::variant<double, VectorXd>&,
        ScaleType>(&ODEPhaseBase::addBoundaryValue),
        py::arg("PhaseRegion"),
        py::arg("Index"),
        py::arg("Value"),
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addDeltaVarEqualCon",
        py::overload_cast<
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addDeltaVarEqualCon),
        py::arg("var"),
        py::arg("value"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addDeltaTimeEqualCon",
        py::overload_cast<
        double,
        double,
        ScaleType>(&ODEPhaseBase::addDeltaTimeEqualCon),
        py::arg("value"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addValueLock",
        py::overload_cast<
        RegionType,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addValueLock),
        py::arg("reg"),
        py::arg("vars"),
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addPeriodicityCon",
        py::overload_cast<
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addPeriodicityCon),
        py::arg("vars"),
        py::arg("AutoScale") = std::string("auto")
    );


    //////////////////////////////////
    /////// InequalCons
    obj.def("addInequalCon",
        py::overload_cast<RegionType,
        VectorFunctionalX,
        VarIndexType,
        VarIndexType,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addInequalCon),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUVars"),
        py::arg("OPVars"),
        py::arg("SPVars"),
        py::arg("AutoScale") = std::string("auto"));

    obj.def("addInequalCon",
        py::overload_cast<RegionType,
        VectorFunctionalX,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addInequalCon),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("InputIndex"),
        py::arg("AutoScale") = std::string("auto"));



    obj.def("addLUVarBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLUVarBound),
        py::arg("PhaseRegion"),
        py::arg("var"),
        py::arg("lowerbound"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );
    obj.def("addLowerVarBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLowerVarBound),
        py::arg("PhaseRegion"),
        py::arg("var"),
        py::arg("lowerbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addUpperVarBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addUpperVarBound),
        py::arg("PhaseRegion"),
        py::arg("var"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );


    obj.def("addLUFuncBound",
        py::overload_cast<RegionType,
        ScalarFunctionalX,
        VarIndexType,
        VarIndexType,
        VarIndexType,
        double,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLUFuncBound),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUVars"),
        py::arg("OPVars"),
        py::arg("SPVars"),
        py::arg("lowerbound"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addLUFuncBound",
        py::overload_cast<RegionType,
        ScalarFunctionalX,
        VarIndexType,
        double,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLUFuncBound),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUPVars"),
        py::arg("lowerbound"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );
    //
    obj.def("addLowerFuncBound",
        py::overload_cast<RegionType,
        ScalarFunctionalX,
        VarIndexType,
        VarIndexType,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLowerFuncBound),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUVars"),
        py::arg("OPVars"),
        py::arg("SPVars"),
        py::arg("lowerbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addLowerFuncBound",
        py::overload_cast<RegionType,
        ScalarFunctionalX,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLowerFuncBound),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUPVars"),
        py::arg("lowerbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addUpperFuncBound",
        py::overload_cast<RegionType,
        ScalarFunctionalX,
        VarIndexType,
        VarIndexType,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addUpperFuncBound),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUVars"),
        py::arg("OPVars"),
        py::arg("SPVars"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addUpperFuncBound",
        py::overload_cast<RegionType,
        ScalarFunctionalX,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addUpperFuncBound),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUPVars"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addLUNormBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLUNormBound),
        py::arg("PhaseRegion"),
        py::arg("XtUPVars"),
        py::arg("lowerbound"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addLUSquaredNormBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLUSquaredNormBound),
        py::arg("PhaseRegion"),
        py::arg("XtUPVars"),
        py::arg("lowerbound"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    //
    obj.def("addLowerNormBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLowerNormBound),
        py::arg("PhaseRegion"),
        py::arg("XtUPVars"),
        py::arg("lowerbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addLowerSquaredNormBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLowerSquaredNormBound),
        py::arg("PhaseRegion"),
        py::arg("XtUPVars"),
        py::arg("lowerbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );
    //
    obj.def("addUpperNormBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addUpperNormBound),
        py::arg("PhaseRegion"),
        py::arg("XtUPVars"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );

    obj.def("addUpperSquaredNormBound",
        py::overload_cast<RegionType,
        VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addUpperSquaredNormBound),
        py::arg("PhaseRegion"),
        py::arg("XtUPVars"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );
    //
    obj.def("addLowerDeltaVarBound",
        py::overload_cast<VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLowerDeltaVarBound),
        py::arg("Var"),
        py::arg("lowerbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );
    obj.def("addLowerDeltaTimeBound",
        py::overload_cast<
        double,
        double,
        ScaleType>(&ODEPhaseBase::addLowerDeltaTimeBound),
        py::arg("lowerbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );
    //
    obj.def("addUpperDeltaVarBound",
        py::overload_cast<VarIndexType,
        double,
        double,
        ScaleType>(&ODEPhaseBase::addUpperDeltaVarBound),
        py::arg("Var"),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );
    obj.def("addUpperDeltaTimeBound",
        py::overload_cast<
        double,
        double,
        ScaleType>(&ODEPhaseBase::addUpperDeltaTimeBound),
        py::arg("upperbound"),
        py::arg("scale") = 1.0,
        py::arg("AutoScale") = std::string("auto")
    );
    //////////////////////////////////
    /////// StateObjectives /////////
    obj.def("addStateObjective",
        py::overload_cast<RegionType,
        ScalarFunctionalX,
        VarIndexType,
        VarIndexType,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addStateObjective),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("XtUVars"),
        py::arg("OPVars"),
        py::arg("SPVars"),
        py::arg("AutoScale") = std::string("auto"));

    obj.def("addStateObjective",
        py::overload_cast<RegionType,
        ScalarFunctionalX,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addStateObjective),
        py::arg("PhaseRegion"),
        py::arg("Func"),
        py::arg("InputIndex"),
        py::arg("AutoScale") = std::string("auto"));


    obj.def("addValueObjective",
        py::overload_cast<
        RegionType,
        VarIndexType,
        double,
        ScaleType>(&ODEPhaseBase::addValueObjective),
        py::arg("PhaseRegion"),
        py::arg("Var"),
        py::arg("scale"),
        py::arg("AutoScale") = std::string("auto"));

    obj.def("addDeltaVarObjective",
        py::overload_cast<
        VarIndexType,
        double,
        ScaleType>(&ODEPhaseBase::addDeltaVarObjective),
        py::arg("Var"),
        py::arg("scale"),
        py::arg("AutoScale") = std::string("auto"));
    obj.def("addDeltaTimeObjective",
        py::overload_cast<
        double,
        ScaleType>(&ODEPhaseBase::addDeltaTimeObjective),
        py::arg("Var"),
        py::arg("AutoScale") = std::string("auto"));
    //////////////////////////////////
    /////// IntegralObjectives /////////
    obj.def("addIntegralObjective",
        py::overload_cast<
        ScalarFunctionalX,
        VarIndexType,
        VarIndexType,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addIntegralObjective),
        py::arg("Func"),
        py::arg("XtUVars"),
        py::arg("OPVars"),
        py::arg("SPVars"),
        py::arg("AutoScale") = std::string("auto"));

    obj.def("addIntegralObjective",
        py::overload_cast<
        ScalarFunctionalX,
        VarIndexType,
        ScaleType>(&ODEPhaseBase::addIntegralObjective),
        py::arg("Func"),
        py::arg("InputIndex"),
        py::arg("AutoScale") = std::string("auto"));
    //////////////////////////////////
    /////// IntegralParamFunction /////////
    obj.def("addIntegralParamFunction",
        py::overload_cast<
        ScalarFunctionalX,
        VarIndexType,
        VarIndexType,
        VarIndexType,
        int,
        ScaleType>(&ODEPhaseBase::addIntegralParamFunction),
        py::arg("Func"),
        py::arg("XtUVars"),
        py::arg("OPVars"),
        py::arg("SPVars"),
        py::arg("IntParam"),
        py::arg("AutoScale") = std::string("auto"));

    obj.def("addIntegralParamFunction",
        py::overload_cast<
        ScalarFunctionalX,
        VarIndexType,
        int,
        ScaleType>(&ODEPhaseBase::addIntegralParamFunction),
        py::arg("Func"),
        py::arg("InputIndex"),
        py::arg("IntParam"),
        py::arg("AutoScale") = std::string("auto"));


    ///////////////////////////////////////////////////////////////////


    obj.def("addEqualCon",
        py::overload_cast<StateConstraint>(&ODEPhaseBase::addEqualCon),
        ODEPhaseBase_addEqualCon1);


    ///////////////////////////////////////////////////////////////////////////////

    obj.def("addInequalCon",
        py::overload_cast<StateConstraint>(&ODEPhaseBase::addInequalCon),
        ODEPhaseBase_addInequalCon1);
    ////////////////////////////////////////////////////////////////////////////
    obj.def("addLUVarBounds",
        py::overload_cast<PhaseRegionFlags, Eigen::VectorXi, double, double, double>(
            &ODEPhaseBase::addLUVarBounds),
        ODEPhaseBase_addLUVarBounds);
    obj.def("addLUVarBounds",
        py::overload_cast<std::string, Eigen::VectorXi, double, double, double>(
            &ODEPhaseBase::addLUVarBounds),
        ODEPhaseBase_addLUVarBounds);

    ////////////////////////////////////////////////////////////////////////////
    obj.def("addStateObjective",
        py::overload_cast<StateObjective>(&ODEPhaseBase::addStateObjective),
        ODEPhaseBase_addStateObjective);

    ////////////////////////////////////////////////////////////////////////////

    obj.def("addIntegralObjective",
        py::overload_cast<StateObjective>(&ODEPhaseBase::addIntegralObjective),
        ODEPhaseBase_addIntegralObjective1);

    ///////////////////////////////////////////////////////////////////////////////
    obj.def("addIntegralParamFunction",
        py::overload_cast<StateObjective, int>(&ODEPhaseBase::addIntegralParamFunction),
        ODEPhaseBase_addIntegralParamFunction1);



    ////////////////////////////////////////////////////
    obj.def("getMeshInfo", &ODEPhaseBase::getMeshInfo);
    obj.def("refineTrajAuto", &ODEPhaseBase::refineTrajAuto);
    obj.def("calc_global_error", &ODEPhaseBase::calc_global_error);
    obj.def("getMeshIters", &ODEPhaseBase::getMeshIters);


    obj.def_readwrite("AdaptiveMesh", &ODEPhaseBase::AdaptiveMesh);
    obj.def_readwrite("AutoScaling", &ODEPhaseBase::AutoScaling);
    obj.def_readwrite("SyncObjectiveScales", &ODEPhaseBase::SyncObjectiveScales);



    obj.def("setAutoScaling", &ODEPhaseBase::setAutoScaling, py::arg("AutoScaling") = true);

    obj.def("setAdaptiveMesh", &ODEPhaseBase::setAdaptiveMesh, py::arg("AdaptiveMesh") = true);


    obj.def("setUnits", py::overload_cast<const py::kwargs&>(&ODEPhaseBase::setUnits));
    obj.def("setUnits", py::overload_cast<const Eigen::VectorXd&>(&ODEPhaseBase::setUnits));


    obj.def("setMeshTol", &ODEPhaseBase::setMeshTol);
    obj.def("setMeshRedFactor", &ODEPhaseBase::setMeshRedFactor);
    obj.def("setMeshIncFactor", &ODEPhaseBase::setMeshIncFactor);
    obj.def("setMeshErrFactor", &ODEPhaseBase::setMeshErrFactor);
    obj.def("setMaxMeshIters", &ODEPhaseBase::setMaxMeshIters);
    obj.def("setMinSegments", &ODEPhaseBase::setMinSegments);
    obj.def("setMaxSegments", &ODEPhaseBase::setMaxSegments);
    obj.def("setMeshErrorCriteria", &ODEPhaseBase::setMeshErrorCriteria);
    obj.def("setMeshErrorEstimator", &ODEPhaseBase::setMeshErrorEstimator);


    obj.def_readwrite("PrintMeshInfo", &ODEPhaseBase::PrintMeshInfo);
    obj.def_readwrite("MaxMeshIters", &ODEPhaseBase::MaxMeshIters);
    obj.def_readwrite("MeshTol", &ODEPhaseBase::MeshTol);
    obj.def_readwrite("MeshErrorEstimator", &ODEPhaseBase::MeshErrorEstimator);
    obj.def_readwrite("MeshErrorCriteria", &ODEPhaseBase::MeshErrorCriteria);


    obj.def_readwrite("SolveOnlyFirst", &ODEPhaseBase::SolveOnlyFirst);
    obj.def_readwrite("ForceOneMeshIter", &ODEPhaseBase::ForceOneMeshIter);
    obj.def_readwrite("NewError", &ODEPhaseBase::NewError);


    obj.def_readwrite("DetectControlSwitches", &ODEPhaseBase::DetectControlSwitches);
    obj.def_readwrite("RelSwitchTol", &ODEPhaseBase::RelSwitchTol);
    obj.def_readwrite("AbsSwitchTol", &ODEPhaseBase::AbsSwitchTol);
    obj.def_readwrite("MeshAbortFlag", &ODEPhaseBase::MeshAbortFlag);


    obj.def_readwrite("NumExtraSegs", &ODEPhaseBase::NumExtraSegs);
    obj.def_readwrite("MeshRedFactor", &ODEPhaseBase::MeshRedFactor);
    obj.def_readwrite("MeshIncFactor", &ODEPhaseBase::MeshIncFactor);
    obj.def_readwrite("MeshErrFactor", &ODEPhaseBase::MeshErrFactor);
    obj.def_readonly("MeshConverged", &ODEPhaseBase::MeshConverged);
}
