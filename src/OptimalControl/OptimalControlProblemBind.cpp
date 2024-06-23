#include "OptimalControlProblem.h"
#include "PyDocString/OptimalControl/OptimalControlProblem_doc.h"

void ASSET::OptimalControlProblem::Build(py::module& m) {
    using namespace doc;
    auto obj =
        py::class_<OptimalControlProblem, std::shared_ptr<OptimalControlProblem>, OptimizationProblemBase>(
            m, "OptimalControlProblem");
    obj.def(py::init<>());



    BuildNewLinkIterface(obj);
    BuildOldLinkIterface(obj);


    //////////////////////////////////////////////////////////////////////////////

    obj.def("addLinkParamEqualCon",
        py::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
            &OptimalControlProblem::addLinkParamEqualCon),
        OptimalControlProblem_addLinkParamEqualCon1);
    obj.def("addLinkParamEqualCon",
        py::overload_cast<VectorFunctionalX, VectorXi>(&OptimalControlProblem::addLinkParamEqualCon),
        OptimalControlProblem_addLinkParamEqualCon2);
    obj.def("addLinkParamInequalCon",
        py::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
            &OptimalControlProblem::addLinkParamInequalCon),
        OptimalControlProblem_addLinkParamInequalCon1);
    obj.def("addLinkParamInequalCon",
        py::overload_cast<VectorFunctionalX, VectorXi>(&OptimalControlProblem::addLinkParamInequalCon),
        OptimalControlProblem_addLinkParamInequalCon2);
    obj.def("addLinkParamObjective",
        py::overload_cast<ScalarFunctionalX, std::vector<VectorXi>>(
            &OptimalControlProblem::addLinkParamObjective),
        OptimalControlProblem_addLinkParamObjective1);
    obj.def("addLinkParamObjective",
        py::overload_cast<ScalarFunctionalX, VectorXi>(&OptimalControlProblem::addLinkParamObjective),
        OptimalControlProblem_addLinkParamObjective2);

    //////////////////////////////////////////////////////////////////////////////

    obj.def("removeLinkEqualCon",
        &OptimalControlProblem::removeLinkEqualCon,
        OptimalControlProblem_removeLinkEqualCon);
    obj.def("removeLinkInequalCon",
        &OptimalControlProblem::removeLinkInequalCon,
        OptimalControlProblem_removeLinkEqualCon);
    obj.def("removeLinkObjective",
        &OptimalControlProblem::removeLinkObjective,
        OptimalControlProblem_removeLinkObjective);

    obj.def("addPhase",
        py::overload_cast<PhasePtr>(&OptimalControlProblem::addPhase),
        OptimalControlProblem_addPhase);


    obj.def("addPhases", &OptimalControlProblem::addPhases);

    obj.def("getPhaseNum", py::overload_cast<PhasePtr>(&OptimalControlProblem::getPhaseNum));


    obj.def("removePhase", &OptimalControlProblem::removePhase, OptimalControlProblem_removePhase);
    obj.def("Phase", &OptimalControlProblem::Phase, OptimalControlProblem_Phase);


    obj.def("setLinkParams", py::overload_cast<VectorXd, VectorXd>(&OptimalControlProblem::setLinkParams));
    obj.def("setLinkParams", py::overload_cast<VectorXd>(&OptimalControlProblem::setLinkParams), OptimalControlProblem_setLinkParams);

    obj.def("addLinkParamVgroups",
        py::overload_cast<std::map<std::string, Eigen::VectorXi>>(&OptimalControlProblem::addLinkParamVgroups));
    obj.def("setLinkParamVgroups",
        py::overload_cast<std::map<std::string, Eigen::VectorXi>>(&OptimalControlProblem::setLinkParamVgroups));
    obj.def("addLinkParamVgroup",
        py::overload_cast<Eigen::VectorXi, std::string>(&OptimalControlProblem::addLinkParamVgroup));
    obj.def("addLinkParamVgroup",
        py::overload_cast<int, std::string>(&OptimalControlProblem::addLinkParamVgroup));


    obj.def(
        "returnLinkParams", &OptimalControlProblem::returnLinkParams, OptimalControlProblem_returnLinkParams);


    obj.def("transcribe",
        py::overload_cast<bool, bool>(&OptimalControlProblem::transcribe),
        OptimalControlProblem_transcribe);

    obj.def_readonly("Phases", &OptimalControlProblem::phases, OptimalControlProblem_Phases);


    ///////////////////////
    obj.def("returnLinkEqualConVals", &OptimalControlProblem::returnLinkEqualConVals);
    obj.def("returnLinkEqualConLmults", &OptimalControlProblem::returnLinkEqualConLmults);

    obj.def("returnLinkInequalConVals", &OptimalControlProblem::returnLinkInequalConVals);
    obj.def("returnLinkInequalConLmults", &OptimalControlProblem::returnLinkInequalConLmults);

    obj.def("returnLinkEqualConScales", &OptimalControlProblem::returnLinkEqualConScales);
    obj.def("returnLinkInequalConScales", &OptimalControlProblem::returnLinkInequalConScales);
    obj.def("returnLinkObjectiveScales", &OptimalControlProblem::returnLinkObjectiveScales);


    ///////////////////////

    obj.def_readwrite("AutoScaling", &OptimalControlProblem::AutoScaling);
    obj.def_readwrite("SyncObjectiveScales", &OptimalControlProblem::SyncObjectiveScales);


    obj.def_readwrite("AdaptiveMesh", &OptimalControlProblem::AdaptiveMesh);
    obj.def_readwrite("PrintMeshInfo", &OptimalControlProblem::PrintMeshInfo);
    obj.def_readwrite("MaxMeshIters", &OptimalControlProblem::MaxMeshIters);
    obj.def_readonly("MeshConverged", &OptimalControlProblem::MeshConverged);
    obj.def_readwrite("SolveOnlyFirst", &OptimalControlProblem::SolveOnlyFirst);

    obj.def_readwrite("MeshAbortFlag", &OptimalControlProblem::MeshAbortFlag);


    obj.def("setAdaptiveMesh",
        &OptimalControlProblem::setAdaptiveMesh,
        py::arg("AdaptiveMesh") = true,
        py::arg("ApplyToPhases") = true);
    obj.def("setAutoScaling",
        &OptimalControlProblem::setAutoScaling,
        py::arg("AutoScaling") = true,
        py::arg("ApplyToPhases") = true);


    obj.def("setMeshTol", &OptimalControlProblem::setMeshTol);
    obj.def("setMeshRedFactor", &OptimalControlProblem::setMeshRedFactor);
    obj.def("setMeshIncFactor", &OptimalControlProblem::setMeshIncFactor);
    obj.def("setMeshErrFactor", &OptimalControlProblem::setMeshErrFactor);
    obj.def("setMaxMeshIters", &OptimalControlProblem::setMaxMeshIters);
    obj.def("setMinSegments", &OptimalControlProblem::setMinSegments);
    obj.def("setMaxSegments", &OptimalControlProblem::setMaxSegments);
    obj.def("setMeshErrorCriteria", &OptimalControlProblem::setMeshErrorCriteria);
    obj.def("setMeshErrorEstimator", &OptimalControlProblem::setMeshErrorEstimator);
}

void ASSET::OptimalControlProblem::BuildNewLinkIterface(py::class_<OptimalControlProblem, std::shared_ptr<OptimalControlProblem>, OptimizationProblemBase>& obj)
{

    //////////// EqualCons////////////////////////////////////////
    {
        obj.def("addLinkEqualCon",
            py::overload_cast<
            VectorFunctionalX,
            std::vector<PhasePack>,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addLinkEqualCon),
            py::arg("func"),
            py::arg("phasepack"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );



        obj.def("addLinkEqualCon",
            py::overload_cast<
            VectorFunctionalX,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            ScaleType> (&OptimalControlProblem::addLinkEqualCon),
            py::arg("func"),
            py::arg("phase0"),
            py::arg("reg0"),
            py::arg("XtUVars0"),
            py::arg("OPVars0"),
            py::arg("SPVars0"),
            py::arg("phase1"),
            py::arg("reg1"),
            py::arg("XtUVars1"),
            py::arg("OPVars1"),
            py::arg("SPVars1"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );



        obj.def("addLinkEqualCon",
            py::overload_cast<
            VectorFunctionalX,
            PhaseRefType,
            RegionType,
            VarIndexType,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addLinkEqualCon),
            py::arg("func"),
            py::arg("phase0"),
            py::arg("reg0"),
            py::arg("v0"),
            py::arg("phase1"),
            py::arg("reg1"),
            py::arg("v1"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );

        obj.def("addLinkParamEqualCon",
            py::overload_cast<
            VectorFunctionalX,
            std::vector<VectorXi>, ScaleType>(
                &OptimalControlProblem::addLinkParamEqualCon),
            py::arg("func"),
            py::arg("LinkParms"),
            py::arg("AutoScale") = std::string("auto")
        );
        obj.def("addLinkParamEqualCon",
            py::overload_cast<
            VectorFunctionalX,
            VectorXi,
            ScaleType>(
                &OptimalControlProblem::addLinkParamEqualCon),
            py::arg("func"),
            py::arg("LinkParms"),
            py::arg("AutoScale") = std::string("auto")
        );


        obj.def("addForwardLinkEqualCon",
            py::overload_cast<
            PhaseRefType,
            PhaseRefType,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addForwardLinkEqualCon),
            py::arg("phase0"),
            py::arg("phase1"),
            py::arg("vars"),
            py::arg("AutoScale") = std::string("auto")
        );

        obj.def("addParamLinkEqualCon",
            py::overload_cast<
            PhaseRefType,
            PhaseRefType,
            RegionType,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addParamLinkEqualCon),
            py::arg("phase0"),
            py::arg("phase1"),
            py::arg("reg0"),
            py::arg("vars"),
            py::arg("AutoScale") = std::string("auto")
        );


        obj.def("addDirectLinkEqualCon",
            py::overload_cast<
            PhaseRefType,
            RegionType,
            VarIndexType,
            PhaseRefType,
            RegionType,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addDirectLinkEqualCon),
            py::arg("phase0"),
            py::arg("reg0"),
            py::arg("v0"),
            py::arg("phase1"),
            py::arg("reg1"),
            py::arg("v1"),
            py::arg("AutoScale") = std::string("auto")
        );
    }


    //////////// InequalCons////////////////////////////////////////
    {
        obj.def("addLinkInequalCon",
            py::overload_cast<
            VectorFunctionalX,
            std::vector<PhasePack>,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addLinkInequalCon),
            py::arg("func"),
            py::arg("phasepack"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );



        obj.def("addLinkInequalCon",
            py::overload_cast<
            VectorFunctionalX,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            ScaleType> (&OptimalControlProblem::addLinkInequalCon),
            py::arg("func"),
            py::arg("phase0"),
            py::arg("reg0"),
            py::arg("XtUVars0"),
            py::arg("OPVars0"),
            py::arg("SPVars0"),
            py::arg("phase1"),
            py::arg("reg1"),
            py::arg("XtUVars1"),
            py::arg("OPVars1"),
            py::arg("SPVars1"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );



        obj.def("addLinkInequalCon",
            py::overload_cast<
            VectorFunctionalX,
            PhaseRefType,
            RegionType,
            VarIndexType,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addLinkInequalCon),
            py::arg("func"),
            py::arg("phase0"),
            py::arg("reg0"),
            py::arg("v0"),
            py::arg("phase1"),
            py::arg("reg1"),
            py::arg("v1"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );

        obj.def("addLinkParamInequalCon",
            py::overload_cast<
            VectorFunctionalX,
            std::vector<VectorXi>, ScaleType>(
                &OptimalControlProblem::addLinkParamInequalCon),
            py::arg("func"),
            py::arg("LinkParms"),
            py::arg("AutoScale") = std::string("auto")
        );
        obj.def("addLinkParamInequalCon",
            py::overload_cast<
            VectorFunctionalX,
            VectorXi,
            ScaleType>(
                &OptimalControlProblem::addLinkParamInequalCon),
            py::arg("func"),
            py::arg("LinkParms"),
            py::arg("AutoScale") = std::string("auto")
        );
    }
    //////////// Objectives ////////////////////////////////////////
    {
        obj.def("addLinkObjective",
            py::overload_cast<
            ScalarFunctionalX,
            std::vector<PhasePack>,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addLinkObjective),
            py::arg("func"),
            py::arg("phasepack"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );



        obj.def("addLinkObjective",
            py::overload_cast<
            ScalarFunctionalX,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            VarIndexType,
            ScaleType> (&OptimalControlProblem::addLinkObjective),
            py::arg("func"),
            py::arg("phase0"),
            py::arg("reg0"),
            py::arg("XtUVars0"),
            py::arg("OPVars0"),
            py::arg("SPVars0"),
            py::arg("phase1"),
            py::arg("reg1"),
            py::arg("XtUVars1"),
            py::arg("OPVars1"),
            py::arg("SPVars1"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );



        obj.def("addLinkObjective",
            py::overload_cast<
            ScalarFunctionalX,
            PhaseRefType,
            RegionType,
            VarIndexType,
            PhaseRefType,
            RegionType,
            VarIndexType,
            VarIndexType,
            ScaleType>(&OptimalControlProblem::addLinkObjective),
            py::arg("func"),
            py::arg("phase0"),
            py::arg("reg0"),
            py::arg("v0"),
            py::arg("phase1"),
            py::arg("reg1"),
            py::arg("v1"),
            py::arg("linkparams") = VectorXi(),
            py::arg("AutoScale") = std::string("auto")
        );

        obj.def("addLinkParamObjective",
            py::overload_cast<
            ScalarFunctionalX,
            std::vector<VectorXi>, ScaleType>(
                &OptimalControlProblem::addLinkParamObjective),
            py::arg("func"),
            py::arg("LinkParms"),
            py::arg("AutoScale") = std::string("auto")
        );
        obj.def("addLinkParamObjective",
            py::overload_cast<
            ScalarFunctionalX,
            VectorXi,
            ScaleType>(
                &OptimalControlProblem::addLinkParamObjective),
            py::arg("func"),
            py::arg("LinkParms"),
            py::arg("AutoScale") = std::string("auto")
        );
    }


}

void ASSET::OptimalControlProblem::BuildOldLinkIterface(py::class_<OptimalControlProblem, std::shared_ptr<OptimalControlProblem>, OptimizationProblemBase>& obj)
{
    using namespace doc;

    {
        ////////////////// Legacy EqualCons//////////
        obj.def("addLinkEqualCon",
            py::overload_cast<LinkConstraint>(&OptimalControlProblem::addLinkEqualCon),
            OptimalControlProblem_addLinkEqualCon1);



        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            RegVec,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            std::vector<std::string>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            LinkFlags,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            std::string,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        /// ///
        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            RegVec,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            std::vector<std::string>,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            LinkFlags,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            std::string,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

        ////////////////////////////


        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            RegVec,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon),
            OptimalControlProblem_addLinkEqualCon2);


        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            RegVec,
            std::vector<std::vector<std::shared_ptr<ODEPhaseBase>>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon),
            OptimalControlProblem_addLinkEqualCon2);


        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                &OptimalControlProblem::addLinkEqualCon),
            OptimalControlProblem_addLinkEqualCon2);
        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>, VectorXi>(
                &OptimalControlProblem::addLinkEqualCon),
            OptimalControlProblem_addLinkEqualCon2);

        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                &OptimalControlProblem::addLinkEqualCon),
            OptimalControlProblem_addLinkEqualCon2);
        obj.def("addLinkEqualCon",
            py::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>, VectorXi>(
                &OptimalControlProblem::addLinkEqualCon),
            OptimalControlProblem_addLinkEqualCon2);


        obj.def("addForwardLinkEqualCon",
            py::overload_cast<int, int, VectorXi>(&OptimalControlProblem::addForwardLinkEqualCon),
            OptimalControlProblem_addForwardLinkEqualCon);


        obj.def("addForwardLinkEqualCon",
            py::overload_cast<PhasePtr, PhasePtr, VectorXi>(&OptimalControlProblem::addForwardLinkEqualCon),
            OptimalControlProblem_addForwardLinkEqualCon);


        obj.def("addDirectLinkEqualCon",
            py::overload_cast<LinkFlags, int, VectorXi, int, VectorXi>(
                &OptimalControlProblem::addDirectLinkEqualCon),
            OptimalControlProblem_addDirectLinkEqualCon);

        obj.def(
            "addDirectLinkEqualCon",
            py::overload_cast<VectorFunctionalX, int, PhaseRegionFlags, VectorXi, int, PhaseRegionFlags, VectorXi>(
                &OptimalControlProblem::addDirectLinkEqualCon),
            OptimalControlProblem_addDirectLinkEqualCon);



        obj.def("addDirectLinkEqualCon",
            py::overload_cast<VectorFunctionalX,
            PhasePtr,
            PhaseRegionFlags,
            VectorXi,
            PhasePtr,
            PhaseRegionFlags,
            VectorXi>(&OptimalControlProblem::addDirectLinkEqualCon),
            OptimalControlProblem_addDirectLinkEqualCon);




        //
        obj.def("addDirectLinkEqualCon",
            py::overload_cast<VectorFunctionalX, int, std::string, VectorXi, int, std::string, VectorXi>(
                &OptimalControlProblem::addDirectLinkEqualCon),
            OptimalControlProblem_addDirectLinkEqualCon);



        obj.def(
            "addDirectLinkEqualCon",
            py::overload_cast<VectorFunctionalX, PhasePtr, std::string, VectorXi, PhasePtr, std::string, VectorXi>(
                &OptimalControlProblem::addDirectLinkEqualCon),
            OptimalControlProblem_addDirectLinkEqualCon);




    }

    //////////////////Legacy  InequalCons//////////
    {

        obj.def("addLinkInequalCon",
            py::overload_cast<LinkConstraint>(&OptimalControlProblem::addLinkInequalCon),
            OptimalControlProblem_addLinkInequalCon);

        ////////////////////////////

        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            RegVec,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            std::vector<std::string>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            LinkFlags,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            std::string,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        /// ///
        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            RegVec,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            std::vector<std::string>,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            LinkFlags,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            std::string,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

        ////////////////////////////


        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                &OptimalControlProblem::addLinkInequalCon),
            OptimalControlProblem_addLinkInequalCon);


        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>, VectorXi>(
                &OptimalControlProblem::addLinkInequalCon),
            OptimalControlProblem_addLinkInequalCon);


        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                &OptimalControlProblem::addLinkInequalCon),
            OptimalControlProblem_addLinkInequalCon);


        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>, VectorXi>(
                &OptimalControlProblem::addLinkInequalCon),
            OptimalControlProblem_addLinkInequalCon);


        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            RegVec,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));


        obj.def("addLinkInequalCon",
            py::overload_cast<VectorFunctionalX,
            RegVec,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));


        ////////////////////////////////////////
        obj.def("addLinkObjective",
            py::overload_cast<LinkObjective>(&OptimalControlProblem::addLinkObjective),
            OptimalControlProblem_addLinkObjective);

    }

    {
        //////////////////Legacy LinkObjectives//////////


        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            RegVec,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            std::vector<std::string>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            LinkFlags,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            std::string,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        /// ///
        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            RegVec,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            std::vector<std::string>,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            LinkFlags,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            std::string,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

        ////////////////////////////


        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
                &OptimalControlProblem::addLinkObjective),
            OptimalControlProblem_addLinkObjective);
        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>, VectorXi>(
                &OptimalControlProblem::addLinkObjective),
            OptimalControlProblem_addLinkObjective);

        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
                &OptimalControlProblem::addLinkObjective),
            OptimalControlProblem_addLinkObjective);
        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX, std::string, std::vector<std::vector<PhasePtr>>, VectorXi>(
                &OptimalControlProblem::addLinkObjective),
            OptimalControlProblem_addLinkObjective);


        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            RegVec,
            std::vector<VectorXi>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective),
            OptimalControlProblem_addLinkObjective);


        obj.def("addLinkObjective",
            py::overload_cast<ScalarFunctionalX,
            RegVec,
            std::vector<std::vector<PhasePtr>>,
            std::vector<VectorXi>,
            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective),
            OptimalControlProblem_addLinkObjective);

    }

}
