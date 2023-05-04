#include "ASSET_OptimalControl.h"

#include "Integrators/Integrator.h"
#include "ODEArguments.h"

namespace ASSET {


  void OptimalControlBuild(FunctionRegistry& reg, py::module& m) {
    auto& oc = reg.getOptimalControlModule();
    RKFlagsBuild(oc);
    OCPFlagsBuild(oc);


    StateFunction<GenericFunction<-1, -1>>::Build(oc, "StateConstraint");
    StateFunction<GenericFunction<-1, 1>>::Build(oc, "StateObjective");
    LinkFunction<GenericFunction<-1, -1>>::Build(oc, "LinkConstraint");
    LinkFunction<GenericFunction<-1, 1>>::Build(oc, "LinkObjective");

    MeshIterateInfo::Build(oc);

    ODEPhaseBase::Build(oc);
    OptimalControlProblem::Build(oc);
    LGLInterpTable::Build(oc);

    reg.Build_Register<InterpFunction<-1>>(oc, "InterpFunction");
    reg.Build_Register<InterpFunction<1>>(oc, "InterpFunction_1");
    reg.Build_Register<InterpFunction<3>>(oc, "InterpFunction_3");
    reg.Build_Register<InterpFunction<6>>(oc, "InterpFunction_6");

    FDDerivArbitrary<Eigen::VectorXd>::Build(oc, "FiniteDiffTable");
    ODEArguments<-1, -1, -1>::Build(oc, "ODEArguments");


    GenericODESBuildPart1(reg, oc);
    GenericODESBuildPart2(reg, oc);
    GenericODESBuildPart3(reg, oc);
    GenericODESBuildPart4(reg, oc);
    GenericODESBuildPart5(reg, oc);
    GenericODESBuildPart6(reg, oc);
  }

}  // namespace ASSET
