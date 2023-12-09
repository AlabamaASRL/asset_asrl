#include "ODEPhaseBase.h"
#include "ODEPhaseBase.h"
#include "LGLControlSplines.h"
#include "LGLIntegrals.h"
#include "MeshSpacingConstraints.h"
#include "PyDocString/OptimalControl/ODEPhaseBase_doc.h"
#include "ValueLock.h"
#include "AutoScalingUtils.h"

int ASSET::ODEPhaseBase::calc_threads() {
  if (this->Threads > 1 && false) {
    return this->Threads;
  }

  auto Cost = [](int N, int T, int Tmax, int V) {
    int N_t = N / T;
    int Nmt = N_t % V;
    int NN = N % T;
    int NN_t = NN / T;
    int NNmt = NN_t % V;

    double tscale = double(T) / double(std::min(T, Tmax));
  };

  return 0;
}

void ASSET::ODEPhaseBase::setUnits(const py::kwargs& kwargs)
{
   

    py::module builtins = py::module::import("builtins");
    py::object py_int = builtins.attr("int");
    py::object py_float = builtins.attr("float");
    py::object py_list = builtins.attr("list");
    py::object np_array = (py::object)py::module::import("numpy").attr("ndarray");
    py::object np_float = (py::object)py::module::import("numpy").attr("float64");
    py::object np_int = (py::object)py::module::import("numpy").attr("int32");


    VectorXd Units(this->XtUPVars());
    Units.setOnes();



    for (auto & kw : kwargs) {
        auto name = kw.first.cast<std::string>();
        auto idxs = this->idx(name);
        VectorXd units(idxs.size());
        units.setOnes();

        if (kw.second.get_type().is(py_int)   ||
            kw.second.get_type().is(py_float) ||
            kw.second.get_type().is(np_float) ||
            kw.second.get_type().is(np_int)) {
            double unit = kw.second.cast<double>();
            units *= unit;
        }
        else if (
            kw.second.get_type().is(np_array) ||
            kw.second.get_type().is(py_list)) {

            int lenvec = kw.second.attr("__len__")().cast<int>();

            if (lenvec != idxs.size()) {
                throw std::invalid_argument(fmt::format("Size of index group {0:} does not match units vector.", name));
            }

            for (int i = 0; i < lenvec; i++) {
                auto elem = kw.second.attr("__getitem__")(py::int_(i)).get_type();
                if (!(elem.is(py_float) || elem.is(py_int) || elem.is(np_int) || elem.is(np_float))) {
                    py::print(py::str(elem));
                    throw std::invalid_argument("Vectors and lists must only contain doubles or floats");
                }
                units[i]= kw.second.attr("__getitem__")(py::int_(i)).cast<double>();
            }
        }
        else {
            throw std::invalid_argument("Invalid unit type");
        }

        for (int i = 0; i < idxs.size(); i++) {
            Units[idxs[i]] = units[i];
        }
    }

    this->setUnits(Units);
}

int ASSET::ODEPhaseBase::addBoundaryValue(RegionType reg, VarIndexType args, const std::variant<double, VectorXd>& value_t, ScaleType scale_t)
{
    Eigen::VectorXd value;
    if (std::holds_alternative<double>(value_t)) {
        value.resize(1);
        value[0] = std::get<double>(value_t);
    }
    else if (std::holds_alternative<VectorXd>(value_t)) {
        value = std::get<VectorXd>(value_t);
    }
    auto Func = Arguments<-1>(value.size()) - value;
    return this->addEqualCon(reg,Func,args,scale_t);
}

int ASSET::ODEPhaseBase::addDeltaVarEqualCon(VarIndexType var, double value, double scale, ScaleType scale_t)
{
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = ((x1 - x0) - value) * scale;
    return this->addEqualCon(PhaseRegionFlags::FrontandBack, func, var,scale_t);
}

int ASSET::ODEPhaseBase::addValueLock(RegionType reg, VarIndexType args, ScaleType scale_t)
{
    int argsize = this->getXtUPVars(getRegion(reg), args).size();
    return this->addEqualCon(reg,LockArgs<-1>(argsize), args,scale_t);
}

int ASSET::ODEPhaseBase::addPeriodicityCon(VarIndexType args, ScaleType scale_t)
{
    int argsize = this->getXtUPVars(PhaseRegionFlags::FrontandBack, args).size();
    auto X = Arguments<-1>(argsize * 2);
    auto Func = X.head(argsize) - X.tail(argsize);
    return this->addEqualCon(PhaseRegionFlags::FrontandBack,Func, args,scale_t);
}
/////////////////////////////////////////////////////////////////////////////////////////////










int ASSET::ODEPhaseBase::addLUVarBound(RegionType reg, VarIndexType var, double lowerbound, double upperbound, double lbscale, double ubscale, ScaleType scale_t)
{
    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
            "Transcription Error!!!\n"
            "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
            lowerbound,
            upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);

    auto x = Arguments<1>();
    auto lowbound = (lowerbound - x) * lbscale;
    auto ubound = (x - upperbound) * ubscale;
    auto lubound = StackedOutputs{ lowbound, ubound };

    return this->addInequalCon(reg,lubound, var,scale_t);

}
int ASSET::ODEPhaseBase::addLUVarBound(
    PhaseRegionFlags reg, int var, double lowerbound, double upperbound, double lbscale, double ubscale) {


    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
            "Transcription Error!!!\n"
            "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
            lowerbound,
            upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);

    auto x = Arguments<1>();
    auto lowbound = (lowerbound - x) * lbscale;
    auto ubound = (x - upperbound) * ubscale;

    auto lubound = StackedOutputs{ lowbound, ubound };
    VectorXi v(1);
    v[0] = var;
    return this->addInequalCon(StateConstraint(lubound, reg, v));
}


int ASSET::ODEPhaseBase::addLowerVarBound(RegionType reg, VarIndexType var, double lowerbound, double lbscale, ScaleType scale_t)
{
    check_lbscale(lbscale);
    auto x = Arguments<1>();
    auto lbound = (lowerbound - x) * lbscale;
    
    return this->addInequalCon(reg,lbound, var, scale_t);
}

int ASSET::ODEPhaseBase::addUpperVarBound(RegionType reg, VarIndexType var, double upperbound, double ubscale, ScaleType scale_t)
{
    check_ubscale(ubscale);
    auto x = Arguments<1>();
    auto ubound = (x - upperbound) * ubscale;
    return this->addInequalCon(reg, ubound, var, scale_t);
}

int ASSET::ODEPhaseBase::addLUFuncBound(RegionType reg, ScalarFunctionalX func, VarIndexType XtUPvars,
    VarIndexType OPvars,
    VarIndexType SPvars,
    double lowerbound, double upperbound, double lbscale, double ubscale, ScaleType scale_t)
{

    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
            "Transcription Error!!!\n"
            "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
            lowerbound,
            upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);
    auto x = Arguments<1>();
    auto lubound = StackedOutputs{ (lowerbound - x) * lbscale, (x - upperbound) * ubscale };
    auto lufun = lubound.eval(func);
    return this->addInequalCon(reg,lufun, XtUPvars,OPvars,SPvars,scale_t);
}

int ASSET::ODEPhaseBase::addLowerFuncBound(RegionType reg, ScalarFunctionalX func, 
    VarIndexType XtUPvars,
    VarIndexType OPvars, 
    VarIndexType SPvars, 
    double lowerbound, double lbscale, ScaleType scale_t)
{
    check_lbscale(lbscale);

    Vector1<double> rhs;
    rhs[0] = lowerbound;
    auto lbfun = ((-1.0 * func) + rhs) * lbscale;

    return this->addInequalCon(reg, lbfun, XtUPvars, OPvars, SPvars, scale_t);
}

int ASSET::ODEPhaseBase::addUpperFuncBound(RegionType reg, ScalarFunctionalX func, 
    VarIndexType XtUPvars, 
    VarIndexType OPvars, 
    VarIndexType SPvars, 
    double upperbound, double ubscale, ScaleType scale_t)
{

    check_ubscale(ubscale);

    Vector1<double> rhs;
    rhs[0] = -upperbound;
    auto ubfun = ((func)+rhs) * ubscale;
    return this->addInequalCon(reg, ubfun, XtUPvars, OPvars, SPvars, scale_t);
}

int ASSET::ODEPhaseBase::addLUNormBound(RegionType reg, VarIndexType XtUPvars,
    double lowerbound, double upperbound, double lbscale, double ubscale, ScaleType scale_t)
{
    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
            "Transcription Error!!!\n"
            "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
            lowerbound,
            upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);

    int size = this->getXtUPVars(getRegion(reg), XtUPvars).size();
    
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto lubound = StackedOutputs{ (lowerbound - x) * lbscale, (x - upperbound) * ubscale };
        auto normfun = lubound.eval(Arguments<sz.value>(size).norm());
        return this->addInequalCon(reg,normfun, XtUPvars,scale_t);
    };

    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
    return 0;
}

int ASSET::ODEPhaseBase::addLUSquaredNormBound(RegionType reg, VarIndexType XtUPvars,
    double lowerbound, double upperbound, double lbscale, double ubscale, ScaleType scale_t)
{
    if (lowerbound > upperbound) {
        fmt::print(fmt::fg(fmt::color::red),
            "Transcription Error!!!\n"
            "Lower-bound({0:.3e}) greater than Upper-bound({1:.3e}) \n",
            lowerbound,
            upperbound);
        throw std::invalid_argument("");
    }
    check_lbscale(lbscale);
    check_ubscale(ubscale);

    int size = this->getXtUPVars(getRegion(reg), XtUPvars).size();

    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto lubound = StackedOutputs{ (lowerbound - x) * lbscale, (x - upperbound) * ubscale };
        auto normfun = lubound.eval(Arguments<sz.value>(size).squared_norm());
        return this->addInequalCon(reg, normfun, XtUPvars, scale_t);
    };

    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
    return 0;
}

int ASSET::ODEPhaseBase::addLowerNormBound(RegionType reg, VarIndexType XtUPvars, double lowerbound, double lbscale, ScaleType scale_t)
{
    check_lbscale(lbscale);

    int size = this->getXtUPVars(getRegion(reg), XtUPvars).size();
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto normfun = (lowerbound - Arguments<sz.value>(size).norm()) * lbscale;
        return this->addInequalCon(reg,normfun, XtUPvars,scale_t);
    };
    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
}

int ASSET::ODEPhaseBase::addLowerSquaredNormBound(RegionType reg, VarIndexType XtUPvars, double lowerbound, double lbscale, ScaleType scale_t)
{
    check_lbscale(lbscale);

    int size = this->getXtUPVars(getRegion(reg), XtUPvars).size();
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto normfun = (lowerbound - Arguments<sz.value>(size).squared_norm()) * lbscale;
        return this->addInequalCon(reg, normfun, XtUPvars, scale_t);
    };
    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
}

int ASSET::ODEPhaseBase::addUpperNormBound(RegionType reg, VarIndexType XtUPvars, double upperbound, double ubscale, ScaleType scale_t)
{
    check_ubscale(ubscale);

    int size = this->getXtUPVars(getRegion(reg), XtUPvars).size();
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto normfun = (Arguments<sz.value>(size).norm() - upperbound) * ubscale;
        return this->addInequalCon(reg, normfun, XtUPvars, scale_t);
    };
    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
    return 0;
}

int ASSET::ODEPhaseBase::addUpperSquaredNormBound(RegionType reg, VarIndexType XtUPvars, double upperbound, double ubscale, ScaleType scale_t)
{
    check_ubscale(ubscale);

    int size = this->getXtUPVars(getRegion(reg), XtUPvars).size();
    auto impl = [&](auto sz) {
        auto x = Arguments<1>();
        auto normfun = (Arguments<sz.value>(size).squared_norm() - upperbound) * ubscale;
        return this->addInequalCon(reg, normfun, XtUPvars, scale_t);
    };
    switch (size) {
    case 2:
        return impl(int_const<2>());
    case 3:
        return impl(int_const<3>());
    case 4:
        return impl(int_const<4>());
    default:
        return impl(int_const<-1>());
    }
    return 0;
}

int ASSET::ODEPhaseBase::addLowerDeltaVarBound(RegionType reg, VarIndexType var, double lowerbound, double lbscale, ScaleType scale_t)
{
    check_lbscale(lbscale);

    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = (lowerbound - (x1 - x0)) * lbscale;
   
    return this->addInequalCon(reg,func,var,scale_t);
}

int ASSET::ODEPhaseBase::addUpperDeltaVarBound(RegionType reg, VarIndexType var, double upperbound, double ubscale, ScaleType scale_t)
{
    check_ubscale(ubscale);
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = ((x1 - x0) - upperbound) * ubscale;
    return this->addInequalCon(reg, func, var, scale_t);
}



//////////////////////////////

int ASSET::ODEPhaseBase::addValueObjective(RegionType reg, VarIndexType var, double scale, ScaleType scale_t)
{
    auto obj = Arguments<1>() * scale;
    return this->addStateObjective(reg, obj, var, scale_t);
}

int ASSET::ODEPhaseBase::addDeltaVarObjective(VarIndexType var, double scale, ScaleType scale_t)
{
    auto args = Arguments<2>();
    auto x0 = args.coeff<0>();
    auto x1 = args.coeff<1>();
    auto func = ((x1 - x0)) * scale;
    
    return this->addStateObjective(PhaseRegionFlags::FrontandBack, func, var,scale_t);
}



std::vector<Eigen::VectorXd> ASSET::ODEPhaseBase::returnCostateTraj() const {
  if (!this->PostOptInfoValid) {
    throw std::invalid_argument("No costates to return,a solve or optimize call must be made before "
                                "returning the costate trajectory ");
  }

  auto TrajTemp = this->indexer.getFuncEqMultipliers(this->DynamicsFuncIndex, this->ActiveEqLmults);

  std::vector<Eigen::VectorXd> tmp;
  int k = 0;
  int stride = (this->numTranCardStates - 1);
  auto getSpace = [&](int i) {
    if (this->numTranCardStates == 4) {
      return LGLCoeffs<4>::InteriorSpacings[i];
    } else if (this->numTranCardStates == 3) {
      return LGLCoeffs<3>::InteriorSpacings[i];
    } else if (this->numTranCardStates == 2) {
      return LGLCoeffs<2>::InteriorSpacings[i];
    } else {
      std::invalid_argument("Costate estimation Not Implemented for specified Transcription "
                            "Mode");
      return 0.0;
    }
  };
  for (auto& T: TrajTemp) {
    VectorXd x0 = this->ActiveTraj[k * (stride)];
    VectorXd xf = this->ActiveTraj[k * (stride) + stride];
    double t0 = x0[this->TVar()];
    double h = xf[this->TVar()] - x0[this->TVar()];
    for (int i = 0; i < stride; i++) {
      VectorXd cs(this->XVars() + 1);
      cs.head(this->XVars()) = T.segment(i * this->XVars(), this->XVars());
      cs[this->TVar()] = t0 + h * getSpace(i);
      tmp.push_back(cs);
    }
    k++;
  }

  std::vector<Eigen::VectorXd> Costates(this->ActiveTraj.size());

  for (int i = 0; i < (this->ActiveTraj.size()); i++) {

    int idx0 = i - 1;
    int idx1 = i;

    if (i == 0) {
      idx0++;
      idx1++;
    } else if (i == (this->ActiveTraj.size() - 1)) {
      idx0--;
      idx1--;
    }
    auto t0 = tmp[idx0][this->TVar()];
    auto t1 = tmp[idx1][this->TVar()];
    auto tm = this->ActiveTraj[i][this->TVar()];
    Costates[i] = tmp[idx0] + ((tm - t0) / (t1 - t0)) * (tmp[idx1] - tmp[idx0]);
  }


  return Costates;
}

std::vector<Eigen::VectorXd> ASSET::ODEPhaseBase::returnTrajError() const {

 if (!this->PostOptInfoValid) {
    throw std::invalid_argument("No trajectory errors to return,a solve or optimize call must be made before "
                                "returning the trajectory error ");
  }

  auto ErrTrajTemp = this->indexer.getFuncEqMultipliers(this->DynamicsFuncIndex, this->ActiveEqCons);


  std::vector<Eigen::VectorXd> ErrTraj;
  int k = 0;
  int stride = (this->numTranCardStates - 1);
  auto getSpace = [&](int i) {
    if (this->numTranCardStates == 4) {
      return LGLCoeffs<4>::InteriorSpacings[i];
    } else if (this->numTranCardStates == 3) {
      return LGLCoeffs<3>::InteriorSpacings[i];
    } else if (this->numTranCardStates == 2) {
      return LGLCoeffs<2>::InteriorSpacings[i];
    } else {
      std::invalid_argument("Error estimation Not Implemented for specified Transcription "
                            "Mode");
      return 0.0;
    }
  };
  for (auto& T: ErrTrajTemp) {
    VectorXd x0 = this->ActiveTraj[k * (stride)];
    VectorXd xf = this->ActiveTraj[k * (stride) + stride];
    double t0 = x0[this->TVar()];
    double h = xf[this->TVar()] - x0[this->TVar()];
    for (int i = 0; i < stride; i++) {
      VectorXd cs(this->XVars() + 1);
      cs.head(this->XVars()) = T.segment(i * this->XVars(), this->XVars());
      cs[this->TVar()] = t0 + h * getSpace(i);
      ErrTraj.push_back(cs);
    }
    k++;
  }
  return ErrTraj;
}

void ASSET::ODEPhaseBase::setTraj(const std::vector<Eigen::VectorXd>& mesh,
                                  Eigen::VectorXd DBS,
                                  Eigen::VectorXi DPB,
                                  bool LerpTraj) {

  if (mesh.size() == 0) {
    throw std::invalid_argument("Input trajectory is empty");
  }
  if (DPB.sum() < 2) {
    throw std::invalid_argument("Phase must have least 2 segments/defects.");
  }
  int msize = mesh[0].size();
  if (msize != this->Table.XtUVars) {
    std::cout << "User Input Error in function setInitTraj for ODE:" << this->Table.ode.name() << std::endl;
    std::cout << " Dimension of Input States(" << msize << ") does not match expected dimensions of the ODE("
              << this->Table.XtUVars << ")" << std::endl;
    throw std::invalid_argument("");
  }
  if ((DBS.size() - 1) != DPB.size()) {
    std::cout << "User Input Error in function setInitTraj for ODE:" << this->Table.ode.name() << std::endl;
    std::cout << "  Size of Defect Bin Spacing(" << DBS.size()
              << ") not consistent with size of Defects Per Bin(" << DPB.size() << ")" << std::endl;
    throw std::invalid_argument("");
  }
  for (auto& X: mesh) {
    if (X.hasNaN()) {
      throw std::invalid_argument("NaN detected in State Vector in ODEPhaseBase::setTraj");
    }
  }


  this->Table.loadUnevenData(DPB.sum(), mesh);

  if (LerpTraj) {

    double t0 = mesh[0][this->TVar()];
    double tf = mesh.back()[this->TVar()];
    Eigen::VectorXd intime_nd(mesh.size());
    for (int i = 0; i < mesh.size(); i++) {
      intime_nd[i] = (mesh[i][this->TVar()] - t0) / (tf - t0);
    }

    int nstates = (this->numTranCardStates - 1) * DPB.sum() + 1;
    Eigen::VectorXd outtime_nd(nstates);
    outtime_nd.setZero();

    Eigen::VectorXd Tspacing = this->Table.Tspacing;
    Eigen::VectorXd cvect(Tspacing.size());

    this->ActiveTraj.resize(nstates);

    for (int i = 0, start = 0; i < DBS.size() - 1; i++) {
      double bint0 = DBS[i];
      double bintf = DBS[i + 1];
      double segdt = (bintf - bint0) / DPB[i];
      for (int j = 0; j < DPB[i]; j++) {
        cvect.setConstant(outtime_nd[start]);
        outtime_nd.segment(start, Tspacing.size()) = cvect + Tspacing * segdt;
        start += (this->numTranCardStates - 1);
      }
    }


    for (int i = 0, elem = 0; i < nstates; i++) {
      double ti = outtime_nd[i];
      auto it = std::upper_bound(intime_nd.cbegin() + elem, intime_nd.cend(), ti);
      elem = int(it - intime_nd.cbegin()) - 1;
      elem = std::clamp(elem, 0, int(intime_nd.size() - 2));
      double helem = intime_nd[elem + 1] - intime_nd[elem];
      double tndlocal = (ti - intime_nd[elem]) / (helem);
      this->ActiveTraj[i] = mesh[elem] * (1.0 - tndlocal) + mesh[elem + 1] * tndlocal;
    }


  } else {
    this->ActiveTraj = this->Table.NDdistribute(DBS, DPB);
  }

  this->DefBinSpacing = DBS;
  this->DefsPerBin = DPB;
  this->numDefects = DPB.sum();
  this->TrajectoryLoaded = true;
  this->resetTranscription();
  this->invalidatePostOptInfo();
}


void ASSET::ODEPhaseBase::refineTrajManual(VectorXd DBS, VectorXi DPB) {
  if ((DBS.size() - 1) != DPB.size()) {
    std::cout << "User Input Error in function setInitTraj for ODE:" << this->Table.ode.name() << std::endl;
    std::cout << "  Size of Defect Bin Spacing(" << DBS.size()
              << ") not consistent with size of Defects Per Bin(" << DPB.size() << ")" << std::endl;
    throw std::invalid_argument("");
  }

  this->Table.loadExactData(this->ActiveTraj);
  this->ActiveTraj = this->Table.NDdistribute(DBS, DPB);
  this->DefBinSpacing = DBS;
  this->DefsPerBin = DPB;
  this->numDefects = DPB.sum();
  this->resetTranscription();
  this->invalidatePostOptInfo();
}

std::vector<Eigen::VectorXd> ASSET::ODEPhaseBase::refineTrajEqual(int n) {
  this->checkMesh();
  this->MeshIters.back().up_numsegs = n;
  Eigen::VectorXi dpb = VectorXi::Ones(n);
  Eigen::VectorXd bins = this->MeshIters.back().calc_bins(n);
  this->refineTrajManual(bins, dpb);

  return this->ActiveTraj;
}


void ASSET::ODEPhaseBase::refineTrajAuto() {
  this->checkMesh();
  this->updateMesh();
  if (this->PrintMeshInfo) {
    this->MeshIters.back().print(0);
  }
}

void ASSET::ODEPhaseBase::subVariables(PhaseRegionFlags reg, VectorXi indices, VectorXd vals) {
  switch (reg) {
    case PhaseRegionFlags::Front: {
      for (int i = 0; i < indices.size(); i++) {
        this->ActiveTraj[0][indices[i]] = vals[i];
      }
      break;
    }
    case PhaseRegionFlags::Back: {
      for (int i = 0; i < indices.size(); i++) {
        this->ActiveTraj.back()[indices[i]] = vals[i];
      }
      break;
    }
    case PhaseRegionFlags::Path: {
      for (int i = 0; i < indices.size(); i++) {
        for (int j = 0; j < this->ActiveTraj.size(); j++) {
          this->ActiveTraj[j][indices[i]] = vals[i];
        }
      }
      break;
    }
    case PhaseRegionFlags::StaticParams: {
      for (int i = 0; i < indices.size(); i++) {
        this->ActiveStaticParams[indices[i]] = vals[i];
      }
      break;
    }
    default: {
      throw std::invalid_argument("Variable Substitution Not Allowed for specified Phase Region");
    }
  }
}

void ASSET::ODEPhaseBase::transcribe_integrals() {
  auto MakeInt = [&](auto cs, auto xv, auto pv, const ScalarFunctionalX& integrand, int xvv, int pvv) {
    auto integral = LGLIntegral<ScalarFunctionalX, cs.value, xv.value, pv.value> {integrand, xvv, pvv};
    integral.EnableVectorization = this->EnableVectorization;

    return ObjectiveInterface(integral);
  };
  auto SwitchX = [&](auto cs, int xv, auto pv, const ScalarFunctionalX& integrand, int xvv, int pvv) {
    switch (xv) {
      case 1:
        return MakeInt(cs, int_const<1>(), pv, integrand, xvv, pvv);
      case 2:
        return MakeInt(cs, int_const<2>(), pv, integrand, xvv, pvv);
      case 3:
        return MakeInt(cs, int_const<3>(), pv, integrand, xvv, pvv);
      default:
        return MakeInt(cs, int_const<-1>(), pv, integrand, xvv, pvv);
    }
  };
  auto SwitchP = [&](auto cs, int xv, int pv, const ScalarFunctionalX& integrand, int xvv, int pvv) {
    switch (pv) {
      case 0:
        return SwitchX(cs, xv, int_const<0>(), integrand, xvv, pvv);
      default:
        return MakeInt(cs, int_const<-1>(), int_const<-1>(), integrand, xvv, pvv);
    }
  };
  auto SwitchC = [&](int cs, int xv, int pv, const ScalarFunctionalX& integrand, int xvv, int pvv) {
    switch (cs) {
      case 2:
        return SwitchP(int_const<2>(), xv, pv, integrand, xvv, pvv);
      case 3:
        return SwitchP(int_const<3>(), xv, pv, integrand, xvv, pvv);
      case 4:
        return SwitchP(int_const<4>(), xv, pv, integrand, xvv, pvv);
      default: {
        throw std::invalid_argument("Integral Type not implemented");
        return SwitchP(int_const<2>(), xv, pv, integrand, xvv, pvv);
      }
    }
  };

  for (auto& [key, ob]: this->userIntegrands) {
    int xp = ob.XtUVars.size();
    int sop = ob.OPVars.size() + ob.SPVars.size();
    VectorXi xtrap(xp + 1);
    xtrap.head(xp) = ob.XtUVars;
    xtrap[xp] = this->TVar();

    ObjectiveInterface obj;
    PhaseRegionFlags PhaseReg = PhaseRegionFlags::PairWisePath;

    auto Func = ob.Func;
    if (this->AutoScaling) {
        VectorXd input_scales = this->get_input_scale(ob.RegionFlag, ob.XtUVars, ob.OPVars, ob.SPVars);
        VectorXd output_scales(Func.ORows());
        output_scales.setOnes(); // Come Back and Fix this
        output_scales = ob.OutputScales;

        Func = IOScaled<decltype(Func)>(ob.Func, input_scales, output_scales);
    }


    if (this->IntegralMode == IntegralModes::BaseIntegral
        && ((this->TranscriptionMode == LGL5) || (this->TranscriptionMode == LGL7))) {
      if (this->TranscriptionMode == LGL5) {
        obj = SwitchC(3, xp, sop, Func, xp, sop);
      } else if (this->TranscriptionMode == LGL7) {
        obj = SwitchC(4, xp, sop, Func, xp, sop);
      }
      PhaseReg = PhaseRegionFlags::DefectPath;
    } else {
      obj = SwitchC(2, xp, sop, Func, xp, sop);
      if (this->TranscriptionMode == LGL3 || this->TranscriptionMode == Trapezoidal
          || this->TranscriptionMode == CentralShooting) {
        PhaseReg = PhaseRegionFlags::DefectPath;

      } else {
        PhaseReg = PhaseRegionFlags::PairWisePath;
      }
    }

    ThreadingFlags ThreadMode =
        ob.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;

    int Gindex = this->indexer.addObjective(obj, PhaseReg, xtrap, ob.OPVars, ob.SPVars, ThreadMode);

    int PLindex = Gindex - this->indexer.StartObj;
    ob.GlobalIndex = Gindex;
    ob.PhaseLocalIndex = PLindex;
  }

  for (auto& [key, ob]: this->userParamIntegrands) {
    int xp = ob.XtUVars.size();
    int sop = ob.OPVars.size() + ob.SPVars.size();
    VectorXi xtrap(xp + 1);
    xtrap.head(xp) = ob.XtUVars;
    xtrap[xp] = this->TVar();

    ObjectiveInterface obj;
    PhaseRegionFlags PhaseReg = PhaseRegionFlags::PairWisePath;

    auto Func = ob.Func;
    if (this->AutoScaling) {
        VectorXd input_scales = this->get_input_scale(ob.RegionFlag, ob.XtUVars, ob.OPVars, ob.SPVars);
        VectorXd output_scales(Func.ORows());
        output_scales.setOnes(); // Come Back and Fix this
        output_scales = ob.OutputScales;
        Func = IOScaled<decltype(Func)>(ob.Func, input_scales, output_scales);
    }


    if (this->IntegralMode == IntegralModes::BaseIntegral
        && ((this->TranscriptionMode == LGL5) || (this->TranscriptionMode == LGL7))) {
      if (this->TranscriptionMode == LGL5) {
        obj = SwitchC(3, xp, sop, Func, xp, sop);
      } else if (this->TranscriptionMode == LGL7) {
        obj = SwitchC(4, xp, sop, Func, xp, sop);
      }
      PhaseReg = PhaseRegionFlags::DefectPath;
    } else {
      obj = SwitchC(2, xp, sop, Func, xp, sop);
      PhaseReg = PhaseRegionFlags::PairWisePath;
    }

    ThreadingFlags ThreadMode =
        ob.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;

    auto AccFunc = Arguments<1>() * -1.0;

    int Gindex = this->indexer.addAccumulation(
        obj, PhaseReg, xtrap, ob.OPVars, ob.SPVars, AccFunc, ob.EXTVars, ThreadMode);

    int PLindex = Gindex - this->indexer.StartObj;
    ob.GlobalIndex = Gindex;
    ob.PhaseLocalIndex = PLindex;
  }
}
void ASSET::ODEPhaseBase::transcribe_basic_funcs() {
  for (auto& [key, eq]: this->userEqualities) {
    ThreadingFlags ThreadMode =
        eq.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
    if (eq.RegionFlag == PhaseRegionFlags::Path || eq.RegionFlag == PhaseRegionFlags::PairWisePath)
      eq.Func.enable_vectorization(this->EnableVectorization);


    auto Func = eq.Func;
    if (this->AutoScaling) {
        VectorXd input_scales = this->get_input_scale(eq.RegionFlag, eq.XtUVars, eq.OPVars, eq.SPVars);
        VectorXd output_scales(Func.ORows());
        output_scales.setOnes(); // Come Back and Fix this
        output_scales = eq.OutputScales;
        Func = IOScaled<decltype(Func)>(eq.Func, input_scales, output_scales);
    }

    int Gindex =
        this->indexer.addEquality(Func, eq.RegionFlag, eq.XtUVars, eq.OPVars, eq.SPVars, ThreadMode);

    int PLindex = Gindex - this->indexer.StartEq;
    eq.GlobalIndex = Gindex;
    eq.PhaseLocalIndex = PLindex;
  }
  for (auto& [key,iq]: this->userInequalities) {
    ThreadingFlags ThreadMode =
        iq.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
    if (iq.RegionFlag == PhaseRegionFlags::Path || iq.RegionFlag == PhaseRegionFlags::PairWisePath) {

      iq.Func.enable_vectorization(this->EnableVectorization);
    }

    auto Func = iq.Func;
    if (this->AutoScaling) {
        VectorXd input_scales = this->get_input_scale(iq.RegionFlag, iq.XtUVars, iq.OPVars, iq.SPVars);
        VectorXd output_scales(Func.ORows());
        output_scales.setOnes(); // Come Back and Fix this
        output_scales = iq.OutputScales;

        Func = IOScaled<decltype(Func)>(iq.Func, input_scales, output_scales);
    }

    int Gindex =
        this->indexer.addInequality(Func, iq.RegionFlag, iq.XtUVars, iq.OPVars, iq.SPVars, ThreadMode);
    int PLindex = Gindex - this->indexer.StartIq;
    iq.GlobalIndex = Gindex;
    iq.PhaseLocalIndex = PLindex;
  }
  for (auto& [key, ob]: this->userStateObjectives) {
    ThreadingFlags ThreadMode =
        ob.Func.thread_safe() ? ThreadingFlags::ByApplication : ThreadingFlags::MainThread;
    if (ob.RegionFlag == PhaseRegionFlags::Path || ob.RegionFlag == PhaseRegionFlags::PairWisePath)
      ob.Func.enable_vectorization(this->EnableVectorization);

    auto Func = ob.Func;
    if (this->AutoScaling) {
        VectorXd input_scales = this->get_input_scale(ob.RegionFlag, ob.XtUVars, ob.OPVars, ob.SPVars);
        VectorXd output_scales(Func.ORows());
        output_scales.setOnes(); // Come Back and Fix this
        output_scales = ob.OutputScales;

        Func = IOScaled<decltype(Func)>(ob.Func, input_scales, output_scales);
    }

    int Gindex =
        this->indexer.addObjective(Func, ob.RegionFlag, ob.XtUVars, ob.OPVars, ob.SPVars, ThreadMode);
    int PLindex = Gindex - this->indexer.StartObj;
    ob.GlobalIndex = Gindex;
    ob.PhaseLocalIndex = PLindex;
  }
}
void ASSET::ODEPhaseBase::transcribe_axis_funcs() {
  VectorXd cspace(this->numDefects + 1);
  int start = 0;
  for (int i = 0; i < this->DefsPerBin.size(); i++) {
    cspace.segment(start, this->DefsPerBin[i] + 1)
        .setLinSpaced(this->DefBinSpacing[i], this->DefBinSpacing[i + 1]);
    start += this->DefsPerBin[i];
  }

  std::vector<ConstraintInterface> AxisFuncs;
  std::vector<int> Tmodes;
  Eigen::VectorXi bins(this->Threads + 1);
  bins.setLinSpaced(0, this->indexer.numNodalStates);

  for (int i = 0; i < this->indexer.numNodalStates - 2; i++) {
    AxisFuncs.emplace_back(SingleMeshSpacing(cspace[i + 1]));
    int thrt = Thread0;
    for (int j = 0; j < this->Threads; j++) {
      if (i >= bins(j) && i < bins(j + 1)) {
        thrt = j;
      }
    }
    Tmodes.push_back(thrt);
  }

  VectorXi tloc(1);
  tloc[0] = this->TVar();

  VectorXi empty(0);
  empty.resize(0);

  if (this->TranscriptionMode == TranscriptionModes::LGL7) {
    LGLMeshSpacing<4> axcon7;
    this->indexer.addEquality(
        axcon7, PhaseRegionFlags::DefectPath, tloc, empty, empty, ThreadingFlags::ByApplication);
  } else if (this->TranscriptionMode == TranscriptionModes::LGL5) {
    LGLMeshSpacing<3> axcon5;
    this->indexer.addEquality(
        axcon5, PhaseRegionFlags::DefectPath, tloc, empty, empty, ThreadingFlags::ByApplication);
  }

  this->indexer.addPartitionedEquality(
      AxisFuncs, PhaseRegionFlags::FrontNodalBackPath, tloc, empty, empty, Tmodes);
}

void ASSET::ODEPhaseBase::transcribe_control_funcs() {
  VectorXi StateT(this->XtUVars());
  for (int i = 0; i < this->XtUVars(); i++)
    StateT[i] = i;
  VectorXi TUvarT = StateT.segment(this->TVar(), 1 + this->UVars());
  VectorXi empty(0);
  empty.resize(0);

  std::vector<VectorXi> TUis(this->UVars());
  for (int i = 0; i < this->UVars(); i++) {
    TUis[i].resize(2);
    TUis[i][0] = this->TVar();
    TUis[i][1] = this->TVar() + 1 + i;
  }
  this->ControlFuncsIndex = -1;

  if (this->TranscriptionMode == TranscriptionModes::LGL7) {
    if (this->UVars() > 0) {
      if (this->ControlMode == ControlModes::HighestOrderSpline) {
        LGLControlSpline<4, -1, 2> lgl7spln2(this->UVars());

        this->ControlFuncsIndex = this->indexer.addEquality(lgl7spln2,
                                  PhaseRegionFlags::DefectPairWisePath,
                                  TUvarT,
                                  empty,
                                  empty,
                                  ThreadingFlags::ByApplication);

      } else if (this->ControlMode == ControlModes::FirstOrderSpline) {
        LGLControlSpline<4, -1, 1> lgl7spln1(this->UVars());

        this->ControlFuncsIndex = this->indexer.addEquality(lgl7spln1,
                                  PhaseRegionFlags::DefectPairWisePath,
                                  TUvarT,
                                  empty,
                                  empty,
                                  ThreadingFlags::ByApplication);
      }
    }
  } else if (this->TranscriptionMode == TranscriptionModes::LGL5) {
    if (this->UVars() > 0) {
      if (this->ControlMode == ControlModes::HighestOrderSpline
          || this->ControlMode == ControlModes::FirstOrderSpline) {
        LGLControlSpline<3, -1, 1> lgl5spln1(this->UVars());

        this->ControlFuncsIndex = this->indexer.addEquality(lgl5spln1,
                                  PhaseRegionFlags::DefectPairWisePath,
                                  TUvarT,
                                  empty,
                                  empty,
                                  ThreadingFlags::ByApplication);
      }
    }
  }
}

void ASSET::ODEPhaseBase::check_functions(int pnum) {
  auto CheckFun = [&](const std::string& type, auto& func) {
    if (func.XtUVars.size() > 0) {
      if (func.XtUVars.maxCoeff() >= this->XtUPVars() || func.XtUVars.minCoeff() < 0) {

        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "{0:} function state variable indices out of bounds in phase:{1:}\n"
                   " Function Storage Index:{2:}\n"
                   " Function Name:{3:}\n",
                   type,
                   pnum,
                   func.StorageIndex,
                   func.Func.name());
        throw std::invalid_argument("");
      }
    }
    if (func.OPVars.size() > 0) {
      if (func.OPVars.maxCoeff() >= this->PVars() || func.OPVars.minCoeff() < 0) {

        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "{0:} function ODE Param variable indices out of bounds in phase:{1:}\n"
                   " Function Storage Index:{2:}\n"
                   " Function Name:{3:}\n",
                   type,
                   pnum,
                   func.StorageIndex,
                   func.Func.name());
        throw std::invalid_argument("");
      }
    }
    if (func.SPVars.size() > 0) {
      if (func.SPVars.maxCoeff() >= this->numStatParams || func.SPVars.minCoeff() < 0) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "{0:} function Static Param variable indices out of bounds in phase:{1:}\n"
                   " Function Storage Index:{2:}\n"
                   " Function Name:{3:}\n",
                   type,
                   pnum,
                   func.StorageIndex,
                   func.Func.name());
        throw std::invalid_argument("");
      }
    }
    if (func.EXTVars.size() > 0) {
      if (func.EXTVars.maxCoeff() >= this->numStatParams || func.EXTVars.minCoeff() < 0) {

        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "{0:} function Integral Static Param variable indices out of bounds in phase:{1:}\n"
                   " Function Storage Index:{2:}\n"
                   " Function Name:{3:}\n",
                   type,
                   pnum,
                   func.StorageIndex,
                   func.Func.name());
        throw std::invalid_argument("");
      }
    }
  };

  std::string eq = "Equality constraint";
  std::string iq = "Inequality constraint";
  std::string sobj = "State objective";
  std::string iobj = "Integral objective";
  std::string ipcon = "Integral parameter";

  for (auto& [key,f]: this->userEqualities)
    CheckFun(eq, f);
  for (auto& [key, f]: this->userInequalities)
    CheckFun(iq, f);
  for (auto& [key, f]: this->userIntegrands)
    CheckFun(iobj, f);
  for (auto& [key, f]: this->userParamIntegrands)
    CheckFun(ipcon, f);
  for (auto& [key, f]: this->userStateObjectives)
    CheckFun(sobj, f);
}

Eigen::VectorXd ASSET::ODEPhaseBase::get_input_scale(PhaseRegionFlags flag, VectorXi XtUV, VectorXi OPV, VectorXi SPV) const
{

    int nloops;
    switch (flag) {
    case Front:
    case Back:
    case Path:
    case Params:
    case ODEParams:
    case StaticParams:
    case InnerPath: {
        nloops = 1;
        break;
    }
    case FrontandBack:
    case BackandFront:
    case PairWisePath: {
        nloops = 2;
        break;
    }
    default: {
        throw std::invalid_argument("Cannot scale this phase region");
        break;
    }
    }


    int isize = XtUV.size() * nloops + OPV.size() + SPV.size();
    VectorXd scales(isize);

    int next = 0;
    for (int n = 0; n < nloops; n++) {
        for (int i = 0; i < XtUV.size(); i++) {
            scales[next] = this->XtUPUnits[XtUV[i]];
            next++;
        }
    }
    for (int i = 0; i < OPV.size(); i++) {
        scales[next] = this->XtUPUnits[OPV[i] + this->XtUVars()];
        next++;
    }
    for (int i = 0; i < SPV.size(); i++) {
        scales[next] = this->SPUnits[SPV[i]];
        next++;
    }

    return scales;
}

std::vector<Eigen::VectorXd> ASSET::ODEPhaseBase::get_test_inputs(PhaseRegionFlags flag, VectorXi XtUV, VectorXi OPV, VectorXi SPV) const
{

    std::vector<std::vector<int>> test_states;

    int nloops = 0;
    switch (flag) {
    case Front: {
        test_states.push_back({ 0 });
        break;
    }
    case Back: {
        test_states.push_back({ int(this->ActiveTraj.size() - 1) });
        break;
    }
    case Params: {
        test_states.push_back({ 0 });
        break;
    }
    case ODEParams: {
        test_states.push_back({ 0 });
        break;
    }
    case StaticParams: {
        test_states.push_back({ 0 });
        break;
    }
    case Path: {
        for (int i = 0; i < this->ActiveTraj.size(); i++)
            test_states.push_back({ i });
        break;
    }
    case InnerPath: {
        for (int i = 1; i < this->ActiveTraj.size() - 1; i++)
            test_states.push_back({ i });
        break;
    }
    case FrontandBack: {
        test_states.push_back({ 0, int(this->ActiveTraj.size() - 1) });
        break;
    }
    case BackandFront: {
        test_states.push_back({ int(this->ActiveTraj.size() - 1),0 });
        break;
    }
    case PairWisePath: {
        for (int i = 0; i < this->ActiveTraj.size() - 1; i++)
            test_states.push_back({ i,i + 1 });
        break;
    }
    default: {
        throw std::invalid_argument("Cannot scale this phase region");
        break;
    }
    }


    int isize = XtUV.size() * test_states[0].size() + OPV.size() + SPV.size();



    std::vector<Eigen::VectorXd> inputs;

    for (int ncalls = 0; ncalls < test_states.size(); ncalls++) {

        VectorXd input(isize);

        int next = 0;

        for (int i = 0; i < test_states[ncalls].size(); i++) {
            int state = test_states[ncalls][i];
            for (int j = 0; j < XtUV.size(); j++) {
                input[next] = this->ActiveTraj[state][XtUV[j]];
                next++;
            }
        }
        for (int j = 0; j < OPV.size(); j++) {
            input[next] = this->ActiveTraj[0][OPV[j] + this->XtUVars()];
            next++;
        }
        for (int j = 0; j < SPV.size(); j++) {
            input[next] = this->ActiveStaticParams[SPV[j]];
            next++;
        }
        inputs.push_back(input);
    }

    return inputs;
}

void ASSET::ODEPhaseBase::calc_auto_scales()
{
    auto calc_impl = [&](auto& funcmap) {
        for (auto& [key, func] : funcmap) {
            if (func.ScaleMode == "auto") {
                VectorXd input_scales = this->get_input_scale(func.RegionFlag, func.XtUVars, func.OPVars, func.SPVars);
                std::vector<VectorXd> test_inputs = this->get_test_inputs(func.RegionFlag, func.XtUVars, func.OPVars, func.SPVars);
                VectorXd output_scales = calc_jacobian_row_scales(func.Func, input_scales, test_inputs, "norm", "mean");
                func.OutputScales = output_scales;
                //std::cout << output_scales << std::endl;
            }
            else {


            }

        }
    };

    calc_impl(this->userEqualities);
    calc_impl(this->userInequalities);
    calc_impl(this->userStateObjectives);
    calc_impl(this->userIntegrands);
    calc_impl(this->userParamIntegrands);


}

void ASSET::ODEPhaseBase::transcribe_phase(
    int vo, int eqo, int iqo, std::shared_ptr<NonLinearProgram> np, int pnum)

{
  this->indexer.begin_indexing(np, vo, eqo, iqo);
  this->check_functions(pnum);

  if (this->AutoScaling) {
      this->calc_auto_scales();
  }

  this->transcribe_dynamics();
  this->transcribe_axis_funcs();
  this->transcribe_control_funcs();
  this->transcribe_integrals();
  this->transcribe_basic_funcs();


  //////DO NOT GET RID OF THIS!!!!!!//
  this->doTranscription = false;
  ////////////////////////////////////
}

void ASSET::ODEPhaseBase::transcribe(bool showstats, bool showfuns) {
  this->nlp = std::make_shared<NonLinearProgram>(this->Threads);

  this->initIndexing();
  this->transcribe_phase(0, 0, 0, this->nlp, 0);
  if (showstats)
    this->indexer.print_stats(showfuns);

  if (this->indexer.numPhaseEqCons > this->indexer.numPhaseVars) {
    fmt::print(fmt::fg(fmt::color::yellow),
               "Transcription Warning!!!\n"
               "Number of Equality Constraints({0:}) in phase exceeds number of free variables({1:}).\n"
               "You likely have a redundant constraint.\n",
               this->indexer.numPhaseEqCons,
               this->indexer.numPhaseVars);
  }

  this->nlp->make_NLP(this->indexer.numPhaseVars, this->indexer.numPhaseEqCons, this->indexer.numPhaseIqCons);

  this->optimizer->setNLP(this->nlp);
}

void ASSET::ODEPhaseBase::test_threads(int i, int j, int n) {
  this->resetTranscription();

  auto nlp1 = std::make_shared<NonLinearProgram>(i);
  this->initIndexing();
  this->transcribe_phase(0, 0, 0, nlp1, 0);
  if (false)
    this->indexer.print_stats(false);
  nlp1->make_NLP(this->indexer.numPhaseVars, this->indexer.numPhaseEqCons, this->indexer.numPhaseIqCons);

  auto nlp2 = std::make_shared<NonLinearProgram>(j);
  this->initIndexing();
  this->transcribe_phase(0, 0, 0, nlp2, 0);
  if (false)
    this->indexer.print_stats(false);
  nlp2->make_NLP(this->indexer.numPhaseVars, this->indexer.numPhaseEqCons, this->indexer.numPhaseIqCons);

  Eigen::VectorXd v = this->makeSolverInput();
  NonLinearProgram::NLPTest(v, n, nlp1, nlp2);

  this->resetTranscription();
}

bool ASSET::ODEPhaseBase::checkMesh() {

  Eigen::VectorXd tsnd;
  Eigen::MatrixXd mesh_errors;
  Eigen::MatrixXd mesh_dist;

  this->Table.loadExactData(this->ActiveTraj);


  if (this->MeshErrorEstimator == "integrator" || this->TranscriptionMode == CentralShooting) {
    this->get_meshinfo_integrator(tsnd, mesh_errors, mesh_dist);
  } else if (this->MeshErrorEstimator == "deboor") {
    this->get_meshinfo_deboor(tsnd, mesh_errors, mesh_dist);
  } else {
    throw std::invalid_argument("Unknown mesh error estimator");
  }


  Eigen::VectorXd error = mesh_errors.colwise().lpNorm<Eigen::Infinity>();
  Eigen::VectorXd dist = mesh_dist.colwise().lpNorm<Eigen::Infinity>();


  this->MeshIters.emplace_back(this->numDefects, this->MeshTol, tsnd, error, dist);

  if (this->MeshErrorCriteria == "endtoend" || this->MeshErrorDistributor == "endtoend") {
    Eigen::VectorXd error_vec = this->calc_global_error();
    this->MeshIters.back().global_error = error_vec.lpNorm<Eigen::Infinity>();
  }


  double error_crit;

  if (this->MeshErrorCriteria == "max") {
    error_crit = this->MeshIters.back().max_error;
  } else if (this->MeshErrorCriteria == "avg") {
    error_crit = this->MeshIters.back().avg_error;
  } else if (this->MeshErrorCriteria == "geometric") {
    error_crit = this->MeshIters.back().gmean_error;
  } else if (this->MeshErrorCriteria == "endtoend") {
    error_crit = this->MeshIters.back().global_error;
  } else {
    throw std::invalid_argument("Unknown mesh error criteria");
  }

  this->MeshConverged = (error_crit < this->MeshTol);
  this->MeshIters.back().converged = this->MeshConverged;

  return this->MeshConverged;
}

void ASSET::ODEPhaseBase::updateMesh() {


  double ntemp = 0;
  for (int i = 0; i < this->MeshIters.back().error.size() - 1; i++) {

    double nsegs = std::pow((this->MeshIters.back().error[i] * this->MeshErrFactor) / this->MeshTol,
                            1 / (this->Order + 1));
    ntemp += std::max(this->MeshRedFactor, nsegs);
  }
  int n = int(std::ceil(ntemp)) + this->NumExtraSegs;

  n = std::clamp(n, int(this->numDefects * this->MeshRedFactor), int(this->numDefects * this->MeshIncFactor));
  n = std::clamp(n, this->MinSegments, this->MaxSegments);

  Eigen::VectorXd bins = this->MeshIters.back().calc_bins(n);

  if (this->DetectControlSwitches && this->UVars() > 0) {
    Eigen::VectorXd switchvec = this->calcSwitches();
    std::vector<double> stmp;


    for (int i = 0; i < bins.size() - 1; i++) {
      for (int j = 0; j < switchvec.size(); j++) {
        if (switchvec[j] > bins[i] && switchvec[j] < bins[i + 1]) {
          stmp.push_back(2 * bins[i] / 3.0 + bins[i + 1] / 3.0);
          stmp.push_back(bins[i] / 3.0 + 2 * bins[i + 1] / 3.0);

          break;
        }
      }
    }
    switchvec.resize(stmp.size());
    for (int i = 0; i < stmp.size(); i++) {
      switchvec[i] = stmp[i];
    }

    Eigen::VectorXd binstmp(bins.size() + switchvec.size());
    binstmp.head(bins.size()) = bins;
    binstmp.tail(switchvec.size()) = switchvec;
    std::sort(binstmp.begin(), binstmp.end());

    bins = binstmp;
  }

  Eigen::VectorXi dpb = VectorXi::Ones(bins.size() - 1);
  this->MeshIters.back().up_numsegs = bins.size() - 1;

  this->refineTrajManual(bins, dpb);
}

Eigen::VectorXd ASSET::ODEPhaseBase::calcSwitches() {

  Eigen::MatrixXd uvals(this->UVars(), this->ActiveTraj.size());
  Eigen::VectorXd tsnd(this->ActiveTraj.size());

  double T0 = this->ActiveTraj[0][this->TVar()];
  double TF = this->ActiveTraj.back()[this->TVar()];

  for (int i = 0; i < this->ActiveTraj.size(); i++) {
    uvals.col(i) = this->ActiveTraj[i].segment(this->XtVars(), this->UVars());
    tsnd[i] = (this->ActiveTraj[i][this->TVar()] - T0) / (TF - T0);
  }

  Eigen::VectorXd umin = uvals.rowwise().minCoeff();
  Eigen::VectorXd umax = uvals.rowwise().maxCoeff();
  Eigen::VectorXd ones(this->UVars());
  ones.setOnes();

  Eigen::MatrixXd und(this->UVars(), this->ActiveTraj.size());

  for (int i = 0; i < this->ActiveTraj.size(); i++) {
    und.col(i) = (uvals.col(i) - umin).cwiseQuotient(ones + umax - umin);
  }

  Eigen::VectorXd udiff;
  Eigen::VectorXd unddiff;
  std::vector<double> switches;

  for (int i = 0; i < this->ActiveTraj.size() - 1; i++) {
    udiff = (uvals.col(i + 1) - uvals.col(i)).cwiseAbs();
    unddiff = (uvals.col(i + 1) - uvals.col(i)).cwiseAbs();
    if (udiff.maxCoeff() > this->AbsSwitchTol && unddiff.maxCoeff() > this->RelSwitchTol) {
      double t = tsnd[i + 1] / 2.0 + tsnd[i] / 2.0;
      switches.push_back(t);
    }
  }


  return stdvector_to_eigenvector(switches);
}


ASSET::PSIOPT::ConvergenceFlags ASSET::ODEPhaseBase::psipot_call_impl(std::string mode) {
  if (this->doTranscription)
    this->transcribe();
  VectorXd Input = this->makeSolverInput();
  VectorXd Output;

  if (mode == "solve") {
    Output = this->optimizer->solve(Input);
  } else if (mode == "optimize") {
    Output = this->optimizer->optimize(Input);
  } else if (mode == "solve_optimize") {
    Output = this->optimizer->solve_optimize(Input);
  } else if (mode == "solve_optimize_solve") {
    Output = this->optimizer->solve_optimize_solve(Input);
  } else if (mode == "optimize_solve") {
    Output = this->optimizer->optimize_solve(Input);
  } else {
    throw std::invalid_argument("Unrecognized PSIOPT mode");
  }

  this->collectSolverOutput(Output);

  this->collectPostOptInfo(this->optimizer->LastEqCons, this->optimizer->LastEqLmults, 
                           this->optimizer->LastIqCons, this->optimizer->LastIqLmults );


  return this->optimizer->ConvergeFlag;
}

ASSET::PSIOPT::ConvergenceFlags ASSET::ODEPhaseBase::phase_call_impl(std::string mode) {

  if (this->PrintMeshInfo && this->AdaptiveMesh) {
    fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
    fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
    fmt::print(": ");
    fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
    fmt::print("\n");
  }

  Utils::Timer Runtimer;

  Runtimer.start();

  PSIOPT::ConvergenceFlags flag = this->psipot_call_impl(mode);

  std::string nextmode = mode;
  if (this->SolveOnlyFirst) {
    if (nextmode.find(std::string("solve_")) != std::string::npos) {
      nextmode.erase(0, 6);
    }
  }

  if (this->AdaptiveMesh) {
    if (flag >= this->MeshAbortFlag) {
      if (this->PrintMeshInfo) {
        fmt::print(fmt::fg(fmt::color::red), "Mesh Iteration 0 Failed to Solve: Aborting\n");
      }
    } else {
      initMeshRefinement();
      for (int i = 0; i < this->MaxMeshIters; i++) {
        if (checkMesh() && !((i == 0) && this->ForceOneMeshIter)) {
          if (this->PrintMeshInfo) {
            MeshIterateInfo::print_header(i);
            this->MeshIters.back().print(0);
            fmt::print(fmt::fg(fmt::color::lime_green), "Mesh Converged\n");
          }
          break;
        } else if (i == this->MaxMeshIters - 1) {
          if (this->PrintMeshInfo) {
            MeshIterateInfo::print_header(i);
            this->MeshIters.back().print(0);
            fmt::print(fmt::fg(fmt::color::red), "Mesh Not Converged\n");
          }
          break;
        } else {
          updateMesh();
          if (this->PrintMeshInfo) {
            MeshIterateInfo::print_header(i);
            this->MeshIters.back().print(0);
          }
        }
        flag = this->psipot_call_impl(nextmode);
        if (flag >= this->MeshAbortFlag) {
          if (this->PrintMeshInfo) {
            fmt::print(fmt::fg(fmt::color::red), "Mesh Iteration {0:} Failed to Solve: Aborting\n", i + 1);
          }
          break;
        }
      }
    }
  }

  if (this->PrintMeshInfo && this->AdaptiveMesh) {

    Runtimer.stop();
    double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
    fmt::print("Total Time:");
    if (tseconds > 0.5) {
      fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.4f} s\n", tseconds);
    } else {
      fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.2f} ms\n", tseconds * 1000);
    }


    fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
    fmt::print(": ");
    fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
    fmt::print("\n");
    fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
  }

  return flag;
}

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

  obj.def("setStaticParams", py::overload_cast<VectorXd,VectorXd>( & ODEPhaseBase::setStaticParams), ODEPhaseBase_setStaticParams);
  obj.def("setStaticParams", py::overload_cast<VectorXd>(&ODEPhaseBase::setStaticParams), ODEPhaseBase_setStaticParams);


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

  obj.def(
      "returnInequalConLmults", &ODEPhaseBase::returnInequalConLmults, ODEPhaseBase_returnInequalConLmults);
  obj.def("returnInequalConVals", &ODEPhaseBase::returnInequalConVals);


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
      py::overload_cast<RegionType ,
      VectorFunctionalX ,
      VarIndexType ,
      VarIndexType ,
      VarIndexType ,
      ScaleType >(&ODEPhaseBase::addEqualCon),
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
      py::arg("AutoScale")=std::string("auto"));

  obj.def("addBoundaryValue",
      py::overload_cast<RegionType , 
      VarIndexType , 
      const std::variant<double, VectorXd>& , 
      ScaleType >(&ODEPhaseBase::addBoundaryValue),
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
      py::arg("scale")=1.0,
      py::arg("AutoScale") = std::string("auto")
  );

  obj.def("addDeltaTimeEqualCon",
      py::overload_cast<
      double,
      double,
      ScaleType>(&ODEPhaseBase::addDeltaTimeEqualCon),
      py::arg("value"),
      py::arg("scale")=1.0,
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
      py::overload_cast<RegionType ,
      VarIndexType , 
      double , 
      double , 
      double , 
      ScaleType >(&ODEPhaseBase::addLUVarBound),
      py::arg("PhaseRegion"),
      py::arg("var"),
      py::arg("lowerbound"),
      py::arg("upperbound"),
      py::arg("scale")=1.0,
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
      py::overload_cast<RegionType ,
      ScalarFunctionalX ,
      VarIndexType ,
      VarIndexType ,
      VarIndexType ,
      double ,
      double ,
      double ,
      ScaleType >(&ODEPhaseBase::addLUFuncBound),
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
      py::overload_cast<VarIndexType , 
      double ,
      double ,
      ScaleType >(&ODEPhaseBase::addLowerDeltaVarBound),
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
