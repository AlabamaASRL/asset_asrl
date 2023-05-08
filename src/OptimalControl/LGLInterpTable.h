#pragma once

#include "FDDerivArbitrary.h"
#include "LGLCoeffs.h"
#include "OptimalControlFlags.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "pch.h"

namespace ASSET {
  struct LGLInterpTable {
    using VectorFunctionalX = GenericFunction<-1, -1>;

    Eigen::MatrixXd XtUData;
    Eigen::MatrixXd XdotData;

    int XVars = 0;
    int UVars = 0;
    int XtUVars = 0;
    int axis = 0;

    double DeltaT = 0.0;
    double TotalT = 0.0;
    double T0 = 0;
    double TF = 0;
    double Eps = 1.0e-4;

    TranscriptionModes Method = TranscriptionModes::LGL3;
    bool BlockedControls = false;
    bool HasOde = false;
    double Order = 3.0;
    double ErrorWeight;

    int BlockSize = 0;
    int NumBlocks = 0;
    int NumStates = 0;
    mutable int LastBlockAccessed = 0;

    bool Periodic = false;
    bool EvenData = false;

    bool WarnOutOfBounds = true;
    bool ThrowOutOfBounds = false;

    Eigen::VectorXd Tspacing;
    Eigen::MatrixXd Xweights;
    Eigen::MatrixXd DXweights;
    Eigen::MatrixXd Uweights;
    VectorFunctionalX ode;
    LGLInterpTable(VectorFunctionalX od, int xv, int uv, TranscriptionModes m) {
      this->XVars = xv;
      this->UVars = uv;
      this->axis = xv;
      this->XtUVars = xv + uv + 1;
      this->ode = od;
      this->HasOde = true;
      this->setMethod(m);
    }
    LGLInterpTable(VectorFunctionalX od,
                   int xv,
                   int uv,
                   TranscriptionModes m,
                   const std::vector<Eigen::VectorXd>& xtudat,
                   int dnum) {
      this->XVars = xv;
      this->UVars = uv;
      this->axis = xv;
      this->XtUVars = xv + uv + 1;
      this->ode = od;
      this->HasOde = true;
      this->setMethod(m);
      this->loadUnevenData(dnum, xtudat);
    }

    LGLInterpTable(VectorFunctionalX od,
                   int xv,
                   int uv,
                   int pv,
                   std::string m,
                   const std::vector<Eigen::VectorXd>& xtudat,
                   int dnum) {
      this->XVars = xv;
      this->UVars = uv + pv;
      this->axis = xv;
      this->XtUVars = xv + uv + 1 + pv;
      this->ode = od;
      this->HasOde = true;
      this->setMethod(strto_TranscriptionMode(m));
      this->loadUnevenData(dnum, xtudat);
    }
    LGLInterpTable(VectorFunctionalX od, int xv, int uv, int pv, const std::vector<Eigen::VectorXd>& xtudat)
        : LGLInterpTable(od, xv, uv, pv, "LGL3", xtudat, xtudat.size() - 1) {
    }
    LGLInterpTable(VectorFunctionalX od, int xv, int uv, const std::vector<Eigen::VectorXd>& xtudat)
        : LGLInterpTable(od, xv, uv, 0, "LGL3", xtudat, xtudat.size() - 1) {
    }

    LGLInterpTable(VectorFunctionalX od,
                   int xv,
                   int uv,
                   std::string m,
                   const std::vector<Eigen::VectorXd>& xtudat,
                   int dnum)
        : LGLInterpTable(od, xv, uv, 0, m, xtudat, dnum) {
    }

    LGLInterpTable(
        int xv, int uv, TranscriptionModes m, const std::vector<Eigen::VectorXd>& xtudat, int dnum) {
      this->XVars = xv;
      this->UVars = uv;
      this->axis = xv;
      this->XtUVars = xv + uv + 1;
      this->setMethod(m);
      this->loadUnevenData(dnum, xtudat);
    }

    LGLInterpTable(int xv, const std::vector<Eigen::VectorXd>& xtudat, int dnum) {
      this->XVars = xv;
      this->UVars = 0;
      this->axis = xv;
      this->XtUVars = xv + 0 + 1;
      this->setMethod(LGL3);
      this->loadUnevenData(dnum, xtudat);
    }
    LGLInterpTable(const std::vector<Eigen::VectorXd>& xtudat) {
      this->XVars = xtudat[0].size() - 1;
      this->UVars = 0;
      this->axis = xtudat[0].size() - 1;
      this->XtUVars = xtudat[0].size();
      this->setMethod(LGL3);
      this->loadUnevenData(xtudat.size() - 1, xtudat);
    }
    LGLInterpTable(int xv, int uv, TranscriptionModes m) {
      this->XVars = xv;
      this->UVars = uv;
      this->axis = xv;
      this->XtUVars = xv + uv + 1;
      this->setMethod(m);
    }
    LGLInterpTable() {
    }
    std::shared_ptr<LGLInterpTable> getTablePtr() {
      return std::shared_ptr<LGLInterpTable>(this);
    }
    void setMethod(TranscriptionModes m);

    void makePeriodic() {
      this->Periodic = true;

      Eigen::VectorXd x0 = XtUData.col(0);
      Eigen::VectorXd xd0 = XdotData.col(0);
      Eigen::VectorXd xf = XtUData.col(XtUData.cols() - 1);

      double diff = (x0.head(this->XVars) - xf.head(this->XVars)).cwiseAbs().maxCoeff();

      if (diff > 1.0e-8) {
        std::cout << "Warning: Calling makePeriodic on non-Periodic table data" << std::endl;
      }
      XtUData.col(XtUData.cols() - 1).head(this->XVars) = x0.head(this->XVars);
      XdotData.col(XdotData.cols() - 1).head(this->XVars) = xd0.head(this->XVars);
    }
    void loadEvenData(const std::vector<Eigen::VectorXd>& xtudat) {
      int msize = xtudat[0].size();
      if (msize != this->XtUVars) {
        std::cout << "User Input Error in supplying data to LGLInterpTable" << std::endl;
        std::cout << " Dimension of Input States(" << msize
                  << ") does not match expected dimensions of the Table(" << this->XtUVars << ")"
                  << std::endl;
        exit(1);
      }
      this->XtUData.resize(this->XtUVars, xtudat.size());
      this->XdotData.resize(this->XVars, xtudat.size());
      this->T0 = xtudat[0][axis];
      this->TF = xtudat.back()[axis];
      this->TotalT = xtudat.back()[axis] - xtudat[0][axis];
      this->NumStates = xtudat.size();
      this->NumBlocks = (this->NumStates - 1) / (this->BlockSize - 1);
      this->DeltaT = this->TotalT / double(this->NumBlocks);
      this->EvenData = true;
      this->LastBlockAccessed = 0;
      Eigen::VectorXd temp(this->XVars);

      for (int i = 0; i < this->NumStates; i++) {
        this->XtUData.col(i) = xtudat[i];
        if (this->HasOde) {
          temp.setZero();
          this->ode.compute(xtudat[i], temp);
          this->XdotData.col(i) = temp;
        }
      }

      if (!this->HasOde) {
        FDDerivArbitrary<Eigen::VectorXd> dterp;
        dterp.setAxis(this->axis);
        dterp.setData(xtudat);
        std::vector<Eigen::VectorXd> datatmp;
        if (xtudat.size() > 8) {
          datatmp = dterp.deriv<Eigen::VectorXd>(1, 4);
        } else {
          datatmp = dterp.deriv<Eigen::VectorXd>(1, 1);
        }

        for (int i = 0; i < this->NumStates; i++) {
          this->XdotData.col(i) = datatmp[i].head(this->XVars);
        }
      }
    }

    void loadEvenData2(const std::vector<Eigen::VectorXd>& xtudat,
                       const std::vector<Eigen::VectorXd>& xdotdat) {
      int msize = xtudat[0].size();
      if (msize != this->XtUVars) {
        std::cout << "User Input Error in supplying data to LGLInterpTable" << std::endl;
        std::cout << " Dimension of Input States(" << msize
                  << ") does not match expected dimensions of the Table(" << this->XtUVars << ")"
                  << std::endl;
        exit(1);
      }
      this->XtUData.resize(this->XtUVars, xtudat.size());
      this->XdotData.resize(this->XVars, xtudat.size());
      this->T0 = xtudat[0][axis];
      this->TF = xtudat.back()[axis];

      this->TotalT = xtudat.back()[axis] - xtudat[0][axis];
      this->NumStates = xtudat.size();
      this->NumBlocks = (this->NumStates - 1) / (this->BlockSize - 1);
      this->DeltaT = this->TotalT / double(this->NumBlocks);
      this->EvenData = true;
      this->LastBlockAccessed = 0;
      Eigen::VectorXd temp(this->XVars);

      for (int i = 0; i < this->NumStates; i++) {
        this->XtUData.col(i) = xtudat[i];
        this->XdotData.col(i) = xdotdat[i];
      }
    }

    void loadUnevenData(int dnum, const std::vector<Eigen::VectorXd>& xtudat) {
      int msize = xtudat[0].size();
      if (msize != this->XtUVars) {
        std::cout << "User Input Error in supplying data to LGLInterpTable" << std::endl;
        std::cout << " Dimension of Input States(" << msize
                  << ") does not match expected dimensions of the Table(" << this->XtUVars << ")"
                  << std::endl;
        exit(1);
      }

      Eigen::VectorXd myspace = this->Tspacing;
      TranscriptionModes mymeth = this->Method;
      this->setMethod(TranscriptionModes::LGL3);

      this->XtUData.resize(this->XtUVars, xtudat.size());
      this->XtUData.setZero();
      this->XdotData.resize(this->XVars, xtudat.size());
      this->XdotData.setZero();
      this->T0 = xtudat[0][axis];
      this->TF = xtudat.back()[axis];

      this->TotalT = xtudat.back()[axis] - xtudat[0][axis];
      this->NumStates = xtudat.size();
      this->NumBlocks = (this->NumStates - 1) / (this->BlockSize - 1);
      this->EvenData = false;
      this->LastBlockAccessed = 0;
      Eigen::VectorXd temp(this->XVars);

      for (int i = 0; i < this->NumStates; i++) {
        this->XtUData.col(i) = xtudat[i];
        if (this->HasOde) {
          temp.setZero();
          this->ode.compute(xtudat[i], temp);
          this->XdotData.col(i) = temp;
        }
      }

      if (!this->HasOde) {
        FDDerivArbitrary<Eigen::VectorXd> dterp;
        dterp.setAxis(this->axis);
        dterp.setData(xtudat);
        std::vector<Eigen::VectorXd> datatmp;
        if (xtudat.size() > 8) {
          dterp.deriv<Eigen::VectorXd>(datatmp, 1, 4);
        } else {
          dterp.deriv<Eigen::VectorXd>(datatmp, 1, 1);
        }

        for (int i = 0; i < this->NumStates; i++) {
          this->XdotData.col(i) = datatmp[i].head(this->XVars);
        }
      }

      std::vector<Eigen::VectorXd> nxs = this->NDequidist(myspace, dnum, 0.0, 1.0);
      std::vector<Eigen::VectorXd> ndxs(nxs.size());

      this->setMethod(mymeth);
      this->loadEvenData(nxs);
    }

    void loadRegularData(int dnum, const std::vector<Eigen::VectorXd>& xtudat) {
      this->XtUData.resize(this->XtUVars, xtudat.size());
      this->XtUData.setZero();
      this->XdotData.resize(this->XVars, xtudat.size());
      this->XdotData.setZero();
      this->T0 = xtudat[0][axis];
      this->TF = xtudat.back()[axis];

      this->TotalT = xtudat.back()[axis] - xtudat[0][axis];
      this->NumStates = xtudat.size();
      this->NumBlocks = (this->NumStates - 1) / (this->BlockSize - 1);
      this->EvenData = false;
      this->LastBlockAccessed = 0;
      Eigen::VectorXd temp(this->XVars);
      for (int i = 0; i < this->NumStates; i++) {
        temp.setZero();
        this->ode.compute(xtudat[i], temp);
        this->XtUData.col(i) = xtudat[i];
        this->XdotData.col(i) = temp;
      }
      std::vector<Eigen::VectorXd> nxs = this->NDequidist(dnum, 0.0, 1.0);
      this->loadEvenData(nxs);
    }
    void loadExactData(const std::vector<Eigen::VectorXd>& xtudat) {
      this->XtUData.resize(this->XtUVars, xtudat.size());
      this->XtUData.setZero();
      this->XdotData.resize(this->XVars, xtudat.size());
      this->XdotData.setZero();
      this->T0 = xtudat[0][axis];
      this->TF = xtudat.back()[axis];

      this->TotalT = xtudat.back()[axis] - xtudat[0][axis];
      this->NumStates = xtudat.size();
      this->NumBlocks = (this->NumStates - 1) / (this->BlockSize - 1);
      this->EvenData = false;
      this->LastBlockAccessed = 0;
      Eigen::VectorXd temp(this->XVars);
      for (int i = 0; i < this->NumStates; i++) {
        temp.setZero();
        this->ode.compute(xtudat[i], temp);
        this->XtUData.col(i) = xtudat[i];
        this->XdotData.col(i) = temp;
      }
    }
    template<class V1, class V2>
    void loadExactData(const std::vector<V1>& xtudat, const std::vector<V2>& xdotdat) {
      this->XtUData.resize(this->XtUVars, xtudat.size());
      this->XtUData.setZero();
      this->XdotData.resize(this->XVars, xtudat.size());
      this->XdotData.setZero();
      this->T0 = xtudat[0][axis];
      this->TF = xtudat.back()[axis];

      this->TotalT = xtudat.back()[axis] - xtudat[0][axis];
      this->NumStates = xtudat.size();
      this->NumBlocks = (this->NumStates - 1) / (this->BlockSize - 1);
      this->EvenData = false;
      this->LastBlockAccessed = 0;
      for (int i = 0; i < this->NumStates; i++) {
        this->XtUData.col(i) = xtudat[i];
        this->XdotData.col(i) = xdotdat[i];
      }
    }


    std::vector<Eigen::VectorXd> NDequidist(Eigen::VectorXd spacing,
                                            int dnum,
                                            double low,
                                            double high) {  // 0 to 1;
      Eigen::VectorXd NodeSpace;
      NodeSpace.setLinSpaced(dnum + 1, low, high);
      Eigen::VectorXd AllSpace(dnum * (spacing.size() - 1) + 1);
      for (int i = 0; i < dnum; i++) {
        AllSpace.segment(i * (spacing.size() - 1), spacing.size()) =
            Eigen::VectorXd::Constant(spacing.size(), NodeSpace[i])
            + spacing * (NodeSpace[i + 1] - NodeSpace[i]);
      }
      AllSpace *= this->TotalT;
      AllSpace += Eigen::VectorXd::Constant(AllSpace.size(), this->T0);
      std::vector<Eigen::VectorXd> mesh(AllSpace.size());
      for (int i = 0; i < AllSpace.size(); i++) {
        mesh[i] = this->Interpolate(AllSpace[i]);
      }
      return mesh;
    }
    std::vector<Eigen::VectorXd> NDequidist(int dnum, double low, double high) {
      return this->NDequidist(this->Tspacing, dnum, low, high);
    }
    std::vector<Eigen::VectorXd> NDdistribute(Eigen::VectorXd ndspacing,
                                              Eigen::VectorXd ndtimes,
                                              Eigen::VectorXi defper) {  // 0-1
      std::vector<Eigen::VectorXd> mesh;
      for (int i = 0; i < defper.size(); i++) {
        std::vector<Eigen::VectorXd> submesh =
            this->NDequidist(ndspacing, defper[i], ndtimes[i], ndtimes[i + 1]);
        int jstart = 1;
        if (i == 0)
          jstart = 0;
        for (int j = jstart; j < (submesh.size()); j++) {
          mesh.push_back(submesh[j]);
        }
      }
      return mesh;
    }
    std::vector<Eigen::VectorXd> NDdistribute(Eigen::VectorXd ndtimes, Eigen::VectorXi defper) {
      return this->NDdistribute(this->Tspacing, ndtimes, defper);
    }

    std::vector<Eigen::VectorXd> InterpRange(int dnum, double tlow, double thig) {
      double frac1 = (tlow - this->T0) / this->TotalT;
      double frac2 = (thig - this->T0) / this->TotalT;
      return this->NDequidist(dnum, frac1, frac2);
    }
    std::vector<Eigen::VectorXd> InterpWholeRange(int dnum) {
      return this->NDequidist(dnum, 0.0, 1.0);
    }

    std::vector<Eigen::VectorXd> ErrorIntegral(int NumSamps) {
      Eigen::VectorXd ts;
      ts.setLinSpaced(NumSamps, this->T0, this->TF);
      std::vector<Eigen::VectorXd> errint(ts.size(), Eigen::VectorXd(2));
      errint[0][0] = 0.0;
      errint[0][1] = ts[0];

      Eigen::Matrix<double, -1, 2> XXd;
      Eigen::VectorXd temp(this->XVars);
      Eigen::VectorXd temp2(this->XVars);

      double h = ts[1] - ts[0];
      for (int i = 1; i < ts.size(); i++) {
        XXd = this->InterpolateDeriv(ts[i]);
        temp.setZero();
        temp2 = XXd.col(0);
        this->ode.compute(temp2, temp);
        errint[i][0] = errint[i - 1][0]
                       + std::pow((temp - XXd.col(1).head(this->XVars)).norm(), 1.0 / (this->Order + 1)) * h;
        errint[i][1] = ts[i];
      }
      return errint;
    }


    std::vector<Eigen::VectorXd> NewErrorIntegral();

    void DeboorMeshError(Eigen::VectorXd& tsnd,
                         Eigen::MatrixXd& mesh_errors,
                         Eigen::MatrixXd& mesh_dist) const;


    template<class Scalar>
    VectorX<Scalar> Interpolate(Scalar tglobal) const {
      VectorX<Scalar> fx(this->XtUVars);
      fx.setZero();
      this->InterpolateRef(tglobal, fx);
      return fx;
    }
    template<class Scalar>
    Eigen::Matrix<Scalar, -1, 2> InterpolateDeriv(Scalar tglobal) const {
      Eigen::Matrix<Scalar, -1, 2> fx(this->XtUVars, 2);
      fx.setZero();
      this->InterpolateDerivRef(tglobal, fx);
      return fx;
    }
    template<class Scalar>
    void FindBlock(Scalar tglobal, Scalar& tnd, int& element) const {
      if (this->EvenData) {
        Scalar tlocal = tglobal - this->T0;

        if (this->Periodic) {


          Scalar frac = tlocal / this->TotalT;
          if (frac > 1.0) {
            tlocal -= int(frac) * this->TotalT;
          } else if (frac < 0.0) {
            tlocal += (int(std::abs(frac)) + 1) * this->TotalT;
          }
        }

        element = int((tlocal / this->DeltaT));
        if (element < 0)
          element = 0;
        element = std::min(element, this->NumBlocks - 1);
        double root = double(element) * (this->DeltaT);
        Scalar remainder = tlocal - root;
        tnd = remainder / this->DeltaT;
        return;
      }
      element = this->LastBlockAccessed;
      int sd = 0;
      do {
        sd = this->CheckIthBlock(tglobal, element);

        element += sd;
        if (element < 0) {
          element = 0;
          break;
        } else if (element == this->NumBlocks) {
          element = this->NumBlocks - 1;
          break;
        }
      } while (sd != 0);
      this->LastBlockAccessed = element;
      double tb0 = this->XtUData.middleCols((this->BlockSize - 1) * element, this->BlockSize)(axis, 0);
      double tbf = this->XtUData.middleCols((this->BlockSize - 1) * element, this->BlockSize)(
          axis, this->BlockSize - 1);
      tnd = (tglobal - tb0) / (tbf - tb0);
    }

    template<class Scalar, class OutType>
    void InterpolateRef(Scalar tglobal, const Eigen::MatrixBase<OutType>& fx) const {
      int element = 0;
      Scalar tnd = 0;
      this->FindBlock(tglobal, tnd, element);

      return InterpIthBlock(tnd, fx, element);
    }
    template<class Scalar, class OutType>
    void InterpolateDerivRef(Scalar tglobal, const Eigen::MatrixBase<OutType>& fx) const {
      int element = 0;
      Scalar tnd = 0;
      this->FindBlock(tglobal, tnd, element);
      return InterpIthBlockDeriv(tnd, fx, element);
    }
    template<class Scalar, class OutType>
    void Interpolate2ndDerivRef(Scalar tglobal, const Eigen::MatrixBase<OutType>& fx) const {
      int element = 0;
      Scalar tnd = 0;
      this->FindBlock(tglobal, tnd, element);
      return InterpIthBlock2ndDeriv(tnd, fx, element);
    }

    template<class Scalar, class OutType>
    void InterpIthBlock(Scalar t, const Eigen::MatrixBase<OutType>& fx, int i) const {
      return this->InterpBlock(t,
                               fx,
                               this->XtUData.middleCols((this->BlockSize - 1) * i, this->BlockSize),
                               this->XdotData.middleCols((this->BlockSize - 1) * i, this->BlockSize));
    }
    template<class Scalar, class OutType>
    void InterpIthBlockDeriv(Scalar t, const Eigen::MatrixBase<OutType>& fx, int i) const {
      return this->InterpBlockDeriv(t,
                                    fx,
                                    this->XtUData.middleCols((this->BlockSize - 1) * i, this->BlockSize),
                                    this->XdotData.middleCols((this->BlockSize - 1) * i, this->BlockSize));
    }
    template<class Scalar, class OutType>
    void InterpIthBlock2ndDeriv(Scalar t, const Eigen::MatrixBase<OutType>& fx, int i) const {
      return this->InterpBlock2ndDeriv(t,
                                       fx,
                                       this->XtUData.middleCols((this->BlockSize - 1) * i, this->BlockSize),
                                       this->XdotData.middleCols((this->BlockSize - 1) * i, this->BlockSize));
    }

    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlockGen(Scalar t,
                        const Eigen::MatrixBase<OutType>& fx,
                        const Eigen::MatrixBase<XtUBlockType>& xtublk,
                        const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      VectorX<Scalar> tpow(Xweights.rows());
      tpow[0] = Scalar(1.0);
      for (int i = 1; i < Xweights.rows(); i++) {
        tpow[i] = tpow[i - 1] * t;
      }

      Eigen::MatrixBase<OutType>& xtuN = fx.const_cast_derived();
      Scalar t0 = xtublk(axis, 0);
      Scalar tf = xtublk(axis, this->BlockSize - 1);
      Scalar h = tf - t0;

      for (int i = 0; i < this->BlockSize; i++) {
        Scalar xsc = tpow.dot(this->Xweights.col(i).template cast<Scalar>());
        Scalar dxsc = tpow.dot(this->DXweights.col(i).template cast<Scalar>());
        xtuN.head(this->XVars) += xtublk.col(i).head(this->XVars).template cast<Scalar>() * xsc
                                  + dxblk.col(i).head(this->XVars).template cast<Scalar>() * dxsc * h;
      }
      xtuN[axis] = t0 + h * t;

      if (this->UVars > 0) {
        if (this->BlockedControls) {
          xtuN.tail(this->UVars) = xtublk.col(0).tail(this->UVars).template cast<Scalar>();
        } else {
          VectorX<Scalar> utpow = tpow.head(Uweights.rows());
          for (int i = 0; i < this->BlockSize; i++) {
            Scalar usc = utpow.dot(this->Uweights.col(i).template cast<Scalar>());
            xtuN.tail(this->UVars) += xtublk.col(i).tail(this->UVars).template cast<Scalar>() * usc;
          }
        }
      }
    }

    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlockDerivGen(Scalar t,
                             const Eigen::MatrixBase<OutType>& fx,
                             const Eigen::MatrixBase<XtUBlockType>& xtublk,
                             const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      VectorX<Scalar> tpow(Xweights.rows());
      tpow[0] = Scalar(1.0);
      VectorX<Scalar> tpow2(Xweights.rows());
      tpow2[0] = Scalar(0.0);
      tpow2[1] = Scalar(1.0);
      for (int i = 1; i < Xweights.rows(); i++) {
        tpow[i] = tpow[i - 1] * t;
      }
      for (int i = 2; i < Xweights.rows(); i++) {
        tpow2[i] = tpow[i - 1] * Scalar(i);
      }

      Eigen::MatrixBase<OutType>& xtuN = fx.const_cast_derived();
      Scalar t0 = xtublk(axis, 0);
      Scalar tf = xtublk(axis, this->BlockSize - 1);
      Scalar h = tf - t0;

      for (int i = 0; i < this->BlockSize; i++) {
        Scalar xsc = tpow.dot(this->Xweights.col(i).template cast<Scalar>());
        Scalar dxsc = tpow.dot(this->DXweights.col(i).template cast<Scalar>());
        Scalar xsc_dt = tpow2.dot(this->Xweights.col(i).template cast<Scalar>());
        Scalar dxsc_dt = tpow2.dot(this->DXweights.col(i).template cast<Scalar>());
        xtuN.col(0).head(this->XVars) += xtublk.col(i).head(this->XVars).template cast<Scalar>() * xsc
                                         + dxblk.col(i).head(this->XVars).template cast<Scalar>() * dxsc * h;
        xtuN.col(1).head(this->XVars) += xtublk.col(i).head(this->XVars).template cast<Scalar>() * xsc_dt / h
                                         + dxblk.col(i).head(this->XVars).template cast<Scalar>() * dxsc_dt;
      }
      xtuN.col(0)[axis] = t0 + h * t;
      xtuN.col(1)[axis] = 1.0;

      if (this->UVars > 0) {
        VectorX<Scalar> utpow = tpow.head(Uweights.rows());
        VectorX<Scalar> utpow2 = tpow2.head(Uweights.rows());

        for (int i = 0; i < this->BlockSize; i++) {
          Scalar usc = utpow.dot(this->Uweights.col(i).template cast<Scalar>());
          Scalar usc_dt = utpow2.dot(this->Uweights.col(i).template cast<Scalar>());

          xtuN.col(0).tail(this->UVars) += xtublk.col(i).tail(this->UVars).template cast<Scalar>() * usc;
          xtuN.col(1).tail(this->UVars) +=
              xtublk.col(i).tail(this->UVars).template cast<Scalar>() * usc_dt / h;
        }
      }
    }

    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlockDeriv2Gen(Scalar t,
                              const Eigen::MatrixBase<OutType>& fx,
                              const Eigen::MatrixBase<XtUBlockType>& xtublk,
                              const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      VectorX<Scalar> tpow(Xweights.rows());
      tpow[0] = Scalar(1.0);
      VectorX<Scalar> tpow2(Xweights.rows());
      tpow2[0] = Scalar(0.0);
      tpow2[1] = Scalar(1.0);
      for (int i = 1; i < Xweights.rows(); i++) {
        tpow[i] = tpow[i - 1] * t;
      }
      for (int i = 2; i < Xweights.rows(); i++) {
        tpow2[i] = tpow[i - 1] * Scalar(i);
      }

      Eigen::MatrixBase<OutType>& xtuN = fx.const_cast_derived();
      Scalar t0 = xtublk(axis, 0);
      Scalar tf = xtublk(axis, this->BlockSize - 1);
      Scalar h = tf - t0;

      for (int i = 0; i < this->BlockSize; i++) {
        Scalar xsc = tpow.dot(this->Xweights.col(i).template cast<Scalar>());
        Scalar dxsc = tpow.dot(this->DXweights.col(i).template cast<Scalar>());
        Scalar xsc_dt = tpow2.dot(this->Xweights.col(i).template cast<Scalar>());
        Scalar dxsc_dt = tpow2.dot(this->DXweights.col(i).template cast<Scalar>());
        xtuN.col(0).head(this->XVars) +=
            xtublk.col(i).head(this->XVars) * xsc + dxblk.col(i).head(this->XVars) * dxsc * h;
        xtuN.col(1).head(this->XVars) +=
            xtublk.col(i).head(this->XVars) * xsc_dt / h + dxblk.col(i).head(this->XVars) * dxsc_dt;
      }
      xtuN.col(0)[axis] = t0 + h * t;
      xtuN.col(1)[axis] = 1.0;

      if (this->UVars > 0) {
        VectorX<Scalar> utpow = tpow.head(Uweights.rows());
        VectorX<Scalar> utpow2 = tpow2.head(Uweights.rows());

        for (int i = 0; i < this->BlockSize; i++) {
          Scalar usc = utpow.dot(this->Uweights.col(i).template cast<Scalar>());
          Scalar usc_dt = utpow2.dot(this->Uweights.col(i).template cast<Scalar>());

          xtuN.col(0).tail(this->UVars) += xtublk.col(i).tail(this->UVars).template cast<Scalar>() * usc;
          xtuN.col(1).tail(this->UVars) +=
              xtublk.col(i).tail(this->UVars).template cast<Scalar>() * usc_dt / h;
        }
      }
    }

    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlockLGL3(Scalar t,
                         const Eigen::MatrixBase<OutType>& fx,
                         const Eigen::MatrixBase<XtUBlockType>& xtublk,
                         const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      Eigen::MatrixBase<OutType>& xtuN = fx.const_cast_derived();
      Scalar t0 = xtublk(axis, 0);
      Scalar tf = xtublk(axis, this->BlockSize - 1);
      Scalar h = tf - t0;
      Scalar t2 = t * t;
      Scalar t3 = t2 * t;
      Scalar xsc0 = (2.0 * t3 - 3.0 * t2 + 1.0);
      Scalar dxsc0 = (t3 - 2.0 * t2 + t) * h;
      Scalar xsc1 = (-2.0 * t3 + 3.0 * t2);
      Scalar dxsc1 = (t3 - t2) * h;

      xtuN.head(this->XVars) = xtublk.col(0).head(this->XVars).template cast<Scalar>() * xsc0
                               + dxblk.col(0).head(this->XVars).template cast<Scalar>() * dxsc0
                               + xtublk.col(1).head(this->XVars).template cast<Scalar>() * xsc1
                               + dxblk.col(1).head(this->XVars).template cast<Scalar>() * dxsc1;
      xtuN[axis] = t0 + h * t;

      if (this->UVars > 0) {

        if (this->BlockedControls) {
          xtuN.tail(this->UVars) = xtublk.col(0).tail(this->UVars).template cast<Scalar>();
        } else {
          Scalar usc0 = 1.0 - t;
          Scalar usc1 = t;
          xtuN.tail(this->UVars) = xtublk.col(0).tail(this->UVars).template cast<Scalar>() * usc0
                                   + xtublk.col(1).tail(this->UVars).template cast<Scalar>() * usc1;
        }
      }
    }

    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlockDerivLGL3(Scalar t,
                              const Eigen::MatrixBase<OutType>& fx,
                              const Eigen::MatrixBase<XtUBlockType>& xtublk,
                              const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      Eigen::MatrixBase<OutType>& xtuN = fx.const_cast_derived();
      Scalar t0 = xtublk(axis, 0);
      Scalar tf = xtublk(axis, this->BlockSize - 1);
      Scalar h = tf - t0;
      Scalar t2 = t * t;
      Scalar t3 = t2 * t;

      Scalar xsc0 = (2.0 * t3 - 3.0 * t2 + 1.0);
      Scalar dxsc0 = (t3 - 2.0 * t2 + t) * h;
      Scalar xsc1 = (-2.0 * t3 + 3.0 * t2);
      Scalar dxsc1 = (t3 - t2) * h;

      Scalar xsc0_dt = (6.0 * t2 - 6.0 * t) / h;
      Scalar dxsc0_dt = (3.0 * t2 - 4.0 * t + 1.0);
      Scalar xsc1_dt = (-6.0 * t2 + 6.0 * t) / h;
      Scalar dxsc1_dt = (3.0 * t2 - 2.0 * t);

      if constexpr (std::is_same<Scalar, double>::value) {
        xtuN.col(0).head(this->XVars) =
            xtublk.col(0).head(this->XVars) * xsc0 + dxblk.col(0).head(this->XVars) * dxsc0
            + xtublk.col(1).head(this->XVars) * xsc1 + dxblk.col(1).head(this->XVars) * dxsc1;
        xtuN.col(1).head(this->XVars) =
            xtublk.col(0).head(this->XVars) * xsc0_dt + dxblk.col(0).head(this->XVars) * dxsc0_dt
            + xtublk.col(1).head(this->XVars) * xsc1_dt + dxblk.col(1).head(this->XVars) * dxsc1_dt;
      } else {
        xtuN.col(0).head(this->XVars) = xtublk.col(0).head(this->XVars).template cast<Scalar>() * xsc0
                                        + dxblk.col(0).head(this->XVars).template cast<Scalar>() * dxsc0
                                        + xtublk.col(1).head(this->XVars).template cast<Scalar>() * xsc1
                                        + dxblk.col(1).head(this->XVars).template cast<Scalar>() * dxsc1;
        xtuN.col(1).head(this->XVars) = xtublk.col(0).head(this->XVars).template cast<Scalar>() * xsc0_dt
                                        + dxblk.col(0).head(this->XVars).template cast<Scalar>() * dxsc0_dt
                                        + xtublk.col(1).head(this->XVars).template cast<Scalar>() * xsc1_dt
                                        + dxblk.col(1).head(this->XVars).template cast<Scalar>() * dxsc1_dt;
      }


      xtuN.col(0)[axis] = t0 + h * t;
      xtuN.col(1)[axis] = 1;

      if (this->UVars > 0) {
        Scalar usc0 = 1.0 - t;
        Scalar usc1 = t;
        Scalar usc0_dt = -1.0 / h;
        Scalar usc1_dt = 1.0 / h;
        xtuN.col(0).tail(this->UVars) = xtublk.col(0).tail(this->UVars).template cast<Scalar>() * usc0
                                        + xtublk.col(1).tail(this->UVars).template cast<Scalar>() * usc1;
        xtuN.col(1).tail(this->UVars) = xtublk.col(0).tail(this->UVars).template cast<Scalar>() * usc0_dt
                                        + xtublk.col(1).tail(this->UVars).template cast<Scalar>() * usc1_dt;
      }
    }

    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlockDeriv2LGL3(Scalar t,
                               const Eigen::MatrixBase<OutType>& fx,
                               const Eigen::MatrixBase<XtUBlockType>& xtublk,
                               const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      Eigen::MatrixBase<OutType>& xtuN = fx.const_cast_derived();
      Scalar t0 = xtublk(axis, 0);
      Scalar tf = xtublk(axis, this->BlockSize - 1);
      Scalar h = tf - t0;
      Scalar t2 = t * t;
      Scalar t3 = t2 * t;

      Scalar xsc0 = (2.0 * t3 - 3.0 * t2 + 1.0);
      Scalar dxsc0 = (t3 - 2.0 * t2 + t) * h;
      Scalar xsc1 = (-2.0 * t3 + 3.0 * t2);
      Scalar dxsc1 = (t3 - t2) * h;

      Scalar xsc0_dt = (6.0 * t2 - 6.0 * t) / h;
      Scalar dxsc0_dt = (3.0 * t2 - 4.0 * t + 1.0);
      Scalar xsc1_dt = (-6.0 * t2 + 6.0 * t) / h;
      Scalar dxsc1_dt = (3.0 * t2 - 2.0 * t);

      Scalar xsc0_dt2 = (12.0 * t - 6.0) / (h * h);
      Scalar dxsc0_dt2 = (6.0 * t - 4.0) / h;
      Scalar xsc1_dt2 = (-12.0 * t + 6.0) / (h * h);
      Scalar dxsc1_dt2 = (6.0 * t - 2.0) / h;

      xtuN.col(0).head(this->XVars) =
          xtublk.col(0).head(this->XVars) * xsc0 + dxblk.col(0).head(this->XVars) * dxsc0
          + xtublk.col(1).head(this->XVars) * xsc1 + dxblk.col(1).head(this->XVars) * dxsc1;
      xtuN.col(1).head(this->XVars) =
          xtublk.col(0).head(this->XVars) * xsc0_dt + dxblk.col(0).head(this->XVars) * dxsc0_dt
          + xtublk.col(1).head(this->XVars) * xsc1_dt + dxblk.col(1).head(this->XVars) * dxsc1_dt;
      xtuN.col(2).head(this->XVars) =
          xtublk.col(0).head(this->XVars) * xsc0_dt2 + dxblk.col(0).head(this->XVars) * dxsc0_dt2
          + xtublk.col(1).head(this->XVars) * xsc1_dt2 + dxblk.col(1).head(this->XVars) * dxsc1_dt2;

      xtuN.col(0)[axis] = t0 + h * t;
      xtuN.col(1)[axis] = 1;
      xtuN.col(2)[axis] = 0;

      if (this->UVars > 0) {
        Scalar usc0 = 1.0 - t;
        Scalar usc1 = t;
        Scalar usc0_dt = -1.0 / h;
        Scalar usc1_dt = 1.0 / h;
        xtuN.col(0).tail(this->UVars) =
            xtublk.col(0).tail(this->UVars) * usc0 + xtublk.col(1).tail(this->UVars) * usc1;
        xtuN.col(1).tail(this->UVars) =
            xtublk.col(0).tail(this->UVars) * usc0_dt + xtublk.col(1).tail(this->UVars) * usc1_dt;
        xtuN.col(2).tail(this->UVars).setZero();
      }
    }

    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlock(Scalar t,
                     const Eigen::MatrixBase<OutType>& fx,
                     const Eigen::MatrixBase<XtUBlockType>& xtublk,
                     const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      if (this->Method == TranscriptionModes::LGL3 || this->Method == TranscriptionModes::Trapezoidal) {
        return this->InterpBlockLGL3(t, fx, xtublk, dxblk);
      }
      return this->InterpBlockGen(t, fx, xtublk, dxblk);
    }
    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlockDeriv(Scalar t,
                          const Eigen::MatrixBase<OutType>& fx,
                          const Eigen::MatrixBase<XtUBlockType>& xtublk,
                          const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      if (this->Method == TranscriptionModes::LGL3 || this->Method == TranscriptionModes::Trapezoidal) {
        return this->InterpBlockDerivLGL3(t, fx, xtublk, dxblk);
      }
      return this->InterpBlockDerivGen(t, fx, xtublk, dxblk);
    }

    template<class Scalar, class OutType, class XtUBlockType, class DXBlockType>
    void InterpBlock2ndDeriv(Scalar t,
                             const Eigen::MatrixBase<OutType>& fx,
                             const Eigen::MatrixBase<XtUBlockType>& xtublk,
                             const Eigen::MatrixBase<DXBlockType>& dxblk) const {
      if (this->Method == TranscriptionModes::LGL3) {
        return this->InterpBlockDeriv2LGL3(t, fx, xtublk, dxblk);
      } else {
        throw std::invalid_argument("Implement LGL Table 2ndDerives");
      }
    }

    template<class Scalar>
    int CheckIthBlock(Scalar tglob, int i) const {
      Scalar t0 = this->XtUData.middleCols((this->BlockSize - 1) * i, this->BlockSize)(axis, 0);
      Scalar tf =
          this->XtUData.middleCols((this->BlockSize - 1) * i, this->BlockSize)(axis, this->BlockSize - 1);


      int sd = 0;

      if (tf > t0) {
        if (t0 <= tglob && tf >= tglob)
          sd = 0;
        else if (tglob > tf)
          sd = 1;
        else if (tglob < t0)
          sd = -1;
      } else {
        if (t0 >= tglob && tf <= tglob)
          sd = 0;
        else if (tglob < tf)
          sd = 1;
        else if (tglob > t0)
          sd = -1;
      }


      return sd;
    }

    static void Build(py::module& m);
  };

}  // namespace ASSET
