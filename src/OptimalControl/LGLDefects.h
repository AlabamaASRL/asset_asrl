#pragma once

#include "LGLCoeffs.h"
#include "TranscriptionSizing.h"
#include "VectorFunctions/VectorFunction.h"

namespace ASSET {
  template<class DODE, int CS>
  struct LGLDefects : VectorFunction<LGLDefects<DODE, CS>,
                                     DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefIRC,
                                     DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefORC> {
    static const int Cardinals = CS;
    static const int Interiors = CS - 1;

    using Base = VectorFunction<LGLDefects<DODE, CS>,
                                DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefIRC,
                                DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefORC>;

    /////////////////////////////////////////////////////////////////////////////////
    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;
    template<class Scalar>
    using Jacobian = typename Base::template Jacobian<Scalar>;
    template<class Scalar>
    using Hessian = typename Base::template Hessian<Scalar>;
    //////////////////////////////////////////////////////////////////////////////////
    template<class Scalar>
    using ODEOutput = typename DODE::template Output<Scalar>;
    template<class Scalar>
    using ODEInput = typename DODE::template Input<Scalar>;
    template<class Scalar>
    using ODEGrad = typename DODE::template Gradient<Scalar>;
    template<class Scalar>
    using ODEJacobian = typename DODE::template Jacobian<Scalar>;
    template<class Scalar>
    using ODEHessian = typename DODE::template Hessian<Scalar>;

    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;
    using Coeffs = LGLCoeffs<CS>;
    /////////////////////////////////////////////////////////////////////////////
    DODE ode;
    static const bool IsVectorizable = DODE::IsVectorizable;

    LGLDefects(const DODE& od) {
      this->setODE(od);
    }
    void setODE(const DODE& od) {
      this->ode = od;
      this->setOutputRows(this->ode.ORows() * (CS - 1));
      this->setInputRows(CS * this->ode.XtUVars() + this->ode.PVars());
    }

    ////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();


      auto Impl = [&](auto& C_XS, auto& C_DXS, auto& I_XS, auto& I_DXS) {
        for (int i = 0; i < Cardinals; i++) {
          C_XS[i].template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
              x.template segment<DODE::XtUV>(i * this->ode.XtUVars(), this->ode.XtUVars());
          if constexpr (DODE::PV >= 0) {
            C_XS[i].template tail<DODE::PV>(this->ode.PVars()) = x.template tail<DODE::PV>(this->ode.PVars());
          } else {
            C_XS[i].tail(this->ode.PVars()) = x.tail(this->ode.PVars());
          }

          this->ode.compute(C_XS[i], C_DXS[i]);
        }

        Scalar h = C_XS.back()[this->ode.TVar()] - C_XS[0][this->ode.TVar()];

        for (int i = 0; i < Interiors; i++) {
          I_XS[i][this->ode.TVar()] = C_XS[0][this->ode.TVar()] + h * Coeffs::InteriorSpacings[i];

          if constexpr (DODE::PV >= 0) {
            I_XS[i].template tail<DODE::PV>(this->ode.PVars()) = x.template tail<DODE::PV>(this->ode.PVars());
          } else {
            I_XS[i].tail(this->ode.PVars()) = x.tail(this->ode.PVars());
          }
          for (int j = 0; j < Cardinals; j++) {
            I_XS[i].template segment<DODE::XV>(0, this->ode.XVars()) +=
                (Scalar(Coeffs::Cardinal_XInterp_Weights[i][j])
                     * C_XS[j].template segment<DODE::XV>(0, this->ode.XVars())
                 + (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) * C_DXS[j]);
            I_XS[i].template segment<DODE::UV>(this->ode.XtVars(), this->ode.UVars()) +=
                Scalar(Coeffs::Cardinal_UPoly_Weights[i][j])
                * C_XS[j].template segment<DODE::UV>(this->ode.XtVars(), this->ode.UVars());

            fx.template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
                (Scalar(Coeffs::Cardinal_XDef_Weights[i][j])
                     * C_XS[j].template segment<DODE::XV>(0, this->ode.XVars())
                 + (Coeffs::Cardinal_DXDef_Weights[i][j] * h) * C_DXS[j]);
          }
          this->ode.compute(I_XS[i], I_DXS[i]);
          fx.template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
              (h * Coeffs::Interior_DXDef_Weights[i] * I_DXS[i]);
        }
      };


      const int crit_size = this->IRows();

      using XType = ODEInput<Scalar>;
      using FXType = ODEOutput<Scalar>;

      const int irowsode = this->ode.IRows();
      const int orowsode = this->ode.ORows();

      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  ArrayOfTempSpecs<XType, Cardinals>(irowsode, 1),
                                  ArrayOfTempSpecs<FXType, Cardinals>(orowsode, 1),
                                  ArrayOfTempSpecs<XType, Interiors>(irowsode, 1),
                                  ArrayOfTempSpecs<FXType, Interiors>(orowsode, 1));
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();


      auto Impl = [&](auto& C_XS,
                      auto& C_DXS,
                      auto& C_JDXS,
                      auto& I_XS,
                      auto& I_DXS,
                      auto& I_JDXS,
                      auto& DI_DCS) {
        for (int i = 0; i < Cardinals; i++) {
          C_XS[i].template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
              x.template segment<DODE::XtUV>(i * this->ode.XtUVars(), this->ode.XtUVars());
          if constexpr (DODE::PV >= 0) {
            C_XS[i].template tail<DODE::PV>(this->ode.PVars()) = x.template tail<DODE::PV>(this->ode.PVars());
          } else {
            C_XS[i].tail(this->ode.PVars()) = x.tail(this->ode.PVars());
          }

          this->ode.compute_jacobian(C_XS[i], C_DXS[i], C_JDXS[i]);
        }

        Scalar h = C_XS.back()[this->ode.TVar()] - C_XS[0][this->ode.TVar()];

        for (int i = 0; i < Interiors; i++) {

          I_XS[i][this->ode.TVar()] = C_XS[0][this->ode.TVar()] + h * Coeffs::InteriorSpacings[i];
          if constexpr (DODE::PV >= 0) {
            I_XS[i].template tail<DODE::PV>(this->ode.PVars()) = x.template tail<DODE::PV>(this->ode.PVars());
          } else {
            I_XS[i].tail(this->ode.PVars()) = x.tail(this->ode.PVars());
          }

          if (i > 0) {
            DI_DCS.setZero();
          }


          DI_DCS(this->ode.TVar(), this->ode.TVar()) = Scalar(1.0 - Coeffs::InteriorSpacings[i]);
          DI_DCS(this->ode.TVar(), this->ode.XtUVars() * (Cardinals - 1) + this->ode.TVar()) =
              Coeffs::InteriorSpacings[i];
          DI_DCS
              .template block<DODE::PV, DODE::PV>(
                  this->ode.XtUVars(), Cardinals * this->ode.XtUVars(), this->ode.PVars(), this->ode.PVars())
              .diagonal()
              .setConstant(Scalar(1.0));

          for (int j = 0; j < Cardinals; j++) {
            ////////////////////////////// Sum up Cardinal Interpolation
            /// Terms/////////////////////////////////////////
            I_XS[i].template segment<DODE::XV>(0, this->ode.XVars()) +=
                (Scalar(Coeffs::Cardinal_XInterp_Weights[i][j])
                     * C_XS[j].template segment<DODE::XV>(0, this->ode.XVars())
                 + (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) * C_DXS[j]);

            DI_DCS
                .template block<DODE::XV, DODE::XV>(
                    0, j * this->ode.XtUVars(), this->ode.XVars(), this->ode.XVars())
                .diagonal()
                .setConstant(Scalar(Coeffs::Cardinal_XInterp_Weights[i][j]));

            DI_DCS.template block<DODE::XV, DODE::XtUV>(
                0, j * this->ode.XtUVars(), this->ode.XVars(), this->ode.XtUVars()) +=
                (Coeffs::Cardinal_DXInterp_Weights[i][j] * h)
                * C_JDXS[j].template leftCols<DODE::XtUV>(this->ode.XtUVars());

            DI_DCS.template block<DODE::XV, DODE::PV>(
                0, Cardinals * this->ode.XtUVars(), this->ode.XVars(), this->ode.PVars()) +=
                (Coeffs::Cardinal_DXInterp_Weights[i][j] * h)
                * C_JDXS[j].template rightCols<DODE::PV>(this->ode.PVars());

            DI_DCS.col(this->ode.TVar()).template segment<DODE::XV>(0, this->ode.XVars()) -=
                Scalar(Coeffs::Cardinal_DXInterp_Weights[i][j]) * C_DXS[j];
            DI_DCS.col(this->ode.XtUVars() * (Cardinals - 1) + this->ode.TVar())
                .template segment<DODE::XV>(0, this->ode.XVars()) +=
                Scalar(Coeffs::Cardinal_DXInterp_Weights[i][j]) * C_DXS[j];

            I_XS[i].template segment<DODE::UV>(this->ode.XtVars(), this->ode.UVars()) +=
                Scalar(Coeffs::Cardinal_UPoly_Weights[i][j])
                * C_XS[j].template segment<DODE::UV>(this->ode.XtVars(), this->ode.UVars());

            DI_DCS
                .template block<DODE::UV, DODE::UV>(this->ode.XtVars(),
                                                    j * this->ode.XtUVars() + this->ode.XtVars(),
                                                    this->ode.UVars(),
                                                    this->ode.UVars())
                .diagonal()
                .setConstant(Scalar(Coeffs::Cardinal_UPoly_Weights[i][j]));

            ////////////////////////////// Sum up Cardinal Output
            /// Terms/////////////////////////////////////////
            fx.template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
                (Scalar(Coeffs::Cardinal_XDef_Weights[i][j])
                     * C_XS[j].template head<DODE::XV>(this->ode.XVars())
                 + (Coeffs::Cardinal_DXDef_Weights[i][j] * h) * C_DXS[j]);

            jx.template block<DODE::XV, DODE::XV>(
                  i * this->ode.XVars(), j * this->ode.XtUVars(), this->ode.XVars(), this->ode.XVars())
                .diagonal()
                .setConstant(Scalar(Coeffs::Cardinal_XDef_Weights[i][j]));

            jx.template block<DODE::XV, DODE::XtUV>(
                i * this->ode.XVars(), j * this->ode.XtUVars(), this->ode.XVars(), this->ode.XtUVars()) +=
                (Coeffs::Cardinal_DXDef_Weights[i][j] * h)
                * C_JDXS[j].template leftCols<DODE::XtUV>(this->ode.XtUVars());

            jx.template block<DODE::XV, DODE::PV>(i * this->ode.XVars(),
                                                  Cardinals * this->ode.XtUVars(),
                                                  this->ode.XVars(),
                                                  this->ode.PVars()) +=
                (Coeffs::Cardinal_DXDef_Weights[i][j] * h)
                * C_JDXS[j].template rightCols<DODE::PV>(this->ode.PVars());

            jx.col(this->ode.TVar()).template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) -=
                Scalar(Coeffs::Cardinal_DXDef_Weights[i][j]) * C_DXS[j];
            jx.col(this->ode.XtUVars() * (Cardinals - 1) + this->ode.TVar())
                .template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
                Scalar(Coeffs::Cardinal_DXDef_Weights[i][j]) * C_DXS[j];
          }

          this->ode.compute_jacobian(I_XS[i], I_DXS[i], I_JDXS[i]);

          fx.template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
              (h * Coeffs::Interior_DXDef_Weights[i] * I_DXS[i]);
          jx.template middleRows<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
              ((h * Coeffs::Interior_DXDef_Weights[i]) * I_JDXS[i]) * (DI_DCS);

          jx.col(this->ode.TVar()).template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) -=
              Scalar(Coeffs::Interior_DXDef_Weights[i]) * I_DXS[i];
          jx.col(this->ode.XtUVars() * (Cardinals - 1) + this->ode.TVar())
              .template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
              Scalar(Coeffs::Interior_DXDef_Weights[i]) * I_DXS[i];
        }
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      };


      const int crit_size = this->IRows();

      using XType = ODEInput<Scalar>;
      using FXType = ODEOutput<Scalar>;
      using JXType = ODEJacobian<Scalar>;

      using DIType = Eigen::Matrix<Scalar, DODE::IRC, Base::IRC>;

      const int irowsode = this->ode.IRows();
      const int orowsode = this->ode.ORows();

      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  ArrayOfTempSpecs<XType, Cardinals>(irowsode, 1),
                                  ArrayOfTempSpecs<FXType, Cardinals>(orowsode, 1),
                                  ArrayOfTempSpecs<JXType, Cardinals>(orowsode, irowsode),
                                  ArrayOfTempSpecs<XType, Interiors>(irowsode, 1),
                                  ArrayOfTempSpecs<FXType, Interiors>(orowsode, 1),
                                  ArrayOfTempSpecs<JXType, Interiors>(orowsode, irowsode),
                                  TempSpec<DIType>(irowsode, this->IRows()));
    }


    template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType>& x,
        Eigen::MatrixBase<OutType> const& fx_,
        Eigen::MatrixBase<JacType> const& jx_,
        Eigen::MatrixBase<AdjGradType> const& adjgrad_,
        Eigen::MatrixBase<AdjHessType> const& adjhess_,
        const Eigen::MatrixBase<AdjVarType>& adjvars) const {
      typedef typename InType::Scalar Scalar;

      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      Eigen::MatrixBase<AdjHessType>& adjhess = adjhess_.const_cast_derived();


      auto Impl = [&](auto& C_XS,
                      auto& C_DXS,
                      auto& C_JDXS,
                      auto& C_AGXS,  // Not an array
                      auto& C_AVS,
                      auto& C_HDXS,  // Not an array
                      auto& I_XS,
                      auto& I_DXS,
                      auto& I_JDXS,
                      auto& I_AGXS,
                      auto& I_AVS,
                      auto& I_HDXS,
                      auto& DI_DCS,
                      auto& HTpar) {
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < Cardinals; i++) {


          C_XS[i].template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
              x.template segment<DODE::XtUV>(i * this->ode.XtUVars(), this->ode.XtUVars());
          if constexpr (DODE::PV >= 0) {
            C_XS[i].template tail<DODE::PV>(this->ode.PVars()) = x.template tail<DODE::PV>(this->ode.PVars());
          } else {
            C_XS[i].tail(this->ode.PVars()) = x.tail(this->ode.PVars());
          }

          this->ode.compute(C_XS[i], C_DXS[i]);
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        Scalar h = C_XS.back()[this->ode.TVar()] - C_XS[0][this->ode.TVar()];

        for (int i = 0; i < Interiors; i++) {

          I_XS[i][this->ode.TVar()] = C_XS[0][this->ode.TVar()] + h * Coeffs::InteriorSpacings[i];
          if constexpr (DODE::PV >= 0) {
            I_XS[i].template tail<DODE::PV>(this->ode.PVars()) = x.template tail<DODE::PV>(this->ode.PVars());
          } else {
            I_XS[i].tail(this->ode.PVars()) = x.tail(this->ode.PVars());
          }


          I_AVS[i] = adjvars.template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars());

          for (int j = 0; j < Cardinals; j++) {
            ////////////////////////////// Sum up Cardinal Interpolation
            /// Terms/////////////////////////////////////////
            I_XS[i].template segment<DODE::XV>(0, this->ode.XVars()) +=
                (Scalar(Coeffs::Cardinal_XInterp_Weights[i][j])
                     * C_XS[j].template segment<DODE::XV>(0, this->ode.XVars())
                 + (Coeffs::Cardinal_DXInterp_Weights[i][j] * h) * C_DXS[j]);

            I_XS[i].template segment<DODE::UV>(this->ode.XtVars(), this->ode.UVars()) +=
                Scalar(Coeffs::Cardinal_UPoly_Weights[i][j])
                * C_XS[j].template segment<DODE::UV>(this->ode.XtVars(), this->ode.UVars());
          }

          this->ode.compute_jacobian_adjointgradient_adjointhessian(
              I_XS[i], I_DXS[i], I_JDXS[i], I_AGXS[i], I_HDXS[i], I_AVS[i]);

          for (int j = 0; j < Cardinals; j++) {
            Scalar scale =
                Scalar(Coeffs::Interior_DXDef_Weights[i] * Coeffs::Cardinal_DXInterp_Weights[i][j]);
            C_AVS[j] += I_AGXS[i].template head<DODE::XV>(this->ode.XVars()) * (scale * h * h);
            C_AVS[j] += I_AVS[i] * (Coeffs::Cardinal_DXDef_Weights[i][j] * h);
          }
        }

        for (int j = 0; j < Cardinals; j++) {

          if (j > 0) {
            C_AGXS.setZero();
            C_HDXS.setZero();
          }
          this->ode.compute_jacobian_adjointgradient_adjointhessian(
              C_XS[j], C_DXS[j], C_JDXS[j], C_AGXS, C_HDXS, C_AVS[j]);

          adjhess.template block<DODE::XtUV, DODE::XtUV>(
              j * this->ode.XtUVars(), j * this->ode.XtUVars(), this->ode.XtUVars(), this->ode.XtUVars()) +=
              C_HDXS.template block<DODE::XtUV, DODE::XtUV>(0, 0, this->ode.XtUVars(), this->ode.XtUVars());
          adjhess.template block<DODE::XtUV, DODE::PV>(j * this->ode.XtUVars(),
                                                       Cardinals * this->ode.XtUVars(),
                                                       this->ode.XtUVars(),
                                                       this->ode.PVars()) +=
              C_HDXS.template block<DODE::XtUV, DODE::PV>(
                  0, this->ode.XtUVars(), this->ode.XtUVars(), this->ode.PVars());
          adjhess.template block<DODE::PV, DODE::XtUV>(Cardinals * this->ode.XtUVars(),
                                                       j * this->ode.XtUVars(),
                                                       this->ode.PVars(),
                                                       this->ode.XtUVars()) +=
              C_HDXS.template block<DODE::PV, DODE::XtUV>(
                  this->ode.XtUVars(), 0, this->ode.PVars(), this->ode.XtUVars());
          adjhess.template bottomRightCorner<DODE::PV, DODE::PV>(this->ode.PVars(), this->ode.PVars()) +=
              C_HDXS.template bottomRightCorner<DODE::PV, DODE::PV>(this->ode.PVars(), this->ode.PVars());
          HTpar.template segment<DODE::XtUV>(j * this->ode.XtUVars(), this->ode.XtUVars()) +=
              C_AGXS.template segment<DODE::XtUV>(0, this->ode.XtUVars()) * (1.0 / h);

          if constexpr (DODE::PV >= 0) {
            HTpar.template tail<DODE::PV>(this->ode.PVars()) +=
                C_AGXS.template tail<DODE::PV>(this->ode.PVars()) * (1.0 / h);
          } else {
            HTpar.tail(this->ode.PVars()) += C_AGXS.tail(this->ode.PVars()) * (1.0 / h);
          }
        }

        for (int i = 0; i < Interiors; i++) {
          if (i > 0)
            DI_DCS.setZero();
          DI_DCS(this->ode.TVar(), this->ode.TVar()) = Scalar(1.0 - Coeffs::InteriorSpacings[i]);
          DI_DCS(this->ode.TVar(), this->ode.XtUVars() * (Cardinals - 1) + this->ode.TVar()) =
              Scalar(Coeffs::InteriorSpacings[i]);
          DI_DCS
              .template block<DODE::PV, DODE::PV>(
                  this->ode.XtUVars(), Cardinals * this->ode.XtUVars(), this->ode.PVars(), this->ode.PVars())
              .diagonal()
              .setConstant(Scalar(1.0));

          for (int j = 0; j < Cardinals; j++) {
            ////////////////////////////// Sum up Cardinal Interpolation
            /// Terms/////////////////////////////////////////

            DI_DCS
                .template block<DODE::XV, DODE::XV>(
                    0, j * this->ode.XtUVars(), this->ode.XVars(), this->ode.XVars())
                .diagonal()
                .setConstant(Scalar(Coeffs::Cardinal_XInterp_Weights[i][j]));

            DI_DCS.template block<DODE::XV, DODE::XtUV>(
                0, j * this->ode.XtUVars(), this->ode.XVars(), this->ode.XtUVars()) +=
                (Coeffs::Cardinal_DXInterp_Weights[i][j] * h)
                * C_JDXS[j].template leftCols<DODE::XtUV>(this->ode.XtUVars());

            DI_DCS.template block<DODE::XV, DODE::PV>(
                0, Cardinals * this->ode.XtUVars(), this->ode.XVars(), this->ode.PVars()) +=
                (Coeffs::Cardinal_DXInterp_Weights[i][j] * h)
                * C_JDXS[j].template rightCols<DODE::PV>(this->ode.PVars());

            DI_DCS.col(this->ode.TVar()).template head<DODE::XV>(this->ode.XVars()) -=
                Scalar(Coeffs::Cardinal_DXInterp_Weights[i][j]) * C_DXS[j];
            DI_DCS.col(this->ode.XtUVars() * (Cardinals - 1) + this->ode.TVar())
                .template head<DODE::XV>(this->ode.XVars()) +=
                Scalar(Coeffs::Cardinal_DXInterp_Weights[i][j]) * C_DXS[j];

            DI_DCS
                .template block<DODE::UV, DODE::UV>(this->ode.XtVars(),
                                                    j * this->ode.XtUVars() + this->ode.XtVars(),
                                                    this->ode.UVars(),
                                                    this->ode.UVars())
                .diagonal()
                .setConstant(Scalar(Coeffs::Cardinal_UPoly_Weights[i][j]));

            ////////////////////////////// Sum up Cardinal Output
            /// Terms/////////////////////////////////////////
            fx.template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
                (Scalar(Coeffs::Cardinal_XDef_Weights[i][j])
                     * C_XS[j].template head<DODE::XV>(this->ode.XVars())
                 + (Coeffs::Cardinal_DXDef_Weights[i][j] * h) * C_DXS[j]);

            jx.template block<DODE::XV, DODE::XV>(
                  i * this->ode.XVars(), j * this->ode.XtUVars(), this->ode.XVars(), this->ode.XVars())
                .diagonal()
                .setConstant(Scalar(Coeffs::Cardinal_XDef_Weights[i][j]));

            jx.template block<DODE::XV, DODE::XtUV>(
                i * this->ode.XVars(), j * this->ode.XtUVars(), this->ode.XVars(), this->ode.XtUVars()) +=
                (Coeffs::Cardinal_DXDef_Weights[i][j] * h)
                * C_JDXS[j].template leftCols<DODE::XtUV>(this->ode.XtUVars());

            jx.template block<DODE::XV, DODE::PV>(i * this->ode.XVars(),
                                                  Cardinals * this->ode.XtUVars(),
                                                  this->ode.XVars(),
                                                  this->ode.PVars()) +=
                (Coeffs::Cardinal_DXDef_Weights[i][j] * h)
                * C_JDXS[j].template rightCols<DODE::PV>(this->ode.PVars());

            jx.col(this->ode.TVar()).template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) -=
                Scalar(Coeffs::Cardinal_DXDef_Weights[i][j]) * C_DXS[j];
            jx.col(this->ode.XtUVars() * (Cardinals - 1) + this->ode.TVar())
                .template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
                Scalar(Coeffs::Cardinal_DXDef_Weights[i][j]) * C_DXS[j];
          }

          fx.template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
              (h * Coeffs::Interior_DXDef_Weights[i] * I_DXS[i]);
          jx.template middleRows<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()).noalias() +=
              ((h * Coeffs::Interior_DXDef_Weights[i]) * I_JDXS[i]) * (DI_DCS);

          jx.col(this->ode.TVar()).template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) -=
              Scalar(Coeffs::Interior_DXDef_Weights[i]) * I_DXS[i];
          jx.col(this->ode.XtUVars() * (Cardinals - 1) + this->ode.TVar())
              .template segment<DODE::XV>(i * this->ode.XVars(), this->ode.XVars()) +=
              Scalar(Coeffs::Interior_DXDef_Weights[i]) * I_DXS[i];

          adjhess.noalias() +=
              DI_DCS.transpose() * (I_HDXS[i] * (h * Coeffs::Interior_DXDef_Weights[i])) * DI_DCS;
          HTpar.noalias() +=
              ((I_AGXS[i].transpose() * Scalar(Coeffs::Interior_DXDef_Weights[i])) * DI_DCS).transpose();
        }

        adjhess.col(this->ode.TVar()) -= HTpar;
        adjhess.col(this->ode.TVar() + this->ode.XtUVars() * (Cardinals - 1)) += HTpar;
        adjhess.row(this->ode.TVar()) -= HTpar;
        adjhess.row(this->ode.TVar() + this->ode.XtUVars() * (Cardinals - 1)) += HTpar;
        adjgrad.noalias() = (adjvars.transpose() * jx).transpose();
        // QED
      };


      const int crit_size = this->IRows();

      using XType = ODEInput<Scalar>;
      using FXType = ODEOutput<Scalar>;
      using JXType = ODEJacobian<Scalar>;
      using AGXType = ODEInput<Scalar>;
      using AVType = ODEOutput<Scalar>;
      using HType = ODEHessian<Scalar>;


      using DIType = Eigen::Matrix<Scalar, DODE::IRC, Base::IRC>;
      using HTParType = Input<Scalar>;

      const int irowsode = this->ode.IRows();
      const int orowsode = this->ode.ORows();

      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  ArrayOfTempSpecs<XType, Cardinals>(irowsode, 1),
                                  ArrayOfTempSpecs<FXType, Cardinals>(orowsode, 1),
                                  ArrayOfTempSpecs<JXType, Cardinals>(orowsode, irowsode),
                                  TempSpec<AGXType>(irowsode, 1),
                                  ArrayOfTempSpecs<AVType, Cardinals>(orowsode, 1),
                                  TempSpec<HType>(irowsode, irowsode),

                                  ArrayOfTempSpecs<XType, Interiors>(irowsode, 1),
                                  ArrayOfTempSpecs<FXType, Interiors>(orowsode, 1),
                                  ArrayOfTempSpecs<JXType, Interiors>(orowsode, irowsode),
                                  ArrayOfTempSpecs<AGXType, Interiors>(irowsode, 1),
                                  ArrayOfTempSpecs<AVType, Interiors>(orowsode, 1),
                                  ArrayOfTempSpecs<HType, Interiors>(irowsode, irowsode),

                                  TempSpec<DIType>(irowsode, this->IRows()),
                                  TempSpec<HTParType>(this->IRows(), 1));
    }
  };


}  // namespace ASSET
