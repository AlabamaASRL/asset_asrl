#pragma once

#include "LGLCoeffs.h"
#include "VectorFunctions/VectorFunction.h"

namespace ASSET {

  template<class Derived, int CSC, int USZ, int Order>
  struct LGLControlSplineSize : VectorFunction<Derived, (2 * CSC - 1) * (USZ + 1), USZ * Order> {
    using Base = VectorFunction<Derived, (2 * CSC - 1) * (USZ + 1), USZ * Order>;
    using Base::Base;

    static const int Usz = USZ;
    static const int tUsz = USZ + 1;

    const int tUsize() const {
      return tUsz;
    }
    const int Usize() const {
      return Usz;
    }

    template<class Scalar>
    using tUVec = Eigen::Matrix<Scalar, USZ + 1, 1>;
    template<class Scalar>
    using UVec = Eigen::Matrix<Scalar, USZ, 1>;

    void setUsize(int u) {
    }
  };

  template<class Derived, int CSC, int Order>
  struct LGLControlSplineSize<Derived, CSC, -1, Order> : VectorFunction<Derived, -1, -1> {
    using Base = VectorFunction<Derived, -1, -1>;
    using Base::Base;

    static const int Usz = -1;
    static const int tUsz = -1;

    int Uszd = -1;
    int tUszd = -1;

    int tUsize() const {
      return tUszd;
    }
    int Usize() const {
      return Uszd;
    }

    template<class Scalar>
    using tUVec = Eigen::Matrix<Scalar, -1, 1>;
    template<class Scalar>
    using UVec = Eigen::Matrix<Scalar, -1, 1>;

    void setUsize(int u) {
      this->Uszd = u;
      this->tUszd = u + 1;
      int irr = (2 * CSC - 1) * (u + 1);
      int orr = u * Order;
      this->setIORows(irr, orr);
    }
  };

  template<int CSC, int USZ, int Order = CSC - 2>
  struct LGLControlSpline : LGLControlSplineSize<LGLControlSpline<CSC, USZ, Order>, CSC, USZ, Order> {
    using SplineBase = LGLControlSplineSize<LGLControlSpline<CSC, USZ, Order>, CSC, USZ, Order>;

    template<class Scalar>
    using tUVec = typename SplineBase::template tUVec<Scalar>;
    template<class Scalar>
    using UVec = typename SplineBase::template UVec<Scalar>;

    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;
    using Coeffs = LGLCoeffs<CSC>;

    static const int tUNum = (2 * CSC - 1);
    static const int UeqNum = Order;

    LGLControlSpline() {
    }

    LGLControlSpline(int usize) {
      this->setUsize(usize);
    }

    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      STDarray<tUVec<Scalar>, tUNum> tUVs;

      for (int i = 0; i < tUNum; i++) {
        tUVs[i] = x.template segment<SplineBase::tUsz>(i * this->tUsize(), this->tUsize());
      }
      Scalar h0 = tUVs[CSC - 1][0] - tUVs[0][0];
      Scalar h1 = tUVs.back()[0] - tUVs[CSC - 1][0];

      for (int j = 0; j < UeqNum; j++) {  // j = 0-> 1st order continuity, j=1 2nd order continuity ...
        for (int i = 0; i < CSC; i++) {
          fx.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              (Coeffs::UOneSpline_Weights[j][i] * tUVs[i].tail(this->Usize()) / (pow(h0, Scalar(j + 1)))
               - Coeffs::UZeroSpline_Weights[j][i] * tUVs[i + CSC - 1].tail(this->Usize())
                     / (pow(h1, Scalar(j + 1))));
        }
      }
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      STDarray<tUVec<Scalar>, tUNum> tUVs;

      for (int i = 0; i < tUNum; i++) {
        tUVs[i] = x.template segment<SplineBase::tUsz>(i * this->tUsize(), this->tUsize());
      }

      Scalar h0 = tUVs[CSC - 1][0] - tUVs[0][0];
      Scalar h1 = tUVs.back()[0] - tUVs[CSC - 1][0];

      for (int j = 0; j < UeqNum; j++) {  // j = 0-> 1st order continuity, j=1 2nd order continuity ...
        Scalar h0pow = 1.0 / pow(h0, Scalar(j + 1));
        Scalar h1pow = 1.0 / pow(h1, Scalar(j + 1));
        Scalar hdt = Scalar(j + 1);

        for (int i = 0; i < CSC; i++) {
          fx.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              ((Coeffs::UOneSpline_Weights[j][i] * h0pow) * tUVs[i].tail(this->Usize())
               - (Coeffs::UZeroSpline_Weights[j][i] * h1pow) * tUVs[i + CSC - 1].tail(this->Usize()));

          jx.col(0).template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              (Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0) * tUVs[i].tail(this->Usize());

          jx.col((CSC - 1) * this->tUsize())
              .template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              -(Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0) * tUVs[i].tail(this->Usize())
              - (Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1)
                    * tUVs[i + CSC - 1].tail(this->Usize());

          jx.col((2 * CSC - 2) * this->tUsize())
              .template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              (Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1) * tUVs[i + CSC - 1].tail(this->Usize());

          jx.template block<SplineBase::Usz, SplineBase::Usz>(
                j * this->Usize(), i * this->tUsize() + 1, this->Usize(), this->Usize())
              .diagonal() += UVec<Scalar>::Constant(this->Usize(), Coeffs::UOneSpline_Weights[j][i] * h0pow);

          jx.template block<SplineBase::Usz, SplineBase::Usz>(
                j * this->Usize(), (i + CSC - 1) * this->tUsize() + 1, this->Usize(), this->Usize())
              .diagonal() +=
              UVec<Scalar>::Constant(this->Usize(), -Coeffs::UZeroSpline_Weights[j][i] * h1pow);
        }
      }
    }

    template<class InType, class OutType, class JacType, class AdjGradType, class AdjVarType>
    inline void compute_jacobian_adjointgradient(const Eigen::MatrixBase<InType>& x,
                                                 Eigen::MatrixBase<OutType> const& fx_,
                                                 Eigen::MatrixBase<JacType> const& jx_,
                                                 Eigen::MatrixBase<AdjGradType> const& adjgrad_,
                                                 const Eigen::MatrixBase<AdjVarType>& adjvars) const {
      this->compute_jacobian(x, fx_, jx_);
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      adjgrad = (adjvars.transpose() * jx_).transpose();
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

      STDarray<tUVec<Scalar>, tUNum> tUVs;

      for (int i = 0; i < tUNum; i++) {
        tUVs[i] = x.template segment<SplineBase::tUsz>(i * this->tUsize(), this->tUsize());
      }

      Scalar h0 = tUVs[CSC - 1][0] - tUVs[0][0];
      Scalar h1 = tUVs.back()[0] - tUVs[CSC - 1][0];

      for (int j = 0; j < UeqNum; j++) {  // j = 0-> 1st order continuity, j=1 2nd order continuity ...
        Scalar h0pow = 1.0 / pow(h0, Scalar(j + 1));
        Scalar h1pow = 1.0 / pow(h1, Scalar(j + 1));
        Scalar hdt = Scalar(j + 1);
        Scalar h2dt = Scalar(j + 2);
        Scalar OTH = 0.0;
        Scalar ZTH = 0.0;

        for (int i = 0; i < CSC; i++) {
          fx.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              ((Coeffs::UOneSpline_Weights[j][i] * h0pow) * tUVs[i].tail(this->Usize())
               - (Coeffs::UZeroSpline_Weights[j][i] * h1pow) * tUVs[i + CSC - 1].tail(this->Usize()));

          OTH += (Coeffs::UOneSpline_Weights[j][i] * hdt * h2dt * h0pow / (h0 * h0))
                 * tUVs[i]
                       .tail(this->Usize())
                       .dot(adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()));
          ZTH += (Coeffs::UZeroSpline_Weights[j][i] * hdt * h2dt * h1pow / (h1 * h1))
                 * tUVs[i + CSC - 1]
                       .tail(this->Usize())
                       .dot(adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()));

          jx.col(0).template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              (Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0) * tUVs[i].tail(this->Usize());

          jx.col((CSC - 1) * this->tUsize())
              .template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              -(Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0) * tUVs[i].tail(this->Usize())
              - (Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1)
                    * tUVs[i + CSC - 1].tail(this->Usize());

          jx.col((2 * CSC - 2) * this->tUsize())
              .template segment<SplineBase::Usz>(j * this->Usize(), this->Usize()) +=
              (Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1) * tUVs[i + CSC - 1].tail(this->Usize());

          jx.template block<SplineBase::Usz, SplineBase::Usz>(
                j * this->Usize(), i * this->tUsize() + 1, this->Usize(), this->Usize())
              .diagonal() += UVec<Scalar>::Constant(this->Usize(), Coeffs::UOneSpline_Weights[j][i] * h0pow);

          jx.template block<SplineBase::Usz, SplineBase::Usz>(
                j * this->Usize(), (i + CSC - 1) * this->tUsize() + 1, this->Usize(), this->Usize())
              .diagonal() +=
              UVec<Scalar>::Constant(this->Usize(), -Coeffs::UZeroSpline_Weights[j][i] * h1pow);
          //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
          adjhess.row(0).template segment<SplineBase::Usz>(i * this->tUsize() + 1, this->Usize()) +=
              adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                  .cwiseProduct(UVec<Scalar>::Constant(this->Usize(),
                                                       Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0))
                  .transpose();

          adjhess.row((CSC - 1) * this->tUsize())
              .template segment<SplineBase::Usz>(i * this->tUsize() + 1, this->Usize()) -=
              adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                  .cwiseProduct(UVec<Scalar>::Constant(this->Usize(),
                                                       Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0))
                  .transpose();

          adjhess.col(0).template segment<SplineBase::Usz>(i * this->tUsize() + 1, this->Usize()) +=
              adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                  .cwiseProduct(UVec<Scalar>::Constant(this->Usize(),
                                                       Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0));

          adjhess.col((CSC - 1) * this->tUsize())
              .template segment<SplineBase::Usz>(i * this->tUsize() + 1, this->Usize()) -=
              adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                  .cwiseProduct(UVec<Scalar>::Constant(this->Usize(),
                                                       Coeffs::UOneSpline_Weights[j][i] * hdt * h0pow / h0));
          /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
          adjhess.row((CSC - 1) * this->tUsize())
              .template segment<SplineBase::Usz>((i + CSC - 1) * this->tUsize() + 1, this->Usize()) -=
              adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                  .cwiseProduct(UVec<Scalar>::Constant(this->Usize(),
                                                       Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1))
                  .transpose();

          adjhess.row((2 * CSC - 2) * this->tUsize())
              .template segment<SplineBase::Usz>((i + CSC - 1) * this->tUsize() + 1, this->Usize()) +=
              adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                  .cwiseProduct(UVec<Scalar>::Constant(this->Usize(),
                                                       Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1))
                  .transpose();

          adjhess.col((CSC - 1) * this->tUsize())
              .template segment<SplineBase::Usz>((i + CSC - 1) * this->tUsize() + 1, this->Usize()) -=
              adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                  .cwiseProduct(UVec<Scalar>::Constant(this->Usize(),
                                                       Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1));

          adjhess.col((2 * CSC - 2) * this->tUsize())
              .template segment<SplineBase::Usz>((i + CSC - 1) * this->tUsize() + 1, this->Usize()) +=
              adjvars.template segment<SplineBase::Usz>(j * this->Usize(), this->Usize())
                  .cwiseProduct(UVec<Scalar>::Constant(this->Usize(),
                                                       Coeffs::UZeroSpline_Weights[j][i] * hdt * h1pow / h1));
        }

        adjhess(0, 0) += OTH;
        adjhess((CSC - 1) * this->tUsize(), (CSC - 1) * this->tUsize()) += OTH;
        adjhess(0, (CSC - 1) * this->tUsize()) += -OTH;
        adjhess((CSC - 1) * this->tUsize(), 0) += -OTH;

        adjhess((CSC - 1) * this->tUsize(), (CSC - 1) * this->tUsize()) -= ZTH;
        adjhess((2 * CSC - 2) * this->tUsize(), (2 * CSC - 2) * this->tUsize()) -= ZTH;
        adjhess((2 * CSC - 2) * this->tUsize(), (CSC - 1) * this->tUsize()) += ZTH;
        adjhess((CSC - 1) * this->tUsize(), (2 * CSC - 2) * this->tUsize()) += ZTH;
      }

      adjgrad = (adjvars.transpose() * jx).transpose();

      // QED
    }
  };


}  // namespace ASSET
