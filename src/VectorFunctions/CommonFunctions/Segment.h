#pragma once

#include "Conditional.h"
#include "Constant.h"
#include "DetectDiagonal.h"
#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, int IR, int OR, int ST>
  struct Segment_Impl;

  template<int IR_OR>
  struct Arguments : Segment_Impl<Arguments<IR_OR>, IR_OR, IR_OR, 0> {
    using Base = Segment_Impl<Arguments<IR_OR>, IR_OR, IR_OR, 0>;
    using Base::Base;
    Arguments(int iror) : Base(iror, iror, 0) {
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<Arguments<IR_OR>>(m, name);
      obj.def(py::init<int>());
      obj.def("Constant", [](const Arguments<IR_OR>& a, Eigen::VectorXd v) {
        return GenericFunction<-1, -1>(Constant<-1, -1>(a.IRows(), v));
      });
      obj.def("Constant", [](const Arguments<IR_OR>& a, double v) {
        Eigen::Matrix<double, 1, 1> vv;
        vv[0] = v;
        return GenericFunction<-1, 1>(Constant<-1, 1>(a.IRows(), vv));
      });
      Base::DenseBaseBuild(obj);
      Base::SegBuild(obj);
    }
  };
  template<int IR, int OR, int ST>
  struct Segment : Segment_Impl<Segment<IR, OR, ST>, IR, OR, ST> {
    using Base = Segment_Impl<Segment<IR, OR, ST>, IR, OR, ST>;
    using Base::Base;

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<Segment<IR, OR, ST>>(m, name);
      obj.def(py::init<int, int, int>());
      Base::DenseBaseBuild(obj);
      Base::SegBuild(obj);
    }
  };

  template<class T>
  struct Is_Segment : std::false_type {};
  template<int IR, int OR, int ST>
  struct Is_Segment<Segment<IR, OR, ST>> : std::true_type {};

  template<class T>
  struct Is_Arguments : std::false_type {};
  template<int IR>
  struct Is_Arguments<Arguments<IR>> : std::true_type {};

  template<class T>
  struct Is_ScaledSegment : std::false_type {};
  template<int IR, int OR, int ST>
  struct Is_ScaledSegment<Scaled<Segment<IR, OR, ST>>> : std::true_type {};
  template<int IR, int OR, int ST, class VALUE>
  struct Is_ScaledSegment<StaticScaled<Segment<IR, OR, ST>, VALUE>> : std::true_type {};

  template<int ST>
  struct SegStartHolder {
    static const int SegStart = ST;
    void setSegStart(int st) {};
  };
  template<>
  struct SegStartHolder<-1> {
    int SegStart = -1;
    void setSegStart(int st) {
      this->SegStart = st;
    };
  };

  template<class Derived, int IR, int OR, int ST>
  struct Segment_Impl : VectorFunction<Derived, IR, OR>, SegStartHolder<ST> {
    using INPUT_DOMAIN = SingleDomain<IR, ST, OR>;
    using Base = VectorFunction<Derived, IR, OR>;
    // using SegStartHolder<ST>::SegStart;
    DENSE_FUNCTION_BASE_TYPES(Base);

    Segment_Impl() {
    }
    Segment_Impl(int irows, int orows, int start) {
      this->setIORows(irows, orows);
      this->setSegStart(start);
      DomainMatrix dmn(2, 1);
      dmn(0, 0) = start;
      dmn(1, 0) = orows;

      this->set_input_domain(irows, {dmn});

      if (start + orows > this->IRows() || start < 0) {
        throw std::invalid_argument("Segment/Element Index Out of Bounds");
      }
    }


    static const bool IsLinearFunction = true;
    static const bool IsVectorizable = true;

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = x.template segment<OR>(this->SegStart, this->ORows());
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      const int OROWS = this->ORows();
      Scalar ONE = Scalar(1.0);
      fx = x.template segment<OR>(this->SegStart, OROWS);
      jx.template middleCols<OR>(this->SegStart, OROWS).diagonal().setConstant(ONE);
    }

    template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_,
        ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_,
        ConstVectorBaseRef<AdjVarType> adjvars) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      //  MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      const int OROWS = this->ORows();
      Scalar ONE = Scalar(1.0);

      fx = x.template segment<OR>(this->SegStart, OROWS);
      jx.template middleCols<OR>(this->SegStart, OROWS).diagonal().setConstant(ONE);
      adjgrad.template segment<OR>(this->SegStart, OROWS) = adjvars;
    }

    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left,
                                       ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      typedef typename Target::Scalar Scalar;

      auto Impl = [&](auto& diag) {
        diag = right.derived().template middleCols<OR>(this->SegStart, this->ORows()).diagonal();

        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
          if constexpr (Is_EigenDiagonalMatrix<
                            typename std::remove_const_reference<decltype(left.derived())>::type>::value) {
            target.template middleCols<OR>(this->SegStart).diagonal() =
                left.derived().diagonal().cwiseProduct(diag);
          } else {
            if constexpr (Aliased)
              target.template middleCols<OR>(this->SegStart, this->ORows()) =
                  left.derived() * diag.asDiagonal();
            else
              target.template middleCols<OR>(this->SegStart, this->ORows()).noalias() =
                  left.derived() * diag.asDiagonal();
          }

        } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
          // target.template middleCols<OR>(this->SegStart).noalias() +=
          //     left.derived() * diag;

          if constexpr (Is_EigenDiagonalMatrix<
                            typename std::remove_const_reference<decltype(left.derived())>::type>::value) {
            target.template middleCols<OR>(this->SegStart, this->ORows()).diagonal() +=
                left.derived().diagonal().cwiseProduct(diag);
          } else {
            if constexpr (Aliased)
              target.template middleCols<OR>(this->SegStart, this->ORows()) +=
                  left.derived() * diag.asDiagonal();
            else
              target.template middleCols<OR>(this->SegStart, this->ORows()).noalias() +=
                  left.derived() * diag.asDiagonal();
          }

        } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {


          if constexpr (Is_EigenDiagonalMatrix<
                            typename std::remove_const_reference<decltype(left.derived())>::type>::value) {
            target.template middleCols<OR>(this->SegStart).diagonal() -=
                left.derived().diagonal().cwiseProduct(diag);
          } else {
            if constexpr (Aliased)
              target.template middleCols<OR>(this->SegStart, this->ORows()) -=
                  left.derived() * diag.asDiagonal();
            else
              target.template middleCols<OR>(this->SegStart, this->ORows()).noalias() -=
                  left.derived() * diag.asDiagonal();
          }

        } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
          target.template middleCols<OR>(this->SegStart, this->ORows()).noalias() =
              assign.value * left.derived() * diag.asDiagonal();
        } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
          target.template middleCols<OR>(this->SegStart, this->ORows()).noalias() +=
              assign.value * left.derived() * diag.asDiagonal();
        } else {
          std::cout << "right_jacobian_product has not been implemented for: " << this->name << std::endl
                    << std::endl;
        }
      };


      const int orows = this->ORows();
      MemoryManager::allocate_run(orows, Impl, TempSpec<Output<Scalar>>(orows, 1));
    }

    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                          ConstEigenBaseRef<Left> left,
                                          ConstEigenBaseRef<Right> right,
                                          Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      typedef typename Target::Scalar Scalar;

      Eigen::DiagonalMatrix<Scalar, OR> diag;
      diag.diagonal() = right.derived().template middleCols<OR>(this->SegStart, this->ORows()).diagonal();
      diag.diagonal() = diag.diagonal().cwiseProduct(diag.diagonal());

      if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
        target.template block<OR, OR>(this->SegStart, this->SegStart, this->ORows(), this->ORows())
            .noalias() = left.derived() * diag;
      } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
        if constexpr (Is_EigenDiagonalMatrix<
                          typename std::remove_const_reference<decltype(left.derived())>::type>::value) {
          target.template block<OR, OR>(this->SegStart, this->SegStart, this->ORows(), this->ORows())
              .diagonal() += left.derived().diagonal().cwiseProduct(diag.diagonal());
        } else {
          target.template block<OR, OR>(this->SegStart, this->SegStart, this->ORows(), this->ORows())
              .noalias() += left.derived() * diag;
        }

      } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
        target.template block<OR, OR>(this->SegStart, this->SegStart, this->ORows(), this->ORows())
            .noalias() -= left.derived() * diag;
      } else if constexpr (std::is_same<Assignment, ScaledDirectAssignment<Scalar>>::value) {
        target.template block<OR, OR>(this->SegStart, this->SegStart, this->ORows(), this->ORows())
            .noalias() = assign.value * left.derived() * diag;
      } else if constexpr (std::is_same<Assignment, ScaledPlusEqualsAssignment<Scalar>>::value) {
        target.template block<OR, OR>(this->SegStart, this->SegStart, this->ORows(), this->ORows())
            .noalias() += assign.value * left.derived() * diag;
      } else {
        std::cout << "symetric_jacobian_product has not been implemented for: " << this->name << std::endl
                  << std::endl;
      }
    }

    template<class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstEigenBaseRef<JacType> right,
                                    Assignment assign) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
        target.template middleCols<OR>(this->SegStart, this->ORows()).diagonal() =
            right.derived().template middleCols<OR>(this->SegStart, this->ORows()).diagonal();
      } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
        target.template middleCols<OR>(this->SegStart, this->ORows()).diagonal() +=
            right.derived().template middleCols<OR>(this->SegStart, this->ORows()).diagonal();
      } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
        target.template middleCols<OR>(this->SegStart, this->ORows()).diagonal() -=
            right.derived().template middleCols<OR>(this->SegStart, this->ORows()).diagonal();
      } else {
      }
    }
    template<class Target, class JacType, class Assignment>
    inline void accumulate_gradient(ConstMatrixBaseRef<Target> target_,
                                    ConstEigenBaseRef<JacType> right,
                                    Assignment assign) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
        target.template segment<OR>(this->SegStart, this->ORows()) =
            right.derived().template segment<OR>(this->SegStart, this->ORows());
      } else if constexpr (std::is_same<Assignment, PlusEqualsAssignment>::value) {
        target.template segment<OR>(this->SegStart, this->ORows()) +=
            right.derived().template segment<OR>(this->SegStart, this->ORows());
      } else if constexpr (std::is_same<Assignment, MinusEqualsAssignment>::value) {
        target.template segment<OR>(this->SegStart, this->ORows()) -=
            right.derived().template segment<OR>(this->SegStart, this->ORows());
      } else {
      }
    }
    template<class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      target.template middleCols<OR>(this->SegStart, this->ORows()).diagonal() *= s;
    }
    template<class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      target.template segment<OR>(this->SegStart, this->ORows()) *= s;
    }
    template<class Func, int FuncIRC>
    decltype(auto) rearged(const DenseFunctionBase<Func, FuncIRC, IR>& f) const {
      return Base::template EVALOP<Func>::make_nested(this->derived(), f.derived());
    }


    template<class PyClass>
    static void SegBuild(PyClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;


      Base::DoubleMathBuild(obj);
      Base::UnaryMathBuild(obj);
      Base::BinaryMathBuild(obj);
      Base::BinaryOperatorsBuild(obj);
      Base::FunctionIndexingBuild(obj);
      Base::ConditionalOperatorsBuild(obj);

      obj.def("tolist", [](const Derived& func) {
        using ELEM = Segment<-1, 1, -1>;
        std::vector<ELEM> elems;
        for (int i = 0; i < func.ORows(); i++) {
          elems.push_back(func.coeff(i));
        }
        return elems;
      });


      obj.def("tolist", [](const Derived& func, std::vector<int> coeffs) {
        using ELEM = Segment<-1, 1, -1>;
        std::vector<ELEM> elems;
        for (const auto& coeff: coeffs) {
          elems.push_back(func.coeff(coeff));
        }
        return elems;
      });

      obj.def("tolist", [](const Derived& func, std::vector<std::tuple<int, int>> seglist) {
        using ELEM = Segment<-1, 1, -1>;
        using SEG2 = Segment<-1, 2, -1>;
        using SEG3 = Segment<-1, 3, -1>;
        using SEG = Segment<-1, -1, -1>;

        std::vector<py::object> segs;
        for (const auto& seg: seglist) {

          int start = std::get<0>(seg);
          int size = std::get<1>(seg);
          py::object pyfun;
          if (size == 1) {
            auto f = func.coeff(start);
            pyfun = py::cast(f);
          } else if (size == 2) {
            auto f = func.template segment<2>(start);
            pyfun = py::cast(f);
          } else if (size == 3) {
            auto f = func.template segment<3>(start);
            pyfun = py::cast(f);
          } else {
            auto f = func.segment(start, size);
            pyfun = py::cast(f);
          }


          segs.push_back(pyfun);
        }
        return segs;
      });
    }
  };

}  // namespace ASSET
