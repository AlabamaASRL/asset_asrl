#pragma once

#include "DeepCopySpecs.h"
#include "DenseFunctionBase.h"
#include "DenseFunctionSpecs.h"
#include "PyDocString/VectorFunctions/VectorFunctionTypeErasure/GenericFunction_doc.h"
#include "SizingSpecs.h"
#include "SolverInterfaceSpecs.h"
#include "VectorFunctions/CommonFunctions/CommonFunctions.h"

namespace ASSET {

template <int IR, int OR>
struct GenericFunction;

template <int IR, int OR>
struct DeepCopySelector {
  using type = DeepCopySpecs<GenericFunction<IR, OR>, GenericFunction<-1, -1>,
                             ConstraintInterface>;
};
template <int IR>
struct DeepCopySelector<IR, 1> {
  using type = DeepCopySpecs<GenericFunction<IR, 1>, GenericFunction<-1, -1>,
                             ConstraintInterface, ObjectiveInterface>;
};
template <>
struct DeepCopySelector<-1, -1> {
  using type = DeepCopySpecs<GenericFunction<-1, -1>, ConstraintInterface>;
};

template <int IR, int OR>
struct GenericFunction : rubber_types::TypeErasure<
                             DenseFunctionSpec<IR, OR>, SizableSpec,
                             typename DeepCopySelector<IR, OR>::type,
                             typename SolverInterfaceSelector<IR, OR>::type> {
  using Base =
      rubber_types::TypeErasure<DenseFunctionSpec<IR, OR>, SizableSpec,
                                typename DeepCopySelector<IR, OR>::type,
                                typename SolverInterfaceSelector<IR, OR>::type>;

  template <class Scalar>
  using Output = Eigen::Matrix<Scalar, OR, 1>;
  template <class Scalar>
  using Input = Eigen::Matrix<Scalar, IR, 1>;
  template <class Scalar>
  using Gradient = Eigen::Matrix<Scalar, IR, 1>;
  template <class Scalar>
  using Jacobian = Eigen::Matrix<Scalar, OR, IR>;
  template <class Scalar>
  using Hessian = Eigen::Matrix<Scalar, IR, IR>;

  template <class Scalar>
  using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;
  template <class Scalar>
  using VectorBaseRef = Eigen::MatrixBase<Scalar>&;
  template <class Scalar>
  using ConstMatrixBaseRef = const Eigen::MatrixBase<Scalar>&;
  template <class Scalar>
  using MatrixBaseRef = Eigen::MatrixBase<Scalar>&;
  template <class Scalar>
  using ConstEigenBaseRef = const Eigen::EigenBase<Scalar>&;
  template <class Scalar>
  using EigenBaseRef = Eigen::EigenBase<Scalar>&;
  template <class Scalar>
  using ConstDiagonalBaseRef = const Eigen::DiagonalBase<Scalar>&;

  using RightJacTarget = Eigen::Ref<Eigen::Matrix<double, -1, IR>>;

  GenericFunction() {}
  template <class T>
  GenericFunction(const T& t) : Base(t) {
    cachedata();
  }
  GenericFunction(const GenericFunction<IR, OR>& obj) {
    // obj.deep_copy_into(*this);
    this->reset_container(obj.get_container());
    cachedata();
  }
  template <int IR1, int OR1>
  GenericFunction(const GenericFunction<IR1, OR1>& obj) {
    obj.deep_copy_into(*this);
    cachedata();
  }

  void cachedata() {
    this->irows = Base::IRows();
    this->orows = Base::ORows();
    this->islinear = Base::is_linear();
    this->SubDomains = Base::input_domain();
  }

  inline int IRows() const {
    if constexpr (IR < 0) {
      return this->irows;
    } else {
      return IR;
    }
  }
  inline int ORows() const {
    if constexpr (OR < 0) {
      return this->orows;
    } else {
      return OR;
    }
  }
  inline bool is_linear() const { return this->islinear; }

  template <class TargetTT, class LeftTT, class RightTT, class Assignment,
            bool Aliased>
  void right_jacobian_product(ConstMatrixBaseRef<TargetTT> target_,
                              ConstMatrixBaseRef<LeftTT> left,
                              ConstMatrixBaseRef<RightTT> right,
                              Assignment assign,
                              std::bool_constant<Aliased> aliased) const {
    typedef typename TargetTT::Scalar Scalar;

    if constexpr (std::is_same<Scalar, double>::value) {
      constexpr bool TargConv =
          std::is_convertible<decltype(target_.const_cast_derived()),
                              RightJacTarget>::value;

      if constexpr (TargConv) {
        Base::right_jacobian_product(target_, left, right, assign, aliased);
      } else {
        ASSET::right_jacobian_product_dynamic_impl(
            this->SubDomains, target_, left, right, assign, aliased);
      }
    } else {
      right_jacobian_product_dynamic_impl(this->SubDomains, target_, left,
                                          right, assign, aliased);
    }
  }

  template <class TargetTT, class LeftTT, class RightTT, class Assignment,
            bool Aliased>
  void right_jacobian_product(ConstMatrixBaseRef<TargetTT> target_,
                              ConstDiagonalBaseRef<LeftTT> left,
                              ConstMatrixBaseRef<RightTT> right,
                              Assignment assign,
                              std::bool_constant<Aliased> aliased) const {
    typedef typename TargetTT::Scalar Scalar;

    if constexpr (std::is_same<Scalar, double>::value) {
      constexpr bool TargConv =
          std::is_convertible<decltype(target_.const_cast_derived()),
                              RightJacTarget>::value;

      if constexpr (TargConv) {
        Base::right_jacobian_product(target_, left, right, assign, aliased);
      } else {
        ASSET::right_jacobian_product_dynamic_impl(
            this->SubDomains, target_, left, right, assign, aliased);
      }
    } else {
      ASSET::right_jacobian_product_dynamic_impl(this->SubDomains, target_,
                                                 left, right, assign, aliased);
    }
  }

  template <class Target, class Left, class Right, class Assignment,
            bool Aliased>
  inline void right_jacobian_domain_product(
      ConstMatrixBaseRef<Target> target_, ConstEigenBaseRef<Left> left,
      ConstEigenBaseRef<Right> right, Assignment assign,
      std::bool_constant<Aliased> aliased) const {
    ASSET::right_jacobian_product_dynamic_impl(this->SubDomains, target_, left,
                                               right, assign, aliased);
  }

  template <class Target, class Left, class Right, class Assignment,
            bool Aliased>
  inline void symetric_jacobian_product(
      ConstMatrixBaseRef<Target> target_, ConstEigenBaseRef<Left> left,
      ConstEigenBaseRef<Right> right, Assignment assign,
      std::bool_constant<Aliased> aliased) const {
    ASSET::symetric_jacobian_product_dynamic_impl(this->SubDomains, target_,
                                                  left, right, assign, aliased);
  }

  template <class TargetTT, class AdjHessTypeTT, class Assignment>
  void accumulate_hessian(ConstMatrixBaseRef<TargetTT> target_,
                          ConstMatrixBaseRef<AdjHessTypeTT> right,
                          Assignment assign) const {
    typedef typename TargetTT::Scalar Scalar;

    if constexpr (std::is_same<Scalar, double>::value) {
      if (!this->is_linear()) {
        if constexpr (OR > 0 && IR > 0) {
          Base::accumulate_hessian(target_, right, assign);
        } else {
          ASSET::accumulate_symetric_matrix_dynamic_domain_impl(
              this->SubDomains, target_, right, assign);
        }
      }
    } else {
        if (!this->is_linear()) ASSET::accumulate_symetric_matrix_dynamic_domain_impl(
          this->SubDomains, target_, right, assign);
    }
  }
  template <class TargetTT, class JacTypeTT, class Assignment>
  void accumulate_jacobian(ConstMatrixBaseRef<TargetTT> target_,
                           ConstMatrixBaseRef<JacTypeTT> right,
                           Assignment assign) const {
    typedef typename TargetTT::Scalar Scalar;

    if constexpr (std::is_same<Scalar, double>::value) {
      if (this->is_linear()) {
        Base::accumulate_jacobian(target_, right, assign);
      } else {
        ASSET::accumulate_matrix_dynamic_domain_impl(this->SubDomains, target_,
                                                     right, assign);
      }
    } else {
      ASSET::accumulate_matrix_dynamic_domain_impl(this->SubDomains, target_,
                                                   right, assign);
    }
  }
  template <class TargetTT, class AdjGradTypeTT, class Assignment>
  void accumulate_gradient(ConstMatrixBaseRef<TargetTT> target_,
                           ConstMatrixBaseRef<AdjGradTypeTT> right,
                           Assignment assign) const {
    ASSET::accumulate_vector_dynamic_domain_impl(this->SubDomains, target_,
                                                 right, assign);
  }

  template <class JacTypeTT, class Scalar>
  void scale_jacobian(ConstMatrixBaseRef<JacTypeTT> target_, Scalar s) const {
    if constexpr (std::is_same<Scalar, ASSET::DefaultSuperScalar>::value) {
      ASSET::scale_matrix_dynamic_domain_impl(this->SubDomains, target_, s);
    } else {
      Base::scale_jacobian(target_, s);
    }
  }
  template <class AdjGradTypeTT, class Scalar>
  void scale_gradient(ConstMatrixBaseRef<AdjGradTypeTT> target_,
                      Scalar s) const {
    if constexpr (std::is_same<Scalar, ASSET::DefaultSuperScalar>::value) {
      ASSET::scale_vector_dynamic_domain_impl(this->SubDomains, target_, s);
    } else {
      Base::scale_gradient(target_, s);
    }
  }
  template <class AdjHessTypeTT, class Scalar>
  void scale_hessian(ConstMatrixBaseRef<AdjHessTypeTT> target_,
                     Scalar s) const {
    if constexpr (std::is_same<Scalar, ASSET::DefaultSuperScalar>::value) {
      ASSET::scale_matrix_dynamic_domain_impl(this->SubDomains, target_, s);
    } else {
      Base::scale_hessian(target_, s);
    }
  }

  template <class T>
  static GenericFunction<IR, OR> PyCopy(const T& obj) {
    GenericFunction<IR, OR> a;
    obj.deep_copy_into(a);

    return a;
  }

  Output<double> compute_python(ConstEigenRef<Input<double>> x) const {
    Output<double> fx(this->ORows());
    fx.setZero();
    this->compute(x, fx);
    return fx;
  }
  Jacobian<double> jacobian_python(ConstEigenRef<Input<double>> x) const {
    Jacobian<double> jx(this->ORows(), this->IRows());
    jx.setZero();
    Output<double> fx(this->ORows());
    fx.setZero();
    this->compute_jacobian(x, fx, jx);
    return jx;
  }
  Gradient<double> adjointgradient_python(
      ConstEigenRef<Input<double>> x, ConstEigenRef<Output<double>> lm) const {
    Jacobian<double> jx(this->ORows(), this->IRows());
    jx.setZero();
    Output<double> fx(this->ORows());
    fx.setZero();
    this->compute_jacobian(x, fx, jx);

    Gradient<double> ax = jx.transpose() * lm;
    return ax;
  }
  Hessian<double> adjointhessian_python(
      ConstEigenRef<Input<double>> x, ConstEigenRef<Output<double>> lm) const {
    Hessian<double> hx(this->IRows(), this->IRows());
    hx.setZero();
    Jacobian<double> jx(this->ORows(), this->IRows());
    jx.setZero();
    Output<double> fx(this->ORows());
    fx.setZero();

    Gradient<double> ax(this->IRows());
    ax.setZero();

    this->compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, ax, hx,
                                                          lm);

    return hx;
  }

  using Derived = GenericFunction<IR, OR>;

  template <class Func>
  using EVALOP = NestedFunctionSelector<Derived, Func>;
  template <class Func>
  using FWDOP = NestedFunctionSelector<Func, Derived>;
  template <int SZ, int ST>
  using SEGMENTOP = NestedFunctionSelector<Segment<OR, SZ, ST>, Derived>;

  template <class Func, int FuncIRC, int FuncORC>
  auto eval(const DenseFunctionBase<Func, FuncIRC, FuncORC>& f) const {
    return EVALOP<Func>::make_nested(*this, f.derived());
  }
  template <int FuncIRC, int FuncORC>
  auto eval(const GenericFunction<FuncIRC, FuncORC>& f) const {
    return EVALOP<GenericFunction<FuncIRC, FuncORC>>::make_nested(*this, f);
  }

  void SuperTest(const Input<double>& xt, int n) {
    Input<double> x = xt;
    Output<double> fx(this->ORows());
    Jacobian<double> jx(this->ORows(), this->IRows());
    Gradient<double> gx(this->IRows());
    Hessian<double> hx(this->IRows(), this->IRows());
    Output<double> l(this->ORows());

    Input<ASSET::DefaultSuperScalar> X =
        xt.template cast<ASSET::DefaultSuperScalar>();
    Output<ASSET::DefaultSuperScalar> FX(this->ORows());
    Jacobian<ASSET::DefaultSuperScalar> JX(this->ORows(), this->IRows());
    Gradient<ASSET::DefaultSuperScalar> GX(this->IRows());
    Hessian<ASSET::DefaultSuperScalar> HX(this->IRows(), this->IRows());
    Output<ASSET::DefaultSuperScalar> L(this->ORows());

    Eigen::BenchTimer t1;
    Eigen::BenchTimer t2;

    double dummy = 0;
    t1.start();
    for (int i = 0; i < n; i++) {
      x[0] += 1.0 / double(n + 1);
      this->compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx,
                                                            l);
      dummy += fx[0] + jx(0, 0) + hx(0, 0);
    }
    t1.stop();

    std::cout << dummy << std::endl;

    ASSET::DefaultSuperScalar dummy2(0);
    t2.start();
    int n2 = n / ASSET::DefaultSuperScalar::SizeAtCompileTime;
    for (int i = 0; i < n2; i++) {
      X[0] += ASSET::DefaultSuperScalar((1.0 / double(n + 1)));
      this->compute_jacobian_adjointgradient_adjointhessian(X, FX, JX, GX, HX,
                                                            L);
      dummy2 += FX[0] + JX(0, 0) + HX(0, 0);
    }
    t2.stop();

    std::cout << dummy2.transpose() << std::endl << std::endl;

    std::cout << "Scalar     : " << t1.total() * 1000.0 << std::endl;
    std::cout << "SuperScalar: " << t2.total() * 1000.0 << std::endl;
  }

  template <class PYClass>
  static void GenericBuild(PYClass& obj) {
    using namespace doc;
    obj.def(py::init<const GenericFunction<IR, OR>&>());
    obj.def("IRows", &Derived::IRows, GenericFunction_IRows);
    obj.def("ORows", &Derived::ORows, GenericFunction_ORows);
    obj.def("name", &Derived::name, GenericFunction_name);
    obj.def("compute", &Derived::compute_python, GenericFunction_compute);
    obj.def("jacobian", &Derived::jacobian_python, GenericFunction_jacobian);
    obj.def("adjointgradient", &Derived::adjointgradient_python,
            GenericFunction_adjointgradient);
    obj.def("adjointhessian", &Derived::adjointhessian_python,
            GenericFunction_adjointhessian);
    obj.def("input_domain", &Derived::input_domain,
            GenericFunction_input_domain);
    obj.def("is_linear", &Derived::is_linear, GenericFunction_is_linear);
    obj.def("SuperTest", &Derived::SuperTest, GenericFunction_SuperTest);

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    // obj.def("eval", &Derived::eval<-1,-1>);

    obj.def(
        "eval",
        [](const GenericFunction<IR, OR>& a, const GenericFunction<-1, -1>& b) {
          return GenericFunction<IR, OR>(a.eval(b));
        },
        GenericFunction_eval1);

    obj.def(
        "eval",
        [](const GenericFunction<IR, OR>& a, Segment<-1, -1, -1> seg) {
          return GenericFunction<IR, OR>(
              NestedFunction<GenericFunction<IR, OR>, Segment<-1, -1, -1>>(
                  a, seg));
        },
        GenericFunction_eval2);
    obj.def(
        "eval",
        [](const GenericFunction<IR, OR>& a, Segment<-1, 1, -1> seg) {
          return GenericFunction<IR, OR>(
              NestedFunction<GenericFunction<IR, OR>, Segment<-1, 1, -1>>(a,
                                                                          seg));
        },
        GenericFunction_eval3);
    obj.def(
        "eval",
        [](const GenericFunction<IR, OR>& a, Segment<-1, 2, -1> seg) {
          return GenericFunction<IR, OR>(
              NestedFunction<GenericFunction<IR, OR>, Segment<-1, 2, -1>>(a,
                                                                          seg));
        },
        GenericFunction_eval4);
    obj.def(
        "eval",
        [](const GenericFunction<IR, OR>& a, Segment<-1, 3, -1> seg) {
          return GenericFunction<IR, OR>(
              NestedFunction<GenericFunction<IR, OR>, Segment<-1, 3, -1>>(a,
                                                                          seg));
        },
        GenericFunction_eval5);
    obj.def(
        "eval",
        [](const GenericFunction<IR, OR>& a, int ir, Eigen::VectorXi v) {
          return GenericFunction<IR, OR>(
              ParsedInput<GenericFunction<IR, OR>, -1, OR>(a, v, ir));
        },
        GenericFunction_eval6);

    obj.def(
        "padded_lower",
        [](const GenericFunction<IR, OR>& a, int lpad) {
          return GenericFunction<IR, -1>(
              PaddedOutput<GenericFunction<IR, OR>, -1, -1>(a, 0, lpad));
        },
        GenericFunction_padded_lower);
    obj.def(
        "padded_upper",
        [](const GenericFunction<IR, OR>& a, int upad) {
          return GenericFunction<IR, -1>(
              PaddedOutput<GenericFunction<IR, OR>, -1, -1>(a, upad, 0));
        },
        GenericFunction_padded_upper);

    obj.def("rpt", &Derived::rpt, GenericFunction_rpt);

    if constexpr (OR == -1 && IR == -1) {
      obj.def(py::init(&Derived::PyCopy<GenericFunction<IR, 1>>));
    }

    obj.def(
        "__mul__",
        [](const GenericFunction<IR, OR>& a, double b) {
          return GenericFunction<IR, OR>(a * b);
        },
        py::is_operator());

    obj.def(
        "__neg__",
        [](const GenericFunction<IR, OR>& a) {
          return GenericFunction<IR, OR>(a * (-1.0));
        },
        py::is_operator());

    obj.def(
        "__rmul__",
        [](const GenericFunction<IR, OR>& a, double b) {
          return GenericFunction<IR, OR>(a * b);
        },
        py::is_operator());
    obj.def(
        "__truediv__",
        [](const GenericFunction<IR, OR>& a, double b) {
          return GenericFunction<IR, OR>(a * (1.0 / b));
        },
        py::is_operator());
    obj.def(
        "__add__",
        [](const GenericFunction<IR, OR>& a, Output<double> b) {
          return GenericFunction<IR, OR>((a + b));
        },
        py::is_operator());
    obj.def(
        "__radd__",
        [](const GenericFunction<IR, OR>& a, Output<double> b) {
          return GenericFunction<IR, OR>((a + b));
        },
        py::is_operator());
    obj.def(
        "__sub__",
        [](const GenericFunction<IR, OR>& a, Output<double> b) {
          return GenericFunction<IR, OR>((a - b));
        },
        py::is_operator());

    obj.def(
        "__add__",
        [](const GenericFunction<IR, OR>& a, const GenericFunction<IR, OR>& b) {
          return GenericFunction<IR, OR>(
              TwoFunctionSum<GenericFunction<IR, OR>, GenericFunction<IR, OR>>(
                  a, b));
        },
        py::is_operator());
    obj.def(
        "__sub__",
        [](const GenericFunction<IR, OR>& a, const GenericFunction<IR, OR>& b) {
          return GenericFunction<IR, OR>(
              FunctionDifference<GenericFunction<IR, OR>,
                                 GenericFunction<IR, OR>>(a, b));
        },
        py::is_operator());

    obj.def(
        "__mul__",
        [](const GenericFunction<IR, OR>& a, const GenericFunction<IR, 1>& b) {
          return GenericFunction<IR, OR>(
              VectorScalarFunctionProduct<GenericFunction<IR, OR>,
                                          GenericFunction<IR, 1>>(a, b));
        },
        py::is_operator());


    obj.def(
        "sum", [](const GenericFunction<IR, OR> & e) { 
            if constexpr (OR == 1) return e;
            else return GenS(CwiseSum< GenericFunction<IR, OR>>(e));
        });

    obj.def(
        "vf", [](const GenericFunction<IR, OR>& e) {
            return Gen(e);
        });

    if constexpr (OR != 1) {

      obj.def("row_scaled", [](const GenericFunction<IR, OR>& a, Eigen::VectorXd scales) {
          return Gen(RowScaled<Gen>(a, scales));
       });

      obj.def(
          "__rmul__",
          [](const GenericFunction<IR, OR>& a,
             const GenericFunction<IR, 1>& b) {
            return GenericFunction<IR, OR>(
                VectorScalarFunctionProduct<GenericFunction<IR, OR>,
                                            GenericFunction<IR, 1>>(a, b));
          },
          py::is_operator());

      obj.def(
          "norm",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return GenS(
                  NestedFunction<Norm<sz>, Gen>(Norm<sz>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_norm);
      obj.def(
          "squared_norm",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return GenS(NestedFunction<SquaredNorm<sz>, Gen>(
                  SquaredNorm<sz>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_squared_norm);
      obj.def(
          "cubed_norm",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return GenS(NestedFunction<NormPower<sz, 3>, Gen>(
                  NormPower<sz, 3>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_cubed_norm);
      obj.def(
          "inverse_norm",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return GenS(NestedFunction<InverseNorm<sz>, Gen>(
                  InverseNorm<sz>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_inverse_norm);
      obj.def(
          "inverse_squared_norm",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return GenS(NestedFunction<InverseSquaredNorm<sz>, Gen>(
                  InverseSquaredNorm<sz>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_inverse_squared_norm);
      obj.def(
          "inverse_cubed_norm",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return GenS(NestedFunction<InverseNormPower<sz, 3>, Gen>(
                  InverseNormPower<sz, 3>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_inverse_cubed_norm);
      obj.def(
          "inverse_four_norm",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return GenS(NestedFunction<InverseNormPower<sz, 4>, Gen>(
                  InverseNormPower<sz, 4>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_inverse_four_norm);

      obj.def(
          "normalized",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return Gen(NestedFunction<Normalized<sz>, Gen>(
                  Normalized<sz>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_normalized);
      obj.def(
          "normalized_power2",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return Gen(NestedFunction<NormalizedPower<sz, 2>, Gen>(
                  NormalizedPower<sz, 2>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_normalized_power2);
      obj.def(
          "normalized_power3",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return Gen(NestedFunction<NormalizedPower<sz, 3>, Gen>(
                  NormalizedPower<sz, 3>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_normalized_power3);
      obj.def(
          "normalized_power4",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return Gen(NestedFunction<NormalizedPower<sz, 4>, Gen>(
                  NormalizedPower<sz, 4>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_normalized_power4);
      obj.def(
          "normalized_power5",
          [](const Gen& fun) {
            int orr = fun.ORows();
            auto Lam = [](const Gen& funt, auto size) {
              constexpr int sz = size.value;
              return Gen(NestedFunction<NormalizedPower<sz, 5>, Gen>(
                  NormalizedPower<sz, 5>(funt.ORows()), funt));
            };
            if (orr == 2)
              return Lam(fun, std::integral_constant<int, 2>());
            else if (orr == 3)
              return Lam(fun, std::integral_constant<int, 3>());
            return Lam(fun, std::integral_constant<int, -1>());
          },
          GenericFunction_normalized_power5);
      obj.def(
          "cross",
          [](const Gen& seg1, const Gen& seg2) {
            return Gen(FunctionCrossProduct<Gen, Gen>(seg1, seg2));
          },
          GenericFunction_cross1);
      obj.def(
          "cross",
          [](const Gen& seg1, const SEG& seg2) {
            return Gen(FunctionCrossProduct<Gen, SEG>(seg1, seg2));
          },
          GenericFunction_cross2);
      obj.def(
          "cross",
          [](const Gen& seg1, const SEG3& seg2) {
            return Gen(FunctionCrossProduct<Gen, SEG3>(seg1, seg2));
          },
          GenericFunction_cross3);
    }
    if constexpr (OR == 1) {
      obj.def(
          "squared", [](const GenS& e) { return GenS(CwiseSquare<GenS>(e)); },
          GenericFunction_squared);
      obj.def(
          "sqrt", [](const GenS& e) { return GenS(CwiseSqrt<GenS>(e)); },
          GenericFunction_sqrt);
      obj.def(
          "exp", [](const GenS& e) { return GenS(CwiseExp<GenS>(e)); },
          GenericFunction_exp);
      obj.def(
          "sin", [](const GenS& e) { return GenS(CwiseSin<GenS>(e)); },
          GenericFunction_sin);
      obj.def(
          "cos", [](const GenS& e) { return GenS(CwiseCos<GenS>(e)); },
          GenericFunction_cos);
      obj.def(
          "inverse", [](const GenS& e) { return GenS(CwiseInverse<GenS>(e)); },
          GenericFunction_inverse);

      obj.def(
          "__pow__",
          [](const GenS& a, double b) {
              return GenS(CwisePow<GenS>(a, b));
          },
          py::is_operator());
      obj.def(
          "__pow__",
          [](const GenS& a, int b) {
              if (b == 1) return GenS(a);
              else if (b == 2) return GenS(CwiseSquare<GenS>(a));
              return GenS(CwisePow<GenS>(a, b));
          },
          py::is_operator());

      obj.def(
          "__add__",
          [](const GenericFunction<IR, OR>& a, double bs) {
            Output<double> b;
            b[0] = bs;
            return GenericFunction<IR, OR>((a + b));
          },
          py::is_operator());
      obj.def(
          "__radd__",
          [](const GenericFunction<IR, OR>& a, double bs) {
            Output<double> b;
            b[0] = bs;
            return GenericFunction<IR, OR>((a + b));
          },
          py::is_operator());
      obj.def(
          "__sub__",
          [](const GenericFunction<IR, OR>& a, double bs) {
            Output<double> b;
            b[0] = bs;
            return GenericFunction<IR, OR>((a - b));
          },
          py::is_operator());
      obj.def(
          "__rsub__",
          [](const GenericFunction<IR, OR>& a, double bs) {
            Output<double> b;
            b[0] = bs;
            return GenericFunction<IR, OR>((b + -1.0 * a));
          },
          py::is_operator());
      obj.def(
          "__rtruediv__",
          [](const GenericFunction<IR, OR>& a, double bs) {
            return GenericFunction<IR, OR>(
                (bs * CwiseInverse<GenericFunction<IR, 1>>(a)));
          },
          py::is_operator());

      obj.def(
          "sf", [](const GenericFunction<IR, OR>& e) {
              return GenS(e);
          });
      ///////////////////////////////////////////////////////
      using GenCon = GenericConditional<IR>;

      obj.def(
          "__lt__",
          [](const Derived& a, const GenS& b) {
              return GenCon(ConditionalStatement<Derived, GenS>(a, ConditionalFlags::LessThanFlag, b));
          },
          py::is_operator());

      obj.def(
          "__gt__",
          [](const Derived& a, const GenS& b) {
              return GenCon(ConditionalStatement<Derived, GenS>(a, ConditionalFlags::GreaterThanFlag, b));
          },
          py::is_operator());

      obj.def(
          "__lt__",
          [](const Derived& a, double b) {
              Vector1<double> tmp;
              tmp[0] = b;
              Constant<IR, 1> bfunc(a.IRows(), tmp);
              return GenCon(ConditionalStatement<Derived, Constant<IR, 1>>(a, ConditionalFlags::LessThanFlag, bfunc));
          },
          py::is_operator());
      obj.def(
          "__gt__",
          [](const Derived& a, double b) {
              Vector1<double> tmp;
              tmp[0] = b;
              Constant<IR, 1> bfunc(a.IRows(), tmp);
              return GenCon(ConditionalStatement<Derived, Constant<IR, 1>>(a, ConditionalFlags::GreaterThanFlag, bfunc));
          },
          py::is_operator());
      obj.def(
          "__rlt__",
          [](const Derived& a, double b) {
              Vector1<double> tmp;
              tmp[0] = b;
              Constant<IR, 1> bfunc(a.IRows(), tmp);
              return GenCon(ConditionalStatement<Derived, Constant<IR, 1>>(a, ConditionalFlags::GreaterThanFlag, bfunc));
          },
          py::is_operator());
      obj.def(
          "__rgt__",
          [](const Derived& a, double b) {
              Vector1<double> tmp;
              tmp[0] = b;
              Constant<IR, 1> bfunc(a.IRows(), tmp);
              return GenCon(ConditionalStatement<Derived, Constant<IR, 1>>(a, ConditionalFlags::LessThanFlag, bfunc));
          },
          py::is_operator());


      ////////////////////////////////////////////////////////



    }

    obj.def(
        "__truediv__",
        [](const GenericFunction<IR, OR>& a, const Segment<-1, 1, -1>& b) {
          return GenericFunction<IR, OR>(
              VectorScalarFunctionProduct<GenericFunction<IR, OR>,
                                          CwiseInverse<Segment<-1, 1, -1>>>(
                  a, b.cwiseInverse()));
        },
        py::is_operator());
    obj.def(
        "__truediv__",
        [](const GenericFunction<IR, OR>& a, const GenericFunction<IR, 1>& b) {
          return GenericFunction<IR, OR>(
              VectorScalarFunctionProduct<GenericFunction<IR, OR>,
                                          CwiseInverse<GenericFunction<IR, 1>>>(
                  a, CwiseInverse<GenericFunction<IR, 1>>(b)));
        },
        py::is_operator());
  }

  static void OperatorBuild(py::module& m) {}
  DomainMatrix SubDomains;


  /// <summary>
  ///  I put
  /// </summary>
  /// <typeparam name="PyClass"></typeparam>
  /// <param name="obj"></param>
  template <class PyClass>
  static void SegBuild2(PyClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;

      using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

      obj.def("segment", [](const Derived& a, int start, int size) {
          return a.segment(start, size);
          });
      obj.def("head",
          [](const Derived& a, int size) { return a.segment(0, size); });
      obj.def("tail", [](const Derived& a, int size) {
          return a.segment(a.ORows() - size, size);
          });

      if constexpr (OR < 0 || OR > 2) {
          obj.def("segment_2", [](const Derived& a, int start) {
              return a.template segment<2>(start);
              });
          obj.def("head_2",
              [](const Derived& a) { return a.template segment<2>(0); });
          obj.def("tail_2", [](const Derived& a) {
              return a.template segment<2>(a.ORows() - 2);
              });

          obj.def("segment2", [](const Derived& a, int start) {
              return a.template segment<2>(start);
              });
          obj.def("head2",
              [](const Derived& a) { return a.template segment<2>(0); });
          obj.def("tail2", [](const Derived& a) {
              return a.template segment<2>(a.ORows() - 2);
              });

      }
      if constexpr (OR < 0 || OR > 3) {
          obj.def("segment_3", [](const Derived& a, int start) {
              return a.template segment<3>(start);
              });
          obj.def("head_3",
              [](const Derived& a) { return a.template segment<3>(0); });
          obj.def("tail_3", [](const Derived& a) {
              return a.template segment<3>(a.ORows() - 3);
              });

          obj.def("segment3", [](const Derived& a, int start) {
              return a.template segment<3>(start);
              });
          obj.def("head3",
              [](const Derived& a) { return a.template segment<3>(0); });
          obj.def("tail3", [](const Derived& a) {
              return a.template segment<3>(a.ORows() - 3);
              });

      }

      obj.def("coeff", [](const Derived& a, int elem) { return a.coeff(elem); });
      obj.def(
          "__getitem__", [](const Derived& a, int elem) { return a.coeff(elem); },
          py::is_operator());
      obj.def(
          "__getitem__", [](const Derived& a, py::slice slice) {
              size_t start, stop, step, slicelength;
              if (!slice.compute(a.ORows(), &start, &stop, &step, &slicelength))
                  throw py::error_already_set();

              if (step != 1) {
                  throw std::invalid_argument("Non continous slices not supported");
              }
              int start_ = start;
              int size_ = stop - start;
              return Segment<-1, -1, -1>(a.IRows(), size_, start_);
          },
          py::is_operator());

      if constexpr (OR != 1) {
          obj.def("norm", [](const Derived& a) { return GenS(a.norm()); });
          obj.def("squared_norm", [](const Derived& a) { return GenS(a.squared_norm()); });

          obj.def("inverse_norm",
              [](const Derived& a) { return GenS(a.inverse_norm()); });
          obj.def("inverse_squared_norm",
              [](const Derived& a) { return GenS(a.inverse_squared_norm()); });
          obj.def("inverse_cubed_norm", [](const Derived& a) {
              return GenS(a.template inverse_norm_power<3>());
              });
          obj.def("inverse_four_norm", [](const Derived& a) {
              return GenS(a.template inverse_norm_power<4>());
              });

          obj.def("normalized", [](const Derived& a) { return Gen(a.normalized()); });
          obj.def("normalized_power2", [](const Derived& a) {
              return Gen(a.template normalized_power<2>());
              });
          obj.def("normalized_power3", [](const Derived& a) {
              return Gen(a.template normalized_power<3>());
              });
          obj.def("normalized_power4", [](const Derived& a) {
              return Gen(a.template normalized_power<4>());
              });
          obj.def("normalized_power5", [](const Derived& a) {
              return Gen(a.template normalized_power<5>());
              });

          obj.def("normalized_power3", [](const Derived& a, Output<double> b) {
              return Gen((a + b).template normalized_power<3>());
              });
          obj.def("normalized_power3",
              [](const Derived& a, Output<double> b, double s) {
                  return Gen(((a + b).template normalized_power<3>()) * s);
              });
      }



      obj.def("dot",
          [](const Derived& a, const Derived& b) { return GenS(a.dot(b)); });

      obj.def(
          "__add__",
          [](const Derived& a, const Derived& b) {
              return BinGen(TwoFunctionSum<Derived, Derived>(a, b));
          },
          py::is_operator());
      obj.def(
          "__sub__",
          [](const Derived& a, const Derived& b) {
              return BinGen(FunctionDifference<Derived, Derived>(a, b));
          },
          py::is_operator());
      if constexpr (OR != 1) {
          obj.def(
              "__sub__",
              [](const Derived& a, const Gen& b) {
                  return Gen(FunctionDifference<Derived, Gen>(a, b));
              },
              py::is_operator());

          /*obj.def("__rsub__",
              [](const Derived& a, const Gen& b) {
              return Gen(FunctionDifference<Gen, Derived>(b, a));
          },
              py::is_operator());*/

      }
      else {
          obj.def(
              "__sub__",
              [](const Derived& a, const GenS& b) {
                  return GenS(FunctionDifference<Derived, GenS>(a, b));
              },
              py::is_operator());

          /* obj.def("__rsub__",
               [](const Derived& a, const GenS& b) {
                   return Gen(FunctionDifference<GenS, Derived>(b, a));
               },
               py::is_operator());*/
      }
      obj.def(
          "__add__",
          [](const Derived& a, Output<double> b) { return BinGen(a + b); },
          py::is_operator());
      obj.def(
          "__radd__",
          [](const Derived& a, Output<double> b) { return BinGen(a + b); },
          py::is_operator());
      obj.def(
          "__sub__",
          [](const Derived& a, Output<double> b) { return BinGen(a - b); },
          py::is_operator());
      obj.def(
          "__rsub__",
          [](const Derived& a, Output<double> b) { return BinGen(b - a); },
          py::is_operator());

      obj.def(
          "__mul__", [](const Derived& a, double b) { return BinGen(a * b); },
          py::is_operator());
      obj.def(
          "__neg__", [](const Derived& a) { return BinGen(a * (-1.0)); },
          py::is_operator());
      obj.def(
          "__rmul__", [](const Derived& a, double b) { return BinGen(a * b); },
          py::is_operator());
      obj.def(
          "__truediv__",
          [](const Derived& a, double b) { return BinGen(a * (1.0 / b)); },
          py::is_operator());
      obj.def(
          "__truediv__",
          [](const Derived& a, const Segment<-1, 1, -1>& b) {
              return BinGen(a / b);
          },
          py::is_operator());

      obj.def(
          "__mul__",
          [](const Derived& a, const Segment<-1, 1, -1>& b) {
              return BinGen(a * b);
          },
          py::is_operator());
      obj.def(
          "__mul__",
          [](const Derived& a, const GenS& b) {
              return BinGen(VectorScalarFunctionProduct<Derived, GenS>(a, b));
          },
          py::is_operator());

      if constexpr (OR != 1) {
          obj.def(
              "__rmul__",
              [](const Derived& a, const Segment<-1, 1, -1>& b) {
                  return BinGen(a * b);
              },
              py::is_operator());

          obj.def(
              "__rmul__",
              [](const Derived& a, const GenS& b) {
                  return BinGen(VectorScalarFunctionProduct<Derived, GenS>(a, b));
              },
              py::is_operator());

          obj.def("row_scaled", [](const Derived& a, Eigen::VectorXd scales) {
              return Gen(RowScaled<Derived>(a, scales));
              });

      }

      obj.def(
          "sum",
          [](const Derived& a) {
              return GenS(a.sum());
          },
          py::is_operator());

      if constexpr (OR == 1) {
          obj.def("sin", [](const Derived& a) { return GenS(a.Sin()); });
          obj.def("cos", [](const Derived& a) { return GenS(a.Cos()); });
          obj.def("tan", [](const Derived& a) { return GenS(a.Tan()); });
          obj.def("sqrt", [](const Derived& a) { return GenS(a.Sqrt()); });
          obj.def("exp", [](const Derived& a) { return GenS(a.Exp()); });
          obj.def("squared", [](const Derived& a) { return GenS(a.Square()); });

          /////////////////////////////////////////////////////
          using GenCon = GenericConditional<IR>;
          obj.def(
              "__lt__",
              [](const Derived& a, double b) {
                  Vector1<double> tmp;
                  tmp[0] = b;
                  Constant<IR, 1> bfunc(a.IRows(), tmp);
                  return GenCon(ConditionalStatement<Derived, Constant<IR, 1>>(a, ConditionalFlags::LessThanFlag, bfunc));
              },
              py::is_operator());
          obj.def(
              "__gt__",
              [](const Derived& a, double b) {
                  Vector1<double> tmp;
                  tmp[0] = b;
                  Constant<IR, 1> bfunc(a.IRows(), tmp);
                  return GenCon(ConditionalStatement<Derived, Constant<IR, 1>>(a, ConditionalFlags::GreaterThanFlag, bfunc));
              },
              py::is_operator());


          obj.def(
              "__rlt__",
              [](const Derived& a, double b) {
                  Vector1<double> tmp;
                  tmp[0] = b;
                  Constant<IR, 1> bfunc(a.IRows(), tmp);
                  return GenCon(ConditionalStatement<Derived, Constant<IR, 1>>(a, ConditionalFlags::GreaterThanFlag, bfunc));
              },
              py::is_operator());
          obj.def(
              "__rgt__",
              [](const Derived& a, double b) {
                  Vector1<double> tmp;
                  tmp[0] = b;
                  Constant<IR, 1> bfunc(a.IRows(), tmp);
                  return GenCon(ConditionalStatement<Derived, Constant<IR, 1>>(a, ConditionalFlags::LessThanFlag, bfunc));
              },
              py::is_operator());


          obj.def(
              "__lt__",
              [](const Derived& a, const Derived& b) {
                  return GenCon(ConditionalStatement<Derived, Derived>(a, ConditionalFlags::LessThanFlag, b));
              },
              py::is_operator());

          obj.def(
              "__gt__",
              [](const Derived& a, const Derived& b) {
                  return GenCon(ConditionalStatement<Derived, Derived>(a, ConditionalFlags::GreaterThanFlag, b));
              },
              py::is_operator());


          obj.def(
              "__lt__",
              [](const Derived& a, const GenS& b) {
                  return GenCon(ConditionalStatement<Derived, GenS>(a, ConditionalFlags::LessThanFlag, b));
              },
              py::is_operator());

          obj.def(
              "__gt__",
              [](const Derived& a, const GenS& b) {
                  return GenCon(ConditionalStatement<Derived, GenS>(a, ConditionalFlags::GreaterThanFlag, b));
              },
              py::is_operator());






          //////////////////////////////////////////////
          obj.def(
              "__pow__",
              [](const Derived& a, double b) {
                  return GenS(CwisePow<Derived>(a, b));
              },
              py::is_operator());
          obj.def(
              "__pow__",
              [](const Derived& a, int b) {
                  if (b == 1) return GenS(a);
                  else if (b == 2) return GenS(a.Square());
                  return GenS(CwisePow<Derived>(a, b));
              },
              py::is_operator());

          obj.def(
              "__add__",
              [](const Derived& a, double b) { return BinGen(a + b); },
              py::is_operator());
          obj.def(
              "__radd__",
              [](const Derived& a, double b) { return BinGen(a + b); },
              py::is_operator());
          obj.def(
              "__sub__",
              [](const Derived& a, double b) { return BinGen(a - b); },
              py::is_operator());
          obj.def(
              "__rsub__",
              [](const Derived& a, double b) { return BinGen(b - a); },
              py::is_operator());
          obj.def(
              "__rtruediv__",
              [](const Derived& a, double b) { return BinGen(b / a); },
              py::is_operator());

          // obj.def("__mul__", [](const Derived& a, const Derived& b) {
          //  return GenS(a.dot(b));
          // });
      }
  }






 private:
  int irows = 0;
  int orows = 0;
  bool islinear = false;
};


/*

*/
}  // namespace ASSET
