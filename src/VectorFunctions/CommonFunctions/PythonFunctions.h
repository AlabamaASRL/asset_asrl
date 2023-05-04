#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<int IRR, int ORR>
  struct PyVectorFunction : VectorFunction<PyVectorFunction<IRR, ORR>, IRR, ORR, FDiffFwd, FDiffFwd> {
    using Base = VectorFunction<PyVectorFunction<IRR, ORR>, IRR, ORR, FDiffFwd, FDiffFwd>;

    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;

    using FType = std::function<Output<double>(ConstEigenRef<Input<double>>, py::detail::args_proxy)>;
    bool threadSafe = false;
    FType pyfun;
    py::args pyargs;


    PyVectorFunction(int irr, int orr, const FType& f, Input<double> js, Input<double> hs, py::tuple p) {
      this->setIORows(irr, orr);
      this->pyfun = f;
      this->pyargs = p;

      if (irr != js.size() || irr != hs.size()) {
        throw std::invalid_argument("Incorrectly sized FD Step Sizes");
      }

      this->setJacFDSteps(js);
      this->setHessFDSteps(hs);
    }

    PyVectorFunction(int irr, int orr, const FType& f, double js, double hs, py::tuple p) {
      this->setIORows(irr, orr);
      this->pyfun = f;
      this->pyargs = p;
      this->setJacFDSteps(js);
      this->setHessFDSteps(hs);
    }

    PyVectorFunction(int irr, const FType& f, Input<double> js, Input<double> hs, py::tuple p)
        : PyVectorFunction(irr, ORR, f, js, hs, p) {
    }
    PyVectorFunction(int irr, const FType& f, double js, double hs, py::tuple p)
        : PyVectorFunction(irr, ORR, f, js, hs, p) {
    }

    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      fx = pyfun(x, *pyargs);
    }

    bool thread_safe() const {
      return threadSafe;
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<PyVectorFunction>(m, name);

      if constexpr (ORR != 1) {
        obj.def(py::init<int, int, const FType&, double, double, py::tuple>(),
                py::arg("IRows"),
                py::arg("ORows"),
                py::arg("Func"),
                py::arg("Jstepsize") = 1.0e-6,
                py::arg("Hstepsize") = 1.0e-4,
                py::arg("args") = py::tuple());
      } else {
        obj.def(py::init<int, const FType&, double, double, py::tuple>(),
                py::arg("IRows"),
                py::arg("Func"),
                py::arg("Jstepsize") = 1.0e-6,
                py::arg("Hstepsize") = 1.0e-4,
                py::arg("args") = py::tuple());
      }

      Base::DenseBaseBuild(obj);
    }
  };


  template<int IRR, int ORR>
  struct NumbaVectorFunction : VectorFunction<NumbaVectorFunction<IRR, ORR>, IRR, ORR, FDiffFwd, FDiffFwd> {
    using Base = VectorFunction<NumbaVectorFunction<IRR, ORR>, IRR, ORR, FDiffFwd, FDiffFwd>;

    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;

    using FType = long long unsigned int;
    typedef void (*FPtr)(double*, double*, int, int);

    bool threadSafe = false;
    bool dojac = false;
    FPtr fun;

    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      Input<double> xt = x;
      Output<double> fxt = fx;

      this->fun(xt.data(), fxt.data(), this->IRows(), this->ORows());
      fx = fxt;
    }
    NumbaVectorFunction(int irr, int orr, const FType& f, double js, double hs) {
      this->setIORows(irr, orr);
      this->fun = (FPtr) f;
      this->setJacFDSteps(js);
      this->setHessFDSteps(hs);
    }
    NumbaVectorFunction(int irr, int orr, const FType& f) : NumbaVectorFunction(irr, orr, f, 1.0e-6, 1.0e-6) {
    }
    NumbaVectorFunction(const FType& f) : NumbaVectorFunction(IRR, ORR, f) {
    }
    NumbaVectorFunction(const FType& f, double js, double hs) : NumbaVectorFunction(IRR, ORR, f, js, hs) {
    }
    NumbaVectorFunction(int irr, const FType& f, double js, double hs)
        : NumbaVectorFunction(irr, ORR, f, js, hs) {
    }
    NumbaVectorFunction(int irr, const FType& f) : NumbaVectorFunction(irr, ORR, f) {
    }

    bool thread_safe() const {
      return threadSafe;
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<NumbaVectorFunction>(m, name);
      obj.def(py::init<int, int, const FType&, double, double>());
      obj.def(py::init<int, int, const FType&>());

      if constexpr (ORR == 1) {
        obj.def(py::init<int, const FType&, double, double>());
        obj.def(py::init<int, const FType&>());
      }

      if constexpr (IRR > 0 && ORR > 0) {
        obj.def(py::init<const FType&, double, double>());
        obj.def(py::init<const FType&>());
      }
      obj.def_readwrite("thread_safe", &NumbaVectorFunction::thread_safe);
      Base::DenseBaseBuild(obj);
    }
  };

}  // namespace ASSET
