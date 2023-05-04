#pragma once
#include "VectorFunctions/CommonFunctions/CommonFunctions.h"
#include "pch.h"

namespace ASSET {

  template<int IR>
  struct ConditionalSpec {

    template<class Scalar>
    using Input = Eigen::Matrix<Scalar, IR, 1>;
    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;
    using InType = Eigen::Ref<const Input<double>>;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Concept {  // abstract base class for model.
      virtual ~Concept() = default;
      // Your (internal) interface goes here.
      virtual std::string name() const = 0;
      virtual int IRows() const = 0;
      virtual bool compute(ConstVectorBaseRef<InType> x) const = 0;
    };
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class Holder>
    struct Model : public Holder, public virtual Concept {
      using Holder::Holder;
      // Pass through to encapsulated value.
      virtual std::string name() const override {
        return rubber_types::model_get(this).name();
      }
      virtual int IRows() const override {
        return rubber_types::model_get(*this).IRows();
      }

      virtual bool compute(ConstVectorBaseRef<InType> x) const override {
        return rubber_types::model_get(this).compute(x);
      }
    };
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class Container>
    struct ExternalInterface : public Container {
      using Container::Container;
      static const bool IsConditional = true;
      static const bool IRC = IR;

      // Define the external interface. Should match encapsulated type.
      std::string name() const {
        return rubber_types::interface_get(this).name();
      }
      int IRows() const {
        return rubber_types::interface_get(*this).IRows();
      }

      template<class InTypeT>
      bool compute(ConstVectorBaseRef<InTypeT> x) const {
        InType xt(x.derived());
        return rubber_types::interface_get(this).compute(xt);
      }
    };
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };


  template<int IR>
  struct GenericConditional : rubber_types::TypeErasure<ConditionalSpec<IR>> {
    using Base = rubber_types::TypeErasure<ConditionalSpec<IR>>;

    GenericConditional() {
    }
    template<class T>
    GenericConditional(const T& t) : Base(t) {
    }
    GenericConditional(const GenericConditional<IR>& obj) {
      this->reset_container(obj.get_container());
    }


    static void ConditionalBuild(py::module& m) {

      using GenCon = GenericConditional<IR>;

      auto obj = py::class_<GenCon>(m, "Conditional");

      obj.def("compute", [](const GenCon& a, ConstEigenRef<Eigen::VectorXd> x) { return a.compute(x); });

      obj.def(
          "__and__",
          [](const GenCon& a, const GenCon& b) {
            return GenCon(ConditionalStatement<GenCon, GenCon>(a, ConditionalFlags::ANDFlag, b));
          },
          py::is_operator());

      obj.def(
          "__or__",
          [](const GenCon& a, const GenCon& b) {
            return GenCon(ConditionalStatement<GenCon, GenCon>(a, ConditionalFlags::ORFlag, b));
          },
          py::is_operator());

      IfElseBuild(obj);
    }

    template<class PYCLASS>
    static void IfElseBuild(PYCLASS& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;
      using ELEM = Segment<-1, 1, -1>;
      using GenCon = GenericConditional<-1>;

      obj.def("ifelse", [](const GenCon& test, const GenS& tf, const GenS& ff) {
        return GenS(IfElseFunction {test, tf, ff});
      });

      obj.def("ifelse", [](const GenCon& test, double tfv, const GenS& ff) {
        Vector1<double> v;
        v[0] = tfv;
        Constant<-1, 1> tf(test.IRows(), v);
        return GenS(IfElseFunction {test, tf, ff});
      });
      obj.def("ifelse", [](const GenCon& test, const GenS& tf, double ffv) {
        Vector1<double> v;
        v[0] = ffv;
        Constant<-1, 1> ff(test.IRows(), v);
        return GenS(IfElseFunction {test, tf, ff});
      });
      obj.def("ifelse", [](const GenCon& test, double tfv, double ffv) {
        Vector1<double> v1;
        v1[0] = tfv;
        Constant<-1, 1> tf(test.IRows(), v1);
        Vector1<double> v2;
        v2[0] = ffv;
        Constant<-1, 1> ff(test.IRows(), v2);
        return GenS(IfElseFunction {test, tf, ff});
      });


      obj.def("ifelse", [](const GenCon& test, const Gen& tf, const Gen& ff) {
        return Gen(IfElseFunction {test, tf, ff});
      });

      obj.def("ifelse", [](const GenCon& test, Eigen::VectorXd v, const Gen& ff) {
        Constant<-1, -1> tf(test.IRows(), v);
        return Gen(IfElseFunction {test, tf, ff});
      });
      obj.def("ifelse", [](const GenCon& test, const Gen& tf, Eigen::VectorXd v) {
        Constant<-1, -1> ff(test.IRows(), v);
        return Gen(IfElseFunction {test, tf, ff});
      });

      obj.def("ifelse", [](const GenCon& test, Eigen::VectorXd v1, Eigen::VectorXd v2) {
        Constant<-1, -1> tf(test.IRows(), v1);
        Constant<-1, -1> ff(test.IRows(), v2);
        return Gen(IfElseFunction {test, tf, ff});
      });
    }
  };


}  // namespace ASSET