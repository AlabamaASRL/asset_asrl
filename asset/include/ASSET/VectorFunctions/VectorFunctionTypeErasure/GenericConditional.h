#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/CommonFunctions.h>
#include <ASSET/pch.h>

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
  };

}  // namespace ASSET
