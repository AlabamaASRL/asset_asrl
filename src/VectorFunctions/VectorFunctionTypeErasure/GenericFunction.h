/*
File Name: GenericFunction.h

File Description: Defines the class GenericFunction which is a vector function that can be constructed form
ANY other dense vector function with compatible input and output sizes. This is probably the most important
class in the vector functions part of the library as it allows us to type erase arbitrarily compilicated
compile time or run time defined vector functions. It holds type easure object that constructable from any
compatible object and forwards its compute calls as well as selected product and accumulation operations to
this type erased function.


////////////////////////////////////////////////////////////////////////////////

Original File Developer : James B. Pezent - jbpezent - jbpezent@crimson.ua.edu

Current File Maintainers:
    1. James B. Pezent - jbpezent         - jbpezent@crimson.ua.edu
    2. Full Name       - GitHub User Name - Current Email
    3. ....


Usage of this source code is governed by the license found
in the LICENSE file in ASSET's top level directory.

*/

#pragma once

#include "DeepCopySpecs.h"
#include "DenseFunctionBase.h"
#include "DenseFunctionSpecs.h"
#include "SizingSpecs.h"
#include "SolverInterfaceSpecs.h"
#include "VectorFunctions/CommonFunctions/CommonFunctions.h"


namespace ASSET {


  namespace detail {

    /*
    Defining the internal type erasure function held by GenericFunction (GFTE)
    */

    template<int IR, int OR>
    struct GFTE;

    template<int IR, int OR>
    struct GFDeepCopySelector {
      using type = DeepCopySpecs<GFTE<IR, OR>, GFTE<-1, -1>, ConstraintInterface>;
    };
    template<int IR>
    struct GFDeepCopySelector<IR, 1> {
      using type = DeepCopySpecs<GFTE<IR, 1>, GFTE<-1, -1>, ConstraintInterface, ObjectiveInterface>;
    };
    template<>
    struct GFDeepCopySelector<-1, -1> {
      using type = DeepCopySpecs<GFTE<-1, -1>, ConstraintInterface>;
    };


    template<int IR, int OR>
    using GenTE = rubber_types::TypeErasure<typename GFDeepCopySelector<IR, OR>::type,
                                            DenseFunctionSpec<IR, OR>,
                                            SizableSpec,
                                            typename SolverInterfaceSelector<IR, OR>::type>;


    /*
     * Final Type
     */
    template<int IR, int OR>
    struct GFTE : GenTE<IR, OR> {
      using Base = GenTE<IR, OR>;
      template<class T>
      GFTE(const T& t) : Base(t) {
      }
      GFTE() {
      }
    };


  }  // namespace detail


  template<int IR, int OR>
  struct GenericFunction : VectorFunction<GenericFunction<IR, OR>, IR, OR> {
    using Base = VectorFunction<GenericFunction<IR, OR>, IR, OR>;
    DENSE_FUNCTION_BASE_TYPES(Base);


    static const bool IsGenericFunction = true;
    static const bool IsVectorizable = true;

    using TE = detail::GFTE<IR, OR>;
    using RightJacTarget = Eigen::Ref<Eigen::Matrix<double, -1, IR>>;
    using Derived = GenericFunction<IR, OR>;

    using Concept = typename TE::Concept;
    using Dspec = DenseFunctionSpec<IR, OR>;

    TE func;
    bool islinear = false;

    GenericFunction() {
    }

    template<class T>
    GenericFunction(const T& t) : func(t) {
      this->cachedata();
    }
    GenericFunction(const GenericFunction<IR, OR>& obj) {
      if (obj.func.get_container()) {
        this->func.reset_container(obj.func.get_container());
        this->cachedata();
      } else {
        throw std::invalid_argument("Attempting to copy null function");
      }
    }
    template<int IR1, int OR1>
    GenericFunction(const GenericFunction<IR1, OR1>& obj) {
      if (obj.func.get_container()) {
        obj.func.deep_copy_into(this->func);
        cachedata();
      } else {
        throw std::invalid_argument("Attempting to copy null function");
      }
    }
    void cachedata() {
      this->setIORows(this->func.IRows(), this->func.ORows());
      this->set_input_domain(this->IRows(), {this->func.input_domain()});
      this->islinear = this->func.is_linear();
    }

    std::string name() const {
      return this->func.name();
    }
    inline bool is_linear() const {
      return this->islinear;
    }
    void enable_vectorization(bool b) const {
      this->func.enable_vectorization(b);
      this->EnableVectorization = b;
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      this->func.compute(x, fx_);
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      this->func.compute_jacobian(x, fx_, jx_);
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
      this->func.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_, adjvars);
    }

    /////////////////////////////////////////////////////////////////////////////////

    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left,
                                       ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
      typedef typename Target::Scalar Scalar;

      if constexpr (std::is_same<Scalar, double>::value) {
        constexpr bool TargConv =
            std::is_convertible<decltype(target_.const_cast_derived()), RightJacTarget>::value;
        if constexpr (TargConv) {
          ConstMatrixBaseRef<Right> right_ref = right.derived();
          if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
            // ConstDiagonalBaseRef<Left> left_ref = left.derived();
            // this->func.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
            Base::right_jacobian_product(target_, left, right, assign, aliased);
          } else {
            ConstMatrixBaseRef<Left> left_ref = left.derived();
            this->func.right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
          }
        } else {
          Base::right_jacobian_product(target_, left, right, assign, aliased);
        }
      } else {
        Base::right_jacobian_product(target_, left, right, assign, aliased);
      }
    }

    template<class Target, class AdjHessType, class Assignment>
    void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                            ConstMatrixBaseRef<AdjHessType> right,
                            Assignment assign) const {
      typedef typename Target::Scalar Scalar;
      if (!this->is_linear())
        Base::accumulate_hessian(target_, right, assign);
    }
    template<class Target, class JacType, class Assignment>
    void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                             ConstMatrixBaseRef<JacType> right,
                             Assignment assign) const {
      typedef typename Target::Scalar Scalar;

      if constexpr (std::is_same<Scalar, double>::value) {
        if (this->is_linear()) {
          this->func.accumulate_jacobian(target_, right, assign);
        } else {
          Base::accumulate_jacobian(target_, right, assign);
        }
      } else {
        Base::accumulate_jacobian(target_, right, assign);
      }
    }

    ////////////////////////////////////////////////////////////////////////////////
    template<class JacType, class Scalar>
    void scale_jacobian(ConstMatrixBaseRef<JacType> target_, Scalar s) const {
      if constexpr (std::is_same<Scalar, ASSET::DefaultSuperScalar>::value) {
        Base::scale_jacobian(target_, s);
      } else {
        this->func.scale_jacobian(target_, s);
      }
    }


    template<class AdjHessType, class Scalar>
    void scale_hessian(ConstMatrixBaseRef<AdjHessType> target_, Scalar s) const {
      if (!this->is_linear())
        Base::scale_hessian(target_, s);
    }
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    template<class T>
    static GenericFunction<IR, OR> PyCopy(const T& obj) {
      return GenericFunction<IR, OR>(obj);
    }

    void SuperTest(const Input<double>& xt, int n) {
      Input<double> x = xt;
      Output<double> fx(this->ORows());
      Jacobian<double> jx(this->ORows(), this->IRows());
      Gradient<double> gx(this->IRows());
      Hessian<double> hx(this->IRows(), this->IRows());
      Output<double> l(this->ORows());

      Input<ASSET::DefaultSuperScalar> X = xt.template cast<ASSET::DefaultSuperScalar>();
      Output<ASSET::DefaultSuperScalar> FX(this->ORows());
      Jacobian<ASSET::DefaultSuperScalar> JX(this->ORows(), this->IRows());
      Gradient<ASSET::DefaultSuperScalar> GX(this->IRows());
      Hessian<ASSET::DefaultSuperScalar> HX(this->IRows(), this->IRows());
      Output<ASSET::DefaultSuperScalar> L(this->ORows());

      l.setOnes();
      fx.setZero();
      jx.setZero();
      hx.setZero();

      Eigen::BenchTimer t1;
      Eigen::BenchTimer t2;

      double dummy = 0;
      t1.start();
      for (int i = 0; i < n; i++) {
        x[0] += 1.0 / double(n + 1);
        this->func.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, l);
        dummy += fx[0] + jx(0, 0) + hx(0, 0);
      }
      t1.stop();

      std::cout << dummy << std::endl;

      ASSET::DefaultSuperScalar dummy2(0);
      t2.start();
      int n2 = n / ASSET::DefaultSuperScalar::SizeAtCompileTime;
      for (int i = 0; i < n2; i++) {
        X[0] += ASSET::DefaultSuperScalar((1.0 / double(n + 1)));
        this->func.compute_jacobian_adjointgradient_adjointhessian(X, FX, JX, GX, HX, L);
        dummy2 += FX[0] + JX(0, 0) + HX(0, 0);
      }
      t2.stop();

      std::cout << dummy2.transpose() << std::endl << std::endl;

      std::cout << "Scalar     : " << t1.total() * 1000.0 << std::endl;
      std::cout << "SuperScalar: " << t2.total() * 1000.0 << std::endl;
    }


    void SpeedTest(const Input<double>& xt, int n) {
      Input<double> x = xt;
      Output<double> fx(this->ORows());
      Jacobian<double> jx(this->ORows(), this->IRows());
      Gradient<double> gx(this->IRows());
      Hessian<double> hx(this->IRows(), this->IRows());
      Output<double> l(this->ORows());

      Input<ASSET::DefaultSuperScalar> X = xt.template cast<ASSET::DefaultSuperScalar>();
      Output<ASSET::DefaultSuperScalar> FX(this->ORows());
      Jacobian<ASSET::DefaultSuperScalar> JX(this->ORows(), this->IRows());
      Gradient<ASSET::DefaultSuperScalar> GX(this->IRows());
      Hessian<ASSET::DefaultSuperScalar> HX(this->IRows(), this->IRows());
      Output<ASSET::DefaultSuperScalar> L(this->ORows());

      Eigen::BenchTimer t1;
      Eigen::BenchTimer t2;
      l.setOnes();
      L.setOnes();
      fx.setZero();
      jx.setZero();
      hx.setZero();


      MemoryManager::resize(64, 64);


      double dummy = 0;
      t1.start();
      for (int i = 0; i < n; i++) {
        // x[0] += 1.0 / double(n*10 + 1);

        fx.setZero();
        jx.setZero();
        hx.setZero();

        this->func.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, l);
        dummy += fx[0] + jx(0, 0) + hx(0, 0);
      }
      t1.stop();

      std::cout << dummy << std::endl;

      ASSET::DefaultSuperScalar dummy2(0.0);
      t2.start();
      int n2 = n / ASSET::DefaultSuperScalar::SizeAtCompileTime;
      for (int i = 0; i < n2; i++) {
        // X[0] += ASSET::DefaultSuperScalar((1.0 / double(n + 1)));

        FX.setZero();
        JX.setZero();
        HX.setZero();

        this->func.compute_jacobian_adjointgradient_adjointhessian(X, FX, JX, GX, HX, L);
        dummy2 += FX[0] + JX(0, 0) + HX(0, 0);
      }
      t2.stop();

      std::cout << dummy2.transpose() << std::endl << std::endl;

      std::cout << " Run Time " << std::endl;

      std::cout << "  Scalar     : " << t1.total() * 1000.0 << std::endl;
      std::cout << "  SuperScalar: " << t2.total() * 1000.0 << std::endl;

      std::cout << " Max Stack Size (Bytes)" << std::endl;

      std::cout << "  Scalar     : " << MemoryManager::size_scalar() * 8 << std::endl;
      std::cout << "  SuperScalar: "
                << MemoryManager::size_super_scalar() * 8 * ASSET::DefaultSuperScalar::SizeAtCompileTime
                << std::endl;

      MemoryManager::resize(64, 64);
    }


    template<class PYClass>
    static void GenericBuild(PYClass& obj) {
      using Gen = GenericFunction<-1, -1>;
      using GenS = GenericFunction<-1, 1>;
      using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

      using SEG = Segment<-1, -1, -1>;
      using SEG2 = Segment<-1, 2, -1>;
      using SEG3 = Segment<-1, 3, -1>;
      using SEG4 = Segment<-1, 4, -1>;
      using ELEM = Segment<-1, 1, -1>;

      obj.def(py::init<const GenericFunction<IR, OR>&>());
      if constexpr (OR == -1 && IR == -1) {
        obj.def(py::init(&Derived::PyCopy<GenericFunction<IR, 1>>));
      }

      obj.def("input_domain", &Derived::input_domain);
      obj.def("is_linear", &Derived::is_linear);
      obj.def("SuperTest", &Derived::SuperTest);
      obj.def("SpeedTest", &Derived::SpeedTest);

      Base::DenseBaseBuild(obj);
    }
  };


}  // namespace ASSET
