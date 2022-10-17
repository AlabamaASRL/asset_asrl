#pragma once

#include "VectorFunctions/VectorFunction.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"
#include "pch.h"
#include "../PyDocString/Integrators/Integrators_doc.h"

namespace ASSET {

template <class Derived, class ODE, class Stepper>
struct IntegratorBase : VectorFunction<Derived, Stepper::IRC, Stepper::ORC> {
  using Base = VectorFunction<Derived, Stepper::IRC, Stepper::ORC>;

  DENSE_FUNCTION_BASE_TYPES(Base);

  // static const int CPV = Stepper::IRC - 1 - ODE::IRC;
  template <class Scalar>
  using ODEState = typename ODE::template Input<Scalar>;
  // Input  [x0,t0,u0,pv,cpv,tf]
  // Output [xf,tf,uf,pv,]
  ODE ode;
  Stepper stepper;
  std::shared_ptr<ctpl::ThreadPool> pool;

  Vector<double, ODE::XV> AbsTols;
  Vector<double, ODE::XV> RelTols;

  double ErrorOrder = 5.0;
  double MinStepSize = 0.1;
  double DefStepSize = 0.1;
  double MaxStepSize = 0.1;
  double MaxStepChange = 2.0;
  bool Adaptive = false;
  bool FastAdaptiveSTM = true;
  bool ModifyInitialState = true;


  GenericFunction<-1, -1> get_stepper() const {
    return GenericFunction<-1, -1>(stepper);
  }
  IntegratorBase() {
    this->pool = std::make_shared<ctpl::ThreadPool>();
    //this->setPoolThreads(1);
  };
  IntegratorBase(const ODE& ode, const Stepper& stepper, double ds) {
    this->init(ode, stepper, ds);
    this->pool = std::make_shared<ctpl::ThreadPool>();
    //this->setPoolThreads(1);

  };

  void init(const ODE& ode, const Stepper& stepper, double ds) {
    this->ode = ode;
    this->stepper = stepper;
    this->DefStepSize = ds;
    this->MinStepSize = this->DefStepSize / 1000;
    this->MaxStepSize = this->DefStepSize * 1000;
    this->AbsTols.setConstant(this->ode.XVars(), 1.0e-12);
    this->setIORows(this->stepper.IRows(), this->stepper.ORows());
  }

  void setPoolThreads(int thrs) {
      if (this->pool->size() < thrs) {
          this->pool->resize(thrs);
      }
  }

  void setAbsTol(double tol) {
    this->AbsTols.setConstant(this->ode.XVars(), tol);
  }

  template <class InType, class OutType>
  inline std::vector<Input<typename InType::Scalar>> compute_constant(ConstVectorBaseRef<InType> x,
                               ConstVectorBaseRef<OutType> fx_,bool storeoutput) const {
    typedef typename InType::Scalar Scalar;
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();

    Scalar t0 = x[this->ode.TVar()];
    Scalar tf = x[this->IRows() - 1];

    Scalar H = tf - t0;
    int numsteps = int(abs(H / DefStepSize)) + 1;

    Scalar h = H / Scalar(numsteps);

    Input<Scalar> xtemp = x;
    xtemp[this->IRows() - 1] = t0 + h;

    Output<Scalar> fxtemp(this->ORows());
    fxtemp.setZero();

    std::vector<Input<typename InType::Scalar>> xs;
    if (storeoutput) {
        xs.reserve(numsteps);
    }

    for (int i = 0; i < numsteps; i++) {
        if (storeoutput) {
            xs.push_back(xtemp);
        }

      this->stepper.compute(xtemp, fxtemp);
      xtemp.head(this->ORows()) = fxtemp;
      xtemp[this->IRows() - 1] += h;
    }
    fx = fxtemp;

    return xs;
  }

  template <class InType, class OutType>
  inline std::vector<Input<typename InType::Scalar>> compute_adaptive(
      ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
      bool storeoutput) const {
    typedef typename InType::Scalar Scalar;
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();

    Scalar t0 = x[this->ode.TVar()];
    Scalar tf = x[this->IRows() - 1];

    Scalar H = tf - t0;
    int numsteps = int(abs(H / DefStepSize)) + 1;

    Scalar hbig = H / Scalar(numsteps);

    Input<Scalar> xtemp1 = x;
    Input<Scalar> xtemp2 = x;
    Input<Scalar> temp;

    Output<Scalar> fxtemp1(this->ORows());
    fxtemp1.setZero();
    Output<Scalar> fxtemp2(this->ORows());
    fxtemp2.setZero();

    bool continueloop = true;

    Vector<Scalar, ODE::XV> error;
    Vector<Scalar, ODE::XV> error2;
    bool HitMinimum = false;
    int MinimumCount = 0;
    int i = 0;

    std::vector<Input<typename InType::Scalar>> xs;
    if (storeoutput) {
      xs.reserve(numsteps);
    }

    while (continueloop) {
      Scalar tnext = xtemp1[this->ode.TVar()] + hbig;
      if (H > 0) {
        if ((tnext - tf) >= 0.0) {
          hbig = tf - xtemp1[this->ode.TVar()];
          tnext = tf;
          continueloop = false;
        }
      } else {
        if ((tnext - tf) <= 0.0) {
          hbig = tf - xtemp1[this->ode.TVar()];
          tnext = tf;
          continueloop = false;
        }
      }
      Scalar hlittle = hbig / 2.0;
      ////////////////////////////////////
      xtemp2 = xtemp1;

      xtemp1[this->IRows() - 1] = tnext;
      this->stepper.compute(xtemp1, fxtemp1);

      xtemp2[this->IRows() - 1] = xtemp2[this->ode.TVar()] + hlittle;
      this->stepper.compute(xtemp2, fxtemp2);
      temp = xtemp2;

      xtemp2.head(this->ORows()) = fxtemp2;
      fxtemp2.setZero();
      xtemp2[this->IRows() - 1] += hlittle;
      this->stepper.compute(xtemp2, fxtemp2);

      ///////////////////////////////////
      error =
          (fxtemp2.head(this->ode.XVars()) - fxtemp1.head(this->ode.XVars()))
              .cwiseAbs();
      error2 = error.cwiseQuotient(this->AbsTols);
      int worst = 0;
      error2.maxCoeff(&worst);

      Scalar err = error[worst];
      Scalar acc = this->AbsTols[worst];

      Scalar hnext = hbig * pow((acc / err), 1.0 / this->ErrorOrder);

      if (hnext / hbig > this->MaxStepChange)
        hbig *= this->MaxStepChange;
      else if (hnext / hbig < 1. / this->MaxStepChange)
        hbig /= this->MaxStepChange;
      else
        hbig = hnext;

      if (abs(hbig) > this->MaxStepSize)
        hbig = this->MaxStepSize * hbig / abs(hbig);

      if (abs(hbig) < this->MinStepSize) {
        hbig = this->MinStepSize * hbig / abs(hbig);
        HitMinimum = true;
        MinimumCount++;
      } else {
        HitMinimum = false;
      }
      if ((err - acc) > 0 && !HitMinimum) {
      } else {
        if (storeoutput) {
          if (this->FastAdaptiveSTM) {
            xs.push_back(xtemp1);
          } else {
            xs.push_back(temp);
            xs.push_back(xtemp2);
          }
        }

        xtemp1.head(this->ORows()) = fxtemp2;
      }
      i++;
    }
    fx = fxtemp2;
    return xs;
  }

  template <class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> x,
                           ConstVectorBaseRef<OutType> fx_) const {
    if (this->Adaptive)
      this->compute_adaptive(x, fx_, false);
    else
      this->compute_constant(x, fx_, false);
  }

  template <class InType, class OutType, class JacType>
  inline void compute_jacobian_constant(ConstVectorBaseRef<InType> x,
                                        ConstVectorBaseRef<OutType> fx_,
                                        ConstMatrixBaseRef<JacType> jx_) const {
    typedef typename InType::Scalar Scalar;
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

    Scalar t0 = x[this->ode.TVar()];
    Scalar tf = x[this->IRows() - 1];

    Scalar H = tf - t0;
    int numsteps = int(abs(H / DefStepSize)) + 1;
    Scalar h = H / numsteps;

    Input<Scalar> xtemp = x;
    xtemp[this->IRows() - 1] = t0 + h;

    Output<Scalar> fxtemp(this->ORows());
    Jacobian<Scalar> jxtemp(this->ORows(), this->IRows());
    fxtemp.setZero();
    jxtemp.setZero();

    Eigen::Matrix<Scalar, Base::IRC, Base::IRC> jxall(this->IRows(),
                                                      this->IRows());
    jxall.setIdentity();

    for (int i = 0; i < numsteps; i++) {
      jxtemp.setZero();
      fxtemp.setZero();
      this->stepper.compute_jacobian(xtemp, fxtemp, jxtemp);
      jxall.template topRows<Base::ORC>(this->ORows()) = jxtemp * jxall;

      xtemp.head(this->ORows()) = fxtemp;
      xtemp[this->IRows() - 1] += h;
    }
    fx = fxtemp;
    jx = jxall.template topRows<Base::ORC>(this->ORows());
  }

  template <class InType, class OutType, class JacType>
  inline void compute_jacobian_adaptive(ConstVectorBaseRef<InType> x,
                                        ConstVectorBaseRef<OutType> fx_,
                                        ConstMatrixBaseRef<JacType> jx_) const {
    MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
    typedef typename InType::Scalar Scalar;

    auto Xs = this->compute_adaptive(x, fx_, true);

    Output<Scalar> fxtemp(this->ORows());
    fxtemp.setZero();

    Jacobian<Scalar> jxtemp(this->ORows(), this->IRows());
    jxtemp.setZero();

    Eigen::Matrix<Scalar, Base::IRC, Base::IRC> jxall(this->IRows(),
                                                      this->IRows());
    jxall.setIdentity();


    
    constexpr int vsize = DefaultSuperScalar::SizeAtCompileTime;
    Input<DefaultSuperScalar> xtempSS(this->IRows());
    Output<DefaultSuperScalar> fxtempSS(this->ORows());
    fxtempSS.setZero();
    Jacobian<DefaultSuperScalar> jxtempSS(this->ORows(), this->IRows());
    jxtempSS.setZero();
    const int IRR = this->IRows();
    const int ORR = this->ORows();

    auto ScalarImpl = [&](int i) {
        jxtemp.setZero();
        fxtemp.setZero();
        this->stepper.compute_jacobian(Xs[i], fxtemp, jxtemp);
        jxall.template topRows<Base::ORC>(this->ORows()) = jxtemp * jxall;
    };
    auto VectorImpl =[&](int i) {
        fxtempSS.setZero();
        jxtempSS.setZero();

        for (int j = 0; j < vsize; j++) {
            for (int k = 0; k < IRR; k++) {
                xtempSS[k][j] = Xs[i + j][k];
            }
        }
        this->stepper.compute_jacobian(xtempSS, fxtempSS, jxtempSS);

        for (int j = 0; j < vsize; j++) {
            for (int k = 0; k < IRR; k++) {
                for (int l = 0; l < ORR; l++) {
                    jxtemp(l,k) = jxtempSS(l, k)[j];
                }
            }
            jxall.template topRows<Base::ORC>(this->ORows()) = jxtemp * jxall;
        }
    };

    int Packs = (this->EnableVectorization) ? Xs.size() / vsize : 0;

    for (int i = 0; i < Packs; i++) {
        VectorImpl(i * vsize);
    }
    for (int i = Packs*vsize; i < Xs.size(); i++) {
        ScalarImpl(i);
    }

    jx = jxall.template topRows<Base::ORC>(this->ORows());
  }

  template <class InType, class OutType, class JacType>
  inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                    ConstVectorBaseRef<OutType> fx_,
                                    ConstMatrixBaseRef<JacType> jx_) const {
    if (this->Adaptive)
      this->compute_jacobian_adaptive(x, fx_, jx_);
    else
      this->compute_jacobian_constant(x, fx_, jx_);
  }

  template <class InType, class OutType, class JacType, class AdjGradType,
            class AdjHessType, class AdjVarType>
  inline void compute_jacobian_adjointgradient_adjointhessian_impl(
      ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
      ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
      ConstMatrixBaseRef<AdjHessType> adjhess_,
      ConstVectorBaseRef<AdjVarType> adjvars) const {
    typedef typename InType::Scalar Scalar;
    //VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
    VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
    MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

    std::vector<Input<Scalar>> Xs;
    
    if (this->Adaptive)
        Xs=this->compute_adaptive(x, fx_, true);
    else
        Xs=this->compute_constant(x, fx_, true);

    int numsteps = Xs.size();

    //Input<Scalar> xtemp = x;
    //xtemp[this->IRows() - 1] = t0 + h;

    Output<Scalar> fxtemp(this->ORows());
    fxtemp.setZero();


    Gradient<Scalar> gxtemp(this->IRows());
    gxtemp.setZero();

    Jacobian<Scalar> jxtemp(this->ORows(), this->IRows());
    Jacobian<Scalar> jxall(this->ORows(), this->IRows());
    jxtemp.setZero();
    jxall.leftCols(this->ORows()).setIdentity();

    Hessian<Scalar> hxtemp(this->IRows(), this->IRows());
    Hessian<Scalar> hxall(this->IRows(), this->IRows());
    hxtemp.setZero();
    hxall.setZero();

    Output<Scalar> adjtemp = adjvars;
    Hessian<Scalar> jtwist(this->IRows(), this->IRows());
    jtwist.setZero();
    jtwist(this->IRows() - 1, this->IRows() - 1) = 1.0;

    for (int i = 0; i < numsteps; i++) {
      jxtemp.setZero();
      hxtemp.setZero();
      gxtemp.setZero();
      fxtemp.setZero();
      this->stepper.compute_jacobian_adjointgradient_adjointhessian(
          Xs[numsteps - i - 1], fxtemp, jxtemp, gxtemp, hxtemp, adjtemp);

      jtwist.topRows(this->ORows()) = jxtemp;
      jxall = jxall * jtwist;
      if (i == 0) {
        jxall.rightCols(1) = jxtemp.rightCols(1);
      }
      hxall = jtwist.transpose() * hxall * jtwist;
      hxall += hxtemp;
      adjtemp = gxtemp.head(this->ORows());
    }

    jx = jxall.topRows(this->ORows());
    adjhess = hxall;
    adjgrad = jx.transpose() * adjvars;
  }

  template <class Scalar>
  std::tuple<Jacobian<Scalar>, Gradient<Scalar>, Hessian<Scalar>>
  back_prop_derivs(const std::vector<Input<Scalar>>& Xs,
                   const Output<Scalar>& adjvars, bool doHessian) const {
    Jacobian<Scalar> jx;
    Gradient<Scalar> adjgrad;
    Jacobian<Scalar> adjhess;

    int numsteps = Xs.size();

    Output<Scalar> fxtemp(this->ORows());
    fxtemp.setZero();

    Gradient<Scalar> gxtemp(this->IRows());
    gxtemp.setZero();

    Jacobian<Scalar> jxtemp(this->ORows(), this->IRows());
    Jacobian<Scalar> jxall(this->ORows(), this->IRows());
    jxtemp.setZero();
    jxall.leftCols(this->ORows()).setIdentity();

    Hessian<Scalar> hxtemp(this->IRows(), this->IRows());
    Hessian<Scalar> hxall(this->IRows(), this->IRows());
    hxtemp.setZero();
    hxall.setZero();

    Output<Scalar> adjtemp = adjvars;
    Hessian<Scalar> jtwist(this->IRows(), this->IRows());
    jtwist.setZero();
    jtwist(this->IRows() - 1, this->IRows() - 1) = 1.0;

    for (int i = 0; i < numsteps; i++) {
      jxtemp.setZero();
      if (doHessian) hxtemp.setZero();
      gxtemp.setZero();
      fxtemp.setZero();

      if (doHessian) {
        this->stepper.compute_jacobian_adjointgradient_adjointhessian(
            Xs[numsteps - i - 1], fxtemp, jxtemp, gxtemp, hxtemp, adjtemp);
      } else {
        this->stepper.compute_jacobian(Xs[numsteps - i - 1], fxtemp, jxtemp);
      }
      jtwist.topRows(this->ORows()) = jxtemp;
      jxall = jxall * jtwist;
      if (i == 0) {
        jxall.rightCols(1) = jxtemp.rightCols(1);
      }
      if (doHessian) {
        hxall = jtwist.transpose() * hxall * jtwist;
        hxall += hxtemp;
      }
      adjtemp = gxtemp.head(this->ORows());
    }
    jx = jxall.topRows(this->ORows());
    adjhess = hxall;
    adjgrad = jx.transpose() * adjvars;

    return std::tuple{jx, adjhess, adjgrad};
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////


  ODEState<double> update_state(const ODEState<double>& x0, const VectorX<double>& cpv) const {
      Input<double> stepper_input(this->IRows());
      stepper_input.head(this->ode.IRows()) = x0;
      stepper_input[this->IRows() - 1] = x0[this->ode.TVar()];
      stepper_input.segment(this->ode.IRows(), cpv.size()) = cpv;
      Output<double> stepper_output(this->ORows());
      stepper_output.setZero();
      this->stepper.compute(stepper_input, stepper_output);
      ODEState<double> x0n = stepper_output.head(x0.size());
      return x0n;
  }

  ODEState<double> update_state(const ODEState<double>& x0) const {
      VectorX<double> cpv;
      cpv.resize(0);
      return this->update_state(x0,cpv);
  }


  ODEState<double> integrate(const ODEState<double>& x0, double tf,
                             const VectorX<double>& cpv) const {

    if (x0.size() != this->ode.IRows()) {
        throw std::invalid_argument("Incorrect ode state size");
    }

    Input<double> stepper_input(this->IRows());
    stepper_input.head(this->ode.IRows()) = x0;
    stepper_input[this->IRows() - 1] = tf;
    stepper_input.segment(this->ode.IRows(), cpv.size()) = cpv;
    Output<double> stepper_output(this->ORows());
    stepper_output.setZero();

    this->compute(stepper_input, stepper_output);

    ODEState<double> xf = stepper_output.head(this->ode.IRows());

    return xf;
  }

  ODEState<double> integrate(const ODEState<double>& x0, double tf) const {
    Eigen::VectorXd empty;
    empty.resize(0);

    return this->integrate(x0, tf, empty);
  }

  std::tuple<ODEState<double>, Jacobian<double>> integrate_stm(
      const ODEState<double>& x0, double tf, const VectorX<double>& cpv) const {

    if (x0.size() != this->ode.IRows()) {
        throw std::invalid_argument("Incorrect ode state size");
    }
    Input<double> stepper_input(this->IRows());
    stepper_input.head(this->ode.IRows()) = x0;
    stepper_input[this->IRows() - 1] = tf;
    stepper_input.segment(this->ode.IRows(), cpv.size()) = cpv;

    Output<double> stepper_output(this->ORows());
    Jacobian<double> stepper_jac(this->ORows(), this->IRows());
    stepper_jac.setZero();
    stepper_output.setZero();

    this->compute_jacobian(stepper_input, stepper_output, stepper_jac);

    ODEState<double> xf = stepper_output.head(this->ode.IRows());

    return std::tuple{xf, stepper_jac};
  }

  std::tuple<ODEState<double>, Jacobian<double>> integrate_stm(
      const ODEState<double>& x0, double tf) const {
    Eigen::VectorXd empty;
    empty.resize(0);
    return this->integrate_stm(x0, tf, empty);
  }

  std::tuple<ODEState<double>, Jacobian<double>> integrate_stm_parallel(
      const ODEState<double>& x0, double tf, const VectorX<double>& cpv,
      int thrs) {
      this->setPoolThreads(thrs);

      if (x0.size() != this->ode.IRows()) {
          throw std::invalid_argument("Incorrect ode state size");
      }

    VectorX<double> ts =
        VectorX<double>::LinSpaced(thrs + 1, x0[this->ode.TVar()], tf);
    std::vector<ODEState<double>> Xs(thrs + 1);
    Xs[0] = x0;

    using RetType = std::tuple<ODEState<double>, Jacobian<double>>;

    std::vector<std::future<RetType>> results(thrs);

    Eigen::MatrixXd jxall(this->IRows(), this->IRows());
    jxall.setIdentity();

    auto stm_op = [&](int id, int i) {
      auto xi = Xs[i];
      auto tf1 = ts[i + 1];
      return this->integrate_stm(xi, tf1, cpv);
    };

    for (int i = 0; i < thrs; i++) {
      results[i] = this->pool->push(stm_op, i);
      if (i < (thrs - 1)) Xs[i + 1] = this->integrate(Xs[i], ts[i + 1], cpv);
    }
    for (int i = 0; i < thrs; i++) {
      auto [xf, jx] = results[i].get();
      jxall.topRows(this->ORows()) = (jx * jxall).eval();
      if (i == (thrs - 1)) Xs[i + 1] = xf;
    }

    RetType tup_final;
    std::get<0>(tup_final) = Xs.back();
    std::get<1>(tup_final) = jxall.topRows(this->ORows());

    return tup_final;
  }

  std::tuple<ODEState<double>, Jacobian<double>> integrate_stm_parallel(
      const ODEState<double>& x0, double tf, int thrs) {
    Eigen::VectorXd empty;
    empty.resize(0);
    return this->integrate_stm_parallel(x0, tf, empty, thrs);
  }

  std::vector<std::tuple<ODEState<double>, Jacobian<double>>>
  integrate_stm_parallel(const std::vector<ODEState<double>>& x0,
                         std::vector<double> tf,
                         const std::vector<VectorX<double>>& cpv, int thrs) {
    using RetType = std::vector<std::tuple<ODEState<double>, Jacobian<double>>>;
    this->setPoolThreads(thrs);

    int n = x0.size();

    auto stm_op = [&](int id, int start, int stop) {
      RetType stms(stop - start);
      for (int i = start; i < stop; i++) {
        stms[i - start] = this->integrate_stm(x0[i], tf[i], cpv[i]);
      }
      return stms;
    };

    std::vector<std::future<RetType>> results(thrs);

    for (int i = 0; i < thrs; i++) {
      int start = (i * n) / thrs;
      int stop = ((i + 1) * n) / thrs;
      results[i] = this->pool->push(stm_op, start, stop);
    }
    RetType Allstms;
    Allstms.reserve(x0.size());

    for (int i = 0; i < thrs; i++) {
      auto stms = results[i].get();
      for (auto& stm : stms) Allstms.push_back(stm);
    }

    return Allstms;
  }

  std::vector<std::tuple<ODEState<double>, Jacobian<double>>>
  integrate_stm_parallel(const std::vector<ODEState<double>>& x0,
                         std::vector<double> tf, int thrs) {
    int n = x0.size();
    VectorX<double> empty;
    empty.resize(0);

    std::vector<VectorX<double>> cpv(n, empty);
    return this->integrate_stm_parallel(x0, tf, cpv, thrs);
  }

  std::vector<ODEState<double>> integrate_parallel(
      const std::vector<ODEState<double>>& x0, std::vector<double> tf,
      const std::vector<VectorX<double>>& cpv, int thrs) {
    using RetType = std::vector<ODEState<double>>;
    this->setPoolThreads(thrs);

    int n = x0.size();

    auto xf_op = [&](int id, int start, int stop) {
      RetType xfs(stop - start);
      for (int i = start; i < stop; i++) {
        xfs[i - start] = this->integrate(x0[i], tf[i], cpv[i]);
      }
      return xfs;
    };

    std::vector<std::future<RetType>> results(thrs);

    for (int i = 0; i < thrs; i++) {
      int start = (i * n) / thrs;
      int stop = ((i + 1) * n) / thrs;
      results[i] = this->pool->push(xf_op, start, stop);
    }
    RetType Allxfs;
    Allxfs.reserve(x0.size());

    for (int i = 0; i < thrs; i++) {
      auto xfs = results[i].get();
      for (auto& xf : xfs) Allxfs.push_back(xf);
    }

    return Allxfs;
  }

  std::vector<ODEState<double>> integrate_parallel(
      const std::vector<ODEState<double>>& x0, std::vector<double> tf,
      int thrs) {
    int n = x0.size();
    VectorX<double> empty;
    empty.resize(0);

    std::vector<VectorX<double>> cpv(n, empty);
    return this->integrate_parallel(x0, tf, cpv, thrs);
  }

  std::vector<ODEState<double>> integrate_dense(
      const ODEState<double>& x0, double tf, int NumStates,
      const VectorX<double>& cpv) const {
    VectorX<double> ts =
        VectorX<double>::LinSpaced(NumStates, x0[this->ode.TVar()], tf);
    std::vector<ODEState<double>> xout(NumStates);

    xout[0] = x0;
    if (this->ModifyInitialState) {
        xout[0] = update_state(x0, cpv);
    }
    for (int i = 1; i < NumStates; i++) {
      xout[i] = this->integrate(xout[i - 1], ts[i], cpv);
    }
    return xout;
  }

  std::vector<ODEState<double>> integrate_dense(const ODEState<double>& x0,
                                                double tf,
                                                int NumStates) const {
    Eigen::VectorXd empty;
    empty.resize(0);
    return this->integrate_dense(x0, tf, NumStates, empty);
  }

  std::vector<ODEState<double>> integrate_dense(const ODEState<double>& x0,double tf) const {
     

      if (this->Adaptive) {

          Input<double> stepper_input(this->IRows());
          stepper_input.head(this->ode.IRows()) = x0;
          stepper_input[this->IRows() - 1] = tf;
          Output<double> stepper_output(this->ORows());
          stepper_output.setZero();

          auto Steps =  this->compute_adaptive(stepper_input, stepper_output,true);
          std::vector<ODEState<double>> states(Steps.size()+1);
          for (int i = 0; i < Steps.size(); i++) {
              states[i] = Steps[i].head(this->ode.IRows());
          }
          states.back() = stepper_output.head(this->ode.IRows());

          if (this->ModifyInitialState) {
              states[0] = update_state(x0);
          }

          return states;
      }
      else {
          Eigen::VectorXd empty;
          empty.resize(0);
          int NumStates = int(std::abs(tf - x0[this->ode.TVar()]) / this->DefStepSize);
          return this->integrate_dense(x0, tf, NumStates, empty);
      }
      
  }

  std::vector<ODEState<double>> integrate_dense(
      const ODEState<double>& x0, double tf, int NumStates,
      const VectorX<double>& cpv,
      std::function<bool(ConstEigenRef<Eigen::VectorXd>)> exitfun) const {
    VectorX<double> ts =
        VectorX<double>::LinSpaced(NumStates, x0[this->ode.TVar()], tf);

    std::vector<ODEState<double>> xout;
    xout.reserve(NumStates);
    xout.push_back(x0);
    if (this->ModifyInitialState) {
        xout[0] = update_state(x0);
    }

    for (int i = 1; i < NumStates; i++) {
      xout.push_back(this->integrate(xout[i - 1], ts[i], cpv));
      if (exitfun(xout.back())) break;
    }
    return xout;
  }

  std::vector<ODEState<double>> integrate_dense(
      const ODEState<double>& x0, double tf, int NumStates,
      std::function<bool(ConstEigenRef<Eigen::VectorXd>)> exitfun) const {
    Eigen::VectorXd empty;
    empty.resize(0);
    return this->integrate_dense(x0, tf, NumStates, empty, exitfun);
  }


  std::vector<std::vector<ODEState<double>>> integrate_dense_parallel(
      const std::vector<ODEState<double>>& x0, std::vector<double> tf,int thrs) {
      using RetType = std::vector <std::vector<ODEState<double>>>;
      this->setPoolThreads(thrs);

      int n = x0.size();

      auto xf_op = [&](int id, int start, int stop) {
          RetType xfs(stop - start);
          for (int i = start; i < stop; i++) {
              xfs[i - start] = this->integrate_dense(x0[i], tf[i]);
          }
          return xfs;
      };

      std::vector<std::future<RetType>> results(thrs);

      for (int i = 0; i < thrs; i++) {
          int start = (i * n) / thrs;
          int stop = ((i + 1) * n) / thrs;
          results[i] = this->pool->push(xf_op, start, stop);
      }
      RetType Allxfs;
      Allxfs.reserve(x0.size());

      for (int i = 0; i < thrs; i++) {
          auto xfs = results[i].get();
          for (auto& xf : xfs) Allxfs.push_back(xf);
      }

      return Allxfs;
  }


  std::vector<std::vector<ODEState<double>>> integrate_dense_parallel(
      const std::vector<ODEState<double>>& x0, std::vector<double> tf, 
      std::vector<int> ns,
      int thrs) {
      using RetType = std::vector <std::vector<ODEState<double>>>;
      this->setPoolThreads(thrs);

      int n = x0.size();

      auto xf_op = [&](int id, int start, int stop) {
          RetType xfs(stop - start);
          for (int i = start; i < stop; i++) {
              xfs[i - start] = this->integrate_dense(x0[i], tf[i],ns[i]);
          }
          return xfs;
      };

      std::vector<std::future<RetType>> results(thrs);

      for (int i = 0; i < thrs; i++) {
          int start = (i * n) / thrs;
          int stop = ((i + 1) * n) / thrs;
          results[i] = this->pool->push(xf_op, start, stop);
      }
      RetType Allxfs;
      Allxfs.reserve(x0.size());

      for (int i = 0; i < thrs; i++) {
          auto xfs = results[i].get();
          for (auto& xf : xfs) Allxfs.push_back(xf);
      }

      return Allxfs;
  }



  template <class PyClass>
  static void IntegratorAPIBuild(PyClass& obj) {
      using namespace doc;
    obj.def("integrate",
            (ODEState<double>(Derived::*)(const ODEState<double>&, double,
                                          const VectorX<double>&) const) &
                Derived::integrate, IntegratorBase_integrate_cpv);
    obj.def("integrate", (ODEState<double>(Derived::*)(const ODEState<double>&,
                                                       double) const) &
                             Derived::integrate, IntegratorBase_integrate);

    obj.def(
        "integrate_stm",
        (std::tuple<ODEState<double>, Jacobian<double>>(Derived::*)(
            const ODEState<double>&, double, const VectorX<double>&) const) &
            Derived::integrate_stm, IntegratorBase_integrate_stm_cpv);
    obj.def("integrate_stm",
            (std::tuple<ODEState<double>, Jacobian<double>>(Derived::*)(
                const ODEState<double>&, double) const) &
                Derived::integrate_stm, IntegratorBase_integrate_stm);

    obj.def("integrate_stm_parallel",
            (std::tuple<ODEState<double>, Jacobian<double>>(Derived::*)(
                const ODEState<double>&, double, const VectorX<double>&, int)) &
                Derived::integrate_stm_parallel,
            py::call_guard<py::gil_scoped_release>(), IntegratorBase_integrate_stm_parallel_single_cpv);
    obj.def("integrate_stm_parallel",
            (std::tuple<ODEState<double>, Jacobian<double>>(Derived::*)(
                const ODEState<double>&, double, int)) &
                Derived::integrate_stm_parallel,
            py::call_guard<py::gil_scoped_release>(), IntegratorBase_integrate_stm_parallel_single);

    obj.def("integrate_stm_parallel",
            (std::vector<std::tuple<ODEState<double>, Jacobian<double>>>(
                Derived::*)(const std::vector<ODEState<double>>&,
                            std::vector<double>,
                            const std::vector<VectorX<double>>&, int)) &
                Derived::integrate_stm_parallel,
            py::call_guard<py::gil_scoped_release>(), IntegratorBase_integrate_stm_parallel_cpv);
    obj.def("integrate_stm_parallel",
            (std::vector<std::tuple<ODEState<double>, Jacobian<double>>>(
                Derived::*)(const std::vector<ODEState<double>>&,
                            std::vector<double>, int)) &
                Derived::integrate_stm_parallel,
            py::call_guard<py::gil_scoped_release>(), IntegratorBase_integrate_stm_parallel);

    obj.def("integrate_parallel",
            (std::vector<ODEState<double>>(Derived::*)(
                const std::vector<ODEState<double>>&, std::vector<double>,
                const std::vector<VectorX<double>>&, int)) &
                Derived::integrate_parallel,
            py::call_guard<py::gil_scoped_release>(), IntegratorBase_integrate_parallel_cpv);
    obj.def(
        "integrate_parallel",
        (std::vector<ODEState<double>>(Derived::*)(
            const std::vector<ODEState<double>>&, std::vector<double>, int)) &
            Derived::integrate_parallel,
        py::call_guard<py::gil_scoped_release>(), IntegratorBase_integrate_parallel);

    obj.def("integrate_dense", (std::vector<ODEState<double>>(Derived::*)(
                                   const ODEState<double>&, double, int,
                                   const VectorX<double>&) const) &
                                   Derived::integrate_dense, IntegratorBase_integrate_dense_cpv);
    obj.def("integrate_dense",
            (std::vector<ODEState<double>>(Derived::*)(const ODEState<double>&,
                                                       double, int) const) &
                Derived::integrate_dense, IntegratorBase_integrate_dense);

    obj.def("integrate_dense",
        (std::vector<ODEState<double>>(Derived::*)(const ODEState<double>&,
            double) const) &
        Derived::integrate_dense, IntegratorBase_integrate_dense);

    obj.def("integrate_dense",
            (std::vector<ODEState<double>>(Derived::*)(
                const ODEState<double>&, double, int,
                std::function<bool(ConstEigenRef<Eigen::VectorXd>)>) const) &
                Derived::integrate_dense, IntegratorBase_integrate_dense_exit);



    obj.def(
        "integrate_dense_parallel",
        (std::vector<std::vector<ODEState<double>>>(Derived::*)(
            const std::vector<ODEState<double>>&, std::vector<double>, int)) &
        Derived::integrate_dense_parallel,
        py::call_guard<py::gil_scoped_release>(), IntegratorBase_integrate_parallel);

    obj.def(
        "integrate_dense_parallel",
        (std::vector<std::vector<ODEState<double>>>(Derived::*)(
            const std::vector<ODEState<double>>&, std::vector<double>, std::vector<int>, int)) &
        Derived::integrate_dense_parallel,
        py::call_guard<py::gil_scoped_release>(), IntegratorBase_integrate_parallel);


    obj.def_readwrite("DefStepSize", &Derived::DefStepSize, IntegratorBase_DefStepSize);
    obj.def_readwrite("MaxStepSize", &Derived::MaxStepSize, IntegratorBase_MaxStepSize);
    obj.def_readwrite("MinStepSize", &Derived::MinStepSize, IntegratorBase_MinStepSize);
    obj.def_readwrite("MaxStepChange", &Derived::MaxStepChange, IntegratorBase_MaxStepChange);
    obj.def_readwrite("Adaptive", &Derived::Adaptive, IntegratorBase_Adaptive);
    obj.def_readwrite("AbsTols", &Derived::AbsTols, IntegratorBase_AbsTols);
    obj.def_readwrite("ModifyInitialState", &Derived::ModifyInitialState);

    obj.def_readwrite("FastAdaptiveSTM", &Derived::FastAdaptiveSTM, IntegratorBase_FastAdaptiveSTM);
    obj.def("get_stepper", &Derived::get_stepper, IntegratorBase_get_stepper);

    obj.def("setAbsTol", &Derived::setAbsTol, IntegratorBase_setAbsTol);
  }
};

}  // namespace ASSET
