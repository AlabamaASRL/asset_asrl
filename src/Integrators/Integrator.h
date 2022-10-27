#include "VectorFunctionTypeErasure/GenericFunction.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"

#include "RKSteppers.h"
#include "pch.h"


namespace ASSET {

template<class DODE>
struct Integrator:VectorFunction<Integrator<DODE>,SZ_SUM<DODE::IRC,1>::value,DODE::IRC> {

	using Base = VectorFunction<Integrator<DODE>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
	DENSE_FUNCTION_BASE_TYPES(Base);


	template <class Scalar>
	using ODEState = typename DODE::template Input<Scalar>;
	template <class Scalar>
	using ODEDeriv = typename DODE::template Output<Scalar>;

	/// <summary>
	/// The type for the differentiable stepper function.
	/// Psuedo ODE is a compostion of the ode and control function(if any)
	/// </summary>
	/// <typeparam name="PseudoODE"></typeparam>
	template<class PseudoODE, RKOptions RKOp>
	using StepperType = RKStepper_NEW< PseudoODE, RKOp>;

	/// <summary>
	/// Wraps stepper types with RKoptions types
	/// </summary>
	using StepperWrapperType    = GenericFunction<SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
	using ControllerType        = GenericFunction<-1, -1>;
	using StopFuncType             = GenericConditional<-1>;

	DODE ode;
	bool usecontroller = false;
	ControllerType ucon;
	StepperWrapperType stepper;
	StopFuncType defaultstop;
	RKOptions RKMethod = RKOptions::DOPRI54;

	std::shared_ptr<ctpl::ThreadPool> pool;

	void setPoolThreads(int thrs) {
		if (this->pool->size() < thrs) {
			this->pool->resize(thrs);
		}
	}
	Integrator() {
		this->pool = std::make_shared<ctpl::ThreadPool>();
		this->defaultstop = ConstantConditional(this->ode.IRows(), false);
	}
	
	Integrator(std::string str, const DODE & dode, double defstep):Integrator() {

		this->ode = dode;
		this->setIORows(this->ode.IRows() + 1, this->ode.IRows());

		Eigen::VectorXi empty;

		this->setMethod(str,true,ControllerType(),empty);
		this->setAbsTol(1.0e-12);
		this->setStepSizes(defstep, defstep / 1000, defstep * 1000);
	}
	
	void setMethod(std::string str, bool nocontrol,
		const GenericFunction<-1, -1>& ucon,
		const Eigen::VectorXi& varlocs) {

		if (str == "DOPRI54") {
			this->RKMethod = RKOptions::DOPRI54;
			this->ErrorOrder = 4;
			this->stepper = this->MakeStepper<RKOptions::DOPRI5>(this->ode, nocontrol, ucon, varlocs);
		}
		else if (str == "DOPRI87") {
			this->RKMethod = RKOptions::DOPRI87;
			this->ErrorOrder = 7;
			this->stepper = this->MakeStepper<RKOptions::DOPRI87>(this->ode, nocontrol, ucon, varlocs);
		}
	}


	template <RKOptions RKOp>
	static auto MakeStepper(const DODE& ode, bool nocontrol,
		const GenericFunction<-1, -1>& ucon,
		const Eigen::VectorXi& varlocs) {

		auto Stepper = StepperType<DODE, RKOp>(ode);
		constexpr int IRC = decltype(Stepper)::IRC;
		constexpr int DUV = (DODE::UV == 1) ? -1 : DODE::UV;
		if constexpr (DODE::UV == 0) {
			return StepperWrapperType(Stepper);

		}
		else {
			if (nocontrol) {
				return StepperWrapperType(Stepper);
			}
			else {

				if (ucon.ORows() != ode.UVars()) {
					throw std::invalid_argument("Controller output size does not match number of ode control variables");
				}

				if (ucon.IRows() != varlocs.size()) {
					throw std::invalid_argument("Controller input size is inconsistent with specified number of input state variables");
				}


				if constexpr (DODE::PV == 0) {
					Arguments<IRC> stepperargs(Stepper.IRows());
					Arguments<DODE::IRC> odeargs(ode.IRows());

					ParsedInput<GenericFunction<-1, -1>, IRC, DUV> stepfunc(ucon, varlocs, Stepper.IRows());
					ParsedInput<GenericFunction<-1, -1>, DODE::IRC, DUV> odefunc(ucon, varlocs, ode.IRows());

					auto ODEargs = StackedOutputs{ odeargs.template head<DODE::XtV>(ode.XtVars()),odefunc };
					auto StepArgs = StackedOutputs{ stepperargs.template head<DODE::XtV>(ode.XtVars()),stepfunc, stepperargs.template tail<1>() };

					auto ODEexpr = NestedFunction<DODE, decltype(ODEargs)>(ode, ODEargs);
					auto GenOde = GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(ODEexpr, ode.XVars(), ode.UVars(), ode.PVars());
					auto Stepper2 = StepperWrapperType(StepperType<decltype(GenOde), RKOp>(GenOde)).eval(StepArgs);

					auto Stepper3 = StepperWrapperType(ODEargs.eval(StepperType<decltype(GenOde), RKOp>(GenOde)));

					return StepperWrapperType(Stepper3);
				}
				else {


					Arguments<IRC> stepperargs(Stepper.IRows());
					Arguments<DODE::IRC> odeargs(ode.IRows());

					ParsedInput<GenericFunction<-1, -1>, IRC, DUV> stepfunc(ucon, varlocs, Stepper.IRows());
					ParsedInput<GenericFunction<-1, -1>, DODE::IRC, DUV> odefunc(ucon, varlocs, ode.IRows());

					auto ODEargs = StackedOutputs{ odeargs.template head<DODE::XtV>(ode.XtVars()),odefunc,odeargs.template tail<-1>(ode.PVars()) };
					auto StepArgs = StackedOutputs{ stepperargs.template head<DODE::XtV>(ode.XtVars()),stepfunc, stepperargs.template tail<-1>(ode.PVars() + 1) };

					auto ODEexpr = NestedFunction<DODE, decltype(ODEargs)>(ode, ODEargs);
					auto GenOde = GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(ODEexpr, ode.XVars(), ode.UVars(), ode.PVars());
					auto Stepper2 = StepperWrapperType(StepperType<decltype(GenOde), RKOp>(GenOde)).eval(StepArgs);

					return StepperWrapperType(Stepper2);

				}

				
			}
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////
	

	double ErrorOrder = 5.0;
	double MinStepSize = 0.1;
	double DefStepSize = 0.1;
	double MaxStepSize = 0.1;
	double MaxStepChange = 2.0;
	bool Adaptive = true;
	bool FastAdaptiveSTM = true;

	Vector<double, DODE::XV> AbsTols;
	Vector<double, DODE::XV> RelTols;

	void setAbsTol(double tol) {
		this->AbsTols.setConstant(this->ode.XVars(), abs(tol));
	}

	void setStepSizes(double defstep, double minstep, double maxstep) {
		if (defstep < minstep) {
			throw::std::invalid_argument("Default integrator stepsize must be greater than minimum stepsize.");
		}
		if (defstep > maxstep) {
			throw::std::invalid_argument("Default integrator stepsize must be less maximum stepsize.");
		}
		if (minstep > maxstep) {
			throw::std::invalid_argument("Minimum integrator stepsize must be greater than minimum stepsize.");
		}
		if (defstep < 0 || minstep < 0 || maxstep < 0) {
			throw::std::invalid_argument("Stepsizes must be positive numbers (this doesnt mean you cant integrate backwards).");
		}

		this->DefStepSize = defstep;
		this->MinStepSize = minstep;
		this->MaxStepSize = maxstep;

	}

	/////////////////////////////////////////////////////////////////////////////////////

	template<class State>
	void update_control(State& xtup) const {
		if constexpr (DODE::UV != 0) {
			if (this->usecontroller) {
				this->ucon.compute(xtup,
					xtup.template segment<DODE::UV>(this->ode.TVar() + 1, this->ode.UVars()));
			}
		}
	}

	template <RKOptions RKOp>
	inline void stepper_compute_impl(const ODEState<double> & x,double tf, ODEState<double> & fx, 
		bool doestimate,ODEState<double>& fx_est, bool usexdot0,bool fillxdot0) const {
		
		using RKData = RKCoeffs<RKOp>;
		constexpr int Stages = RKData::Stages;
		constexpr int Stgsm1 = RKData::Stages - 1;
		constexpr bool isDiag = RKData::isDiag;


		auto Impl = [&](auto& Kvals, auto& xtup) {

			xtup = x;
			double t0 = xtup[this->ode.TVar()];
			double h = tf - t0;

			if (usexdot0) {
				//Kvals[0]=xdot0*h
			}
			else {
				this->update_control(xtup);
				this->ode.compute(xtup, Kvals[0]);
				Kvals[0] *= h;
			}
			
			for (int i = 0; i < Stgsm1; i++) {
				double ti = t0 + RKData::Times[i] * h;
				xtup = x;
				xtup[this->ode.TVar()] = ti;
				const int ip1 = i + 1;
				const int js = isDiag ? i : 0;
				for (int j = js; j < ip1; j++) {
					xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
						RKData::ACoeffs[i][j] * Kvals[j];
				}

				this->update_control(xtup);

				this->ode.compute(xtup, Kvals[ip1]);

				Kvals[ip1] *= h;
			}
			xtup = x;
			xtup[this->ode.TVar()] = tf;
			for (int i = 0; i < Stages; i++) {
				xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
					RKData::BCoeffs[i] * Kvals[i];
			}


			this->update_control(xtup);

			fx = xtup;

			if (doestimate) {
				xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
				for (int i = 0; i < Stages; i++) {
					xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
						RKData::CCoeffs[i] * Kvals[i];
				}

				this->update_control(xtup);

				fx_est = xtup;
			}
			if (fillxdot0) {
				//xdot0 = Kvals.back() * (1.0 / h);
			}
		};


		MemoryManager::allocate_run(this->ode.IRows(), Impl,
			ArrayOfTempSpecs<ODEDeriv<double>, Stages>(this->ode.ORows(), 1),
			TempSpec<ODEState<double>>(this->ode.IRows(), 1));

		
	}

	void stepper_compute(const ODEState<double>& x, double tf, ODEState<double>& fx, ODEState<double>& fx_est, bool doestimate) const{

		switch (this->RKMethod) {
		case RKOptions::DOPRI54: {
			this->stepper_compute_impl<RKOptions::DOPRI54>(x, tf, fx, doestimate, fx_est, false, false);
		}break;
		case RKOptions::DOPRI87: {
			this->stepper_compute_impl<RKOptions::DOPRI87>(x, tf, fx, doestimate, fx_est, false, false);
		}break;
		default: {

		}
		}


	}

	std::vector<ODEState<double>> integrate_impl(const Input<double>& x, double tf, Output<double>& fx, 
		const StopFuncType& stopfunc,
		bool storestates) const {

		

		double t0     = x[this->ode.TVar()];
		double H      = tf - t0;
		int numsteps  = int(abs(H / this->DefStepSize)) + 1;
		double h      = H / double(numsteps);

		Input<double> xi = x;
		this->update_control(xi); 

		Input<double> xnext = xi;
		Input<double> xnext_est = xi;

		std::vector<ODEState<double>> states;

		if (storestates) {
			states.reserve(numsteps + 1);
			states.push_back(xi);
		}

		Vector<double, DODE::XV> Abserror;
		Vector<double, DODE::XV> Abserror_max;

		bool HitMinimum = false;
		int MinimumCount = 0;
		int i = 0;
		bool continueloop = true;

		while (continueloop) {

			double tnext = xi[this->ode.TVar()] + h;

			if (H > 0.0) {
				if ((tnext - tf) >= 0.0) {
					h = tf - xi[this->ode.TVar()];
					tnext = tf;
					continueloop = false;
				}
			}
			else {
				if ((tnext - tf) <= 0.0) {
					h = tf - xi[this->ode.TVar()];
					tnext = tf;
					continueloop = false;
				}
			}

			this->stepper_compute(xi, tnext, xnext, xnext_est, this->Adaptive);

			if (this->Adaptive) {
				Abserror =
					(xnext.head(this->ode.XVars()) - xnext_est.head(this->ode.XVars()))
					.cwiseAbs();

				Abserror_max = Abserror.cwiseQuotient(this->AbsTols);
				int worst = 0;
				Abserror_max.maxCoeff(&worst);

				double err = Abserror[worst];
				double acc = this->AbsTols[worst];
				double hnext = h * pow((acc / err), 1.0 / this->ErrorOrder);

				if (hnext / h > this->MaxStepChange)
					h *= this->MaxStepChange;
				else if (hnext / h < 1. / this->MaxStepChange)
					h /= this->MaxStepChange;
				else
					h = hnext;

				if (abs(h) > this->MaxStepSize)
					h = this->MaxStepSize * h / abs(h);

				if (abs(h) < this->MinStepSize) {
					h = this->MinStepSize * h / abs(h);
					HitMinimum = true;
					MinimumCount++;
				}
				else {
					HitMinimum = false;
				}
				if ((err - acc) > 0 && !HitMinimum) {
					continue;
				}
			}

			xi = xnext;
			if (storestates) {
				states.push_back(xi);
			}
			
			if (stopfunc.compute(xi)) break;
			
			i++;
		}
		fx = xi;
		return states;
	}


	/////////////////////////////////////////////////////////////////////////////////////

	ODEState<double> integrate(const ODEState<double>& x0, double tf) const {
		ODEState<double> xf(this->ode.IRows());
		xf.setZero();
		integrate_impl(x0, tf, xf, this->defaultstop, false);
		return xf;
	}
	ODEState<double> integrate(const ODEState<double>& x0, double tf, const StopFuncType& stopfunc) const {
		ODEState<double> xf(this->ode.IRows());
		xf.setZero();
		integrate_impl(x0, tf, xf, stopfunc, false);
		return xf;
	}

	std::vector<ODEState<double>> integrate_parallel(
		const std::vector<ODEState<double>>& x0, std::vector<double> tf,
		int thrs,
		const StopFuncType& stopfunc) {
		using RetType = std::vector<ODEState<double>>;
		this->setPoolThreads(thrs);

		int n = x0.size();

		auto xf_op = [&](int id, int start, int stop) {
			RetType xfs(stop - start);
			for (int i = start; i < stop; i++) {
				xfs[i - start] = this->integrate(x0[i], tf[i], stopfunc);
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
	    int thrs)  {
		return this->integrate_parallel(x0, tf, thrs, this->defaultstop);
	}

	std::vector<ODEState<double>> integrate_dense(const ODEState<double>& x0, double tf, const StopFuncType& stopfunc) const{
		ODEState<double> xf(this->ode.IRows());
		xf.setZero();
		return integrate_impl(x0, tf, xf, stopfunc, true);
	}

	std::vector<ODEState<double>> integrate_dense(const ODEState<double>& x0, double tf) const{
		ODEState<double> xf(this->ode.IRows());
		xf.setZero();
		return integrate_impl(x0, tf, xf, this->defaultstop, true);
	}


	std::tuple<ODEState<double>, Jacobian<double>> integrate_stm(const ODEState<double>& x0, double tf, const StopFuncType& stopfunc) const {

		ODEState<double> xf(this->ode.IRows());
		xf.setZero();

		

		Jacobian<double> jx(this->IRows(), this->ORows());
		jx.setZero();

		Eigen::Matrix<double, Base::IRC, Base::IRC> jxall(this->IRows(),this->IRows());
		jxall.setIdentity();

		Input<double> stepper_input(this->IRows());
		ODEState<double> stepper_output(this->ode.IRows());
		Jacobian<double> stepper_jacobian(this->ORows(), this->IRows());

		auto xs = integrate_impl(x0, tf, xf, stopfunc, true);



		for (int i = 0; i < xs.size() - 1; i++) {
			stepper_input.head(this->ode.IRows()) = xs[i];
			stepper_input[this->ode.IRows()] = xs[i + 1][this->ode.TVar()];

			stepper_output.setZero();
			stepper_jacobian.setZero();

			this->stepper.compute_jacobian(stepper_input, stepper_output, stepper_jacobian);
			jxall.template topRows<Base::ORC>(this->ORows()) = stepper_jacobian * jxall;

		}

		jx = jxall.template topRows<Base::ORC>(this->ORows());

		return std::tuple{ xf,jx };
	}

	std::tuple<ODEState<double>, Jacobian<double>> integrate_stm(const ODEState<double>& x0, double tf) const {
		return this->integrate_stm(x0, tf, this->defaultstop);
	}

	std::tuple<ODEState<double>, Jacobian<double>,Hessian<double>> 
		integrate_stm_hessian(const ODEState<double>& x0, double tf, const ODEState<double>& lf) const {

		ODEState<double> xf(this->ode.IRows());
		xf.setZero();



		Jacobian<double> jx(this->IRows(), this->ORows());
		jx.setZero();

		Eigen::Matrix<double, Base::IRC, Base::IRC> jxall(this->IRows(), this->IRows());
		jxall.setIdentity();

		Eigen::Matrix<double, Base::IRC, Base::IRC> hxall(this->IRows(), this->IRows());
		hxall.setZero();

		Input<double> stepper_input(this->IRows());
		Input<double> stepper_grad(this->IRows());

		ODEState<double> stepper_output(this->ode.IRows());
		Jacobian<double> stepper_jacobian(this->ORows(), this->IRows());
		Hessian<double> stepper_hessian(this->ORows(), this->IRows());

		ODEState<double> stepper_adjvars = lf;

		Hessian<double> jtwist(this->IRows(), this->IRows());
		jtwist.setZero();
		jtwist(this->IRows() - 1, this->IRows() - 1) = 1.0;



		auto xs = integrate_impl(x0, tf, xf, this->defaultstop, true);

		int numsteps = xs.size()-1;


		for (int i = 0; i < numsteps; i++) {
			stepper_input.head(this->ode.IRows()) = xs[numsteps - i - 1];
			stepper_input[this->ode.IRows()] = xs[numsteps - i][this->ode.TVar()];

			stepper_output.setZero();
			stepper_jacobian.setZero();
			stepper_grad.setZero();
			stepper_hessian.setZero();

			this->stepper.compute_jacobian_adjointgradient_adjointhessian(
				stepper_input, stepper_output, stepper_jacobian, stepper_grad, stepper_hessian, stepper_adjvars);
			jxall.template topRows<Base::ORC>(this->ORows()) = stepper_jacobian * jxall;


			jtwist.topRows(this->ORows()) = stepper_jacobian;
			jxall = jxall * jtwist;

			if (i == 0) {
				jxall.rightCols(1) = stepper_jacobian.rightCols(1);
			}
			hxall = jtwist.transpose() * hxall * jtwist;
			hxall += stepper_hessian;
			stepper_adjvars = stepper_grad.head(this->ORows());

		}

		jx = jxall.template topRows<Base::ORC>(this->ORows());

		return std::tuple{ xf,jx,hxall };
	}



	/////////////////////////////////////////////////////////////////////////////////////

	template <class InType, class OutType>
	inline void compute_impl(ConstVectorBaseRef<InType> x,
		ConstVectorBaseRef<OutType> fx_) const {
		typedef typename InType::Scalar Scalar;
		VectorBaseRef<OutType> fx = fx_.const_cast_derived();
	}
	template <class InType, class OutType, class JacType>
	inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
		ConstVectorBaseRef<OutType> fx_,
		ConstMatrixBaseRef<JacType> jx_) const {
		typedef typename InType::Scalar Scalar;
		VectorBaseRef<OutType> fx = fx_.const_cast_derived();
		MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
	}
	template <class InType, class OutType, class JacType, class AdjGradType,
		class AdjHessType, class AdjVarType>
		inline void compute_jacobian_adjointgradient_adjointhessian_impl(
			ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
			ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
			ConstMatrixBaseRef<AdjHessType> adjhess_,
			ConstVectorBaseRef<AdjVarType> adjvars) const {
		typedef typename InType::Scalar Scalar;
		VectorBaseRef<OutType> fx = fx_.const_cast_derived();
		MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
		VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
		MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();
	}


	/////////////////////////////////////////////////////////////////////////////////////

	static void Build(py::module& m, const char* name) {
		auto obj =
			py::class_<Integrator>(m, name).def(py::init<std::string, const DODE&, double>());
		

		obj.def("integrate_dense",
			(std::vector<ODEState<double>>(Integrator::*)(const ODEState<double>&,
				double) const) &
			Integrator::integrate_dense);

		obj.def("integrate_stm",
			(std::tuple<ODEState<double>, Jacobian<double>>(Integrator::*)(
				const ODEState<double>&, double) const) &
			Integrator::integrate_stm);
		obj.def("integrate_stm_hessian",
			(std::tuple<ODEState<double>, Jacobian<double>,Hessian<double>>(Integrator::*)(
				const ODEState<double>&, double, const ODEState<double>&) const) &
			Integrator::integrate_stm_hessian);

		obj.def_readwrite("DefStepSize", &Integrator::DefStepSize);
		obj.def_readwrite("MaxStepSize", &Integrator::MaxStepSize);
		obj.def_readwrite("MinStepSize", &Integrator::MinStepSize);
		obj.def_readwrite("MaxStepChange", &Integrator::MaxStepChange);
		obj.def_readwrite("Adaptive", &Integrator::Adaptive);
		obj.def_readwrite("AbsTols", &Integrator::AbsTols);
		obj.def("setAbsTol", &Integrator::setAbsTol);

		
	}


};



}
