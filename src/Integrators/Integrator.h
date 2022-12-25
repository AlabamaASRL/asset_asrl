#pragma once
#include "OptimalControl/LGLInterpTable.h"
#include "OptimalControl/LGLInterpFunctions.h"

#include "VectorFunctionTypeErasure/GenericFunction.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"

#include "RKSteppers.h"
#include "pch.h"
#include <sstream>

namespace ASSET {

	template <class BaseType, int _XV, int _UV, int _PV>
	struct GenericODE;
	template<int OR>
	struct InterpFunction;


	


template<class DODE>
struct Integrator:VectorFunction<Integrator<DODE>,SZ_SUM<DODE::IRC,1>::value,DODE::IRC> {

	using Base = VectorFunction<Integrator<DODE>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
	DENSE_FUNCTION_BASE_TYPES(Base);


	template <class Scalar>
	using ODEState = typename DODE::template Input<Scalar>;
	template <class Scalar>
	using ODEDeriv = typename DODE::template Output<Scalar>;


	using EventPack     = std::tuple<GenericFunction<-1, 1>, int, int>;
	using EventLocsType = std::vector<std::vector<ODEState<double>>>;



	/// <summary>
	/// The type for the differentiable stepper function.
	/// Psuedo ODE is a compostion of the ode and control function(if any)
	/// </summary>
	/// <typeparam name="PseudoODE"></typeparam>
	template<class PseudoODE, RKOptions RKOp>
	using StepperType = RKStepper< PseudoODE, RKOp>;

	/// <summary>
	/// Wraps stepper types with RKoptions types
	/// </summary>
	using StepperWrapperType    = GenericFunction<SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
	using ControllerType        = GenericFunction<-1, -1>;
	using StopFuncType             = GenericConditional<-1>;


	using IntegRet = ODEState<double>;
	using DenseRet = std::vector<ODEState<double>>;
	using STMRet = std::tuple< IntegRet, Jacobian<double> >;

	using IntegEventRet = std::tuple< IntegRet, EventLocsType >;
	using DenseEventRet = std::tuple< DenseRet, EventLocsType >;
	using STMEventRet = std::tuple< IntegRet, Jacobian<double>, EventLocsType >;


protected:
	DODE ode;
	bool usecontroller = false;
	ControllerType controller;
	StepperWrapperType stepper;
	RKOptions RKMethod = RKOptions::DOPRI54;
	std::shared_ptr<ctpl::ThreadPool> pool;
	

	void setPoolThreads(int thrs) {
		if (this->pool->size() < thrs) {
			this->pool->resize(thrs);
		}
	}
public:

	Integrator() {
		this->pool = std::make_shared<ctpl::ThreadPool>();
	}
	
	Integrator(const DODE & dode, std::string meth,double defstep):Integrator() {
		Eigen::VectorXi empty;
		
		this->setMethod(meth,dode,defstep,false,ControllerType(),empty);
		this->setAbsTol(1.0e-12); // Must Be called after setMethod!!!
		this->setRelTol(0); // Must Be called after setMethod!!!
	}
	Integrator(const DODE& dode, double defstep) :Integrator(dode,"DOPRI87",defstep) {

	}
	Integrator(const DODE& dode, std::string meth, double defstep, const ControllerType& ucon,const Eigen::VectorXi& varlocs)
		:Integrator() {
		this->setMethod(meth, dode, defstep, true, ucon, varlocs);
		this->setAbsTol(1.0e-12); // Must Be called after setMethod!!!
		this->setRelTol(0); // Must Be called after setMethod!!!
	}
	Integrator(const DODE& dode, double defstep, const ControllerType& ucon,const Eigen::VectorXi& varlocs) 
		:Integrator(dode, "DOPRI87", defstep, ucon, varlocs) {

	}
	Integrator(const DODE& dode, double defstep, const Eigen::VectorXd& v)
		:Integrator() {

		Eigen::VectorXi tloc(1);
		tloc[0] = dode.TVar();
		GenericFunction<-1, -1> ucon = Constant<-1, -1>(1, v);
		this->setMethod("DOPRI87", dode, defstep, true, ucon, tloc);
		this->setAbsTol(1.0e-12); // Must Be called after setMethod!!!
		this->setRelTol(0); // Must Be called after setMethod!!!
	}
	Integrator(const DODE& dode, std::string meth, double defstep, std::shared_ptr<LGLInterpTable> tab,const Eigen::VectorXi& ulocs)
		:Integrator() {

		Eigen::VectorXi varlocs(1);
		varlocs[0] = dode.TVar();
		ControllerType ucon = InterpFunction<-1>(tab, ulocs);
		this->setMethod(meth, dode, defstep, true, ucon, varlocs);
		this->setAbsTol(1.0e-12); // Must Be called after setMethod!!!
		this->setRelTol(0); // Must Be called after setMethod!!!

	}
	Integrator(const DODE& dode, double defstep, std::shared_ptr<LGLInterpTable> tab,const Eigen::VectorXi& ulocs)
		:Integrator(dode,"DOPRI87", defstep, tab, ulocs) {
	}

	Integrator(const DODE& dode, std::string meth, double defstep, std::shared_ptr<LGLInterpTable> tab)
		:Integrator() {

		// Bug waiting to happen when LGL interp table is re-factored
		if (dode.IRows() != tab->XtUVars||dode.XVars()!=tab->XVars) {
			throw std::invalid_argument("Table data does not match expected dimension of ODE."
				" Please provide the indices variables in the table you want to interpret as controls.\n");
		}
		Eigen::VectorXi ulocs;
		ulocs.setLinSpaced(dode.UVars(), dode.TVar() + 1, dode.TVar() + dode.UVars());


		Eigen::VectorXi varlocs(1);
		varlocs[0] = dode.TVar();
		ControllerType ucon = InterpFunction<-1>(tab, ulocs);
		this->setMethod(meth, dode, defstep, true, ucon, varlocs);
		this->setAbsTol(1.0e-12); // Must Be called after setMethod!!!
		this->setRelTol(0); // Must Be called after setMethod!!!

	}
	Integrator(const DODE& dode, double defstep, std::shared_ptr<LGLInterpTable> tab)
		:Integrator(dode, "DOPRI87", defstep, tab) {
	}

	void setMethod(std::string str, const DODE& dode, double defstep, bool usecontrol,
		const GenericFunction<-1, -1>& ucon,
		const Eigen::VectorXi& varlocs) {

		this->setStepSizes(defstep, defstep / 10000, defstep * 10000);

		if (str == "DOPRI54"||str =="DP54") {
			this->RKMethod = RKOptions::DOPRI54;
			this->ErrorOrder = 4;
			// Using DOPRI5 rather than DOPRI54 here is not a mistake
			this->initStepperAndController<RKOptions::DOPRI5>(dode, usecontrol, ucon, varlocs);
		}
		else if (str == "DOPRI87"||str=="DP87") {
			this->RKMethod = RKOptions::DOPRI87;
			this->ErrorOrder = 7;
			this->initStepperAndController<RKOptions::DOPRI87>(dode, usecontrol, ucon, varlocs);
		}
		else {
			throw std::invalid_argument("Invalid integration method '{0:}'.");
		}

		

	}


	template <RKOptions RKOp>
	void initStepperAndController(const DODE& odet, bool usecontrol,
		const GenericFunction<-1, -1>& ucon,
		const Eigen::VectorXi& varlocs) {

		this->ode = odet;
		this->setIORows(this->ode.IRows() + 1, this->ode.IRows()); 

		this->usecontroller = usecontrol;

		auto Stepper = StepperType<DODE, RKOp>(ode);
		constexpr int IRC = decltype(Stepper)::IRC;
		constexpr int DUV = (DODE::UV == 1) ? -1 : DODE::UV;
		if constexpr (DODE::UV == 0) {
			this->stepper =  StepperWrapperType(Stepper);
			this->controller = Arguments<-1>(ode.UVars());
		}
		else {
			if (!this->usecontroller) {
				this->stepper = StepperWrapperType(Stepper);
				this->controller = Arguments<-1>(ode.UVars());

			}
			else {

				if (ucon.ORows() != ode.UVars()) {
					throw std::invalid_argument("Controller output size does not match number of ode control variables");
				}

				if (ucon.IRows() != varlocs.size()) {
					throw std::invalid_argument("Controller input size is inconsistent with specified number of input state variables");
				}

				Arguments<DODE::IRC> odeargs(ode.IRows());
				ParsedInput<GenericFunction<-1, -1>, DODE::IRC, DUV> controllerfunc(ucon, varlocs, ode.IRows());
				this->controller = controllerfunc;

				if constexpr (DODE::PV == 0) {
					
					auto ODEargs = StackedOutputs{ odeargs.template head<DODE::XtV>(ode.XtVars()),controllerfunc };
					auto ODEexpr = NestedFunction<DODE, decltype(ODEargs)>(ode, ODEargs);
					auto GenOde = GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(ODEexpr, ode.XVars(), ode.UVars(), ode.PVars());
					auto StepperU = ODEargs.eval(StepperType<decltype(GenOde), RKOp>(GenOde));
					this->stepper = StepperWrapperType(StepperU);
				}
				else {
					
					auto ODEargs = StackedOutputs{ odeargs.template head<DODE::XtV>(ode.XtVars()),controllerfunc,odeargs.template tail<-1>(ode.PVars()) };
					auto ODEexpr   = NestedFunction<DODE, decltype(ODEargs)>(ode, ODEargs);
					auto GenOde    = GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(ODEexpr, ode.XVars(), ode.UVars(), ode.PVars());
					auto StepperUP = ODEargs.eval(StepperType<decltype(GenOde), RKOp>(GenOde));
					this->stepper =  StepperWrapperType(StepperUP);

				}
			}
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////
	
	GenericFunction<-1, -1> getstepper() {
		return this->stepper;
	}


	double ErrorOrder = 7.0;
	double MinStepSize = 0.1;
	double DefStepSize = 0.1;
	double MaxStepSize = 0.1;
	double MaxStepChange = 3.0;
	bool Adaptive = true;
	bool FastAdaptiveSTM = true;
	double EventTol = 1.0e-6;
	int MaxEventIters = 10;


	ODEDeriv<double> AbsTols;
	ODEDeriv<double> RelTols;

	void setAbsTol(double tol) {
		this->AbsTols.setConstant(this->ode.XVars(), abs(tol));
	}
	void setRelTol(double tol) {
		this->RelTols.setConstant(this->ode.XVars(), abs(tol));
	}

	void setAbsTols(ODEDeriv<double> tol) {
		if (tol.size() != this->ode.XVars()) {
			throw std::invalid_argument("Incorrectly sized tolerance vector.");
		}
		this->AbsTols=tol;
	}
	void setRelTols(ODEDeriv<double> tol) {
		if (tol.size() != this->ode.XVars()) {
			throw std::invalid_argument("Incorrectly sized tolerance vector.");
		}
		this->RelTols=tol;
	}

	ODEDeriv<double> getAbsTols() const {
		
		return this->AbsTols;
	}
	ODEDeriv<double> getRelTols() const {
		
		return this->RelTols;
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

protected:

	template<class State>
	void update_control(State& xtup) const {
		if constexpr (DODE::UV != 0) {
			if (this->usecontroller) {
				this->controller.compute(xtup,
					xtup.template segment<DODE::UV>(this->ode.TVar() + 1, this->ode.UVars()));
			}
		}
	}

	template <RKOptions RKOp>
	inline void stepper_compute_impl( 
		const ODEState<double>& x, double tf, ODEState<double>& xf, ODEState<double>& xf_est,
		bool dofsal,     ODEDeriv<double>& xdot_prev,
		bool domidpoint, ODEState<double>& xf_mid) const {

		using RKData = RKCoeffs<RKOp>;
		constexpr int Stages = RKData::Stages;
		constexpr int Stgsm1 = RKData::Stages - 1;
		constexpr bool isDiag = RKData::isDiag;

		auto Impl = [&](auto& Kvals, auto& xtup) {

			xtup = x;
			double t0 = xtup[this->ode.TVar()];
			double h = tf - t0;

			if (dofsal||domidpoint) {
				Kvals[0] = xdot_prev * h;
			}
			else {
				this->update_control(xtup);
				this->ode.compute(xtup, Kvals[0]);
				Kvals[0] *= h;
			}

			if constexpr (true) {
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
			}
			else {
				const int tvar = this->ode.TVar();

				ASSET::constexpr_for_loop(
					std::integral_constant<int, 0>(),
					std::integral_constant<int, Stgsm1>(), [&](auto i) {
						double ti = t0 + RKData::Times[i.value] * h;
						xtup = x;
						xtup[tvar] = ti;
						constexpr int ip1 = i.value + 1;
						const int js = isDiag ? i.value : 0;

						ASSET::constexpr_for_loop(
							std::integral_constant<int, 0>(),
							std::integral_constant<int, ip1>(), [&](auto j) {
								if constexpr (RKData::ACoeffs[i.value][j.value] != 0.0) {
									xtup.template segment<DODE::XV>(0, tvar) +=
										RKData::ACoeffs[i.value][j.value] * Kvals[j.value];
								}
								
							});

						this->update_control(xtup);
						this->ode.compute(xtup, Kvals[ip1]);

						Kvals[ip1] *= h;
					});


			}


			xtup = x;
			xtup[this->ode.TVar()] = tf;
			for (int i = 0; i < Stages; i++) {
				xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
					RKData::BCoeffs[i] * Kvals[i];
			}


			this->update_control(xtup);
			xf = xtup; // Next State

			if (dofsal||domidpoint) {
				if constexpr (RKData::FSAL) {
					xdot_prev = Kvals.back() * (1.0 / h);
				}
			}

			xtup = x;
			xtup[this->ode.TVar()] = tf;

			for (int i = 0; i < Stages; i++) {
				xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
					RKData::CCoeffs[i] * Kvals[i];
			}

			xf_est = xtup; // Estimate
			

			if (domidpoint) {

				xtup = x;
				xtup[this->ode.TVar()] = t0+h/2.0;

				for (int i = 0; i < Stages; i++) {
					xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
						(RKData::MidCoeffs[i]/2.0) * Kvals[i];
				}

				if constexpr (!RKData::FSAL) {
					Kvals.back().setZero();
					this->ode.compute(xf, Kvals.back());
					xtup.template segment<DODE::XV>(0, this->ode.XVars()) +=
						(RKData::MidCoeffs.back()/2.0) * Kvals.back() * h;
					xdot_prev = Kvals.back();
				}

				this->update_control(xtup);

				xf_mid = xtup;

			}

		};

		MemoryManager::allocate_run(this->ode.IRows(), Impl,
			ArrayOfTempSpecs<ODEDeriv<double>, Stages>(this->ode.ORows(), 1),
			TempSpec<ODEState<double>>(this->ode.IRows(), 1));

	}



	inline void stepper_compute(
		const ODEState<double>& x, double tf, 
		ODEState<double>& xf, ODEState<double>& xf_est,
		ODEDeriv<double>& xdot_prev,
		bool domidpoint, ODEState<double>& xf_mid) const {


		switch (this->RKMethod) {
		case RKOptions::DOPRI54: {
			this->stepper_compute_impl<RKOptions::DOPRI54>(x, tf, xf, xf_est, true, xdot_prev, domidpoint, xf_mid);
		}break;
		case RKOptions::DOPRI87: {
			this->stepper_compute_impl<RKOptions::DOPRI87>(x, tf, xf, xf_est, false, xdot_prev, domidpoint, xf_mid);
		}break;
		default: {
		}
		}
		

	}

	Output<double> integrate_impl(
		const ODEState<double>& x,
		double tf,
		const std::vector<EventPack> & events,
		std::vector<std::vector<Eigen::Vector2d>> & eventtimes,
		bool storestates,
		bool storederivs,
		bool storemidpoints,
		std::vector<ODEState<double>> & states,
		std::vector<ODEDeriv<double>> & derivs) const {

		if (x.size() != this->ode.IRows()) {
			throw std::invalid_argument("Incorrectly sized input state.");
		}

		double t0 = x[this->ode.TVar()];
		double H = tf - t0;
		int numsteps = int(abs(H / this->DefStepSize)) + 1;
		double h = .9*(H / double(numsteps));
		
		ODEState<double> xi = x;
		this->update_control(xi);

		ODEState<double> xnext = xi;
		ODEState<double> xnext_est = xi;
		ODEState<double> xnext_mid = xi;

		ODEDeriv<double> xdoti(this->ode.ORows());
		xdoti.setZero();
		this->ode.compute(xi, xdoti);
		ODEDeriv<double> xdotnext = xdoti;



		std::vector<Vector1<double>> prev_event_vals(events.size());
		std::vector<Vector1<double>> next_event_vals(events.size());

		for (int j = 0; j < events.size();j++) {
			prev_event_vals[j].setZero();
			next_event_vals[j].setZero();

			if (std::get<0>(events[j]).IRows()!=this->ode.IRows()) {
				throw std::invalid_argument("Input size of event function must equal input size of ode.");
			}

			std::get<0>(events[j]).compute(xi, prev_event_vals[j]);
		}

		eventtimes.resize(events.size());

		if (storestates) {
			states.resize(0);
			derivs.resize(0);
			if (storemidpoints) {
				states.reserve(numsteps*2 + 2);
				if (storederivs) derivs.reserve(numsteps * 2 + 2);
			}
			else {
				
				states.reserve(numsteps + 2);
				
				
				
				if (storederivs) derivs.reserve(numsteps + 2);

			}
			states.push_back(xi);
			if (storederivs) derivs.push_back(xdoti);
		}

		ODEDeriv<double> Abserror;
		ODEDeriv<double> Abserror_max;
		ODEDeriv<double> Errvec;

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

			xnext.setZero();
			xnext_est.setZero();
			xnext_mid.setZero();
			xdotnext = xdoti;

			
			this->stepper_compute(xi, tnext, xnext, xnext_est, xdotnext, storemidpoints||storederivs, xnext_mid);

			
			if (this->Adaptive) {
				Abserror =
					(xnext.head(this->ode.XVars()) - xnext_est.head(this->ode.XVars()))
					.cwiseAbs();

				Errvec = this->AbsTols + xnext.head(this->ode.XVars()).cwiseAbs().cwiseProduct(this->RelTols);

				Abserror_max = Abserror.cwiseQuotient(Errvec);
				int worst = 0;
				Abserror_max.maxCoeff(&worst);

				double err = Abserror[worst];
				double acc = Errvec[worst];
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
					continueloop = true;
					continue;
				}
			}


			bool eventbreak = false;
			for (int j = 0; j < events.size(); j++) {
				next_event_vals[j].setZero();
				std::get<0>(events[j]).compute(xnext, next_event_vals[j]);

				double vprev = prev_event_vals[j][0];
				double vnext = next_event_vals[j][0];

				int dir = std::get<1>(events[j]);

				double vprod = vprev * vnext;

				if (vprod < 0.0) {
					if ((dir > 0 && vnext>0) || (dir < 0 && vnext < 0)||dir==0) {
						Eigen::Vector2d times;
						times[0] = xi[this->ode.TVar()];
						times[1] = xnext[this->ode.TVar()];
						eventtimes[j].push_back(times);
						int stop = std::get<2>(events[j]);

						if (stop != 0) { 
							if (eventtimes[j].size() == stop) {
								eventbreak = true;
							}
						}
					}
				}
			}

			

			xi = xnext;
			xdoti = xdotnext;
			prev_event_vals = next_event_vals;

			if (storestates) {
				if (storemidpoints) {
					states.push_back(xnext_mid);
					if (storederivs) {
						xdotnext.setZero();
						this->ode.compute(xnext_mid, xdotnext);
						derivs.push_back(xdotnext);
					}
				}
				states.push_back(xi);
				if (storederivs) derivs.push_back(xdoti);
			}

			if (eventbreak) break;
			i++;
		}
		return xi;
	}



	std::vector<std::vector<ODEState<double>>>
		find_events(std::shared_ptr<LGLInterpTable> tab,
			const std::vector<EventPack>& events,
			const std::vector<std::vector<Eigen::Vector2d>>& eventtimes) const{


		
		Eigen::VectorXi vars;
		vars.setLinSpaced(this->ode.IRows(), 0, this->ode.IRows() - 1);

		InterpFunction<-1> tabfunc(tab, vars);

		Vector1<double> x;
		Vector1<double> fx;
		Vector1<double> jx;

		std::vector<std::vector<ODEState<double>>> eventstates(events.size());

		for (int i = 0; i < events.size(); i++) {
			if (eventtimes[i].size() > 0) {

				auto func = std::get<0>(events[i]).eval(tabfunc);

				auto rootfind = [&](auto x0) {
					x[0] = x0;
					for (int k = 0; k < MaxEventIters; k++) {
						fx.setZero();
						jx.setZero();
						func.compute_jacobian(x, fx, jx);
						if (abs(fx[0]) < abs(EventTol)) {
							break;
						}
						x[0] = x[0] - fx[0] / jx[0];
					}
					return x[0];
				};

				for (auto & eventtime : eventtimes[i]) {


					double tlow  = eventtime[0];
					double thigh = eventtime[1];

					double x0 = (tlow + thigh) / 2.0;

					double tevent = rootfind(x0);

					if (tlow < thigh) {
						if (tevent > tlow && tevent < thigh) {
							ODEState<double> ei(this->ode.IRows());
							ei.setZero();
							tab->InterpolateRef(tevent, ei);
							eventstates[i].push_back(ei);
						}
					}
					else {
						if (tevent < tlow && tevent > thigh) {
							ODEState<double> ei(this->ode.IRows());
							ei.setZero();
							tab->InterpolateRef(tevent, ei);
							eventstates[i].push_back(ei);
						}
					}

				}


			}
		}

		return eventstates;
	}


	std::shared_ptr<LGLInterpTable> make_table(const std::vector<ODEState<double>>& Xs, bool fifthorder) const {
		std::vector<Eigen::VectorXd> Xsin;
		for (auto& X : Xs) {
			Xsin.push_back(X);
		}

		GenericFunction<-1, -1> odetemp;
		if constexpr (DODE::IsGenericODE) {
			odetemp = this->ode.func;
		}
		else {
			odetemp = this->ode;
		}
		TranscriptionModes m = fifthorder ? LGL5 : LGL3;
		std::shared_ptr<LGLInterpTable> tab = std::make_shared<LGLInterpTable>(odetemp,
			this->ode.XVars(),
			this->ode.UVars() + this->ode.PVars(), m);

		tab->loadExactData(Xsin);

		return tab;
	}

	std::shared_ptr<LGLInterpTable> make_table(const std::vector<ODEState<double>>& Xs, 
		const std::vector<ODEDeriv<double>>& dXs, bool fifthorder) const {
		
		GenericFunction<-1, -1> odetemp;
		if constexpr (DODE::IsGenericODE) {
			odetemp = this->ode.func;
		}
		else {
			odetemp = this->ode;
		}
		TranscriptionModes m = fifthorder ? LGL5 : LGL3;
		std::shared_ptr<LGLInterpTable> tab = std::make_shared<LGLInterpTable>(odetemp,
			this->ode.XVars(),
			this->ode.UVars() + this->ode.PVars(), m);

		tab->loadExactData(Xs,dXs);

		return tab;
	}

	std::vector<ODEState<double>> midpoints_removed(const std::vector<ODEState<double>>& Xs) const{
		std::vector<ODEState<double>> Xnew;
		Xnew.reserve((Xs.size() - 1) / 2.0);
		for (int i = 0; i < Xs.size(); i += 2) {
			Xnew.push_back(Xs[i]);
		}
		return Xnew;
	}

	Jacobian<double> calculate_jacobian(const std::vector<ODEState<double>> & Xs) const {

		Jacobian<double> jx(this->ORows(), this->IRows());
		jx.setZero();
		Hessian<double> jxall(this->IRows(), this->IRows());
		jxall.setIdentity();

		Input<double> stepper_input(this->IRows());
		ODEState<double> stepper_output(this->ode.IRows());
		Jacobian<double> stepper_jacobian(this->ORows(), this->IRows());
		Jacobian<double> jactmp(this->ORows(), this->IRows());

		int n = Xs.size();
		int numsteps = Xs.size() - 1;
		
		constexpr int vsize = DefaultSuperScalar::SizeAtCompileTime;

		Input<DefaultSuperScalar> stepper_inputSS(this->IRows());
		ODEState<DefaultSuperScalar> stepper_outputSS(this->ode.IRows());
		Jacobian<DefaultSuperScalar> stepper_jacobianSS(this->ORows(), this->IRows());


		auto ScalarImpl = [&](int i) {
			stepper_input.head(this->ode.IRows()) = Xs[i];
			stepper_input[this->ode.IRows()] = Xs[i + 1][this->ode.TVar()];

			stepper_output.setZero();
			stepper_jacobian.setZero();

			this->stepper.compute_jacobian(stepper_input, stepper_output, stepper_jacobian);
			jactmp.noalias() = stepper_jacobian * jxall;
			jxall.template topRows<Base::ORC>(this->ORows()) = jactmp;

		};
		
		auto VectorImpl = [&](int i) {
			stepper_outputSS.setZero();
			stepper_jacobianSS.setZero();

			for (int j = 0; j < vsize; j++) {
				for (int k = 0; k < this->ode.IRows(); k++) {
					stepper_inputSS.head(this->ode.IRows())[k][j] = Xs[i + j][k];
				}
				stepper_inputSS[this->ode.IRows()][j] = Xs[i + j + 1][this->ode.TVar()];
			}
			this->stepper.compute_jacobian(stepper_inputSS, stepper_outputSS, stepper_jacobianSS);

			for (int j = 0; j < vsize; j++) {
				for (int k = 0; k < this->IRows(); k++) {
					for (int l = 0; l < this->ORows(); l++) {
						stepper_jacobian(l, k) = stepper_jacobianSS(l, k)[j];
					}
				}
				jactmp.noalias() = stepper_jacobian * jxall;
				jxall.template topRows<Base::ORC>(this->ORows()) = jactmp;
			}
		};

		int Packs = (this->EnableVectorization) ? numsteps / vsize : 0;

		for (int i = 0; i < Packs; i++) {
			VectorImpl(i * vsize);
		}
		for (int i = Packs * vsize; i < numsteps; i++) {
			ScalarImpl(i);
		}

		jx = jxall.template topRows<Base::ORC>(this->ORows());

		return jx;
	}

	std::tuple < Jacobian<double>, Hessian < double> > calculate_jacobian_hessian(const std::vector<ODEState<double>>& Xs, const ODEState<double>& lf) const {
		ODEState<double> xf(this->ode.IRows());
		xf.setZero();



		Jacobian<double> jx(this->ORows(), this->IRows());
		jx.setZero();

		Jacobian<double> jxall(this->ORows(), this->IRows());
		jxall.setZero();
		jxall.leftCols(this->ORows()).setIdentity();

		Hessian<double> hxall(this->IRows(), this->IRows());
		hxall.setZero();

		Input<double> stepper_input(this->IRows());
		Input<double> stepper_grad(this->IRows());

		ODEState<double> stepper_output(this->ode.IRows());
		Jacobian<double> stepper_jacobian(this->ORows(), this->IRows());
		Hessian<double>  stepper_hessian(this->IRows(), this->IRows());

		ODEState<double> stepper_adjvars = lf;

		Hessian<double> jtwist(this->IRows(), this->IRows());
		jtwist.setZero();
		jtwist(this->IRows() - 1, this->IRows() - 1) = 1.0;



		int numsteps = Xs.size() - 1;


		for (int i = 0; i < numsteps; i++) {
			stepper_input.head(this->ode.IRows()) = Xs[numsteps - i - 1];
			stepper_input[this->ode.IRows()] = Xs[numsteps - i][this->ode.TVar()];

			stepper_output.setZero();
			stepper_jacobian.setZero();
			stepper_grad.setZero();
			stepper_hessian.setZero();

			this->stepper.compute_jacobian_adjointgradient_adjointhessian(
				stepper_input, stepper_output, stepper_jacobian, stepper_grad, stepper_hessian, stepper_adjvars);


			
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
		
		

		return std::tuple < Jacobian<double>, Hessian < double> >{jx,hxall};


	}

	


public:
	////////////////////////////////////////////////////////////////////////////////////

	IntegRet integrate(const ODEState<double>& x0, double tf) const {

		ODEState<double> xf;
		std::vector<EventPack> events;
		std::vector<std::vector<Eigen::Vector2d>> eventtimes;

		

		bool storestates = false;
		bool storederivs = false;
		bool storemidpoints = false;
		std::vector<ODEState<double>> Xs;
		std::vector<ODEDeriv<double>> dXs;

		xf = this->integrate_impl(x0, tf, events, eventtimes,
								  storestates, storederivs, storemidpoints, Xs, dXs);
		return xf;
	}

	IntegEventRet integrate(const ODEState<double>& x0, double tf, const std::vector<EventPack>& events) const {

		ODEState<double> xf;
		std::vector<std::vector<Eigen::Vector2d>> eventtimes;

		bool storestates = true;
		bool storederivs = true;
		bool storemidpoints = true;
		std::vector<ODEState<double>> Xs;
		std::vector<ODEDeriv<double>> dXs;

		xf = this->integrate_impl(x0, tf, events, eventtimes,
			storestates, storederivs, storemidpoints, Xs, dXs);

		std::vector<std::vector<ODEState<double>>> eventlocs(events.size());
		for (auto etimes : eventtimes) {
			if (etimes.size() > 0) {
				auto tab = this->make_table(Xs, dXs, false);
				eventlocs = this->find_events(tab, events, eventtimes);
				break;
			}
		}

		return std::tuple{xf,eventlocs};
	}

	
	template<class ... Args>
	auto integrate_parallel_impl(
		const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd & tfs, int thrs, Args && ... args)->
		std::vector<decltype(Integrator::integrate(x0s[0], tfs[0], args...))>
	{

		if (x0s.size() != tfs.size()) {
			throw std::invalid_argument("List of initial states and final times must be the same size");
		}

		using SingleRetType = decltype(Integrator::integrate(x0s[0], tfs[0], args...));
		using RetType = std::vector<SingleRetType>;

		this->setPoolThreads(thrs);
		int n = x0s.size();
		RetType results(n);
		std::vector<std::future<void>> futures(thrs);

		auto job = [&](int id, int start, int stop) {
			for (int i = start; i < stop; i++) {
				results[i] = this->integrate(x0s[i], tfs[i], args...);
			}
		};

		for (int i = 0; i < thrs; i++) {
			int start = (i * n) / thrs;
			int stop = ((i + 1) * n) / thrs;
			futures[i] = this->pool->push(job, start, stop);
		}
		for (int i = 0; i < thrs; i++) {
			futures[i].get();
		}
		return results;
	}

	std::vector<IntegRet> integrate_parallel(const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs, int thrs) {
		return this->integrate_parallel_impl(x0s, tfs, thrs);
	}
	std::vector<IntegEventRet> integrate_parallel(const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs,
		const std::vector<EventPack>& events,int thrs) {
		return this->integrate_parallel_impl(x0s, tfs,thrs,events);
	}
	/////////////////////////////////////////////////////////////////////////////////////

	DenseRet integrate_dense(const ODEState<double>& x0, double tf) const {

		ODEState<double> xf;
		std::vector<EventPack> events;
		std::vector<std::vector<Eigen::Vector2d>> eventtimes;

		bool storestates = true;
		bool storederivs = false;
		bool storemidpoints = false;
		std::vector<ODEState<double>> Xs;
		std::vector<ODEDeriv<double>> dXs;

		xf = this->integrate_impl(x0, tf, events, eventtimes,
			storestates, storederivs, storemidpoints, Xs, dXs);
		return Xs;
	}

	DenseEventRet integrate_dense(const ODEState<double>& x0, double tf, const std::vector<EventPack>& events,  bool  alloutput) const {

		ODEState<double> xf;
		std::vector<std::vector<Eigen::Vector2d>> eventtimes;

		bool storestates = true;
		bool storederivs = true;
		bool storemidpoints = true;
		std::vector<ODEState<double>> Xs;
		std::vector<ODEDeriv<double>> dXs;
		
		xf = this->integrate_impl(x0, tf, events, eventtimes, 
			storestates, storederivs, storemidpoints,Xs,dXs);

		std::vector<std::vector<ODEState<double>>> eventlocs(events.size());
		for (auto etimes : eventtimes) {
			if (etimes.size() > 0) {
				auto tab = this->make_table(Xs,dXs, false);
				eventlocs = this->find_events(tab, events, eventtimes);
				break;
			}
		}
		if(alloutput) return std::tuple{ Xs, eventlocs };
		else return std::tuple{ midpoints_removed(Xs), eventlocs};
	}

	DenseEventRet integrate_dense(const ODEState<double>& x0, double tf, int n, const std::vector<EventPack>& events) const {

		ODEState<double> xf;
		std::vector<std::vector<Eigen::Vector2d>> eventtimes;

		bool storestates = true;
		bool storederivs = true;
		bool storemidpoints = true;
		std::vector<ODEState<double>> Xs;
		std::vector<ODEDeriv<double>> dXs;

		xf = this->integrate_impl(x0, tf, events, eventtimes,
			storestates, storederivs, storemidpoints, Xs, dXs);

		auto tab = this->make_table(Xs,dXs, true);
		std::vector<std::vector<ODEState<double>>> eventlocs = this->find_events(tab, events, eventtimes);

		Eigen::VectorXd ts;
		ts.setLinSpaced(n, Xs[0][this->ode.TVar()], Xs.back()[this->ode.TVar()]);

		std::vector<ODEState<double>> Xinterp(n);

		for (int i = 0; i < n; i++) {
			Xinterp[i].resize(this->ode.IRows());
			tab->InterpolateRef(ts[i], Xinterp[i]);
		}

		return std::tuple{ Xinterp, eventlocs };
	}

	DenseRet integrate_dense(const ODEState<double>& x0, double tf, int n) const {

		ODEState<double> xf;
		std::vector<std::vector<Eigen::Vector2d>> eventtimes;

		bool storestates = true;
		bool storederivs = true;
		bool storemidpoints = true;
		std::vector<ODEState<double>> Xs;
		std::vector<ODEDeriv<double>> dXs;
		std::vector<EventPack> events;

		xf = this->integrate_impl(x0, tf, events, eventtimes,
			storestates, storederivs, storemidpoints, Xs, dXs);

		auto tab = this->make_table(Xs, dXs, true);
		

		Eigen::VectorXd ts;
		ts.setLinSpaced(n, Xs[0][this->ode.TVar()], Xs.back()[this->ode.TVar()]);

		std::vector<ODEState<double>> Xinterp(n);

		for (int i = 0; i < n; i++) {
			Xinterp[i].resize(this->ode.IRows());
			tab->InterpolateRef(ts[i], Xinterp[i]);
		}

		return  Xinterp ;
	}


	DenseRet integrate_dense(
		const ODEState<double>& x0, double tf, int NumStates,
		std::function<bool(ConstEigenRef<Eigen::VectorXd>)> exitfun) const {
		VectorX<double> ts =
			VectorX<double>::LinSpaced(NumStates, x0[this->ode.TVar()], tf);

		std::vector<ODEState<double>> xout;
		xout.reserve(NumStates);
		xout.push_back(x0);
		for (int i = 1; i < NumStates; i++) {
			xout.push_back(this->integrate(xout[i - 1], ts[i]));
			if (exitfun(xout.back())) break;
		}
		return xout;
	}

	template<class ... Args>
	auto integrate_dense_parallel_impl(
		const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs, int thrs,   Args &&... args) ->
		std::vector< decltype(Integrator::integrate_dense(x0s[0], tfs[0], args...))>
	{

		if (x0s.size() != tfs.size()) {
			throw std::invalid_argument("List of initial states and final times must be the same size");
		}

		using SingleRetType = decltype(Integrator::integrate_dense(x0s[0], tfs[0], args...));
		using RetType = std::vector<SingleRetType>;

		this->setPoolThreads(thrs);
		int n = x0s.size();
		RetType results(n);
		std::vector<std::future<void>> futures(thrs);

		auto job = [&](int id, int start, int stop) {
			for (int i = start; i < stop; i++) {
				results[i] = this->integrate_dense(x0s[i], tfs[i], args...);
			}
		};

		for (int i = 0; i < thrs; i++) {
			int start = (i * n) / thrs;
			int stop = ((i + 1) * n) / thrs;
			futures[i] = this->pool->push(job, start, stop);
		}
		for (int i = 0; i < thrs; i++) {
			futures[i].get();
		}
		return results;
	}

	std::vector< DenseRet> integrate_dense_parallel(const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs, int thrs) {
		return this->integrate_dense_parallel_impl(x0s, tfs, thrs);
	}
	std::vector<DenseEventRet> integrate_dense_parallel(const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs,
		const std::vector<EventPack>& events, int thrs) {
		return this->integrate_dense_parallel_impl(x0s, tfs, thrs, events ,false);
	}
	/////////////////////////////////////////////////////////////////////////////////////

	template<class ... Args>
	auto integrate_dense_parallel_impl_n(
		const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs, const std::vector<int>& ns, int thrs, Args &&... args)->
		std::vector<decltype(Integrator::integrate_dense(x0s[0], tfs[0], ns[0], args...))>
	{

		if (x0s.size() != tfs.size()) {
			throw std::invalid_argument("List of initial states and final times must be the same size");
		}
		if (x0s.size() != ns.size()) {
			throw std::invalid_argument("List of initial states and state numbers must be the same size");
		}

		using SingleRetType = decltype(Integrator::integrate_dense(x0s[0],tfs[0],ns[0], args...));
		using RetType = std::vector<SingleRetType>;

		this->setPoolThreads(thrs);
		int n = x0s.size();
		RetType results(n);
		std::vector<std::future<void>> futures(thrs);

		auto job = [&](int id, int start, int stop) {
			for (int i = start; i < stop; i++) {
				results[i] = this->integrate_dense(x0s[i], tfs[i],ns[i], args...);
			}
		};

		for (int i = 0; i < thrs; i++) {
			int start = (i * n) / thrs;
			int stop = ((i + 1) * n) / thrs;
			futures[i] = this->pool->push(job, start, stop);
		}
		for (int i = 0; i < thrs; i++) {
			futures[i].get();
		}
		return results;
	}

	std::vector<DenseEventRet>  integrate_dense_parallel(const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs, const std::vector<int> & ns,
		const std::vector<EventPack>& events, int thrs) {
		return this->integrate_dense_parallel_impl_n(x0s, tfs,ns, thrs, events);
	}
	std::vector<DenseRet>  integrate_dense_parallel(const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs, const std::vector<int>& ns, int thrs) {
		return this->integrate_dense_parallel_impl_n(x0s, tfs,ns,thrs);
	}
	/////////////////////////////////////////////////////////////////////////////////////

	

	STMRet integrate_stm(const ODEState<double>& x0, double tf) const {
		auto Xs = this->integrate_dense(x0, tf);
		Jacobian<double> jx = this->calculate_jacobian(Xs);
		return std::tuple{ Xs.back(),jx};
	}
	STMEventRet integrate_stm(const ODEState<double>& x0, double tf, const std::vector<EventPack>& events) const {
		auto [Xs,eventlocs] = this->integrate_dense(x0, tf, events,false);
		Jacobian<double> jx = this->calculate_jacobian(Xs);
		return std::tuple{Xs.back(),jx,eventlocs };
	}

	template<class ... Args>
	auto integrate_stm_parallel_impl(
		const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs, int thrs, Args && ... args) ->
		std::vector<decltype(Integrator::integrate_stm(x0s[0], tfs[0], args...))>
	{

		if (x0s.size() != tfs.size()) {
			throw std::invalid_argument("List of initial states and final times must be the same size");
		}

		using SingleRetType = decltype(Integrator::integrate_stm(x0s[0], tfs[0], args...));
		using RetType = std::vector<SingleRetType>;

		this->setPoolThreads(thrs);
		int n = x0s.size();
		RetType results(n);
		std::vector<std::future<void>> futures(thrs);

		auto job = [&](int id, int start, int stop) {
			for (int i = start; i < stop; i++) {
				results[i] = this->integrate_stm(x0s[i], tfs[i], args...);
			}
		};

		for (int i = 0; i < thrs; i++) {
			int start = (i * n) / thrs;
			int stop = ((i + 1) * n) / thrs;
			futures[i] = this->pool->push(job, start, stop);
		}
		for (int i = 0; i < thrs; i++) {
			futures[i].get();
		}
		return results;
	}
	
	std::vector<STMRet> integrate_stm_parallel(const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs, int thrs) {
		return this->integrate_stm_parallel_impl(x0s, tfs, thrs);
	}
	std::vector<STMEventRet> integrate_stm_parallel(const std::vector<ODEState<double>>& x0s, const Eigen::VectorXd& tfs,
		const std::vector<EventPack>& events, int thrs) {
		return this->integrate_stm_parallel_impl(x0s, tfs, thrs, events);
	}

	STMRet integrate_stm_parallel(const ODEState<double>& x0, double tf,int thrs) {
		this->setPoolThreads(thrs);

		
		VectorX<double> ts =VectorX<double>::LinSpaced(thrs + 1, x0[this->ode.TVar()], tf);
		std::vector<ODEState<double>> Xs(thrs + 1);
		Xs[0] = x0;


		std::vector<std::future<STMRet>> results(thrs);

		Eigen::MatrixXd jxall(this->IRows(), this->IRows());
		jxall.setIdentity();

		auto stm_op = [&](int id, int i) {
			auto xi = Xs[i];
			auto tf1 = ts[i + 1];
			return this->integrate_stm(xi, tf1);
		};

		for (int i = 0; i < thrs; i++) {
			results[i] = this->pool->push(stm_op, i);
			if (i < (thrs - 1)) Xs[i + 1] = this->integrate(Xs[i], ts[i + 1]);
		}
		for (int i = 0; i < thrs; i++) {
			auto [xf, jx] = results[i].get();
			jxall.topRows(this->ORows()) = (jx * jxall).eval();
			if (i == (thrs - 1)) Xs[i + 1] = xf;
		}

		STMRet tup_final;
		std::get<0>(tup_final) = Xs.back();
		std::get<1>(tup_final) = jxall.topRows(this->ORows());
		return tup_final;
	}

	/////////////////////////////////////////////////////////////////////////////////////

	template <class InType, class OutType>
	inline void compute_impl(ConstVectorBaseRef<InType> x,
		ConstVectorBaseRef<OutType> fx_) const {
		typedef typename InType::Scalar Scalar;
		VectorBaseRef<OutType> fx = fx_.const_cast_derived();

		ODEState<Scalar> x0 = x.head(this->ode.IRows());
		Scalar tf = x[this->ode.IRows()];
		fx = this->integrate(x0, tf);

	}
	template <class InType, class OutType, class JacType>
	inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
		ConstVectorBaseRef<OutType> fx_,
		ConstMatrixBaseRef<JacType> jx_) const {
		typedef typename InType::Scalar Scalar;
		VectorBaseRef<OutType> fx = fx_.const_cast_derived();
		MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

		ODEState<Scalar> x0 = x.head(this->ode.IRows());
		Scalar tf = x[this->ode.IRows()];
		auto Xs = this->integrate_dense(x0, tf);
		fx = Xs.back();
		jx = this->calculate_jacobian(Xs);

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

		ODEState<Scalar> x0 = x.head(this->ode.IRows());
		ODEState<Scalar> lf = adjvars;
		Scalar tf = x[this->ode.IRows()];

		
		

		auto Xs = this->integrate_dense(x0, tf);
		fx = Xs.back();


		std::tuple<Jacobian<double>,Hessian<double>> res= this->calculate_jacobian_hessian(Xs, lf);

		
		jx = std::get<0>(res);
		adjhess = std::get<1>(res);
		adjgrad = jx.transpose() * adjvars;

	}
	
	
	/////////////////////////////////////////////////////////////////////////////////////

	template<class PyDODE>
	static void BuildConstructors(PyDODE& obj) {

		obj.def("integrator", [](const DODE& od, double ds) {
			return Integrator<DODE>(od, ds);
			});

		obj.def("integrator", [](const DODE& od, std::string meth, double ds) {
			return Integrator<DODE>(od, meth, ds);
			});
		if constexpr (DODE::UV != 0) {
			obj.def("integrator", [](const DODE& od, std::string meth, double ds, const ControllerType& u,
				const Eigen::VectorXi& v) {
					return Integrator<DODE>(od,meth, ds, u, v);
				});
			obj.def("integrator", [](const DODE& od, std::string meth, double ds, std::shared_ptr<LGLInterpTable> u,
				const Eigen::VectorXi& v) {
					return Integrator<DODE>(od,meth, ds, u, v);
				});
			obj.def("integrator", [](const DODE& od, std::string meth, double ds, std::shared_ptr<LGLInterpTable> u) {
					return Integrator<DODE>(od, meth, ds, u);
				});
			obj.def("integrator", [](const DODE& od,  double ds, const ControllerType& u,
				const Eigen::VectorXi& v) {
					return Integrator<DODE>(od, ds, u, v);
				});
			obj.def("integrator", [](const DODE& od, double ds, const Eigen::VectorXd& v) {
					return Integrator<DODE>(od, ds, v);
				});
			obj.def("integrator", [](const DODE& od,  double ds, std::shared_ptr<LGLInterpTable> u,
				const Eigen::VectorXi& v) {
					return Integrator<DODE>(od, ds, u, v);
				});
			obj.def("integrator", [](const DODE& od, double ds, std::shared_ptr<LGLInterpTable> u) {
					return Integrator<DODE>(od, ds, u);
				});
		}
	}

	static void Build(py::module& m, const char* name) {
		auto obj =py::class_<Integrator>(m, name);
		
		obj.def(py::init<const DODE&, std::string, double>());
		obj.def(py::init<const DODE&, double>());


		if constexpr (DODE::UV != 0) {
			obj.def(py::init<const DODE&, std::string, double, const ControllerType&, const Eigen::VectorXi&>());
			obj.def(py::init<const DODE&, double, const ControllerType&, const Eigen::VectorXi&>());
			obj.def(py::init<const DODE&, double,const Eigen::VectorXd&>());

			obj.def(py::init<const DODE&, std::string, double, std::shared_ptr<LGLInterpTable>, const Eigen::VectorXi&>());
			obj.def(py::init<const DODE&, double, std::shared_ptr<LGLInterpTable>, const Eigen::VectorXi&>());

			obj.def(py::init<const DODE&, std::string, double, std::shared_ptr<LGLInterpTable>>());
			obj.def(py::init<const DODE&, double, std::shared_ptr<LGLInterpTable>>());

		}

		

		obj.def("integrate",
			(IntegRet(Integrator::*)(const ODEState<double>&,double) const) &Integrator::integrate,
			py::arg("Xt0UP"), py::arg("tf"));

		obj.def("integrate",
			(IntegEventRet(Integrator::*)(const ODEState<double>&, double, const std::vector<EventPack>&) const) & Integrator::integrate,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"));

		//obj.def("integrate",
		//	py::overload_cast<const ODEState<double>&, double, const std::vector<EventPack>&>(&Integrator::integrate, py::const_),
		//	py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"));


		obj.def("integrate_parallel",
			(std::vector<IntegRet>(Integrator::*)
				(const std::vector<ODEState<double>>&, const Eigen::VectorXd&, int) )& Integrator::integrate_parallel,
			py::arg("Xt0UPs"), py::arg("tfs"), py::arg("threads"), py::call_guard<py::gil_scoped_release>());

		obj.def("integrate_parallel",
			(std::vector<IntegEventRet>(Integrator::*)
				(const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<EventPack>&,int) )& Integrator::integrate_parallel,
				py::arg("Xt0UPs"), py::arg("tfs"), py::arg("Events"), py::arg("threads"), py::call_guard<py::gil_scoped_release>());

		/*
		obj.def("integrate_parallel",
			py::overload_cast<const std::vector<ODEState<double>>&, const Eigen::VectorXd&, int>(&Integrator::integrate_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("threads"), py::call_guard<py::gil_scoped_release>());

		obj.def("integrate_parallel",
			py::overload_cast<const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<EventPack>&,int>(&Integrator::integrate_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"), py::arg("threads"), py::call_guard<py::gil_scoped_release>());*/

		////////////////////////////////////////////////////////////////////////////


		obj.def("integrate_dense",
			(DenseRet(Integrator::*)(const ODEState<double>&, double) const) & Integrator::integrate_dense,
			py::arg("Xt0UP"), py::arg("tf"));

		obj.def("integrate_dense",
			(DenseRet(Integrator::*)(const ODEState<double>&, double,int) const) & Integrator::integrate_dense,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("n"));

		obj.def("integrate_dense",
			(DenseRet(Integrator::*)(const ODEState<double>&, double,int, std::function<bool(ConstEigenRef<Eigen::VectorXd>)>) const) & Integrator::integrate_dense,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("n"), py::arg("StopFunc"));

		obj.def("integrate_dense",
			(DenseEventRet(Integrator::*)(const ODEState<double>&, double, const std::vector<EventPack>&,  bool ) const) & Integrator::integrate_dense,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"), py::arg("alloutput") = false);

		obj.def("integrate_dense",
			(DenseEventRet(Integrator::*)(const ODEState<double>&, double, int, const std::vector<EventPack>&) const) & Integrator::integrate_dense,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("nstates"), py::arg("Events"));


		/*obj.def("integrate_dense",
			py::overload_cast<const ODEState<double>& , double >(&Integrator::integrate_dense,py::const_),
			py::arg("Xt0UP"), py::arg("tf"));

		obj.def("integrate_dense",
			py::overload_cast<const ODEState<double>&, double,int>(&Integrator::integrate_dense, py::const_),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("n"));

		obj.def("integrate_dense",
			py::overload_cast<const ODEState<double>&, double, int, std::function<bool(ConstEigenRef<Eigen::VectorXd>)>>(&Integrator::integrate_dense, py::const_),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("n"),py::arg("StopFunc"));

		obj.def("integrate_dense",
			py::overload_cast<const ODEState<double>&, double, const std::vector<EventPack>&, const bool&>(&Integrator::integrate_dense, py::const_),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"), py::arg("alloutput") = false);

		obj.def("integrate_dense",
			py::overload_cast<const ODEState<double>&, double,int, const std::vector<EventPack>&>(&Integrator::integrate_dense, py::const_),
			py::arg("Xt0UP"),py::arg("tf"),py::arg("nstates"),py::arg("Events"));*/

		////////////////////////////////////////////////////////////////////////////

		obj.def("integrate_dense_parallel",
			(std::vector<DenseRet>(Integrator::*)
				(const std::vector<ODEState<double>>&, const Eigen::VectorXd&, int))&Integrator::integrate_dense_parallel,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());

		obj.def("integrate_dense_parallel",
			(std::vector<DenseEventRet>(Integrator::*)
				(const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<EventPack>&, int))& Integrator::integrate_dense_parallel,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());

		obj.def("integrate_dense_parallel",
			(std::vector<DenseRet>(Integrator::*)
				(const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<int>&,
			int))& Integrator::integrate_dense_parallel,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("ns"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());

		obj.def("integrate_dense_parallel",
			(std::vector<DenseEventRet>(Integrator::*)
				(const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<int>&,
			const std::vector<EventPack>&, int))& Integrator::integrate_dense_parallel,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("ns"), py::arg("Events")
			, py::arg("threads"), py::call_guard<py::gil_scoped_release>());


		/*obj.def("integrate_dense_parallel",
			py::overload_cast<const std::vector<ODEState<double>>& , const Eigen::VectorXd& , int>(&Integrator::integrate_dense_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());

		obj.def("integrate_dense_parallel",
			py::overload_cast<const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<EventPack>&, int>(&Integrator::integrate_dense_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());

		obj.def("integrate_dense_parallel",
			py::overload_cast<const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<int> &, 
			int>(&Integrator::integrate_dense_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("ns"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());

		obj.def("integrate_dense_parallel",
			py::overload_cast<const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<int>&,
			const std::vector<EventPack>&, int>(&Integrator::integrate_dense_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("ns"), py::arg("Events")
			, py::arg("threads"), py::call_guard<py::gil_scoped_release>());*/

		/////////////////////////////////////////////////////

		obj.def("integrate_stm",
			(STMRet(Integrator::*)(const ODEState<double>&, double) const)&Integrator::integrate_stm,
			py::arg("Xt0UP"), py::arg("tf"));
		obj.def("integrate_stm",
			(STMEventRet(Integrator::*)(const ODEState<double>&, double, const std::vector<EventPack>&) const)& Integrator::integrate_stm,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"));
		obj.def("integrate_stm_parallel",
			(STMRet(Integrator::*)(const ODEState<double>&, double, int))&Integrator::integrate_stm_parallel,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());
		obj.def("integrate_stm_parallel",
			(std::vector<STMRet>(Integrator::*)(const std::vector<ODEState<double>>&, const Eigen::VectorXd&, int))& Integrator::integrate_stm_parallel,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());
		obj.def("integrate_stm_parallel",
			(std::vector<STMEventRet>(Integrator::*)(const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<EventPack>&, int))& Integrator::integrate_stm_parallel,
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events")
			, py::arg("threads"), py::call_guard<py::gil_scoped_release>());


		/*obj.def("integrate_stm",
			py::overload_cast<const ODEState<double>&, double>(&Integrator::integrate_stm, py::const_),
			py::arg("Xt0UP"), py::arg("tf"));
		obj.def("integrate_stm",
			py::overload_cast<const ODEState<double>&, double, const std::vector<EventPack>&>(&Integrator::integrate_stm, py::const_),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events"));
		obj.def("integrate_stm_parallel",
			py::overload_cast<const ODEState<double>&, double, int>(&Integrator::integrate_stm_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());
		obj.def("integrate_stm_parallel",
			py::overload_cast<const std::vector<ODEState<double>>&, const Eigen::VectorXd&, int>(&Integrator::integrate_stm_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("threads")
			, py::call_guard<py::gil_scoped_release>());
		obj.def("integrate_stm_parallel",
			py::overload_cast<const std::vector<ODEState<double>>&, const Eigen::VectorXd&, const std::vector<EventPack>&, int>(&Integrator::integrate_stm_parallel),
			py::arg("Xt0UP"), py::arg("tf"), py::arg("Events")
			, py::arg("threads"), py::call_guard<py::gil_scoped_release>());*/


		/////////////////////////////////////////////////////


		Base::DenseBaseBuild(obj);


		obj.def_readwrite("EnableVectorization", &Integrator::EnableVectorization);
		



		obj.def_readwrite("DefStepSize", &Integrator::DefStepSize);
		obj.def_readwrite("MaxStepSize", &Integrator::MaxStepSize);
		obj.def_readwrite("MinStepSize", &Integrator::MinStepSize);
		obj.def_readwrite("MaxStepChange", &Integrator::MaxStepChange);
		obj.def_readwrite("FastAdaptiveSTM", &Integrator::FastAdaptiveSTM);

		obj.def_readwrite("Adaptive", &Integrator::Adaptive);
		obj.def_readwrite("AbsTols", &Integrator::AbsTols);

		obj.def("setAbsTol", &Integrator::setAbsTol);
		obj.def("setAbsTols", &Integrator::setAbsTols);
		obj.def("getAbsTols", &Integrator::getAbsTols);

		obj.def("setRelTol", &Integrator::setRelTol);
		obj.def("setRelTols", &Integrator::setRelTols);
		obj.def("getRelTols", &Integrator::getRelTols);

		obj.def("setStepSizes", &Integrator::setStepSizes,
			py::arg("DefStepSize"),py::arg("MinStepSize"),py::arg("MaxStepSize"));


		obj.def_readwrite("EventTol", &Integrator::EventTol);
		obj.def_readwrite("MaxEventIters", &Integrator::MaxEventIters);

	}


};



}
