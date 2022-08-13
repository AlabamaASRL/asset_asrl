#pragma once
#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {


	void LambertSolverBuild(FunctionRegistry& reg, py::module& m);



	// I took this from Izzo

	template<class Scalar>
	void lambert_izzo_impl(const Scalar* r1_in, const Scalar* r2_in, Scalar t, const Scalar& mu, //INPUT
		bool lw, //INPUT
		Scalar* v1, Scalar* v2, Scalar& a, Scalar& p, Scalar& theta, int& iter)//OUTPUT
	{

		//using namespace std;


		Scalar	V, T,
			r2_mod = 0.0,    // R2 module
			dot_prod = 0.0, // dot product
			c,		        // non-dimensional chord
			s,		        // non dimensional semi-perimeter
			am,		        // minimum energy ellipse semi major axis
			lambda,	        //lambda parameter defined in Battin's Book
			x, x1, x2, y1, y2, x_new = 0, y_new, err, alfa, beta, psi, eta, eta2, sigma1, vr1, vt1, vt2, vr2, R = 0.0;
		int i_count, i;
		const Scalar tolerance = 1e-11;
		Scalar r1[3], r2[3], r2_vers[3];
		Scalar ih_dum[3], ih[3], dum[3];

		auto x2tof = [](Scalar x, Scalar s, Scalar c, const int & lw) {

			Scalar am, a, alfa, beta;

			am = s / 2;
			a = am / (1 - x * x);
			if (x < 1)//ellpise
			{
				beta = 2 * asin(sqrt((s - c) / (2 * a)));
				if (lw) beta = -beta;
				alfa = 2 * acos(x);
			}
			else
			{
				
				alfa = 2. * acosh(x);
				beta = 2. * asinh(sqrt((s - c) / (-2 * a)));
				if (lw) beta = -beta;
			}

			if (a > 0)
			{
				return (a * sqrt(a) * ((alfa - sin(alfa)) - (beta - sin(beta))));
			}
			else
			{
				return (-a * sqrt(-a) * ((sinh(alfa) - alfa) - (sinh(beta) - beta)));
			}


		};



		auto vers = [](const Scalar* V_in, Scalar* Ver_out)
		{
			Scalar v_mod = 0;
			int i;

			for (i = 0; i < 3; i++)
			{
				v_mod += V_in[i] * V_in[i];
			}

			Scalar sqrtv_mod = sqrt(v_mod);

			for (i = 0; i < 3; i++)
			{
				Ver_out[i] = V_in[i] / sqrtv_mod;
			}
		};

		auto vett = [](const Scalar* vet1, const Scalar* vet2, Scalar* prod)
		{
			prod[0] = (vet1[1] * vet2[2] - vet1[2] * vet2[1]);
			prod[1] = (vet1[2] * vet2[0] - vet1[0] * vet2[2]);
			prod[2] = (vet1[0] * vet2[1] - vet1[1] * vet2[0]);
		};



		// Increasing the tolerance does not bring any advantage as the
		// precision is usually greater anyway (due to the rectification of the tof
		// graph) except near particular cases such as parabolas in which cases a
		// lower precision allow for usual convergence.

		if (t <= 0)
		{
			//pagmo_throw(value_error, "ERROR in Lambert Solver: Negative Time in input.");
		}

		for (i = 0; i < 3; i++)
		{
			r1[i] = r1_in[i];
			r2[i] = r2_in[i];
			R += r1[i] * r1[i];
		}

		R = sqrt(R);
		V = sqrt(mu / R);
		T = R / V;

		// working with non-dimensional radii and time-of-flight
		t /= T;
		for (i = 0; i < 3; i++)  // r1 dimension is 3
		{
			r1[i] /= R;
			r2[i] /= R;
			r2_mod += r2[i] * r2[i];
		}

		// Evaluation of the relevant geometry parameters in non dimensional units
		r2_mod = sqrt(r2_mod);

		for (i = 0; i < 3; i++)
			dot_prod += (r1[i] * r2[i]);

		theta = acos(dot_prod / r2_mod);

		if (lw)
			theta = 2 * acos(-1.0) - theta;

		c = sqrt(1 + r2_mod * (r2_mod - 2.0 * cos(theta)));
		s = (1 + r2_mod + c) / 2.0;
		am = s / 2.0;
		lambda = sqrt(r2_mod) * cos(theta / 2.0) / s;

		// We start finding the log(x+1) value of the solution conic:
		// NO MULTI REV --> (1 SOL)
		//	inn1=-.5233;    //first guess point
		//  inn2=.5233;     //second guess point
		x1 = log(0.4767);
		x2 = log(1.5233);
		y1 = log(x2tof(-.5233, s, c, lw)) - log(t);
		y2 = log(x2tof(.5233, s, c, lw)) - log(t);

		// Regula-falsi iterations
		err = 1;
		i_count = 0;
		while ((err > tolerance) && (y1 != y2))
		{
			i_count++;
			x_new = (x1 * y2 - y1 * x2) / (y2 - y1);
			y_new = log(x2tof(exp(x_new) - 1, s, c, lw)) - log(t);
			x1 = x2;
			y1 = y2;
			x2 = x_new;
			y2 = y_new;
			err = fabs(x1 - x_new);
		}
		iter = i_count;
		x = exp(x_new) - 1;

		// The solution has been evaluated in terms of log(x+1) or tan(x*pi/2), we
		// now need the conic. As for transfer angles near to pi the lagrange
		// coefficient technique goes singular (dg approaches a zero/zero that is
		// numerically bad) we here use a different technique for those cases. When
		// the transfer angle is exactly equal to pi, then the ih unit vector is not
		// determined. The remaining equations, though, are still valid.

		a = am / (1 - x * x);

		// psi evaluation
		if (x < 1)  // ellipse
		{
			beta = 2 * asin(sqrt((s - c) / (2 * a)));
			if (lw) beta = -beta;
			alfa = 2 * acos(x);
			psi = (alfa - beta) / 2;
			eta2 = 2 * a * pow(sin(psi), 2) / s;
			eta = sqrt(eta2);
		}
		else       // hyperbola
		{
			beta = 2 * asinh(sqrt((c - s) / (2 * a)));
			if (lw) beta = -beta;
			alfa = 2 * acosh(x);
			psi = (alfa - beta) / 2;
			eta2 = -2 * a * pow(sinh(psi), 2) / s;
			eta = sqrt(eta2);
		}

		// parameter of the solution
		p = (r2_mod / (am * eta2)) * pow(sin(theta / 2), 2);
		sigma1 = (1 / (eta * sqrt(am))) * (2 * lambda * am - (lambda + x * eta));
		vett(r1, r2, ih_dum);
		vers(ih_dum, ih);

		if (lw)
		{
			for (i = 0; i < 3; i++)
				ih[i] = -ih[i];
		}
		
		vr1 = sigma1;
		vt1 = sqrt(p);
		vett(ih, r1, dum);

		for (i = 0; i < 3; i++)
			v1[i] = vr1 * r1[i] + vt1 * dum[i];

		vt2 = vt1 / r2_mod;
		vr2 = -vr1 + (vt1 - vt2) / tan(theta / 2);
		vers(r2, r2_vers);
		vett(ih, r2_vers, dum);
		for (i = 0; i < 3; i++)
			v2[i] = vr2 * r2[i] / r2_mod + vt2 * dum[i];

		for (i = 0; i < 3; i++)
		{
			v1[i] *= V;
			v2[i] *= V;
		}
		a *= R;
		p *= R;
	}

	template<class Scalar>
	std::array<Vector3<Scalar>, 2> lambert_izzo(const Vector3<Scalar>& R1t, const Vector3<Scalar>& R2t, Scalar tf, Scalar mu, bool lw) {
		Vector3<Scalar> R1 = R1t;
		Vector3<Scalar> R2 = R2t;

		Vector3<Scalar> V1;
		Vector3<Scalar> V2;

		Scalar a;
		Scalar p;
		Scalar theta;
		int iter;

		lambert_izzo_impl(R1.data(), R2.data(), tf, mu, lw,
			V1.data(), V2.data(), a, p, theta, iter);
		return std::array<Vector3<Scalar>, 2>{ V1, V2 };
	}


	
 }