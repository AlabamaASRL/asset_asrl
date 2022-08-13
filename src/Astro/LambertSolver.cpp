#include "LambertSolver.h"

void ASSET::LambertSolverBuild(FunctionRegistry& reg, py::module& m) {


	m.def("lambert_izzo",
		[](const Vector3<double>& R1t, const Vector3<double>& R2t, double tf, double mu, bool lw) {
			return lambert_izzo(R1t, R2t, tf, mu, lw);
		});



	




}
