#include "LambertSolverMultiRevTest.h"

void ASSET::LambertSolverBuildMultiRev(FunctionRegistry& reg, py::module& m) {


	m.def("lambert_izzo_multirev",
		[](const Vector3<double>& R1t, const Vector3<double>& R2t, double tf, double mu, bool lw, double N, bool right) {
			return lambert_izzo_multirev(R1t, R2t, tf, mu, lw, N, right);
		});



	


}
