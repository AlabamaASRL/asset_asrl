#pragma once
#include "pch.h"



namespace ASSET {

	void OCUtilsBuild(py::module& m);

	Eigen::VectorXd jump_function(const Eigen::VectorXd& tsin, const Eigen::VectorXd& usin, const Eigen::VectorXd& tsout, int m);

	Eigen::VectorXd jump_function_mmod(const Eigen::VectorXd& tsin, const Eigen::VectorXd& usin, const Eigen::VectorXd& tsout, Eigen::VectorXi ms);


}