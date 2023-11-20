#pragma once
#include "pch.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "CommonFunctions/IOScaled.h"


namespace ASSET {

	template<class Func>
	Eigen::VectorXd calc_jacobian_row_scales(const Func & func, const Eigen::VectorXd & input_scales, 
		std::vector<Eigen::VectorXd> & test_inputs,
		std::string normtype,
		std::string avgtype) {

		Eigen::MatrixXd rownorms(func.ORows(), test_inputs.size());
		Eigen::VectorXd output_scales(func.ORows());
		output_scales.setOnes();
		IOScaled<Func> scaled_func(func, input_scales, output_scales);


		Eigen::VectorXd fx(func.ORows());
		Eigen::MatrixXd jx(func.ORows(), func.IRows());

		for (int i = 0; i < test_inputs.size(); i++) {
			fx.setZero();
			jx.setZero();
			scaled_func.compute_jacobian(test_inputs[i], fx, jx);
			for (int j = 0; j < func.ORows(); j++) {
				if (normtype == "norm") {
					rownorms(j, i) = jx.row(j).norm();
				}
				else if (normtype == "infnorm") {
					rownorms(j, i) = jx.row(j).lpNorm<Eigen::Infinity>();
				}
				else {
					throw std::invalid_argument("Unknown row norm type");
				}
			}
		}

		for (int j = 0; j < func.ORows(); j++) {
			double avg = 1.0;
			if (rownorms.row(j).minCoeff() < 1.0e-12) {
				// dont scale it
			}
			else if (avgtype == "mean") {
				avg = rownorms.row(j).mean();
			}
			else if (avgtype == "geomean") {
				avg = std::exp(rownorms.row(j).array().log().sum()/double(rownorms.cols()));
			}
			else {
				throw std::invalid_argument("Unknown row average type");
			}

			output_scales[j] = 1.0 / avg;
		}

		return output_scales;


	}




}

