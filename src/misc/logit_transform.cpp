/*################################################################################
  ##
  ##   Copyright (C) 2016-2017 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   OptimLib is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   OptimLib is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ################################################################################*/

/*
 * Generalized logit transform
 *
 * Keith O'Hara
 * 11/28/2014
 *
 * This version:
 * 07/19/2017
 */

#include "optim.hpp"

arma::vec optim::logit_trans(const arma::vec& pars, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
	//
	arma::vec pars_trans = arma::log((pars - lower_bounds)/(upper_bounds - pars));
	//
	if (pars_trans.has_inf()) {
		arma::uvec inf_ind = arma::find_nonfinite(pars_trans);
		int n_inf = inf_ind.n_elem;
		double small_num = 1E-08;
		
		for (int i=0; i < n_inf; i++) {
			int inf_ind_i = inf_ind(i);
			if (pars_trans(inf_ind_i) < 0) {
				pars_trans(inf_ind_i) = std::log((pars(inf_ind_i) + small_num - lower_bounds(inf_ind_i))/(upper_bounds(inf_ind_i) - pars(inf_ind_i) - small_num));
			} else {
				pars_trans(inf_ind_i) = std::log((pars(inf_ind_i) - small_num - lower_bounds(inf_ind_i))/(upper_bounds(inf_ind_i) - pars(inf_ind_i) + small_num));
			}
		} 
	}
	//
	return pars_trans;
}

// logit_trans with [0,1] support
arma::vec optim::logit_trans(const arma::vec& pars)
{
	return arma::log(pars/(1 - pars));
}

double optim::logit_trans(const double pars, const double lower_bounds, const double upper_bounds)
{
	return std::log((pars - lower_bounds)/(upper_bounds - pars));
}

/*
 * inverse transform
 */

arma::vec optim::logit_inv_trans(const arma::vec& pars_trans, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
	return (lower_bounds + upper_bounds % arma::exp(pars_trans)) / (1 + arma::exp(pars_trans));
}

// logit_inv_trans with [0,1] support
arma::vec optim::logit_inv_trans(const arma::vec& pars_trans)
{
	return arma::exp(pars_trans) / (1 + arma::exp(pars_trans));
}

double optim::logit_inv_trans(const double pars_trans, const double lower_bounds, const double upper_bounds)
{
	return (lower_bounds + upper_bounds * std::exp(pars_trans)) / (1 + std::exp(pars_trans));
}
