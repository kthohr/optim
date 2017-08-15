/*################################################################################
  ##
  ##   Copyright (C) 2011-2017 Keith O'Hara
  ##
  ##   This file is part of the Optim C++ library.
  ##
  ##   Optim is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   Optim is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ################################################################################*/
 
/*
 * unit test
 *
 * Keith O'Hara
 * 08/12/2017
 *
 * This version:
 * 08/12/2017
 */

#include "optim.hpp"

int main()
{   
    const bool vals_bound = true;
    const int n_vals = 4;

    arma::vec lb(n_vals);
    lb(0) = 0;
    lb(1) = 0;
    lb(2) = -arma::datum::inf;
    lb(3) = -arma::datum::inf;

    arma::vec ub(n_vals);
    ub(0) = 2;
    ub(1) = arma::datum::inf;
    ub(2) = 2;
    ub(3) = arma::datum::inf;

    arma::uvec bounds_type = optim::determine_bounds_type(vals_bound,n_vals,lb,ub);

    arma::vec initial_vals = arma::ones(n_vals,1);

    arma::vec vals_trans = optim::transform(initial_vals,bounds_type,lb,ub);

    double lj_val = optim::jacobian_adjust(vals_trans,bounds_type,lb,ub);

    arma::vec vals_inv_trans = optim::inv_transform(vals_trans,bounds_type,lb,ub);

    return 0;
}
