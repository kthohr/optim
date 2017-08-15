/*################################################################################
  ##
  ##   Copyright (C) 2011-2017 Keith O'Hara
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
 * Jacobian adjustment
 *
 * Keith O'Hara
 * 05/01/2012
 *
 * This version:
 * 08/13/2017
 */

inline
arma::mat
jacobian_adjust(const arma::vec& vals_trans_inp, const arma::uvec& bounds_type, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    const int n_vals = bounds_type.n_elem;

    arma::mat ret_mat = arma::eye(n_vals,n_vals);

    for (int i=0; i < n_vals; i++) {
        switch (bounds_type(i)) {
            case 2: // lower bound only
                ret_mat(i,i) = std::exp(vals_trans_inp(i));
                break;
            case 3: // upper bound only
                ret_mat(i,i) = std::exp(-vals_trans_inp(i));
                break;
            case 4: // upper and lower bounds
                ret_mat(i,i) =  std::exp(vals_trans_inp(i))*(upper_bounds(i) - lower_bounds(i)) / std::pow(std::exp(vals_trans_inp(i)) + 1,2);
                break;
        }
    }
    //
    return ret_mat;
}
