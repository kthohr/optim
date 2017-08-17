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
 * transform values
 *
 * Keith O'Hara
 * 05/01/2012
 *
 * This version:
 * 08/12/2017
 */

inline
arma::vec
transform(const arma::vec& vals_inp, const arma::uvec& bounds_type, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    const int n_vals = bounds_type.n_elem;

    arma::vec vals_trans_out(n_vals);

    for (int i=0; i < n_vals; i++) {
        switch (bounds_type(i)) {
            case 1: // no bounds
                vals_trans_out(i) = vals_inp(i);
                break;
            case 2: // lower bound only
                vals_trans_out(i) = std::log(vals_inp(i) - lower_bounds(i));
                break;
            case 3: // upper bound only
                vals_trans_out(i) = - std::log(upper_bounds(i) - vals_inp(i));
                break;
            case 4: // upper and lower bounds
                vals_trans_out(i) = std::log(vals_inp(i) - lower_bounds(i)) - std::log(upper_bounds(i) - vals_inp(i));
                break;
        }
    }
    //
    return vals_trans_out;
}

inline
arma::vec
inv_transform(const arma::vec& vals_trans_inp, const arma::uvec& bounds_type, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    const int n_vals = bounds_type.n_elem;

    arma::vec vals_out(n_vals);

    for (int i=0; i < n_vals; i++) {
        switch (bounds_type(i)) {
            case 1: // no bounds
                vals_out(i) = vals_trans_inp(i);
                break;
            case 2: // lower bound only
                vals_out(i) = lower_bounds(i) + std::exp(vals_trans_inp(i));
                break;
            case 3: // upper bound only
                vals_out(i) = upper_bounds(i) - std::exp(-vals_trans_inp(i));
                break;
            case 4: // upper and lower bounds
                if (!std::isfinite(vals_trans_inp(i))) {
                    if (vals_trans_inp(i) < 0.0) {
                        vals_out(i) = lower_bounds(i);
                    } else {
                        vals_out(i) = upper_bounds(i);
                    }
                } else {
                    vals_out(i) = ( lower_bounds(i) + upper_bounds(i)*std::exp(vals_trans_inp(i)) ) / ( 1 + std::exp(vals_trans_inp(i)) );
                }
                break;
        }
    }
    //
    return vals_out;
}
