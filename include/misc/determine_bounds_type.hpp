/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/
 
/*
 * Determine the upper-lower bounds combo type
 */

// note: std::isfinite is not true for: NaN, - Inf, or + Inf

inline
arma::uvec
determine_bounds_type(const bool vals_bound, const size_t n_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    arma::uvec ret_vec(n_vals);

    ret_vec.fill(1); // base case: 1 - no bounds imposed

    if (vals_bound)
    {
        for (size_t i=0; i < n_vals; i++)
        {
            if ( std::isfinite(lower_bounds(i)) && std::isfinite(upper_bounds(i)) ) {
                // lower and upper bound imposed
                ret_vec(i) = 4;
            } else if ( std::isfinite(lower_bounds(i)) && !std::isfinite(upper_bounds(i)) ) {
                // lower bound only
                ret_vec(i) = 2;
            } else if ( !std::isfinite(lower_bounds(i)) && std::isfinite(upper_bounds(i)) ) {
                // upper bound only
                ret_vec(i) = 3;
            }
        }
    }

    //
    
    return ret_vec;
}
