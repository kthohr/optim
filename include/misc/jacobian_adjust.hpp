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
 * Jacobian adjustment
 */

inline
arma::mat
jacobian_adjust(const arma::vec& vals_trans_inp, const arma::uvec& bounds_type, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    const size_t n_vals = bounds_type.n_elem;

    arma::mat ret_mat = arma::eye(n_vals,n_vals);

    for (size_t i=0; i < n_vals; i++)
    {
        switch (bounds_type(i)) 
        {
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
