/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
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
Mat_t
jacobian_adjust(const Vec_t& vals_trans_inp, 
                const VecInt_t& bounds_type, 
                const Vec_t& lower_bounds, 
                const Vec_t& upper_bounds)
{
    const size_t n_vals = OPTIM_MATOPS_SIZE(bounds_type);

    Mat_t ret_mat = OPTIM_MATOPS_EYE(n_vals);

    for (size_t i = 0; i < n_vals; ++i) {
        switch (bounds_type(i)) {
            case 2: // lower bound only
                ret_mat(i,i) = std::exp(vals_trans_inp(i));
                break;
            case 3: // upper bound only
                ret_mat(i,i) = std::exp(-vals_trans_inp(i));
                break;
            case 4: // upper and lower bounds
                // ret_mat(i,i) =  std::exp(vals_trans_inp(i))*(upper_bounds(i) - lower_bounds(i)) / std::pow(std::exp(vals_trans_inp(i)) + 1,2);
                ret_mat(i,i) = std::exp(vals_trans_inp(i)) * (2*eps_dbl + upper_bounds(i) - lower_bounds(i)) \
                                / (std::exp(2 * vals_trans_inp(i)) + 2*std::exp(vals_trans_inp(i)) + 1);
                break;
        }
    }

    return ret_mat;
}
