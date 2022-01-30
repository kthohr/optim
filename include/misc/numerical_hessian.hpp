/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
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
 * Numerical Hessian, using Abramowitz and Stegun (1972, p. 884, 25.3.24 and 25.3.26)
 */

inline
Mat_t
numerical_hessian(
    const ColVec_t& vals_inp, 
    const fp_t* step_size_inp, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* objfn_data)> objfn, 
    void* objfn_data
)
{
    const size_t n_vals = BMO_MATOPS_SIZE(vals_inp);

    const fp_t step_size = (step_size_inp) ? *step_size_inp : 1e-04;
    const fp_t mach_eps = std::numeric_limits<fp_t>::epsilon();

    // const ColVec_t step_vec = arma::max(arma::abs(vals_inp), std::sqrt(step_size) * std::pow(mach_eps,1.0/6.0) * arma::ones(n_vals,1));
    const ColVec_t step_vec = BMO_MATOPS_MAX( BMO_MATOPS_ABS(vals_inp), std::sqrt(step_size) * std::pow(mach_eps, fp_t(1.0/6.0)) * BMO_MATOPS_ONE_COLVEC(n_vals) );
    
    ColVec_t x_orig = vals_inp, x_term_1, x_term_2, x_term_3, x_term_4;
    Mat_t hessian_mat = BMO_MATOPS_ZERO_MAT(n_vals,n_vals);

    const fp_t f_orig = - fp_t(30.0) * objfn(x_orig, nullptr, objfn_data);

    //
    
    for (size_t i = 0; i < n_vals; ++i) {
        for (size_t j = i; j < n_vals; ++j) {
            x_term_1 = x_orig;
            x_term_2 = x_orig;
            x_term_3 = x_orig;
            x_term_4 = x_orig;

            if (i == j) {
                x_term_1(i) += fp_t(2.0) * step_vec(i);
                x_term_2(i) +=   step_vec(i);
                x_term_3(i) -=   step_vec(i);
                x_term_4(i) -= fp_t(2.0) * step_vec(i);
                
                //

                fp_t term_1 = - objfn(x_term_1, nullptr, objfn_data);
                fp_t term_2 = fp_t(16.0) * objfn(x_term_2, nullptr, objfn_data);
                fp_t term_3 = fp_t(16.0) * objfn(x_term_3, nullptr, objfn_data);
                fp_t term_4 = - objfn(x_term_4, nullptr, objfn_data);

                fp_t denom_term = fp_t(12.0) * step_vec(i) * step_vec(i);
                
                hessian_mat(i,j) = (term_1 + term_2 + f_orig + term_3 + term_4) / denom_term;
            } else {
                x_term_1(i) += step_vec(i);
                x_term_1(j) += step_vec(j);

                x_term_2(i) += step_vec(i);
                x_term_2(j) -= step_vec(j);

                x_term_3(i) -= step_vec(i);
                x_term_3(j) += step_vec(j);

                x_term_4(i) -= step_vec(i);
                x_term_4(j) -= step_vec(j);
                
                //

                fp_t term_1 =  objfn(x_term_1, nullptr, objfn_data);
                fp_t term_2 = -objfn(x_term_2, nullptr, objfn_data);
                fp_t term_3 = -objfn(x_term_3, nullptr, objfn_data);
                fp_t term_4 =  objfn(x_term_4, nullptr, objfn_data);

                fp_t denom_term = fp_t(4.0) * step_vec(i) * step_vec(j);
                
                hessian_mat(i,j) = (term_1 + term_2 + term_3 + term_4) / denom_term;
                hessian_mat(j,i) = hessian_mat(i,j);
            }
        }
    }

    //

    return hessian_mat;
}
