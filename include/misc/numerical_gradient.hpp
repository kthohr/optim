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
 * Numerical Gradient, using Abramowitz and Stegun (1972, p. 883, 25.3.21)
 */

inline
arma::vec
numerical_gradient(const arma::vec& vals_inp, const double* step_size_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* objfn_data)> objfn, void* objfn_data)
{
    const size_t n_vals = vals_inp.n_elem;

    const double step_size = (step_size_inp) ? *step_size_inp : 1e-04;
    const double mach_eps = std::numeric_limits<double>::epsilon();

    const arma::vec step_vec = arma::max(arma::abs(vals_inp), std::sqrt(step_size)*arma::ones(n_vals,1)) * std::pow(mach_eps,1.0/6.0);
    
    arma::vec x_orig = vals_inp, x_term_1, x_term_2;
    arma::vec grad_vec = arma::zeros(n_vals,1);

    //
    
    for (size_t i=0; i < n_vals; i++) 
    {
        x_term_1 = x_orig;
        x_term_2 = x_orig;

        x_term_1(i) += step_vec(i);
        x_term_2(i) -= step_vec(i);

        
        //

        double term_1 =  objfn(x_term_1, nullptr, objfn_data);
        double term_2 = -objfn(x_term_2, nullptr, objfn_data);

        double denom_term = 2.0 * step_vec(i);
        
        grad_vec(i) = (term_1 + term_2) / denom_term;
    }

    //

    return grad_vec;
}
