/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
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
 * Numerical Hessian, using Abramowitz and Stegun (1972, p. 884)
 */

inline
arma::mat
numerical_hessian(const arma::vec& vals_inp, const double* step_size_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* objfn_data)> objfn, void* objfn_data)
{
    const size_t n_vals = vals_inp.n_elem;
    const double step_size = (step_size_inp) ? *step_size_inp : 1e-04;
    const double mach_eps = std::numeric_limits<double>::epsilon();

    const arma::vec step_vec = arma::max(arma::abs(vals_inp), std::sqrt(step_size)*arma::ones(n_vals,1)) * std::pow(mach_eps,1.0/6.0);
    
    arma::vec x_orig = vals_inp, x_term_1, x_term_2, x_term_3, x_term_4;
    arma::mat hessian_mat = arma::zeros(n_vals,n_vals);

    const double f_orig = -30.0*objfn(x_orig, nullptr, objfn_data);

    //
    
    for (size_t i=0; i < n_vals; i++) 
    {
        for (size_t j=i; j < n_vals; j++)
        {
            x_term_1 = x_orig;
            x_term_2 = x_orig;
            x_term_3 = x_orig;
            x_term_4 = x_orig;

            if (i==j)
            {
                x_term_1(i) += 2*step_vec(i);
                x_term_2(i) +=   step_vec(i);
                x_term_3(i) -=   step_vec(i);
                x_term_4(i) -= 2*step_vec(i);
                
                //

                double term_1 = - objfn(x_term_1, nullptr, objfn_data);
                double term_2 = 16.0*objfn(x_term_2, nullptr, objfn_data);
                double term_3 = 16.0*objfn(x_term_3, nullptr, objfn_data);
                double term_4 = - objfn(x_term_4, nullptr, objfn_data);

                double denom_term = 12.0 * step_vec(i) * step_vec(i);
                
                hessian_mat(i,j) = (term_1 + term_2 + f_orig + term_3 + term_4) / denom_term;
            }
            else
            {
                x_term_1(i) += step_vec(i);
                x_term_1(j) += step_vec(j);

                x_term_2(i) += step_vec(i);
                x_term_2(j) -= step_vec(j);

                x_term_3(i) -= step_vec(i);
                x_term_3(j) += step_vec(j);

                x_term_4(i) -= step_vec(i);
                x_term_4(j) -= step_vec(j);
                
                //

                double term_1 = objfn(x_term_1, nullptr, objfn_data);
                double term_2 = -objfn(x_term_2, nullptr, objfn_data);
                double term_3 = -objfn(x_term_3, nullptr, objfn_data);
                double term_4 = objfn(x_term_4, nullptr, objfn_data);

                double denom_term = 4.0 * step_vec(i) * step_vec(j);
                
                hessian_mat(i,j) = (term_1 + term_2 + term_3 + term_4) / denom_term;
                hessian_mat(j,i) = hessian_mat(i,j);
            }
        }
    }

    //

    return hessian_mat;
}
