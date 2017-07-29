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
 *  Wolfe's method for line search
 *
 * – the starting point t_init of the line-search;
 * – the direction of search d;
 * – a merit-function t |-> q(t), defined for t >= 0, representing f(x + td).
 *
 * Keith O'Hara
 * 12/23/2016
 *
 * This version:
 * 01/01/2017
 */

#ifndef _optim_line_search_wolfe_HPP
#define _optim_line_search_wolfe_HPP

double line_search_wolfe_simple(double t_init, const arma::vec& x, const arma::vec& d, double* c_1_inp, double* c_2_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);
double line_search_wolfe_cubic(double t_init, const arma::vec& x, const arma::vec& d, double* c_1_inp, double* c_2_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);

#endif
