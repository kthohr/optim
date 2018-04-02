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
 * unit test
 */

#include "optim.hpp"

int main()
{   

    std::cout << "\n     ***** Begin JACOBIAN tests. *****     \n" << std::endl;
    
    //

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

    arma::mat j_mat = optim::jacobian_adjust(vals_trans,bounds_type,lb,ub);

    arma::vec vals_inv_trans = optim::inv_transform(vals_trans,bounds_type,lb,ub);

    //

    std::cout << "\n     ***** END JACOBIAN tests. *****     \n" << std::endl;

    return 0;
}
