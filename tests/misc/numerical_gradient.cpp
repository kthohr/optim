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

//
// numerical gradient tests
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{

    std::cout << "\n     ***** Begin numerical_gradient tests. *****     \n" << std::endl;

    //
    // test 1
    arma::vec x_1 = arma::ones(2,1);

    arma::vec grad_vec_1 = optim::numerical_gradient(x_1,nullptr,unconstr_test_fn_1,nullptr);

    arma::cout << "gradient 1:\n" << grad_vec_1 << arma::endl;

    //
    // test 2

    arma::vec x_2 = arma::ones(2,1);

    arma::vec grad_vec_2 = optim::numerical_gradient(x_2,nullptr,unconstr_test_fn_2,nullptr);

    arma::cout << "gradient 2:\n" << grad_vec_2 << arma::endl;

    //
    // test 3

    arma::vec x_3 = arma::ones(2,1);

    arma::vec grad_vec_3 = optim::numerical_gradient(x_3,nullptr,unconstr_test_fn_3,nullptr);

    arma::cout << "gradient 3:\n" << grad_vec_3 << arma::endl;

    //
    // test 4

    arma::vec x_4 = arma::ones(2,1);

    arma::vec grad_vec_4 = optim::numerical_gradient(x_4,nullptr,unconstr_test_fn_4,nullptr);

    arma::cout << "gradient 4:\n" << grad_vec_4 << arma::endl;

    //
    // test 5

    arma::vec x_5 = arma::ones(2,1);

    arma::vec grad_vec_5 = optim::numerical_gradient(x_5,nullptr,unconstr_test_fn_5,nullptr);

    arma::cout << "gradient 5:\n" << grad_vec_5 << arma::endl;

    //

    std::cout << "\n     ***** end numerical_gradient tests. *****     \n" << std::endl;

    return 0;
}
