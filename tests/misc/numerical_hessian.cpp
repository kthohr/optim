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

//
// numerical hessian tests
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{

    std::cout << "\n     ***** Begin numerical_hessian tests. *****     \n" << std::endl;

    //
    // test 1
    optim::Vec_t x_1 = OPTIM_MATOPS_ONE_VEC(2);

    optim::Mat_t hess_mat_1 = optim::numerical_hessian(x_1,nullptr,unconstr_test_fn_1,nullptr);

    OPTIM_MATOPS_COUT << "hessian 1:\n" << hess_mat_1 << OPTIM_MATOPS_ENDL;

    //
    // test 2

    optim::Vec_t x_2 = OPTIM_MATOPS_ONE_VEC(2);

    optim::Mat_t hess_mat_2 = optim::numerical_hessian(x_2,nullptr,unconstr_test_fn_2,nullptr);

    OPTIM_MATOPS_COUT << "hessian 2:\n" << hess_mat_2 << OPTIM_MATOPS_ENDL;

    //
    // test 3

    optim::Vec_t x_3 = OPTIM_MATOPS_ONE_VEC(2);

    optim::Mat_t hess_mat_3 = optim::numerical_hessian(x_3,nullptr,unconstr_test_fn_3,nullptr);

    OPTIM_MATOPS_COUT << "hessian 3:\n" << hess_mat_3 << OPTIM_MATOPS_ENDL;

    //
    // test 4

    optim::Vec_t x_4 = OPTIM_MATOPS_ONE_VEC(2);

    optim::Mat_t hess_mat_4 = optim::numerical_hessian(x_4,nullptr,unconstr_test_fn_4,nullptr);

    OPTIM_MATOPS_COUT << "hessian 4:\n" << hess_mat_4 << OPTIM_MATOPS_ENDL;

    //
    // test 5

    optim::Vec_t x_5 = OPTIM_MATOPS_ONE_VEC(2);

    optim::Mat_t hess_mat_5 = optim::numerical_hessian(x_5,nullptr,unconstr_test_fn_5,nullptr);

    OPTIM_MATOPS_COUT << "hessian 5:\n" << hess_mat_5 << OPTIM_MATOPS_ENDL;

    //

    std::cout << "\n     ***** end numerical_hessian tests. *****     \n" << std::endl;

    return 0;
}
