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
    optim::ColVec_t x_1 = BMO_MATOPS_ONE_COLVEC(2);

    optim::ColVec_t grad_vec_1 = optim::numerical_gradient(x_1,nullptr,unconstr_test_fn_1,nullptr);

    BMO_MATOPS_COUT << "gradient 1:\n" << grad_vec_1 << BMO_MATOPS_ENDL;

    //
    // test 2

    optim::ColVec_t x_2 = BMO_MATOPS_ONE_COLVEC(2);

    optim::ColVec_t grad_vec_2 = optim::numerical_gradient(x_2,nullptr,unconstr_test_fn_2,nullptr);

    BMO_MATOPS_COUT << "gradient 2:\n" << grad_vec_2 << BMO_MATOPS_ENDL;

    //
    // test 3

    optim::ColVec_t x_3 = BMO_MATOPS_ONE_COLVEC(2);

    optim::ColVec_t grad_vec_3 = optim::numerical_gradient(x_3,nullptr,unconstr_test_fn_3,nullptr);

    BMO_MATOPS_COUT << "gradient 3:\n" << grad_vec_3 << BMO_MATOPS_ENDL;

    //
    // test 4

    optim::ColVec_t x_4 = BMO_MATOPS_ONE_COLVEC(2);

    optim::ColVec_t grad_vec_4 = optim::numerical_gradient(x_4,nullptr,unconstr_test_fn_4,nullptr);

    BMO_MATOPS_COUT << "gradient 4:\n" << grad_vec_4 << BMO_MATOPS_ENDL;

    //
    // test 5

    optim::ColVec_t x_5 = BMO_MATOPS_ONE_COLVEC(2);

    optim::ColVec_t grad_vec_5 = optim::numerical_gradient(x_5,nullptr,unconstr_test_fn_5,nullptr);

    BMO_MATOPS_COUT << "gradient 5:\n" << grad_vec_5 << BMO_MATOPS_ENDL;

    //

    std::cout << "\n     ***** end numerical_gradient tests. *****     \n" << std::endl;

    return 0;
}
