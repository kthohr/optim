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
 * Newton tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    std::cout << "\n     ***** Begin Newton tests. *****     \n" << std::endl;

    //
    // test 1

    Vec_t x_1 = OPTIM_MATOPS_ZERO_VEC(2);

    bool success_1 = optim::newton(x_1, unconstr_test_fn_1_whess, nullptr);

    if (success_1) {
        std::cout << "\nnewton: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "\nnewton: test_1 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_1:\n" \
              << OPTIM_MATOPS_L2NORM(x_1 - unconstr_test_sols::test_1()) << std::endl;

    //
    // test 2

    Vec_t x_2 = OPTIM_MATOPS_ZERO_VEC(2);

    bool success_2 = optim::newton(x_2, unconstr_test_fn_2_whess, nullptr);

    if (success_2) {
        std::cout << "\nnewton: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "\nnewton: test_2 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_2:\n" \
              << OPTIM_MATOPS_L2NORM(x_2 - unconstr_test_sols::test_2()) << std::endl;

    //
    // test 3

    int test_3_dim = 5;
    Vec_t x_3 = OPTIM_MATOPS_ONE_VEC(test_3_dim);

    bool success_3 = optim::newton(x_3, unconstr_test_fn_3_whess, nullptr);

    if (success_3) {
        std::cout << "\nnewton: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "\nnewton: test_3 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_3:\n" \
              << OPTIM_MATOPS_L2NORM(x_3 - unconstr_test_sols::test_3(test_3_dim)) << std::endl;


    //
    // test 5

    // optim::algo_settings_t settings;
    // settings.err_tol = 1.0e-12;
    // settings.print_level = 4;

    Vec_t x_5 = OPTIM_MATOPS_ZERO_VEC(2);
    x_5(0) = 2.0;
    x_5(1) = 2.0;

    bool success_5 = optim::newton(x_5, unconstr_test_fn_5_whess, nullptr);

    if (success_5) {
        std::cout << "\nnewton: test_5 completed successfully." << std::endl;
    } else {
        std::cout << "\nnewton: test_5 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_5:\n" \
              << OPTIM_MATOPS_L2NORM(x_5 - unconstr_test_sols::test_5()) << std::endl;

    //

    std::cout << "\n     ***** End Newton tests. *****     \n" << std::endl;

    return 0;
}
