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
 * L-BFGS tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{

    std::cout << "\n     ***** Begin L-BFGS tests. *****     \n" << std::endl;

    optim::algo_settings_t settings;

    //
    // test 1

    Vec_t x_1 = OPTIM_MATOPS_ONE_VEC(2);

    bool success_1 = optim::lbfgs(x_1, unconstr_test_fn_1, nullptr);

    if (success_1) {
        std::cout << "lbfgs: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "lbfgs: test_1 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_1:\n" \
              << OPTIM_MATOPS_L2NORM(x_1 - unconstr_test_sols::test_1()) << std::endl;

    //
    // test 2

    // settings.print_level = 3;

    Vec_t x_2 = OPTIM_MATOPS_ZERO_VEC(2);

    bool success_2 = optim::lbfgs(x_2, unconstr_test_fn_2, nullptr, settings);

    if (success_2) {
        std::cout << "\nlbfgs: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "\nlbfgs: test_2 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_2:\n" \
              << OPTIM_MATOPS_L2NORM(x_2 - unconstr_test_sols::test_2()) << std::endl;

    //
    // test 3

    int test_3_dim = 5;
    Vec_t x_3 = OPTIM_MATOPS_ONE_VEC(test_3_dim);

    bool success_3 = optim::lbfgs(x_3, unconstr_test_fn_3, nullptr);

    if (success_3) {
        std::cout << "\nlbfgs: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "\nlbfgs: test_3 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_3:\n" \
              << OPTIM_MATOPS_L2NORM(x_3 - unconstr_test_sols::test_3(test_3_dim)) << std::endl;

    //
    // test 4

    optim::algo_settings_t settings_4;

    // settings_4.rel_sol_change_tol = -1.0;
    // settings_4.print_level = 4;

    Vec_t x_4 = OPTIM_MATOPS_ONE_VEC(2);
    x_4(0) = 4.0;
    x_4(1) = 0.0;

    bool success_4 = optim::lbfgs(x_4, unconstr_test_fn_4, nullptr, settings_4);

    if (success_4) {
        std::cout << "\nlbfgs: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\nlbfgs: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << OPTIM_MATOPS_L2NORM(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //
    // test 5

    Vec_t x_5 = OPTIM_MATOPS_ZERO_VEC(2);

    bool success_5 = optim::lbfgs(x_5, unconstr_test_fn_5, nullptr);

    if (success_5) {
        std::cout << "\nlbfgs: test_5 completed successfully." << std::endl;
    } else {
        std::cout << "\nlbfgs: test_5 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_5:\n" \
              << OPTIM_MATOPS_L2NORM(x_5 - unconstr_test_sols::test_5()) << std::endl;

    //
    // for coverage

    // optim::algo_settings_t settings;

    optim::lbfgs(x_1, unconstr_test_fn_1, nullptr);
    optim::lbfgs(x_1, unconstr_test_fn_1 ,nullptr, settings);

    settings.vals_bound = true;
    settings.lower_bounds = OPTIM_MATOPS_ARRAY_ADD_SCALAR( OPTIM_MATOPS_ZERO_VEC(2), - 4.5);
    settings.upper_bounds = OPTIM_MATOPS_ARRAY_ADD_SCALAR( OPTIM_MATOPS_ZERO_VEC(2), + 4.5);

    x_4 = OPTIM_MATOPS_ONE_VEC(2);
    
    success_4 = optim::lbfgs(x_4, unconstr_test_fn_4, nullptr, settings);

    if (success_4) {
        std::cout << "\nlbfgs with box constraints: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\nlbfgs with box constraints: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << OPTIM_MATOPS_L2NORM(x_4 - unconstr_test_sols::test_4()) << std::endl;

    std::cout << "\n     ***** End L-BFGS tests. *****     \n" << std::endl;

    return 0;
}
