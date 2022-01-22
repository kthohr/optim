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
 * Nonlinear CG tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{

    std::cout << "\n     ***** Begin CG tests. *****     \n" << std::endl;
    
    //
    // test 1

    optim::algo_settings_t settings_1;

    settings_1.iter_max = 2000;
    // settings_1.print_level = 4;
    settings_1.conv_failure_switch = 1;
    settings_1.cg_settings.method = 5;

    ColVec_t x_1 = BMO_MATOPS_ONE_COLVEC(2);
    x_1(1) = - 1.0;

    bool success_1 = optim::cg(x_1,unconstr_test_fn_1,nullptr,settings_1);

    if (success_1) {
        std::cout << "cg: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "cg: test_1 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_1:\n" \
              << BMO_MATOPS_L2NORM(x_1 - unconstr_test_sols::test_1()) << std::endl;

    //
    // test 2

    ColVec_t x_2 = BMO_MATOPS_ZERO_COLVEC(2);

    bool success_2 = optim::cg(x_2,unconstr_test_fn_2,nullptr);

    if (success_2) {
        std::cout << "\ncg: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "\ncg: test_2 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_2:\n" \
              << BMO_MATOPS_L2NORM(x_2 - unconstr_test_sols::test_2()) << std::endl;

    //
    // test 3

    int test_3_dim = 5;
    ColVec_t x_3 = BMO_MATOPS_ONE_COLVEC(test_3_dim);

    bool success_3 = optim::cg(x_3,unconstr_test_fn_3,nullptr);

    if (success_3) {
        std::cout << "\ncg: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "\ncg: test_3 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_3:\n" \
              << BMO_MATOPS_L2NORM(x_3 - unconstr_test_sols::test_3(test_3_dim)) << std::endl;

    //
    // test 4

    ColVec_t x_4 = BMO_MATOPS_ONE_COLVEC(2);

    bool success_4 = optim::cg(x_4,unconstr_test_fn_4,nullptr);

    if (success_4) {
        std::cout << "\ncg: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\ncg: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << BMO_MATOPS_L2NORM(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //
    // test 5

    optim::algo_settings_t settings_5;
    settings_5.iter_max = 10000;
    settings_5.cg_settings.method = 5;

    ColVec_t x_5 = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), 2);

    bool success_5 = optim::cg(x_5,unconstr_test_fn_5,nullptr,settings_5);

    if (success_5) {
        std::cout << "\ncg: test_5 completed successfully." << std::endl;
    } else {
        std::cout << "\ncg: test_5 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_5:\n" \
              << BMO_MATOPS_L2NORM(x_5 - unconstr_test_sols::test_5()) << std::endl;

    //
    // for coverage

    optim::algo_settings_t settings_cov;

    x_1 = BMO_MATOPS_ZERO_COLVEC(2);
    settings_cov.cg_settings.method = 1;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings_cov);

    BMO_MATOPS_COUT << "\ncg: solution to test_2 using cg_method = 1\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ZERO_COLVEC(2);
    settings_cov.cg_settings.method = 2;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings_cov);

    BMO_MATOPS_COUT << "cg: solution to test_2 using cg_method = 2\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ZERO_COLVEC(2);
    settings_cov.cg_settings.method = 3;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings_cov);

    BMO_MATOPS_COUT << "cg: solution to test_2 using cg_method = 3\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ZERO_COLVEC(2);
    settings_cov.cg_settings.method = 4;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings_cov);

    BMO_MATOPS_COUT << "cg: solution to test_2 using cg_method = 4\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ZERO_COLVEC(2);
    settings_cov.cg_settings.method = 5;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings_cov);

    BMO_MATOPS_COUT << "cg: solution to test_2 using cg_method = 5\n" << x_1 << "\n";
    
    x_1 = BMO_MATOPS_ZERO_COLVEC(2);
    settings_cov.cg_settings.method = 6;
    optim::cg(x_1,unconstr_test_fn_2,nullptr,settings_cov);

    BMO_MATOPS_COUT << "cg: solution to test_2 using cg_method = 6\n" << x_1 << "\n";

    //

    optim::algo_settings_t settings_extra;

    settings_extra.vals_bound = true;
    settings_extra.lower_bounds = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), -4.5);
    settings_extra.upper_bounds = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2),  4.5);

    x_4 = BMO_MATOPS_ONE_COLVEC(2);
    
    success_4 = optim::cg(x_4,unconstr_test_fn_4,nullptr,settings_extra);

    if (success_4) {
        std::cout << "\ncg with box constraints: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\ncg with box constraints: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << BMO_MATOPS_L2NORM(x_4 - unconstr_test_sols::test_4()) << std::endl;

    std::cout << "\n     ***** End CG tests. *****     \n" << std::endl;

    return 0;
}
