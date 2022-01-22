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
 * Gradient Descent tests
 */

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{

    std::cout << "\n     ***** Begin GD tests. *****     \n" << std::endl;
    
    //
    // test 1

    optim::algo_settings_t settings_1;

    settings_1.iter_max = 2000;
    settings_1.gd_settings.method = 0;
    settings_1.gd_settings.par_step_size = 0.1;

    ColVec_t x_1 = BMO_MATOPS_ONE_COLVEC(2);

    bool success_1 = optim::gd(x_1,unconstr_test_fn_1,nullptr,settings_1);

    if (success_1) {
        std::cout << "gd: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "gd: test_1 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_1:\n" \
              << BMO_MATOPS_L2NORM(x_1 - unconstr_test_sols::test_1()) << std::endl;

    //
    // test 2

    settings_1.gd_settings.par_step_size = 0.001;
    settings_1.iter_max = 50000;
    // settings_1.gd_settings.step_decay = true;
    // settings_1.print_level = 4;

    ColVec_t x_2 = BMO_MATOPS_ZERO_COLVEC(2);

    bool success_2 = optim::gd(x_2,unconstr_test_fn_2,nullptr,settings_1);

    if (success_2) {
        std::cout << "\ngd: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "\ngd: test_2 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_2:\n" \
              << BMO_MATOPS_L2NORM(x_2 - unconstr_test_sols::test_2()) << std::endl;

    //
    // test 3

    settings_1.print_level = 0;
    settings_1.iter_max = 2000;
    settings_1.gd_settings.par_step_size = 0.01;
    settings_1.gd_settings.step_decay = false;

    int test_3_dim = 5;
    ColVec_t x_3 = BMO_MATOPS_ONE_COLVEC(test_3_dim);

    bool success_3 = optim::gd(x_3,unconstr_test_fn_3,nullptr,settings_1);

    if (success_3) {
        std::cout << "\ngd: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "\ngd: test_3 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_3:\n" \
              << BMO_MATOPS_L2NORM(x_3 - unconstr_test_sols::test_3(test_3_dim)) << std::endl;

    //
    // test 4

    settings_1.iter_max = 10000;

    ColVec_t x_4 = BMO_MATOPS_ONE_COLVEC(2);

    bool success_4 = optim::gd(x_4,unconstr_test_fn_4,nullptr,settings_1);

    if (success_4) {
        std::cout << "\ngd: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\ngd: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << BMO_MATOPS_L2NORM(x_4 - unconstr_test_sols::test_4()) << std::endl;

    //
    // test 5

    optim::algo_settings_t settings_5;
    settings_5.iter_max = 10000;
    settings_5.gd_settings.method = 1;

    ColVec_t x_5 = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), 2);

    bool success_5 = optim::gd(x_5,unconstr_test_fn_5,nullptr,settings_5);

    if (success_5) {
        std::cout << "\ngd: test_5 completed successfully." << std::endl;
    } else {
        std::cout << "\ngd: test_5 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_5:\n" \
              << BMO_MATOPS_L2NORM(x_5 - unconstr_test_sols::test_5()) << std::endl;

    //
    // for coverage

    optim::algo_settings_t settings;

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 0;
    settings.gd_settings.par_step_size = 0.1;
    
    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "\ngd: solution to test_3 using gd_method = 0 (basic)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 1;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 1 (momentum)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 2;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 2 (NAG)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 3;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 3 (AdaGrad)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 4;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 4 (RMSProp)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 5;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 5 (Adadelta)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 6;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 6 (Adam)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 6;
    settings.gd_settings.ada_max = true;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    settings.gd_settings.ada_max = false;

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 6 with max (AdaMax)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 7;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 7 (Nadam)\n" << x_1 << "\n";

    x_1 = BMO_MATOPS_ONE_COLVEC(2);
    settings.gd_settings.method = 7;
    settings.gd_settings.ada_max = true;

    optim::gd(x_1,unconstr_test_fn_3,nullptr,settings);

    BMO_MATOPS_COUT << "gd: solution to test_3 using gd_method = 7 with max (NadaMax)\n" << x_1 << "\n";

    //

    optim::algo_settings_t settings_bound;

    // settings_bound.print_level = 4;
    settings_bound.gd_settings.method = 3;
    settings_bound.iter_max = 10000;

    settings_bound.vals_bound = true;
    settings_bound.lower_bounds = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2), -1.0);
    settings_bound.upper_bounds = BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ZERO_COLVEC(2),  4.5);

    x_4 = BMO_MATOPS_ONE_COLVEC(2);
    x_4(0) = 3.5;
    x_4(1) = 1.0;
    
    success_4 = optim::gd(x_4,unconstr_test_fn_4,nullptr,settings_bound);

    if (success_4) {
        std::cout << "\ngd with box constraints: test_4 completed successfully." << std::endl;
    } else {
        std::cout << "\ngd with box constraints: test_4 completed unsuccessfully." << std::endl;
    }

    std::cout << "Distance from the actual solution to test_4:\n" \
              << BMO_MATOPS_L2NORM(x_4 - unconstr_test_sols::test_4()) << std::endl;

    std::cout << "\n     ***** End GD tests. *****     \n" << std::endl;

    return 0;
}
