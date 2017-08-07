//
// constrained optimization tests
//
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include generic_constr_optim.cpp -o generic_constr_optim.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    //
    arma::vec lower_bounds(2), upper_bounds(2);
    lower_bounds.fill(0.0);
    upper_bounds.fill(10.0);

    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::generic_constr_optim(x_1,lower_bounds,upper_bounds,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr);

    if (success_1) {
        std::cout << "generic_constr_optim: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "generic_constr_optim: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "generic_constr_optim: solution to test_1:\n" << x_1 << arma::endl;

    //
    // coverage tests

    arma::vec x_c = arma::ones(2,1);
    optim::opt_settings settings;
    double val_out;

    optim::generic_constr_optim(x_c,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr);
    optim::generic_constr_optim(x_c,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr,settings);
    optim::generic_constr_optim(x_c,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr,val_out);
    optim::generic_constr_optim(x_c,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr,val_out,settings);

    optim::generic_constr_optim(x_1,lower_bounds,upper_bounds,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr);
    optim::generic_constr_optim(x_1,lower_bounds,upper_bounds,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr,settings);
    optim::generic_constr_optim(x_1,lower_bounds,upper_bounds,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr,val_out);
    optim::generic_constr_optim(x_1,lower_bounds,upper_bounds,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr,val_out,settings);

    return 0;
}
