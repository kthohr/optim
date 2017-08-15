//
// SUMT test
//
// g++-mp-7 -O2 -Wall -std=c++11 -I/opt/local/include sumt_test.cpp -o sumt.test -L/opt/local/lib -loptim -framework Accelerate
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include sumt.cpp -o sumt.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    //
    arma::vec x_1 = arma::ones(2,1);

    bool success_1 = optim::sumt(x_1,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr);

    if (success_1) {
        std::cout << "sumt: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "sumt: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "sumt: solution to test_1:\n" << x_1 << arma::endl;

    //
    arma::vec x_2 = arma::ones(2,1);

    bool success_2 = optim::sumt(x_2,constr_test_objfn_2,nullptr,constr_test_constrfn_2,nullptr);

    if (success_2) {
        std::cout << "sumt: test_2 completed successfully." << std::endl;
    } else {
        std::cout << "sumt: test_2 completed unsuccessfully." << std::endl;
    }

    arma::cout << "sumt: solution to test_2:\n" << x_2 << arma::endl;

    // this is particularly troublesome
    arma::vec x_3 = arma::zeros(2,1) + 1.2;

    bool success_3 = optim::sumt(x_3,constr_test_objfn_3,nullptr,constr_test_constrfn_3,nullptr);

    if (success_3) {
        std::cout << "sumt: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "sumt: test_3 completed unsuccessfully." << std::endl;
    }

    arma::cout << "sumt: solution to test_3:\n" << x_3 << arma::endl;

    //
    // coverage tests

    optim::opt_settings settings;

    optim::sumt(x_1,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr);
    optim::sumt(x_1,constr_test_objfn_1,nullptr,constr_test_constrfn_1,nullptr,settings);

    return 0;
}
