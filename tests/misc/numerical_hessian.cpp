//
// numerical hessian tests
//
// g++-mp-7 -O2 -Wall -std=c++11 -I/opt/local/include bfgs_test.cpp -o bfgs.test -L/opt/local/lib -loptim -framework Accelerate
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include bfgs.cpp -o bfgs.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    //
    // test 1
    arma::vec x_1 = arma::ones(2,1);

    arma::mat hess_mat_1 = optim::numerical_hessian(x_1,nullptr,unconstr_test_fn_1,nullptr);

    arma::cout << "hessian 1:\n" << hess_mat_1 << arma::endl;

    //
    // test 2

    arma::vec x_2 = arma::ones(2,1);

    arma::mat hess_mat_2 = optim::numerical_hessian(x_2,nullptr,unconstr_test_fn_2,nullptr);

    arma::cout << "hessian 2:\n" << hess_mat_2 << arma::endl;

    //
    // test 3

    arma::vec x_3 = arma::ones(2,1);

    arma::mat hess_mat_3 = optim::numerical_hessian(x_3,nullptr,unconstr_test_fn_3,nullptr);

    arma::cout << "hessian 3:\n" << hess_mat_3 << arma::endl;

    //
    // test 4

    arma::vec x_4 = arma::ones(2,1);

    arma::mat hess_mat_4 = optim::numerical_hessian(x_4,nullptr,unconstr_test_fn_4,nullptr);

    arma::cout << "hessian 4:\n" << hess_mat_4 << arma::endl;

    //
    // test 5

    arma::vec x_5 = arma::ones(2,1);

    arma::mat hess_mat_5 = optim::numerical_hessian(x_5,nullptr,unconstr_test_fn_5,nullptr);

    arma::cout << "hessian 5:\n" << hess_mat_5 << arma::endl;

    return 0;
}
