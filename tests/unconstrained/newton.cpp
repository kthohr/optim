//
// Newton tests
//
// g++-mp-7 -O2 -Wall -std=c++11 -I/opt/local/include newton_test.cpp -o newton.test -L/opt/local/lib -loptim -framework Accelerate
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include newton.cpp -o newton.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "./../test_fns/test_fns.hpp"

int main()
{
    //
    // test 3

    int test_3_dim = 5;
    arma::vec x_3 = arma::ones(test_3_dim,1);

    bool success_3 = optim::newton(x_3,unconstr_test_fn_3_whess,nullptr);

    if (success_3) {
        std::cout << "newton: test_3 completed successfully." << std::endl;
    } else {
        std::cout << "newton: test_3 completed unsuccessfully." << std::endl;
    }

    arma::cout << "newton: solution to test_3:\n" << x_3 << arma::endl;

    //

    return 0;
}
