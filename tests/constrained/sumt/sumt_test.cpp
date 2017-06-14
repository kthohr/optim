//
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fminunc.html
//
// f(x) = (x_1 - 6)^2 + (x_2 - 7)^2
// g(x) = -3*x_1 - 2*x_2 + 6 <= 0
// 
// solution is: (6,7)
//
// g++-mp-5 -O2 -Wall -std=c++11 -I/opt/local/include sumt_test.cpp -o sumt.test -L/opt/local/lib -loptim -framework Accelerate
// g++-mp-5 -O2 -Wall -std=c++11 -I./../../../include -I./../../../ sumt_test.cpp -o sumt.test -L./../../.. -loptim -framework Accelerate
//

#include "optim.hpp"
#include "tests/test_fns/test_fns.hpp"

int main()
{
    //
    arma::vec x = arma::ones(2,1);

    bool success = optim::sumt(x,constr_test_objfn_1,NULL,constr_test_constrfn_1,NULL);

    arma::cout << x << arma::endl;

    return 0;
}
