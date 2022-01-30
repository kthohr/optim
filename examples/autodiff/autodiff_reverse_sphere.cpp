/*
 * Reverse-mode autodiff test with Sphere function
 *
 * $CXX -Wall -std=c++17 -mcpu=native -O3 -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I$AUTODIFF_INCLUDE_PATH -I$OPTIM/include autodiff_reverse_sphere.cpp -o autodiff_reverse_sphere.test -L$OPTIM -loptim -framework Accelerate
 */

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

//

autodiff::var
opt_fnd(const autodiff::ArrayXvar& x)
{
    return x.cwiseProduct(x).sum();
}

double
opt_fn(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data)
{
    autodiff::ArrayXvar xd = x.eval();

    autodiff::var y = opt_fnd(xd);

    if (grad_out) {
        Eigen::VectorXd grad_tmp = autodiff::gradient(y, xd);

        *grad_out = grad_tmp;
    }

    return autodiff::val(y);
}

int main()
{
    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;

    bool success = optim::bfgs(x, opt_fn, nullptr);

    if (success) {
        std::cout << "bfgs: reverse-mode autodiff test completed successfully.\n" << std::endl;
    } else {
        std::cout << "bfgs: reverse-mode autodiff test completed unsuccessfully.\n" << std::endl;
    }

    std::cout << "solution: x = \n" << x << std::endl;

    return 0;
}
