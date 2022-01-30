/*
 * Forward-mode autodiff test with Sphere function
 *
 * $CXX -Wall -std=c++17 -mcpu=native -O3 -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I$AUTODIFF_INCLUDE_PATH -I$OPTIM/include autodiff_forward_sphere.cpp -o autodiff_forward_sphere.test -L$OPTIM -loptim -framework Accelerate
 */

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

//

autodiff::real
opt_fnd(const autodiff::ArrayXreal& x)
{
    return x.cwiseProduct(x).sum();
}

double
opt_fn(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* opt_data)
{
    autodiff::real u;
    autodiff::ArrayXreal xd = x.eval();

    if (grad_out) {
        Eigen::VectorXd grad_tmp = autodiff::gradient(opt_fnd, autodiff::wrt(xd), autodiff::at(xd), u);

        *grad_out = grad_tmp;
    } else {
        u = opt_fnd(xd);
    }

    return u.val();
}

int main()
{
    Eigen::VectorXd x(5);
    x << 1, 2, 3, 4, 5;

    bool success = optim::bfgs(x, opt_fn, nullptr);

    if (success) {
        std::cout << "bfgs: forward-mode autodiff test completed successfully.\n" << std::endl;
    } else {
        std::cout << "bfgs: forward-mode autodiff test completed unsuccessfully.\n" << std::endl;
    }

    std::cout << "solution: x = \n" << x << std::endl;

    return 0;
}
