// Use Ceres & g2o to Fit curve
// y = exp(ax^2 + bx + c) + w
// min 0.5*||y - exp(ax^2 + bx + c)||^2
#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

// cost function, Functor,
struct CurveFittingCost     // TODO: need to learn grammar !!!
{
    CurveFittingCost(double x, double y): _x(x), _y(y) {}
    // compute residual
    template<typename T>
    bool operator() (   // () operator with template parameters
            const T* const param, // model parameter, 3-D
            T* residual) const  // residual
    {
        // y = exp(ax^2 + bx + c)
        residual[0] = T(_y) - ceres::exp(param[0]*T(_x)*T(_x) + param[1]*T(_x) + param[2]);
        return true;
    }
    const double _x, _y;    // data set
};

void ceresCurveFitting()
{
    // 0.891943, 2.17039, 0.944142
    double a = 1.0, b = 2.0, c = 1.0;   // real parameter
    int N = 100;                        // data total
    double w_sigma = 1.0;               // noise sigma value
    cv::RNG rng;                        // random number generator
    double param[3] = {0, 0, 0};        // estimated parameter
    std::vector<double> x_data, y_data; // data set

    // generate data with noise
    std::cout << "Start generating data ..." << std::endl;
    for (int i = 0; i < N; ++i){
        double x = 1.0 * i / double(N);
        double y = exp(a*x*x + b*x + c) + rng.gaussian(w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
        std::cout << "(" << x << ", " << y << ")" << std::endl;
    }

    // construct least squares problem
    ceres::Problem problem;
    for (int i = 0; i < N; ++i){
        problem.AddResidualBlock(
            // TODO: auto derivative function ????
            // param: cost function, input dim, output dim
            // input: residual
            // output: estimated params
            new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3> (
                new CurveFittingCost(x_data[i], y_data[i])
            ),
            nullptr,        // kernel function
            param                      // parameters to be estimated
        );
    }

    // set solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;   //
    options.minimizer_progress_to_stdout = true;    // print to cout
    ceres::Solver::Summary summary;                 // optimize info

    // solving problem
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_cost = std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time);

    // print result
    std::cout << "Time cost: " << time_cost.count() << std::endl;
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Estimated parameter: ";
    for (auto k : param)
        std::cout << k << ", ";
    std::cout << std::endl;

    // plot figure, using matplotlib cpp version
    std::vector<double> cx, cy_real, cy_est;
    for (int i = 0; i < 10*N; ++i) {
        double px = 1.0 * i / double(10*N);
        double py_real = exp(a*px*px + b*px + c);
        double py_est = exp(param[0]*px*px + param[1]*px + param[2]);
        cx.push_back(px);
        cy_real.push_back(py_real);
        cy_est.push_back(py_est);
    }
    plt::scatter(x_data, y_data);
    plt::plot(cx, cy_real, "r--");
    plt::plot(cx, cy_est, "b");
    plt::show();
}

int main()
{
    ceresCurveFitting();
    return 0;
}
