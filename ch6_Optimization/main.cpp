// Use Ceres & g2o to Fit curve
// y = exp(ax^2 + bx + c) + w
// min 0.5*||y - exp(ax^2 + bx + c)||^2
#include <iostream>
#include <chrono>
#include <cmath>
#include "matplotlibcpp.h"

#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>


namespace plt = matplotlibcpp;

/**************** g2o ****************/
// param, optimized parameter dim and type
// TODO: Learn grammar
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ??
    void setToOriginImpl() override    // reset
    {
        _estimate << 0, 0, 0;
    }

    void oplusImpl(const double* update) override    // update
    {
        _estimate += Eigen::Vector3d(update);
    }
    // read & write, remain empty
    bool read(std::istream& in) override {}
    bool write(std::ostream& out) const override {}
};
// error model, observation dim and type, vertex type
class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit CurveFittingEdge(double x): BaseUnaryEdge(), _x(x) {}
    // compute curve model error
    void computeError() override
    {
        const auto* v = dynamic_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d param = v->estimate();
        _error(0,0) = _measurement - exp(param[0]*_x*_x + param[1]*_x + param[2]);
    }
    // read & write, remain empty
    bool read(std::istream& in) override {}
    bool write(std::ostream& out) const override {}

public:
    double _x;  // _y is _measurement
};

void g2oCurveFitting(const std::vector<double>& x_data,
                     const std::vector<double>& y_data,
                     int N,
                     double param[],
                     double w_sigma)
{
    // set g2o
    // Matrix block: each error has 3-dim optimized variable and
    // 1-dim error value
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> Block;
    // Linear equation solver: dense increment equation
    std::unique_ptr<Block::LinearSolverType> linear_solver(new g2o::LinearSolverDense<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr (new Block (std::move(linear_solver)));
//    auto* linear_solver = new g2o::LinearSolverDense<Block::PoseMatrixType>();    // old version
//    Block* solver_ptr = new Block (linear_solver);   // matrix block solver
    // Gradient descent method, selected from GN, LM, dogleg
    auto* solver = new g2o::OptimizationAlgorithmLevenberg (std::move(solver_ptr));
//    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);    // old version
    // Use GN
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr );
    // Use Dogleg
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
    g2o::SparseOptimizer optimizer;     // graph model
    optimizer.setAlgorithm(solver);     // set solver
    optimizer.setVerbose(true);     // open debug output
    int iter = 100;

    // Add vertex to graph
    auto* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0,0,0));
    v->setId(0);
    optimizer.addVertex(v);

    // Add edge to graph
    for (int i = 0; i < N; ++i){
        auto* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        // Information matrix: inverse of covariance matrix
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma));
        optimizer.addEdge(edge);
    }

    // Optimize
    std::cout << "Start optimization ..." << std::endl;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(iter);
    Eigen::Vector3d param_vec = v->estimate();
    param[0] = param_vec(0);
    param[1] = param_vec(1);
    param[2] = param_vec(2);
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_cost = std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time);

    // print details
    std::cout << "Time cost: " << time_cost.count() << std::endl;
}

/**************** Ceres ****************/
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

void ceresCurveFitting(const std::vector<double>& x_data,
                       const std::vector<double>& y_data,
                       int N,
                       double param[])
{
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

    // print details
    std::cout << "Time cost: " << time_cost.count() << std::endl;
    std::cout << summary.BriefReport() << std::endl;
}

int main()
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

    // Solving Problem
//    ceresCurveFitting(x_data, y_data, N, param);
    g2oCurveFitting(x_data, y_data, N, param, w_sigma);

    // print result
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
    plt::scatter(x_data, y_data);       // how to change color?
    plt::plot(cx, cy_real, "r--");
    plt::plot(cx, cy_est, "b");
    plt::show();

    return 0;
}
