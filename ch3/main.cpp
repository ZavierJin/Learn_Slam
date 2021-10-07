#include <iostream>
#include <ctime>

// add Eigen
#include <Eigen/Core>
// Dense matrix computation
#include <Eigen/Dense>

#define MATRIX_SIZE 100

int main(int argc, char** argv)
{
    // Eigen is modular. params is type, row and col.
    Eigen::Matrix<float, 2, 3> matrix_23;
    // the same as Eigen::Matrix<double, 3, 1>
    Eigen::Vector3d v_3d;
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    // dynamic size
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_x;

    // operation
    // input
    matrix_23 << 1, 2, 3, 4, 5, 6;
    // output
    std::cout << matrix_23 << std::endl;

    // access element
    for (int i = 0; i < 1; ++i)
        for (int j = 0; j < 2; ++j)
            std::cout << matrix_23 << std::endl;

    v_3d << 3, 2, 1;
    // matrix multiply vector
    // need to transfer type !!!
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    std::cout << result << std::endl;

    matrix_33 = Eigen::Matrix3d::Random();
    std::cout << matrix_33 << std::endl << std::endl;
    std::cout << matrix_33.inverse() << std::endl;

    // solve equation
    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();
    // direct
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    auto time_cost = 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC;
    std::cout << "Time use in normal inverse is "
        << time_cost << " ms." << std::endl;

    // QR decompose
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    time_cost = 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC;
    std::cout << "Time use in QR decompose is "
        << time_cost << " ms." << std::endl;




    return 0;
}