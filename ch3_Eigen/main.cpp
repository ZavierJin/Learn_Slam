#include <iostream>
#include <ctime>
#include <cmath>

// add Eigen
#include <Eigen/Core>
// Dense matrix computation
#include <Eigen/Dense>
// Geometry, such as rotation

#define MATRIX_SIZE 100

void geometryEigen()
{
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    // AngleAxis can compute like matrix
    // Rotate 45 degree along z axis
    Eigen::AngleAxisd rotation_vector(M_PI/4, Eigen::Vector3d(0,0,1));
    std::cout .precision(3); // ???
    std::cout << "Rotation matrix: \n" << rotation_vector.matrix() << std::endl;
    // another way
    rotation_matrix = rotation_vector.toRotationMatrix();

    Eigen::Vector3d v(1,0,0);
    Eigen::Vector3d v_rotated = rotation_vector * v; // why?
    // or Eigen::Vector3d v_rotated = rotation_matrix * v;
    std::cout << "(1,0,0) after rotation is " << v_rotated.transpose() << std::endl;

    // Euler angles, ZYX, yaw, pitch, roll
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);
    std::cout << "yaw pitch row: " << euler_angles.transpose() << std::endl;

    // Transform Matrix, 3d means 4*4
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);  // rotation
    T.pretranslate(Eigen::Vector3d(1,3,4)); // translation
    std::cout << "Transform matrix: \n" << T.matrix() << std::endl;
    // Use transform matrix, as R*v+t
    Eigen::Vector3d v_transformed = T * v;
    std::cout << "v transformed: " << v_transformed.transpose() << std::endl;

    // Quaternion
    // Use AngleAxis to assign
    // (x,y,z,w), w is real.
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    // or q = Eigen::Quaterniond(rotation_matrix);
    std::cout << "Quaternion: \n" << q.coeffs() << std::endl;
    // use quaternion, means qvq^{-1}
    v_rotated = q * v;
    std::cout << "(1,0,0) after rotation is " << v_rotated.transpose() << std::endl;







}

void basicEigen()
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
}


int main(int argc, char** argv)
{
    // basicEigen();
    geometryEigen();

    return 0;
}