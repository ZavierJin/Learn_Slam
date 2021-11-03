//
// Created by zavier on 11/3/21.
//
/*
 * The dense depth estimation of monocular camera under known trajectory
 * Use epipolar search + NCC matching\
 */

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using Sophus::SE3d;

// for eigen

using namespace Eigen;
using namespace cv;

#define DATA_PATH "./test_data"


// ------------------------------------------------------------------
// parameters
const int boarder = 20;
const int width = 640;
const int height = 480;
const double fx = 481.2f;
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1);
const double min_cov = 0.1;     // Convergence judgment: minimum variance
const double max_cov = 10;

// ------------------------------------------------------------------
// Function
/// Read data from remode dataset
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

/**
 * Update the depth estimation according to the new image
 * @param ref           reference image
 * @param curr          current image
 * @param T_C_R         The pose of the reference image to the current image
 * @param depth         depth
 * @param depth_cov     depth variance
 * @return              success
 */
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R,
            Mat &depth, Mat &depth_cov2);

/**
 * Polar search
 * @param ref           reference image
 * @param curr          current image
 * @param T_C_R         The pose of the reference image to the current image
 * @param pt_ref        The position of the midpoint of the reference image
 * @param depth_mu      depth mean
 * @param depth_cov     depth variance
 * @param pt_curr       current point
 * @param epi_direction polar direction
 * @return              success
 */
bool epipolarSearch(const Mat &ref, const Mat &curr, const SE3d &T_C_R,
                    const Vector2d &pt_ref, const double &depth_mu,
                    const double &depth_cov, Vector2d &pt_curr,
                    Vector2d &epi_direction);

/**
 * Update depth filter
 * @param pt_ref        reference image point
 * @param pt_curr       current image point
 * @param T_C_R         The pose of the reference image to the current image
 * @param epi_direction polar direction
 * @param depth         depth mean
 * @param depth_cov2    depth direction
 * @return              success
 */
bool updateDepthFilter(const Vector2d &pt_ref, const Vector2d &pt_curr, const SE3d &T_C_R,
                       const Vector2d &epi_direction, Mat &depth, Mat &depth_cov2);

/**
 * Calculate NCC score
 * @param ref           reference image
 * @param curr          current image
 * @param pt_ref        reference point
 * @param pt_curr       current point
 * @return              NCC score
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// Bilinear gray interpolation
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt)
{
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------------------------
// Utils
// Show estimated depth map
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// Pixel to camera coordinate system
inline Vector3d px2cam(const Vector2d& px)
{
    return {(px(0, 0) - cx) / fx,(px(1, 0) - cy) / fy,1};
}

// Camera coordinate system to pixel
inline Vector2d cam2px(const Vector3d& p_cam)
{
    return {p_cam(0, 0) * fx / p_cam(2, 0) + cx,p_cam(1, 0) * fy / p_cam(2, 0) + cy};
}

// Detects whether a point is within the image border
inline bool inside(const Vector2d &pt)
{
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

// Show polar matching
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// Display polar
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

/// Evaluation depth estimation
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


int main()
{
    // Read data
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(DATA_PATH, color_image_files, poses_TWC, ref_depth);
    if (!ret) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // First image
    Mat ref = imread(color_image_files[0], 0);  // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;
    double init_cov2 = 3.0;
    Mat depth(height, width, CV_64F, init_depth);             // depth figure
    Mat depth_cov2(height, width, CV_64F, init_cov2);         // depth variance figure

    for (int index = 1; index < 2; index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // T_C_W * T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(0);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth_result.png", depth);
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(const string &path, vector<string> &color_image_files,
                      std::vector<SE3d> &poses, cv::Mat &ref_depth)
{
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // Data format: image name tx, ty, tz, qx, qy, qz, qw
        // Note: TWC, not TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
                SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                     Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good())
            break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}

// Update the entire depth map
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    for (int x = boarder; x < width - boarder; x++)
        for (int y = boarder; y < height - boarder; y++) {
            // Traverse each pixel
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) // 深度已收敛或发散
                continue;
            // Search for the match of (x, y) on the epipolar line
            Vector2d pt_curr;
            Vector2d epi_direction;
            bool ret = epipolarSearch(
                    ref,
                    curr,
                    T_C_R,
                    Vector2d(x, y),
                    depth.ptr<double>(y)[x],
                    sqrt(depth_cov2.ptr<double>(y)[x]),
                    pt_curr,
                    epi_direction
            );

            if (!ret) // Matching failed
                continue;

            // Uncomment the note to show the match
//            showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // Matching succeeded, update depth map
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epi_direction, depth, depth_cov2);
        }
}

// Polar search
// See 12.2 12.3
bool epipolarSearch(const Mat &ref, const Mat &curr, const SE3d &T_C_R, const Vector2d &pt_ref,
                    const double &depth_mu, const double &depth_cov, Vector2d &pt_curr, Vector2d &epi_direction)
{
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;    // P vector of reference frame

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // Pixels projected by depth mean
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));    // Pixels projected at minimum depth
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));

    Vector2d epipolar_line = px_max_curr - px_min_curr;
    epi_direction = epipolar_line;
    epi_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();    // Half length of polar line segment
    if (half_length > 100) half_length = 100;   // We don't want to search too much

    // Uncomment this sentence to display polar lines (segments)
//    showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // Search on the epipolar line, take the depth mean point as the center, and take half the length on the left and right
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epi_direction;
        if (!inside(px_curr))
            continue;
        // Calculate the NCC between the point to be matched and the reference frame
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)      // Only believe that NCC has a high match
        return false;
    pt_curr = best_px_curr;
    return true;
}

double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr)
{
    // Zero mean normalized cross correlation
    // mean
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // mean
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // calc Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   // prevent zero
}

bool updateDepthFilter(const Vector2d &pt_ref, const Vector2d &pt_curr, const SE3d &T_C_R,
                       const Vector2d &epi_direction, Mat &depth, Mat &depth_cov2)
{
    // Calc depth use triangulation
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    // Formula
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // transform into
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;               // ref result
    Vector3d xn = t + ans[1] * f2;              // cur result
    Vector3d p_esti = (xm + xn) / 2.0;          // P position, use mean
    double depth_estimation = p_esti.norm();    // depth

    // Calc uncertainty, pixel
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epi_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // Gauss integration
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

void plotDepth(const Mat &depth_truth, const Mat &depth_estimate)
{
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate)
{
    double ave_depth_error = 0;
    double ave_depth_error_sq = 0;
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr)
{
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr)
{
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}