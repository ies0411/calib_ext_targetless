#include "lidar_camera_calib.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "ceres/ceres.h"
#include "common.h"
#include "optimize.hpp"

using namespace std;

// Data path
string image_file;
string pcd_file;
string result_file;

// Camera config
vector<double> camera_matrix;
vector<double> dist_coeffs;

// Calib config
string calib_config_file;

int main(int argc, char **argv) {
    ros::init(argc, argv, "lidarCamCalib");
    ros::NodeHandle nh("~");

    Calibration calibra(nh);

    std::vector<VPnPData> vpnp_list;
    calibra.rotation_matrix_ = calibra.init_rotation_matrix_;
    calibra.translation_vector_ = calibra.init_translation_vector_;

    cv::Mat init_img = calibra.getProjectionImg();
    for (int dis_threshold = 20; dis_threshold > 8; dis_threshold -= 1) {
        // For each distance, do twice optimization
        for (int cnt = 0; cnt < 2; cnt++) {
            calibra.buildVPnp(dis_threshold, true, calibra.rgb_egde_cloud_, calibra.plane_line_cloud_, vpnp_list);

            cv::Mat projection_img = calibra.getProjectionImg();
            cv::imshow("Optimization", projection_img);
            cv::waitKey(100);
            Eigen::Quaterniond q(calibra.rotation_matrix_);
            double ext[7];
            ext[0] = q.x();
            ext[1] = q.y();
            ext[2] = q.z();
            ext[3] = q.w();
            ext[4] = calibra.translation_vector_[0];
            ext[5] = calibra.translation_vector_[1];
            ext[6] = calibra.translation_vector_[2];

            Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
            Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);

            ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
            ceres::Problem problem;

            problem.AddParameterBlock(ext, 4, q_parameterization);
            problem.AddParameterBlock(ext + 4, 3);

            for (auto val : vpnp_list) {
                ceres::CostFunction *cost_function;
                cost_function = vpnp_calib::Create(val, calibra.camera_intrinsic_, calibra.dist_coeffs_);
                problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
            }
            ceres::Solver::Options options;
            options.preconditioner_type = ceres::JACOBI;
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.minimizer_progress_to_stdout = true;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;

            calibra.rotation_matrix_ = m_q.toRotationMatrix();  ////////////////////

            calibra.translation_vector_[0] = m_t(0);
            calibra.translation_vector_[1] = m_t(1);
            calibra.translation_vector_[2] = m_t(2);
        }
    }

    return 0;
}