#ifndef LIDAR_CAMERA_CALIB_HPP
#define LIDAR_CAMERA_CALIB_HPP

#include <cv_bridge/cv_bridge.h>
#include <open3d/Open3D.h>
#include <open3d/io/PointCloudIO.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Header.h>
#include <stdio.h>
#include <time.h>

#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <unordered_map>

// #include "CustomMsg.h"
#include "common.h"

class Calibration {
   public:
    int rgb_edge_minLen_ = 200;
    int rgb_canny_threshold_ = 20;
    int min_depth_ = 2.5;
    int max_depth_ = 50;
    int plane_max_size_ = 5;
    int line_number_ = 0;
    int color_intensity_threshold_ = 5;
    Eigen::Matrix3d rotation_matrix_;
    Eigen::Vector3d translation_vector_;
    Eigen::Vector3d adjust_euler_angle_;
    Calibration(const ros::NodeHandle &priv_nh);

    bool loadCalibConfig(const std::string &config_file);

    bool checkFov(const cv::Point2d &p);

    void edgeDetector();
    void projection(const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud, cv::Mat &projection_img);
    void calcLine(const std::vector<Plane> &plane_list, const double voxel_size_, const Eigen::Vector3d origin, std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list);

    void buildVPnp(const int dis_threshold, const bool show_residual,
                   const pcl::PointCloud<pcl::PointXYZ>::Ptr &cam_edge_cloud_2d, const pcl::PointCloud<pcl::PointXYZI>::Ptr &plane_line_cloud_,
                   std::vector<VPnPData> &pnp_list);

    cv::Mat getConnectImg(const int dis_threshold, const pcl::PointCloud<pcl::PointXYZ>::Ptr &rgb_edge_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr &depth_edge_cloud);
    cv::Mat getProjectionImg();
    void initVoxel();
    void LiDAREdgeExtraction();
    void calcDirection(const std::vector<Eigen::Vector2d> &points, Eigen::Vector2d &direction);
    void setParam(const ros::NodeHandle &priv_nh);

    void loadImg();
    void loadPCD();

    float fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_, k3_, s_;
    int width_, height_;

    cv::Mat init_extrinsic_;

    float voxel_size_ = 1.0;
    float down_sample_size_ = 0.02;
    float ransac_dis_threshold_ = 0.02;
    float plane_size_threshold_ = 60;
    float theta_min_;
    float theta_max_;
    float direction_theta_min_;
    float direction_theta_max_;
    float min_line_dis_threshold_ = 0.03;
    float max_line_dis_threshold_ = 0.06;

    cv::Mat rgb_image_;
    cv::Mat image_;
    cv::Mat grey_image_;
    cv::Mat cut_grey_image_;

    Eigen::Matrix3d init_rotation_matrix_;
    Eigen::Vector3d init_translation_vector_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_lidar_cloud_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud_;
    std::vector<int> plane_line_number_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_cloud_;

    std::string calib_config_file_, image_file_, pcd_file_, result_file_;
    Eigen::Matrix3d camera_intrinsic_;
    Eigen::Vector4d dist_coeffs_;

    std::unordered_map<VOXEL_LOC, Voxel *> voxel_map_;
};
void Calibration::setParam(const ros::NodeHandle &priv_nh) {
    std::vector<double> camera_intrinsic, dist_coeffs;
    priv_nh.param<std::string>("calib/alib_config_file", calib_config_file_, "/home/catkin_ws/src/calib-extrinsic-targetless/config/config_indoor.yaml");
    priv_nh.param<std::string>("common/image_file", image_file_, "/home/data/superb/3.png");
    priv_nh.param<std::string>("common/pcd_file", pcd_file_, "/home/data/superb/3.png");
    priv_nh.param<std::string>("common/result_file", result_file_, "/home/data/extrinsic.txt");
    priv_nh.param<std::vector<double>>("camera/camera_matrix", camera_intrinsic, std::vector<double>());
    priv_nh.param<std::vector<double>>("camera/dist_coeffs", dist_coeffs, std::vector<double>());

    camera_intrinsic_ << camera_intrinsic[0], camera_intrinsic[1], camera_intrinsic[2],
        camera_intrinsic[3], camera_intrinsic[4], camera_intrinsic[5],
        camera_intrinsic[6], camera_intrinsic[7], camera_intrinsic[8];

    dist_coeffs_ << dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3];

    loadCalibConfig(calib_config_file_);
}
void Calibration::loadImg() {
    image_ = cv::imread(image_file_, cv::IMREAD_UNCHANGED);
    if (!image_.data) {
        std::string msg = "Can not load image from " + image_file_;
        ROS_ERROR_STREAM(msg.c_str());
        exit(-1);
    } else {
        std::string msg = "Sucessfully load image!";
        ROS_INFO_STREAM(msg.c_str());
    }
    width_ = image_.cols;
    height_ = image_.rows;
    // check rgb or gray
    if (image_.type() == CV_8UC1) {
        grey_image_ = image_;
    } else if (image_.type() == CV_8UC3) {
        cv::cvtColor(image_, grey_image_, cv::COLOR_BGR2GRAY);
    } else {
        std::string msg = "Unsupported image type, please use CV_8UC3 or CV_8UC1";
        ROS_ERROR_STREAM(msg.c_str());
        exit(-1);
    }
}
void Calibration::loadPCD() {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    raw_lidar_cloud_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    ROS_INFO_STREAM("Loading point cloud from pcd file.");
    if (!pcl::io::loadPCDFile(pcd_file_, *raw_lidar_cloud_)) {
        std::string msg = "Sucessfully load pcd, pointcloud size: " + std::to_string(raw_lidar_cloud_->size());
        ROS_INFO_STREAM(msg.c_str());
    } else {
        std::string msg = "Unable to load " + pcd_file_;
        ROS_ERROR_STREAM(msg.c_str());
        exit(-1);
    }
    std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
    std::cout << "duration : " << sec.count() << "sec" << std::endl;

    std::chrono::system_clock::time_point start2 = std::chrono::system_clock::now();

    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    open3d::io::ReadPointCloud(pcd_file_, *pcd);
    std::chrono::duration<double> sec2 = std::chrono::system_clock::now() - start2;
    std::cout << "duration : " << sec2.count() << "sec" << std::endl;
}

Calibration::Calibration(const ros::NodeHandle &priv_nh) {
    setParam(priv_nh);
    loadImg();
    loadPCD();
    edgeDetector();
    std::string msg = "Sucessfully extract edge from image, edge size:" + std::to_string(rgb_egde_cloud_->size());
    ROS_INFO_STREAM(msg.c_str());

    initVoxel();
    LiDAREdgeExtraction();
};

bool Calibration::loadCalibConfig(const std::string &config_file) {
    cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
    if (!fSettings.isOpened()) {
        std::cerr << "Failed to open settings file at: " << config_file
                  << std::endl;
        exit(-1);
    } else {
        ROS_INFO("Sucessfully load calib config file");
    }
    fSettings["ExtrinsicMat"] >> init_extrinsic_;
    init_rotation_matrix_ << init_extrinsic_.at<double>(0, 0),
        init_extrinsic_.at<double>(0, 1), init_extrinsic_.at<double>(0, 2),
        init_extrinsic_.at<double>(1, 0), init_extrinsic_.at<double>(1, 1),
        init_extrinsic_.at<double>(1, 2), init_extrinsic_.at<double>(2, 0),
        init_extrinsic_.at<double>(2, 1), init_extrinsic_.at<double>(2, 2);
    init_translation_vector_ << init_extrinsic_.at<double>(0, 3),
        init_extrinsic_.at<double>(1, 3), init_extrinsic_.at<double>(2, 3);
    rgb_canny_threshold_ = fSettings["Canny.gray_threshold"];
    rgb_edge_minLen_ = fSettings["Canny.len_threshold"];
    voxel_size_ = fSettings["Voxel.size"];
    down_sample_size_ = fSettings["Voxel.down_sample_size"];
    plane_size_threshold_ = fSettings["Plane.min_points_size"];
    plane_max_size_ = fSettings["Plane.max_size"];
    ransac_dis_threshold_ = fSettings["Ransac.dis_threshold"];
    min_line_dis_threshold_ = fSettings["Edge.min_dis_threshold"];
    max_line_dis_threshold_ = fSettings["Edge.max_dis_threshold"];
    theta_min_ = fSettings["Plane.normal_theta_min"];
    theta_max_ = fSettings["Plane.normal_theta_max"];
    theta_min_ = cos(DEG2RAD(theta_min_));
    theta_max_ = cos(DEG2RAD(theta_max_));
    direction_theta_min_ = cos(DEG2RAD(30.0));
    direction_theta_max_ = cos(DEG2RAD(150.0));
    color_intensity_threshold_ = fSettings["Color.intensity_threshold"];
    return true;
};

// Detect edge by canny, and filter by edge length

void Calibration::edgeDetector() {
    int gaussian_size = 5;
    cv::GaussianBlur(grey_image_, grey_image_, cv::Size(gaussian_size, gaussian_size), 0, 0);
    cv::Mat canny_result = cv::Mat::zeros(height_, width_, CV_8UC1);
    cv::Canny(grey_image_, canny_result, rgb_canny_threshold_, rgb_canny_threshold_ * 3, 3,
              true);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(canny_result, contours, hierarchy, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

    rgb_egde_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() > rgb_edge_minLen_) {
            cv::Mat debug_img = cv::Mat::zeros(height_, width_, CV_8UC1);
            for (size_t j = 0; j < contours[i].size(); j++) {
                pcl::PointXYZ p;
                p.x = contours[i][j].x;
                p.y = -contours[i][j].y;
                p.z = 0;
                rgb_egde_cloud_->points.push_back(p);
            }
        }
    }

    rgb_egde_cloud_->width = rgb_egde_cloud_->points.size();
    rgb_egde_cloud_->height = 1;
}

void Calibration::projection(const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud, cv::Mat &projection_img) {
    std::vector<cv::Point3f> pts_3d;
    std::vector<float> intensity_list;

    for (size_t i = 0; i < lidar_cloud->size(); i++) {
        std::shared_ptr<pcl::PointXYZI> point_3d = std::make_shared<pcl::PointXYZI>(lidar_cloud->points[i]);
        float depth = sqrt(pow(point_3d->x, 2) + pow(point_3d->y, 2) + pow(point_3d->z, 2));
        if (depth > min_depth_ && depth < max_depth_) {
            pts_3d.emplace_back(cv::Point3f(point_3d->x, point_3d->y, point_3d->z));
            intensity_list.emplace_back(lidar_cloud->points[i].intensity);
        }
    }

    std::vector<cv::Point2f> pts_2d;
    cv::Mat R_cv, intrinsic_cv, distor_cv, t_vec;
    cv::Mat c_R_l = cv::Mat::zeros(3, 3, CV_64F);
    cv::eigen2cv(rotation_matrix_, c_R_l);
    cv::eigen2cv(camera_intrinsic_, intrinsic_cv);
    cv::eigen2cv(dist_coeffs_, distor_cv);

    cv::eigen2cv(translation_vector_, t_vec);
    cv::Rodrigues(c_R_l, R_cv);
    cv::projectPoints(pts_3d, R_cv, t_vec, intrinsic_cv, distor_cv, pts_2d);
    projection_img = cv::Mat::zeros(height_, width_, CV_16UC1);

    for (size_t i = 0; i < pts_2d.size(); ++i) {
        if (pts_2d[i].x <= 0 || pts_2d[i].x >= width_ || pts_2d[i].y <= 0 || pts_2d[i].y >= height_) {
            continue;
        } else {
            float intensity = intensity_list[i];
            if (intensity > 100) {
                intensity = 65535;
            } else {
                intensity = (intensity / 150.0) * 65535;
            }
            projection_img.at<ushort>(pts_2d[i].y, pts_2d[i].x) = intensity;
        }
    }

    projection_img.convertTo(projection_img, CV_8UC1, 1 / 256.0);
}

cv::Mat Calibration::getConnectImg(const int dis_threshold, const pcl::PointCloud<pcl::PointXYZ>::Ptr &rgb_edge_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr &depth_edge_cloud) {
    cv::Mat connect_img = cv::Mat::zeros(height_, width_, CV_8UC3);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
        new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    kdtree->setInputCloud(rgb_edge_cloud);
    tree_cloud = rgb_edge_cloud;
    for (size_t i = 0; i < depth_edge_cloud->points.size(); i++) {
        cv::Point2d p2(depth_edge_cloud->points[i].x, -depth_edge_cloud->points[i].y);
        if (checkFov(p2)) {
            pcl::PointXYZ p = depth_edge_cloud->points[i];
            search_cloud->points.push_back(p);
        }
    }

    int line_count = 0;
    int K = 1;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for (size_t i = 0; i < search_cloud->points.size(); i++) {
        pcl::PointXYZ searchPoint = search_cloud->points[i];
        if (kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            for (int j = 0; j < K; j++) {
                float distance = sqrt(pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) + pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
                if (distance < dis_threshold) {
                    cv::Scalar color = cv::Scalar(0, 255, 0);
                    line_count++;
                    if ((line_count % 3) == 0) {
                        cv::line(connect_img, cv::Point(search_cloud->points[i].x, -search_cloud->points[i].y), cv::Point(tree_cloud->points[pointIdxNKNSearch[j]].x, -tree_cloud->points[pointIdxNKNSearch[j]].y), color, 1);
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < rgb_edge_cloud->size(); i++) {
        connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y, rgb_edge_cloud->points[i].x)[0] = 255;
        connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y, rgb_edge_cloud->points[i].x)[1] = 0;
        connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y, rgb_edge_cloud->points[i].x)[2] = 0;
    }
    for (size_t i = 0; i < search_cloud->size(); i++) {
        connect_img.at<cv::Vec3b>(-search_cloud->points[i].y, search_cloud->points[i].x)[0] = 0;
        connect_img.at<cv::Vec3b>(-search_cloud->points[i].y, search_cloud->points[i].x)[1] = 0;
        connect_img.at<cv::Vec3b>(-search_cloud->points[i].y, search_cloud->points[i].x)[2] = 255;
    }
    int expand_size = 2;
    cv::Mat expand_edge_img;
    expand_edge_img = connect_img.clone();
    for (int x = expand_size; x < connect_img.cols - expand_size; x++) {
        for (int y = expand_size; y < connect_img.rows - expand_size; y++) {
            if (connect_img.at<cv::Vec3b>(y, x)[0] == 255) {
                for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
                    for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 255;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 0;
                    }
                }
            } else if (connect_img.at<cv::Vec3b>(y, x)[2] == 255) {
                for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
                    for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
                        expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 255;
                    }
                }
            }
        }
    }
    return connect_img;
}

bool Calibration::checkFov(const cv::Point2d &p) {
    if (p.x > 0 && p.x < width_ && p.y > 0 && p.y < height_) {
        return true;
    } else {
        return false;
    }
}

void Calibration::initVoxel() {
    ROS_INFO_STREAM("Building Voxel");

    pcl::PointCloud<pcl::PointXYZRGB> test_cloud;
    for (size_t i = 0; i < raw_lidar_cloud_->size(); i++) {
        const pcl::PointXYZI &p_c = raw_lidar_cloud_->points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_c.data[j] / voxel_size_;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = voxel_map_.find(position);
        if (iter != voxel_map_.end()) {
            voxel_map_[position]->cloud->push_back(p_c);
            pcl::PointXYZRGB p_rgb;
            p_rgb.x = p_c.x;
            p_rgb.y = p_c.y;
            p_rgb.z = p_c.z;
            p_rgb.r = voxel_map_[position]->voxel_color(0);
            p_rgb.g = voxel_map_[position]->voxel_color(1);
            p_rgb.b = voxel_map_[position]->voxel_color(2);
            test_cloud.push_back(p_rgb);
        } else {
            Voxel *voxel = new Voxel(voxel_size_);
            voxel_map_[position] = voxel;
            voxel_map_[position]->voxel_origin[0] = position.x * voxel_size_;
            voxel_map_[position]->voxel_origin[1] = position.y * voxel_size_;
            voxel_map_[position]->voxel_origin[2] = position.z * voxel_size_;
            voxel_map_[position]->cloud->push_back(p_c);
            int r = rand() % 256;
            int g = rand() % 256;
            int b = rand() % 256;
            voxel_map_[position]->voxel_color << r, g, b;
        }
    }
    for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++) {
        if (iter->second->cloud->size() > 20) {
            down_sampling_voxel(*(iter->second->cloud), 0.02);
        }
    }
}

void Calibration::LiDAREdgeExtraction() {
    plane_line_cloud_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++) {
        if (iter->second->cloud->size() > 50) {
            std::vector<Plane> plane_list;
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::copyPointCloud(*iter->second->cloud, *cloud_filter);
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZI> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(ransac_dis_threshold_);

            pcl::PointCloud<pcl::PointXYZRGB> color_planner_cloud;
            int plane_index = 0;
            while (cloud_filter->points.size() > 10) {
                pcl::PointCloud<pcl::PointXYZI> planner_cloud;
                pcl::ExtractIndices<pcl::PointXYZI> extract;
                seg.setInputCloud(cloud_filter);
                seg.setMaxIterations(500);
                seg.segment(*inliers, *coefficients);
                if (inliers->indices.size() == 0) {
                    ROS_INFO_STREAM(
                        "Could not estimate a planner model for the given dataset");
                    break;
                }
                extract.setIndices(inliers);
                extract.setInputCloud(cloud_filter);
                extract.filter(planner_cloud);

                if (planner_cloud.size() > plane_size_threshold_) {
                    pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
                    std::vector<unsigned int> colors;
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    pcl::PointXYZ p_center(0, 0, 0);
                    for (size_t i = 0; i < planner_cloud.points.size(); i++) {
                        pcl::PointXYZRGB p;
                        p.x = planner_cloud.points[i].x;
                        p.y = planner_cloud.points[i].y;
                        p.z = planner_cloud.points[i].z;
                        p_center.x += p.x;
                        p_center.y += p.y;
                        p_center.z += p.z;
                        p.r = colors[0];
                        p.g = colors[1];
                        p.b = colors[2];
                        color_cloud.push_back(p);
                        color_planner_cloud.push_back(p);
                    }
                    p_center.x = p_center.x / planner_cloud.size();
                    p_center.y = p_center.y / planner_cloud.size();
                    p_center.z = p_center.z / planner_cloud.size();
                    Plane single_plane;
                    single_plane.cloud = planner_cloud;
                    single_plane.p_center = p_center;
                    single_plane.normal << coefficients->values[0], coefficients->values[1], coefficients->values[2];
                    single_plane.index = plane_index;
                    plane_list.push_back(single_plane);
                    plane_index++;
                }
                extract.setNegative(true);
                pcl::PointCloud<pcl::PointXYZI> cloud_f;
                extract.filter(cloud_f);
                *cloud_filter = cloud_f;
            }

            std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud_list;
            calcLine(plane_list, voxel_size_, iter->second->voxel_origin, line_cloud_list);
            // ouster 5,normal 3
            if (line_cloud_list.size() > 0 && line_cloud_list.size() <= 8) {
                for (size_t cloud_index = 0; cloud_index < line_cloud_list.size(); cloud_index++) {
                    for (size_t i = 0; i < line_cloud_list[cloud_index].size(); i++) {
                        pcl::PointXYZI p = line_cloud_list[cloud_index].points[i];
                        plane_line_cloud_->points.push_back(p);
                        plane_line_number_.push_back(line_number_);
                    }
                    line_number_++;
                }
            }
        }
    }
}

void Calibration::calcLine(const std::vector<Plane> &plane_list, const double voxel_size_, const Eigen::Vector3d origin, std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list) {
    if (plane_list.size() >= 2 && plane_list.size() <= plane_max_size_) {
        pcl::PointCloud<pcl::PointXYZI> temp_line_cloud;
        for (size_t plane_index1 = 0; plane_index1 < plane_list.size() - 1; plane_index1++) {
            for (size_t plane_index2 = plane_index1 + 1; plane_index2 < plane_list.size(); plane_index2++) {
                float a1 = plane_list[plane_index1].normal[0];
                float b1 = plane_list[plane_index1].normal[1];
                float c1 = plane_list[plane_index1].normal[2];
                float x1 = plane_list[plane_index1].p_center.x;
                float y1 = plane_list[plane_index1].p_center.y;
                float z1 = plane_list[plane_index1].p_center.z;
                float a2 = plane_list[plane_index2].normal[0];
                float b2 = plane_list[plane_index2].normal[1];
                float c2 = plane_list[plane_index2].normal[2];
                float x2 = plane_list[plane_index2].p_center.x;
                float y2 = plane_list[plane_index2].p_center.y;
                float z2 = plane_list[plane_index2].p_center.z;
                float theta = a1 * a2 + b1 * b2 + c1 * c2;
                //
                float point_dis_threshold = 0.00;
                if (theta > theta_max_ && theta < theta_min_) {
                    // for (int i = 0; i < 6; i++) {
                    if (plane_list[plane_index1].cloud.size() > 0 && plane_list[plane_index2].cloud.size() > 0) {
                        float matrix[4][5];
                        matrix[1][1] = a1;
                        matrix[1][2] = b1;
                        matrix[1][3] = c1;
                        matrix[1][4] = a1 * x1 + b1 * y1 + c1 * z1;
                        matrix[2][1] = a2;
                        matrix[2][2] = b2;
                        matrix[2][3] = c2;
                        matrix[2][4] = a2 * x2 + b2 * y2 + c2 * z2;
                        // six types
                        std::vector<Eigen::Vector3d> points;
                        Eigen::Vector3d point;
                        matrix[3][1] = 1;
                        matrix[3][2] = 0;
                        matrix[3][3] = 0;
                        matrix[3][4] = origin[0];
                        calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold && point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold && point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold && point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 0;
                        matrix[3][2] = 1;
                        matrix[3][3] = 0;
                        matrix[3][4] = origin[1];
                        calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold && point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold && point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold && point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 0;
                        matrix[3][2] = 0;
                        matrix[3][3] = 1;
                        matrix[3][4] = origin[2];
                        calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold && point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold && point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold && point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 1;
                        matrix[3][2] = 0;
                        matrix[3][3] = 0;
                        matrix[3][4] = origin[0] + voxel_size_;
                        calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold && point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold && point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold && point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 0;
                        matrix[3][2] = 1;
                        matrix[3][3] = 0;
                        matrix[3][4] = origin[1] + voxel_size_;
                        calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold && point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold && point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold && point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
                            points.push_back(point);
                        }
                        matrix[3][1] = 0;
                        matrix[3][2] = 0;
                        matrix[3][3] = 1;
                        matrix[3][4] = origin[2] + voxel_size_;
                        calc<float>(matrix, point);
                        if (point[0] >= origin[0] - point_dis_threshold && point[0] <= origin[0] + voxel_size_ + point_dis_threshold &&
                            point[1] >= origin[1] - point_dis_threshold && point[1] <= origin[1] + voxel_size_ + point_dis_threshold &&
                            point[2] >= origin[2] - point_dis_threshold && point[2] <= origin[2] + voxel_size_ + point_dis_threshold) {
                            points.push_back(point);
                        }

                        if (points.size() == 2) {
                            pcl::PointCloud<pcl::PointXYZI> line_cloud;
                            pcl::PointXYZ p1(points[0][0], points[0][1], points[0][2]);
                            pcl::PointXYZ p2(points[1][0], points[1][1], points[1][2]);
                            float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                            int K = 1;

                            std::vector<int> pointIdxNKNSearch1(K);
                            std::vector<float> pointNKNSquaredDistance1(K);
                            std::vector<int> pointIdxNKNSearch2(K);
                            std::vector<float> pointNKNSquaredDistance2(K);
                            pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(new pcl::search::KdTree<pcl::PointXYZI>());
                            pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZI>());
                            kdtree1->setInputCloud(plane_list[plane_index1].cloud.makeShared());
                            kdtree2->setInputCloud(plane_list[plane_index2].cloud.makeShared());
                            for (float inc = 0; inc <= length; inc += 0.01) {
                                pcl::PointXYZI p;
                                p.x = p1.x + (p2.x - p1.x) * inc / length;
                                p.y = p1.y + (p2.y - p1.y) * inc / length;
                                p.z = p1.z + (p2.z - p1.z) * inc / length;
                                p.intensity = 100;
                                if ((kdtree1->nearestKSearch(p, K, pointIdxNKNSearch1, pointNKNSquaredDistance1) > 0) && (kdtree2->nearestKSearch(p, K, pointIdxNKNSearch2, pointNKNSquaredDistance2) > 0)) {
                                    float dis1 = pow(p.x - plane_list[plane_index1].cloud.points[pointIdxNKNSearch1[0]].x, 2) + pow(p.y - plane_list[plane_index1].cloud.points[pointIdxNKNSearch1[0]].y, 2) +
                                                 pow(p.z - plane_list[plane_index1].cloud.points[pointIdxNKNSearch1[0]].z, 2);
                                    float dis2 = pow(p.x - plane_list[plane_index2].cloud.points[pointIdxNKNSearch2[0]].x, 2) + pow(p.y - plane_list[plane_index2].cloud.points[pointIdxNKNSearch2[0]].y, 2) +
                                                 pow(p.z - plane_list[plane_index2].cloud.points[pointIdxNKNSearch2[0]].z, 2);
                                    if ((dis1 < min_line_dis_threshold_ * min_line_dis_threshold_ && dis2 < max_line_dis_threshold_ * max_line_dis_threshold_) ||
                                        ((dis1 < max_line_dis_threshold_ * max_line_dis_threshold_ && dis2 < min_line_dis_threshold_ * min_line_dis_threshold_))) {
                                        line_cloud.push_back(p);
                                    }
                                }
                            }
                            if (line_cloud.size() > 10) {
                                line_cloud_list.push_back(line_cloud);
                            }
                        }
                    }
                }
            }
        }
    }
}

void Calibration::buildVPnp(const int dis_threshold, const bool show_residual, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cam_edge_cloud_2d,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr &plane_line_cloud_, std::vector<VPnPData> &pnp_list) {
    pnp_list.clear();
    std::vector<std::vector<std::vector<pcl::PointXYZI>>> img_pts_container;
    for (int y = 0; y < height_; y++) {
        std::vector<std::vector<pcl::PointXYZI>> row_pts_container;
        for (int x = 0; x < width_; x++) {
            std::vector<pcl::PointXYZI> col_pts_container;
            row_pts_container.push_back(col_pts_container);
        }
        img_pts_container.push_back(row_pts_container);
    }
    std::vector<cv::Point3d> pts_3d;

    for (size_t i = 0; i < plane_line_cloud_->size(); i++) {
        pcl::PointXYZI point_3d = plane_line_cloud_->points[i];
        pts_3d.emplace_back(cv::Point3d(point_3d.x, point_3d.y, point_3d.z));
    }

    cv::Mat r_vec, t_vec, R_cv, intrinsic, distor;
    std::vector<cv::Point2d> pts_2d;

    cv::Mat c_R_l = cv::Mat::zeros(3, 3, CV_64F);
    cv::eigen2cv(rotation_matrix_, c_R_l);
    cv::eigen2cv(translation_vector_, t_vec);
    cv::eigen2cv(camera_intrinsic_, intrinsic);
    cv::eigen2cv(dist_coeffs_, distor);
    cv::Rodrigues(c_R_l, R_cv);
    cv::projectPoints(pts_3d, R_cv, t_vec, intrinsic, distor, pts_2d);
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_edge_cloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> line_edge_cloud_2d_number;
    for (size_t i = 0; i < pts_2d.size(); i++) {
        pcl::PointXYZ p;
        p.x = pts_2d[i].x;
        p.y = -pts_2d[i].y;
        p.z = 0;
        pcl::PointXYZI pi_3d;
        pi_3d.x = pts_3d[i].x;
        pi_3d.y = pts_3d[i].y;
        pi_3d.z = pts_3d[i].z;
        pi_3d.intensity = 1;
        if (p.x > 0 && p.x < width_ && pts_2d[i].y > 0 && pts_2d[i].y < height_) {
            if (img_pts_container[pts_2d[i].y][pts_2d[i].x].size() == 0) {
                line_edge_cloud_2d->points.push_back(p);
                line_edge_cloud_2d_number.push_back(plane_line_number_[i]);
                img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
            } else {
                img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
            }
        }
    }
    if (show_residual) {
        cv::Mat residual_img = getConnectImg(dis_threshold, cam_edge_cloud_2d, line_edge_cloud_2d);
        cv::imshow("residual", residual_img);
        cv::waitKey(100);
    }
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_lidar(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud_lidar = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    kdtree->setInputCloud(cam_edge_cloud_2d);
    kdtree_lidar->setInputCloud(line_edge_cloud_2d);
    tree_cloud = cam_edge_cloud_2d;
    tree_cloud_lidar = line_edge_cloud_2d;
    search_cloud = line_edge_cloud_2d;
    int K = 5;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    std::vector<int> pointIdxNKNSearchLidar(K);
    std::vector<float> pointNKNSquaredDistanceLidar(K);
    int match_count = 0;
    double mean_distance;
    int line_count = 0;
    std::vector<cv::Point2d> lidar_2d_list;
    std::vector<cv::Point2d> img_2d_list;
    std::vector<Eigen::Vector2d> camera_direction_list;
    std::vector<Eigen::Vector2d> lidar_direction_list;
    std::vector<int> lidar_2d_number;
    for (size_t i = 0; i < search_cloud->points.size(); i++) {
        pcl::PointXYZ searchPoint = search_cloud->points[i];
        if ((kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) && (kdtree_lidar->nearestKSearch(searchPoint, K, pointIdxNKNSearchLidar, pointNKNSquaredDistanceLidar) > 0)) {
            bool dis_check = true;
            for (int j = 0; j < K; j++) {
                float distance = sqrt(pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) + pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
                if (distance > dis_threshold) {
                    dis_check = false;
                }
            }
            if (dis_check) {
                cv::Point p_l_2d(search_cloud->points[i].x, -search_cloud->points[i].y);
                cv::Point p_c_2d(tree_cloud->points[pointIdxNKNSearch[0]].x, -tree_cloud->points[pointIdxNKNSearch[0]].y);
                Eigen::Vector2d direction_cam(0, 0);
                std::vector<Eigen::Vector2d> points_cam;
                for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
                    Eigen::Vector2d p(tree_cloud->points[pointIdxNKNSearch[i]].x, -tree_cloud->points[pointIdxNKNSearch[i]].y);
                    points_cam.push_back(p);
                }
                calcDirection(points_cam, direction_cam);
                Eigen::Vector2d direction_lidar(0, 0);
                std::vector<Eigen::Vector2d> points_lidar;
                for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
                    Eigen::Vector2d p(tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].x, -tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].y);
                    points_lidar.push_back(p);
                }
                calcDirection(points_lidar, direction_lidar);
                // direction.normalize();
                if (checkFov(p_l_2d)) {
                    lidar_2d_list.push_back(p_l_2d);
                    img_2d_list.push_back(p_c_2d);
                    camera_direction_list.push_back(direction_cam);
                    lidar_direction_list.push_back(direction_lidar);
                    lidar_2d_number.push_back(line_edge_cloud_2d_number[i]);
                }
            }
        }
    }
    for (size_t i = 0; i < lidar_2d_list.size(); i++) {
        int y = lidar_2d_list[i].y;
        int x = lidar_2d_list[i].x;
        int pixel_points_size = img_pts_container[y][x].size();
        if (pixel_points_size > 0) {
            VPnPData pnp;
            pnp.x = 0;
            pnp.y = 0;
            pnp.z = 0;
            pnp.u = img_2d_list[i].x;
            pnp.v = img_2d_list[i].y;
            for (size_t j = 0; j < pixel_points_size; j++) {
                pnp.x += img_pts_container[y][x][j].x;
                pnp.y += img_pts_container[y][x][j].y;
                pnp.z += img_pts_container[y][x][j].z;
            }
            pnp.x = pnp.x / pixel_points_size;
            pnp.y = pnp.y / pixel_points_size;
            pnp.z = pnp.z / pixel_points_size;
            pnp.direction = camera_direction_list[i];
            pnp.direction_lidar = lidar_direction_list[i];
            pnp.number = lidar_2d_number[i];
            float theta = pnp.direction.dot(pnp.direction_lidar);
            if (theta > direction_theta_min_ || theta < direction_theta_max_) {
                pnp_list.push_back(pnp);
            }
        }
    }
}

void Calibration::calcDirection(const std::vector<Eigen::Vector2d> &points, Eigen::Vector2d &direction) {
    Eigen::Vector2d mean_point(0, 0);
    for (size_t i = 0; i < points.size(); i++) {
        mean_point(0) += points[i](0);
        mean_point(1) += points[i](1);
    }
    mean_point(0) = mean_point(0) / points.size();
    mean_point(1) = mean_point(1) / points.size();
    Eigen::Matrix2d S;
    S << 0, 0, 0, 0;
    for (size_t i = 0; i < points.size(); i++) {
        Eigen::Matrix2d s = (points[i] - mean_point) * (points[i] - mean_point).transpose();
        S += s;
    }
    Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
    Eigen::MatrixXcd evecs = es.eigenvectors();
    Eigen::MatrixXcd evals = es.eigenvalues();
    Eigen::MatrixXd evalsReal;
    evalsReal = evals.real();
    Eigen::MatrixXf::Index evalsMax;
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax);
}

cv::Mat Calibration::getProjectionImg() {
    cv::Mat depth_projection_img;
    projection(raw_lidar_cloud_, depth_projection_img);
    cv::Mat map_img = cv::Mat::zeros(height_, width_, CV_8UC3);
    for (int x = 0; x < map_img.cols; x++) {
        for (int y = 0; y < map_img.rows; y++) {
            uint8_t r, g, b;
            float norm = depth_projection_img.at<uchar>(y, x) / 256.0;
            mapJet(norm, 0, 1, r, g, b);
            map_img.at<cv::Vec3b>(y, x)[0] = b;
            map_img.at<cv::Vec3b>(y, x)[1] = g;
            map_img.at<cv::Vec3b>(y, x)[2] = r;
        }
    }
    cv::Mat merge_img;
    if (image_.type() == CV_8UC3) {
        merge_img = 0.5 * map_img + 0.8 * image_;
    } else {
        cv::Mat src_rgb;
        cv::cvtColor(image_, src_rgb, cv::COLOR_GRAY2BGR);
        merge_img = 0.5 * map_img + 0.8 * src_rgb;
    }
    return merge_img;
}

#endif