#include "stereo_depth.h"
#include "opencv2/ximgproc.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h> // SACSegmentation
#include <pcl/filters/extract_indices.h>       // ExtractIndices
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>

using namespace cv;
using namespace std;

StereoDepth::StereoDepth(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh), it_(nh_),
      left_sub_(nh_, "/cam0/image_raw", 1),
      right_sub_(nh_, "/cam1/image_raw", 1),
      sync_(SyncPolicy(10), left_sub_, right_sub_)
{
    // publishers
    pub_disparity_ = nh_.advertise<sensor_msgs::Image>("stereo/disparity", 1);
    pub_depth_ = nh_.advertise<sensor_msgs::Image>("stereo/depth", 1);
    pub_depth_color_ = nh_.advertise<sensor_msgs::Image>("stereo/depth_color", 1);
    pub_points_ = nh_.advertise<sensor_msgs::PointCloud2>("stereo/points", 1);

    // read SGBM params (with defaults)
    pnh_.param("min_disparity", min_disp_, 0);
    pnh_.param("num_disparities", num_disp_, 128); // must be divisible by 16
    pnh_.param("block_size", block_size_, 5);

    // create SGBM (tweak P1,P2 for your image size)
    sgbm_ = cv::StereoSGBM::create(min_disp_, num_disp_, block_size_);
    int cn = 1;
    int P1 = 8 * cn * block_size_ * block_size_;
    int P2 = 32 * cn * block_size_ * block_size_;
    sgbm_->setP1(P1);
    sgbm_->setP2(P2);
    sgbm_->setMode(cv::StereoSGBM::MODE_SGBM);
    sgbm_->setPreFilterCap(63);
    sgbm_->setUniquenessRatio(10);
    sgbm_->setSpeckleWindowSize(100);
    sgbm_->setSpeckleRange(32);
    sgbm_->setDisp12MaxDiff(1);

    // load camera params and init rectify maps
    if (!loadCameraParameters())
    {
        ROS_ERROR("Failed to load camera parameters. Node will run but rectification is disabled.");
    }

    // sync tolerance example (set std::max interval to 20ms)
    sync_.setMaxIntervalDuration(ros::Duration(0.02));
    sync_.registerCallback(boost::bind(&StereoDepth::stereoCallback, this, _1, _2));

    ROS_INFO("StereoDepth initialized.");
}

bool StereoDepth::loadCameraParameters()
{
    // Expect parameters in private namespace as vectors
    std::vector<double> K1v, D1v, K2v, D2v, Rv, Tv;
    if (!pnh_.getParam("K_left", K1v) ||
        !pnh_.getParam("D_left", D1v) ||
        !pnh_.getParam("K_right", K2v) ||
        !pnh_.getParam("D_right", D2v) ||
        !pnh_.getParam("R", Rv) ||
        !pnh_.getParam("T", Tv) ||
        !pnh_.getParam("image_width", image_size_.width) ||
        !pnh_.getParam("image_height", image_size_.height))
    {
        ROS_WARN("Camera params not fully provided on param server.");
        return false;
    }

    K1_ = cv::Mat(3, 3, CV_64F, K1v.data()).clone();
    K2_ = cv::Mat(3, 3, CV_64F, K2v.data()).clone();
    D1_ = cv::Mat(D1v).clone();
    D2_ = cv::Mat(D2v).clone();
    R_ = cv::Mat(3, 3, CV_64F, Rv.data()).clone();
    T_ = cv::Mat(3, 1, CV_64F, Tv.data()).clone();

    // fill intrinsics for depth formula
    fx_ = K1_.at<double>(0, 0);
    fy_ = K1_.at<double>(1, 1);
    cx_ = K1_.at<double>(0, 2);
    cy_ = K1_.at<double>(1, 2);
    baseline_ = std::abs(T_.at<double>(0, 0));

    // compute rectification maps
    cv::stereoRectify(K1_, D1_, K2_, D2_, image_size_, R_, T_, R1_, R2_, P1_, P2_, Q_,
                      cv::CALIB_ZERO_DISPARITY, -1, image_size_);
    std::cout << "init R1_:\n"
              << R1_ << "\n\n";
    std::cout << "init P1_:\n"
              << P1_ << "\n\n";
    std::cout << "init R2_:\n"
              << R2_ << "\n\n";
    std::cout << "init P2_:\n"
              << P2_ << "\n\n";
    std::cout << "init Q_:\n"
              << Q_ << "\n\n";
    cv::initUndistortRectifyMap(K1_, D1_, R1_, P1_, image_size_, CV_32FC1, map11_, map12_);
    cv::initUndistortRectifyMap(K2_, D2_, R2_, P2_, image_size_, CV_32FC1, map21_, map22_);

    // pre-allocate buffers
    left_rect_gray_.create(image_size_, CV_8UC1);
    right_rect_gray_.create(image_size_, CV_8UC1);
    left_rect_color_.create(image_size_, CV_8UC3);
    disp_raw_.create(image_size_, CV_16S);
    disp_float_.create(image_size_, CV_32F);
    depth_float_.create(image_size_, CV_32F);

    ROS_INFO("Loaded camera parameters and prepared rectification maps.");
    return true;
}

void StereoDepth::stereoCallback(const sensor_msgs::ImageConstPtr &left_msg,
                                 const sensor_msgs::ImageConstPtr &right_msg)
{
    // convert once to cv::Mat (BGR)
    cv::Mat left_bgr, right_bgr;
    try
    {
        left_bgr = cv_bridge::toCvShare(left_msg, "bgr8")->image;
        right_bgr = cv_bridge::toCvShare(right_msg, "bgr8")->image;
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat left_rect_color_, right_rect_color_;
    // rectify + undistort (remap)
    if (!is_initial_R_)
    {
        R_ = RefineR(left_bgr, right_bgr, K1_);
        cv::stereoRectify(K1_, D1_, K2_, D2_, image_size_, R_, T_, R1_, R2_, P1_, P2_, Q_,
                          cv::CALIB_ZERO_DISPARITY, -1, image_size_);
        cv::initUndistortRectifyMap(K1_, D1_, R1_, P1_, image_size_, CV_32FC1, map11_, map12_);
        cv::initUndistortRectifyMap(K2_, D2_, R2_, P2_, image_size_, CV_32FC1, map21_, map22_);
        std::cout << "refine R1_:\n"
                  << R1_ << "\n\n";
        std::cout << "refine P1_:\n"
                  << P1_ << "\n\n";
        std::cout << "refine R2_:\n"
                  << R2_ << "\n\n";
        std::cout << "refine P2_:\n"
                  << P2_ << "\n\n";
        std::cout << "refine Q_:\n"
                  << Q_ << "\n\n";
        is_initial_R_ = true;
    }
    if (!map11_.empty())
    {
        cv::remap(left_bgr, left_rect_color_, map11_, map12_, cv::INTER_LINEAR);
        cv::remap(right_bgr, right_rect_color_, map21_, map22_, cv::INTER_LINEAR);
    }
    else
    {
        // if maps not available, use raw images
        left_rect_color_ = left_bgr;
        right_rect_color_ = right_bgr;
    }
    // ensure grayscale versions
    if (left_rect_color_.channels() == 3)
    {
        cv::cvtColor(left_rect_color_, left_rect_gray_, cv::COLOR_BGR2GRAY);
    }
    else
    {
        left_rect_gray_ = left_rect_color_.clone();
    }

    if (right_rect_color_.channels() == 3)
    {
        cv::cvtColor(right_rect_color_, right_rect_gray_, cv::COLOR_BGR2GRAY);
    }
    else
    {
        right_rect_gray_ = right_rect_color_.clone();
    }

    if (left_rect_gray_.empty() || right_rect_gray_.empty())
    {
        ROS_WARN("Empty rectified images, skipping frame.");
        return;
    }

    cv::Mat mask;
    // compute disparity
    computeDisparity(left_rect_gray_, right_rect_gray_, disp_float_, mask);

    // insertDepth32f(disp_float_);
    // compute depth map and pointcloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    computeDepthAndPointcloud(disp_float_, left_rect_color_, depth_float_, cloud);

    // create colored depth for rviz
    double max_depth_display = 20.0; // 最远显示 10m
    cv::Mat depth_norm, depth_color, depth_clipped;
    cv::threshold(depth_float_, depth_clipped, max_depth_display, max_depth_display, cv::THRESH_TRUNC);
    cv::normalize(depth_clipped, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_JET); // depth_color 为彩色
    // publish images and cloud
    publishImagesAndCloud(left_msg, disp_float_, depth_float_, depth_color, cloud);
}

void StereoDepth::computeDisparity(const cv::Mat &left_gray, const cv::Mat &right_gray, cv::Mat &disp_out, cv::Mat &mask)
{
    cv::Mat leftImageCorr, rightImageCorr;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, Size(8, 8));
    clahe->apply(left_gray, left_gray);
    clahe->apply(right_gray, right_gray);

    cv::GaussianBlur(left_gray, left_gray, Size(3, 3), 0);
    cv::GaussianBlur(right_gray, right_gray, Size(3, 3), 0);

    cv::medianBlur(left_gray, leftImageCorr, 9);
    cv::medianBlur(right_gray, rightImageCorr, 9);

    int blockSize = 5;
    int cn = leftImageCorr.channels();
    Size img_size = leftImageCorr.size();
    int P1 = 8 * cn * blockSize * blockSize;
    int P2 = 32 * cn * blockSize * blockSize;
    int window_size = 3; // 对应你的 window_size
    // int numberOfDisparities = ((img_size.width / 8) + 15) & -16;
    auto left_matcher = cv::StereoSGBM::create(
        0,          // minDisparity
        5 * 16,     // numDisparities
        window_size // blockSize
    );

    left_matcher->setP1(8 * 3 * window_size);
    left_matcher->setP2(32 * 3 * window_size);
    left_matcher->setMinDisparity(0);
    left_matcher->setDisp12MaxDiff(1);
    left_matcher->setUniquenessRatio(10);
    left_matcher->setSpeckleWindowSize(100);
    left_matcher->setSpeckleRange(32);
    left_matcher->setPreFilterCap(63);
    left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    auto right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

    cv::Mat left_disp, right_disp;
    left_matcher->compute(leftImageCorr, rightImageCorr, left_disp);
    right_matcher->compute(rightImageCorr, leftImageCorr, right_disp);
    double lambda = 8000.0;
    double sigma = 1.5;
    cv::Mat filtered_disp;
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    wls_filter->setLRCthresh(3);
    wls_filter->filter(left_disp, leftImageCorr, filtered_disp, right_disp);
    // cv::Mat filtered_disp_16;
    filtered_disp.convertTo(disp_out, CV_32F, 1. / 16);
    mask = cv::Mat::zeros(left_disp.size(), CV_8U);
    float lr_thresh = 3.0f; // 左右一致性阈值（像素）
    for (int y = 0; y < left_disp.rows; y++)
    {
        for (int x = 0; x < left_disp.cols; x++)
        {
            float dL = left_disp.at<short>(y, x);
            if (dL <= 0.0f)
                continue; // 无效点

            int xr = cvRound(x - (dL / 16.0)); // 在右图的匹配位置
            if (xr >= 0 && xr < right_disp.cols)
            {
                float dR = right_disp.at<short>(y, xr);
                if (dR >= 0.0f)
                    continue;
                // 检查左右视差差异
                if (std::fabs(dL + dR) < lr_thresh * 16)
                {
                    mask.at<uchar>(y, x) = 255; // 有效点
                }
            }
        }
    }
}

void StereoDepth::computeDepthAndPointcloud(const cv::Mat &disp, const cv::Mat &left_color,
                                            cv::Mat &depth_out, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, const cv::Mat mask)
{
    // reproject to 3D using Q matrix (efficient native op)
    cv::Mat points3d;                                 // CV_32FC3
    cv::reprojectImageTo3D(disp, points3d, Q_, true); // handleMissingValues=true

    // prepare cloud
    cloud->header.frame_id = "stereo_camera";
    cloud->is_dense = false;
    cloud->points.clear();
    cloud->points.reserve(points3d.rows * points3d.cols / 8); // reserve some

    // fill depth_out and cloud in single pass (pointer access for speed)
    depth_out.create(disp.size(), CV_32F);
    for (int y = 0; y < disp.rows; ++y)
    {
        const float *dp = disp.ptr<float>(y);
        const cv::Vec3f *p3 = points3d.ptr<cv::Vec3f>(y);
        const cv::Vec3b *lc = left_color.ptr<cv::Vec3b>(y);
        float *depth_ptr = depth_out.ptr<float>(y);

        for (int x = 0; x < disp.cols; ++x)
        {
            float d = dp[x];
            cv::Vec3f pw = p3[x];
            float Z = pw[2];
            if ((!mask.empty() && mask.at<uchar>(y, x) == 0) || d <= 0 || !std::isfinite(Z) || Z <= 0.25f || Z > 20.0f)
            { // clamp invalid / too far
                depth_ptr[x] = 10000.0f;
                continue;
            }
            depth_ptr[x] = Z;

            pcl::PointXYZRGB pt;
            pt.x = pw[0];
            pt.y = pw[1];
            pt.z = Z;
            cv::Vec3b c = lc[x];
            pt.r = c[2];
            pt.g = c[1];
            pt.b = c[0];
            cloud->points.push_back(pt);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
}

void StereoDepth::publishImagesAndCloud(const sensor_msgs::ImageConstPtr &left_msg,
                                        const cv::Mat &disp,
                                        const cv::Mat &depth_float, const cv::Mat &depth_color,
                                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    // disparity float (32FC1)
    sensor_msgs::ImagePtr disp_msg = cv_bridge::CvImage(left_msg->header, "32FC1", disp).toImageMsg();
    pub_disparity_.publish(disp_msg);

    // depth float (32FC1) - for algorithms (do NOT show in RViz)
    sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(left_msg->header, "32FC1", depth_float).toImageMsg();
    pub_depth_.publish(depth_msg);

    // depth color (bgr8) - safe for RViz
    sensor_msgs::ImagePtr depth_color_msg = cv_bridge::CvImage(left_msg->header, "bgr8", depth_color).toImageMsg();
    pub_depth_color_.publish(depth_color_msg);

    // publish cloud
    sensor_msgs::PointCloud2 pc2;
    pcl::toROSMsg(*cloud, pc2);
    pc2.header = left_msg->header;
    pc2.header.frame_id = "map";
    pub_points_.publish(pc2);
}

void StereoDepth::insertDepth32f(cv::Mat &depth)
{
    const int width = depth.cols;
    const int height = depth.rows;
    float *data = (float *)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double *integral = (double *)integralMap.data;
    int *ptsIntegral = (int *)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 1e-3)
            {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
    // 积分区间
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 2;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = std::max(0, left);
                right = std::min(right, width - 1);
                top = std::max(0, top);
                bot = std::min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0)
                {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201)
        {
            s = 201;
        }
        cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
}

cv::Mat StereoDepth::RefineR(const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &K1)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kptsL;
    cv::Mat descL; // 光流不使用描述子，但用ORB检测关键点位置
    orb->detectAndCompute(img_left, cv::noArray(), kptsL, descL);

    // 提取左图关键点坐标
    std::vector<cv::Point2f> pts1;
    pts1.reserve(kptsL.size());
    for (auto &kp : kptsL)
        pts1.push_back(kp.pt);

    // -------------------------------
    // 2. 使用光流跟踪关键点到右图
    // -------------------------------
    std::vector<cv::Point2f> pts2;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        img_left, img_right, pts1, pts2, status, err,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001);

    // -------------------------------
    // 3. 筛选跟踪成功点
    // -------------------------------
    std::vector<cv::Point2f> good_pts1, good_pts2;
    for (size_t i = 0; i < pts1.size(); i++)
    {
        if (status[i])
        {
            good_pts1.push_back(pts1[i]);
            good_pts2.push_back(pts2[i]);
        }
    }

    std::cout << "Total keypoints: " << pts1.size() << std::endl;
    std::cout << "Tracked points: " << good_pts1.size() << std::endl;

    // 2. 用匹配点估计本质矩阵，自动剔除外点
    cv::Mat mask;
    cv::Mat E_est = findEssentialMat(pts1, pts2, K1, cv::RANSAC, 0.999, 1.0, mask);

    // 3. 分解本质矩阵恢复旋转和平移方向
    cv::Mat R_est, t_est;
    int inliers = recoverPose(E_est, pts1, pts2, K1, R_est, t_est, mask);

    std::cout << "recoverPose inliers: " << inliers << " / " << (int)pts1.size() << std::endl;
    std::cout << "R est \n"
              << R_est << std::endl;

    // 4. 只用估计旋转，保持标定平移
    return R_est;
}
