#ifndef STEREO_DEPTH_STEREO_DEPTH_H
#define STEREO_DEPTH_STEREO_DEPTH_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

class StereoDepth
{
public:
    StereoDepth(ros::NodeHandle &nh, ros::NodeHandle &pnh);
    ~StereoDepth() = default;
    void processStereo(cv::Mat &left, cv::Mat &right);

private:
    // callbacks
    void stereoCallback(const sensor_msgs::ImageConstPtr &left,
                        const sensor_msgs::ImageConstPtr &right);

    // setup & utils
    bool loadCameraParameters();
    void computeDisparity(const cv::Mat &left_gray, const cv::Mat &right_gray, cv::Mat &disp, cv::Mat &mask);
    void computeDepthAndPointcloud(const cv::Mat &disp, const cv::Mat &left_rect_color,
                                   cv::Mat &depth_float, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, const cv::Mat mask = cv::Mat());
    void publishImagesAndCloud(const sensor_msgs::ImageConstPtr &left_msg,
                               const cv::Mat &disp, const cv::Mat &depth_float, const cv::Mat &depth_color,
                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
    cv::Mat RefineR(const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &K1);
    void insertDepth32f(cv::Mat &depth);

    // ROS
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    image_transport::ImageTransport it_;
    // message_filters ApproximateTime subscribers
    message_filters::Subscriber<sensor_msgs::Image> left_sub_;
    message_filters::Subscriber<sensor_msgs::Image> right_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync_;

    ros::Publisher pub_disparity_;
    ros::Publisher pub_depth_;
    ros::Publisher pub_depth_color_;
    ros::Publisher pub_points_;

    // Camera / rectification
    cv::Mat K1_, D1_, K2_, D2_, R_, T_;
    cv::Mat R1_, R2_, P1_, P2_, Q_; // stereoRectify outputs
    cv::Mat map11_, map12_, map21_, map22_;
    cv::Size image_size_;
    double fx_, fy_, cx_, cy_;
    double baseline_; // in same units as T (usually meters)

    // SGBM
    cv::Ptr<cv::StereoSGBM> sgbm_;
    int min_disp_, num_disp_, block_size_;
    bool is_initial_R_ = false;

    // reusable buffers to avoid allocations
    cv::Mat left_rect_gray_, right_rect_gray_, left_rect_color_;
    cv::Mat disp_raw_;    // CV_16S
    cv::Mat disp_float_;  // CV_32F (disparity in pixels)
    cv::Mat depth_float_; // CV_32F depth (meters)
};

#endif // STEREO_DEPTH_STEREO_DEPTH_H
