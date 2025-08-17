#include "stereo_depth.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "stereo_depth_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    StereoDepth stereo_depth(nh, pnh);
    ros::spin();
    return 0;
}
