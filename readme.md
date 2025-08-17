# Stereo Depth Demo


euroc数据集，基于opencv，先进行本质矩阵计算，重新估计一次外参中的R，在进行立体校正及去畸变，最后使用sgbm进行立体匹配，基于视差图恢复深度，但目前感觉视差图仍然有些问题

# Useage

roslaunch stereo_depth stereo_depth.launch 

![demo](./data/finally.png "demo")