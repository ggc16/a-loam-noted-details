// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);    // 文件指针指向文件末尾
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);    // 统计一下文件有多少float数据
    lidar_data_file.seekg(0, std::ios::beg);    // 再把指针指向文件开始

    std::vector<float> lidar_data_buffer(num_elements);
    // 读取所有的数据
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

int main(int argc, char** argv)
{
    //初始化节点
    ros::init(argc, argv, "kitti_helper");
    // ~ 发布的topic  会 以  节点名字+话题名字  发布  
    ros::NodeHandle n("~"); //n 是局部命名空间
    
    // 程序从参数服务器中读取的参数都是从 launch文件中设置的 需要设置的已注释
    
    // 声明字符串  相关的参数设置 在launch中设置的  数据的路径  数据的sequence 输出的bag路径
    std::string dataset_folder, sequence_number, output_bag_file;
    //从参数服务器获取 数据路径
    n.getParam("dataset_folder", dataset_folder);
    //从参数服务器获取 数据的sequence
    n.getParam("sequence_number", sequence_number);
    std::cout << "Reading sequence " << sequence_number << " from " << dataset_folder << '\n';
    // 声明 变量 是否需要 生成bag
    bool to_bag;
    //从参数服务器获取 是否需要 生成bag
    n.getParam("to_bag", to_bag);
    //如果生成bag
    if (to_bag)
        n.getParam("output_bag_file", output_bag_file);//从参数服务器获取 
    //声明 变量 发布延时
    int publish_delay;
    //从参数服务器获取 发布延时
    n.getParam("publish_delay", publish_delay);
    //如果小于0 则 设置为1
    publish_delay = publish_delay <= 0 ? 1 : publish_delay;
    
    // 声明设置 要通过 ros 发布的topic 句柄
    // 有:点云的\图像的\里程计的\路径的

    //lidar 的点云 发布句柄
    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 2);
    //图像 的 发布句柄
    image_transport::ImageTransport it(n);
    image_transport::Publisher pub_image_left = it.advertise("/image_left", 2);
    image_transport::Publisher pub_image_right = it.advertise("/image_right", 2);
    //里程计的发布句柄
    ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry> ("/odometry_gt", 5);
    //声明一个 里程计
    nav_msgs::Odometry odomGT;
    odomGT.header.frame_id = "/camera_init";   //赋值里程计的坐标系 父
    odomGT.child_frame_id = "/ground_truth";   //赋值里程计的坐标系 子
    //声明路径的 消息 发布句柄
    ros::Publisher pubPathGT = n.advertise<nav_msgs::Path> ("/path_gt", 5);
    nav_msgs::Path pathGT;  //声明一个路径变量
    pathGT.header.frame_id = "/camera_init";  //该坐标系下的位姿
    
    //读取 时间戳的文件，一个时间戳对应一组数据 ,后面的 点云\图像的消息的时间戳则赋值现在设置的 
    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt"; //时间戳文件路径
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);
    //读取 路径的 文件
    std::string ground_truth_path = "results/" + sequence_number + ".txt";
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);
    //记录 rosbag，根据参数设置 决定 是否 生成 rosbag 文件
    rosbag::Bag bag_out;
    if (to_bag)
        bag_out.open(output_bag_file, rosbag::bagmode::Write);
    //方向设置矩阵  本质上就是一个旋转矩阵，位姿传感器的坐标系 方向 和 lidar 图像 的坐标系 方向 不一致
    Eigen::Matrix3d R_transform;
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Quaterniond q_transform(R_transform);

    // 声明 读取行数后保存的字符串
    // 声明读取的行数 控制读取的 图像 lidar 的 数据序号
    // 控制发布频率 publish_delay 这个值 从参数服务器中来 也就是 launch中

    std::string line; //声明 读取行数后保存的字符
    std::size_t line_num = 0;//声明读取的行数 控制读取的 图像  lidar 的 数据序号
    //设置话题的发布频率
    ros::Rate r(10.0 / publish_delay);
    // 遍历时间戳这个文本文件
    std::cout << "timestamp_file " << std::string(dataset_folder + timestamp_path) << std::endl;
    //循环读 时间戳 文件 的 每一行 
    while (std::getline(timestamp_file, line) && ros::ok())
    {
        // 把string转成浮点型float
        float timestamp = stof(line);
        //声明 左右图像的路径
        std::stringstream left_image_path, right_image_path;
        // 找到对应的左右目的图片，设置 左图片的路径
        left_image_path << dataset_folder << "sequences/" + sequence_number + "/image_0/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        //通过opencv的方法 读取图片
        cv::Mat left_image = cv::imread(left_image_path.str(), CV_LOAD_IMAGE_GRAYSCALE);
        //设置 右图片的路径
        right_image_path << dataset_folder << "sequences/" + sequence_number + "/image_1/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        //通过opencv的方法 读取图片
        cv::Mat right_image = cv::imread(left_image_path.str(), CV_LOAD_IMAGE_GRAYSCALE);
        
        // 得到ground truth的文件，读取 ground_truth的一行
        std::getline(ground_truth_file, line);
        std::stringstream pose_stream(line);
        std::string s;
        Eigen::Matrix<double, 3, 4> gt_pose; //声明位姿 矩阵 3行4列[R,t]
        // 得到这个变换矩阵T
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                std::getline(pose_stream, s, ' ');
                gt_pose(i, j) = stof(s); //给位姿矩阵赋值
            }
        }
        // 相机坐标系转到前左上的坐标系
        Eigen::Quaterniond q_w_i(gt_pose.topLeftCorner<3, 3>()); //提取位姿矩阵的角度部分 并转成四元数
        Eigen::Quaterniond q = q_transform * q_w_i;  //进行 方向的 旋转
        q.normalize();//归一化
        Eigen::Vector3d t = q_transform * gt_pose.topRightCorner<3, 1>();//提取位姿矩阵的位置部分
        // 赋值里程计的 内容 发布里程计
        odomGT.header.stamp = ros::Time().fromSec(timestamp);
        odomGT.pose.pose.orientation.x = q.x();
        odomGT.pose.pose.orientation.y = q.y();
        odomGT.pose.pose.orientation.z = q.z();
        odomGT.pose.pose.orientation.w = q.w();
        odomGT.pose.pose.position.x = t(0);
        odomGT.pose.pose.position.y = t(1);
        odomGT.pose.pose.position.z = t(2);
        //发布里程计
        pubOdomGT.publish(odomGT);
        
        //赋值路径的内容 发布路径
        geometry_msgs::PoseStamped poseGT;
        poseGT.header = odomGT.header;
        poseGT.pose = odomGT.pose.pose;
        pathGT.header.stamp = odomGT.header.stamp;
        pathGT.poses.push_back(poseGT); 
        //发布路径
        pubPathGT.publish(pathGT);

        // 读取 lidar 的点云 
        std::stringstream lidar_data_path; //声明点云路径字符串
        // 获取lidar数据的文件名
        lidar_data_path << dataset_folder << "velodyne/sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";
        //读取lidar 的数据  以 [x y z i x y z i ....] 存储在向量中
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        //4个元素是一个点 打印一共多少个点
        std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n";
       
        // 将lidar 的信息 存入 PCL形式的点云中
        
        std::vector<Eigen::Vector3d> lidar_points; //声明雷达点 的位置 向量
        std::vector<float> lidar_intensities;    //声明雷达点 的 强度 的 向量
        pcl::PointCloud<pcl::PointXYZI> laser_cloud; //声明 一个 PCL 的 点云
        // 每个点数据占四个float数据，分别是xyz，intensity
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            //存入lidar位置
            lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
            //存入lidar 强度
            lidar_intensities.push_back(lidar_data[i+3]);
            // 构建pcl的点云格式
            pcl::PointXYZI point;  //声明一个PCL的点
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            point.intensity = lidar_data[i + 3];
            laser_cloud.push_back(point); //将点存入点云
        }
        // 转成ros的消息格式
        sensor_msgs::PointCloud2 laser_cloud_msg; //声明一个ros的点云
        pcl::toROSMsg(laser_cloud, laser_cloud_msg); //将pcl的点云转成ros的
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);//赋值时间戳
        laser_cloud_msg.header.frame_id = "/camera_init"; //声明坐标系
        pub_laser_cloud.publish(laser_cloud_msg);// 发布点云数据
        // 图片也转成ros的消息发布出去
        sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", left_image).toImageMsg();
        sensor_msgs::ImagePtr image_right_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", right_image).toImageMsg();
        pub_image_left.publish(image_left_msg);
        pub_image_right.publish(image_right_msg);
        // 也可以写到rosbag包中去
        if (to_bag)
        {   
            //写入 rosbag
            bag_out.write("/image_left", ros::Time::now(), image_left_msg);
            bag_out.write("/image_right", ros::Time::now(), image_right_msg);
            bag_out.write("/velodyne_points", ros::Time::now(), laser_cloud_msg);
            bag_out.write("/path_gt", ros::Time::now(), pathGT);
            bag_out.write("/odometry_gt", ros::Time::now(), odomGT);
        }

        line_num ++;  //更新行数  也相当于 更新 读取的 文件 序号
        r.sleep();    //进行延时 控制发布频率
    }
    bag_out.close();  // 关闭rosbag
    std::cout << "Done \n";


    return 0;
}