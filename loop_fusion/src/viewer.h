#ifndef VIEWER_H_
#define VIEWER_H_

#include <thread>
#include <mutex>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>

#include <iostream>
#include <unistd.h>

using namespace std;

//  Visualization for DSO

/**
 * viewer implemented by pangolin
 */
class PangolinDSOViewer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    PangolinDSOViewer(int w, int h, bool startRunThread = true);

    ~PangolinDSOViewer();

    void run();

    void close();

    void publishPointPoseFrame(std::vector<Sophus::SE3f>& trajs, std::vector<Eigen::Vector3f>& points3d, cv::Mat& _frame);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);
    // void pushLiveFrame( shared_ptr<Frame> image);

    /* call on finish */
    void join();

private:

    thread runThread;
    bool running = true;
    int w, h;
    //std::vector<Eigen::Vector3d> allFramePoses;  // trajectory
    std::vector<Sophus::SE3f> allFramePoses;  // trajectory
    std::vector<Eigen::Vector3f> mapPoints;
    cv::Mat frame;

    bool videoImgChanged = true;
    // 3D model rendering
    mutex model3DMutex;

    // timings
    struct timeval last_track;
    struct timeval last_map;

    std::deque<float> lastNTrackingMs;
    std::deque<float> lastNMappingMs;

};



#endif // LDSO_VIEWER_H_
