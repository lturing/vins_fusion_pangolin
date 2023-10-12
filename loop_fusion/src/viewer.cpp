#include <thread>
#include <pangolin/pangolin.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>

#include "viewer.h"

using namespace std;

// =================================================================================

PangolinDSOViewer::PangolinDSOViewer(int w, int h, bool startRunThread) {

    this->w = w;
    this->h = h;
    running = true;

    if (startRunThread)
        runThread = thread(&PangolinDSOViewer::run, this);
}

PangolinDSOViewer::~PangolinDSOViewer() {
    close();
    if (runThread.joinable()) {
        runThread.join();
    }
}

void PangolinDSOViewer::run() {

    pangolin::CreateWindowAndBind("Main", 3 * w, 2 * h);
    std::cout << "Create Pangolin DSO viewer" << endl;
    const int UI_WIDTH = 180;

    glEnable(GL_DEPTH_TEST);

    // 3D visualization
    pangolin::OpenGlRenderState Visualization3D_camera(
        pangolin::ProjectionMatrix(w, h, 400, 400, w / 2, h / 2, 0.1, 1000),
        pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY)
    );

    pangolin::View &Visualization3D_display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w / (float) h)
        .SetHandler(new pangolin::Handler3D(Visualization3D_camera));

    pangolin::View &d_video = pangolin::Display("imgVideo")
        .SetAspect(2 * w / (float) h);

    pangolin::GlTexture texVideo(2 * w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.2, pangolin::Attach::Pix(UI_WIDTH), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(d_video);

    // parameter reconfigure gui
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::Var<bool> settings_showLiveVideo("ui.showVideo", true, true);
    pangolin::Var<bool> settings_followCamera("ui.followCamera", false, true);
    pangolin::Var<bool> settings_showTrajectory("ui.showTrajectory", true, true);
    pangolin::Var<bool> settings_showPoints("ui.show3DPoint", true, true);

    // Default hooks for exiting (Esc) and fullscreen (tab).
    std::cout << "Looping viewer thread" << std::endl;
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();

    while (!pangolin::ShouldQuit() && running) {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        {
            unique_lock<mutex> lk3d(model3DMutex);
            {
                // allFramePoses
                //if (allFramePoses.size() > 0) Visualization3D_camera.Follow();
                
                Visualization3D_display.Activate(Visualization3D_camera);

                // trajectory
                if (settings_showTrajectory)
                {
                    if (settings_followCamera)
                    {
                        GetCurrentOpenGLCameraMatrix(Twc, Ow);
                        Visualization3D_camera.Follow(Twc);
                    }
                    
                    float colorRed[3] = {1, 0, 0};
                    glColor3f(colorRed[0], colorRed[1], colorRed[2]);
                    glLineWidth(3);
                    glBegin(GL_LINE_STRIP);
                    for (unsigned int i = 0; i < allFramePoses.size(); i++) {
                        Sophus::SE3f pose = allFramePoses[i];
                        glVertex3d(pose.translation()[0], pose.translation()[1], pose.translation()[2]);
                    }
                    glEnd(); 
                }

                if (settings_showPoints)
                {
                    float pointSize = 0.2;
                    glPointSize(pointSize);
                    glBegin(GL_POINTS);
                    glColor3f(0.0,0.0,0.0);

                    for(size_t i=0, iend=mapPoints.size(); i<iend;i++)
                    {
                        Eigen::Vector3f pos = mapPoints[i];
                        glVertex3f(pos(0),pos(1),pos(2));
                    }
                    glEnd();
                }

            }
            
            /*
            {
                model3DMutex.lock();
                float sd = 0;
                for (float d : lastNTrackingMs) sd += d;
                settings_trackFps = lastNTrackingMs.size() * 1000.0f / sd;
                model3DMutex.unlock();
            }
            */

            if (settings_showLiveVideo) {
                // https://github.com/stevenlovegrove/Pangolin/issues/682
                glPixelStorei(GL_UNPACK_ALIGNMENT,1);
                texVideo.Upload(frame.data, GL_BGR, GL_UNSIGNED_BYTE);
                d_video.Activate();
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                texVideo.RenderToViewportFlipY();
            }

        }
        // Swap frames and Process Events
        pangolin::FinishFrame();

        usleep(5000);
    }

    std::cout << "QUIT Pangolin thread!" << std::endl;
    std::cout << "So Long, and Thanks for All the Fish!" << std::endl;
}

void PangolinDSOViewer::close() {
    running = false;
}

void PangolinDSOViewer::join() {
    runThread.join();
    std::cout << "JOINED Pangolin thread!" << std::endl;
}

/*
void PangolinDSOViewer::publishPointPoseFrame(OdomStatus& status) {
    unique_lock<mutex> lk3d(model3DMutex);
    auto points = status.kf.points();
    float max_depth = 100.;

    for (int gr = 0; gr < points.rows(); ++gr) {
        for (int gc = 0; gc < points.cols(); ++gc) {
            const auto& point = points.at(gr, gc);
            // Only draw points with max info and within max depth
            if (!point.InfoMax() || (1.0 / point.idepth()) > max_depth) {
                continue;
            }
            if (point.PixelOk() && point.DepthOk())
            {
                // transform to fixed frame
                Eigen::Vector3f p_w = (status.kf.Twc() * point.pt()).cast<float>();
                mapPoints.push_back(p_w);
            }
        }
    }
    allFramePoses.push_back(status.kf.Twc().cast<float>());
    frame = status.frame_left.clone();
}
*/

void PangolinDSOViewer::publishPointPoseFrame(vector<Sophus::SE3f>& trajs, cv::Mat& _frame)
{
    unique_lock<mutex> lk3d(model3DMutex);
    allFramePoses.clear();
    allFramePoses.resize(trajs.size());
    for (int i = 0; i < trajs.size(); i++)
    {
        allFramePoses.push_back(trajs[i]); 
    }
    //cv::cvtColor(_frame, frame, cv::COLOR_GRAY2BGR);
    frame = _frame.clone();

}


void PangolinDSOViewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    if(allFramePoses.size() == 0)
        return; 
    Eigen::Matrix4f Twc;
    {
        //unique_lock<mutex> lock(model3DMutex); // dead lock
        Twc = allFramePoses.back().matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}



