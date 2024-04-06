/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "pose_graph.h"

PoseGraph::PoseGraph()
{
    earliest_loop_index = -1;
    t_drift = Eigen::Vector3d(0, 0, 0);
    yaw_drift = 0;
    r_drift = Eigen::Matrix3d::Identity();
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    global_index = 0;
    sequence_cnt = 0;
    sequence_loop.push_back(0);
    base_sequence = 1;
    use_imu = 0;
    is_running = true;
    flag_detect_loop_ = false;

    //myViewer = boost::shared_ptr<PangolinDSOViewer> (new PangolinDSOViewer(COL, ROW));

}

PoseGraph::~PoseGraph()
{
    myViewer->close();
    t_optimization.join();
    t_loopdetect.join();
}

void PoseGraph::setIMUFlag(bool _use_imu)
{
    use_imu = _use_imu;
    if(use_imu)
    {
        std::cout << "VIO input, perfrom 4 DoF (x, y, z, yaw) pose graph optimization" << std::endl;
        t_optimization = std::thread(&PoseGraph::optimize4DoF, this);
    }
    else
    {
        std::cout << "VO input, perfrom 6 DoF pose graph optimization" << std::endl;
        t_optimization = std::thread(&PoseGraph::optimize6DoF, this);
    }

    t_loopdetect = std::thread(&PoseGraph::processKeyFrame, this);

}

void PoseGraph::loadVocabulary(std::string voc_path)
{
    voc = new BriefVocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}

void PoseGraph::addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
    keyFrameBuf.push(cur_kf);
    flag_detect_loop_ = flag_detect_loop;

}

void PoseGraph::processKeyFrame()
{
    while(true)
    {
        if (keyFrameBuf.size() > 0)
        {
            KeyFrame* cur_kf = keyFrameBuf.front();
            keyFrameBuf.pop();
            //shift to base frame
            Vector3d vio_P_cur;
            Matrix3d vio_R_cur;
            if (sequence_cnt != cur_kf->sequence)
            {
                sequence_cnt++;
                sequence_loop.push_back(0);
                w_t_vio = Eigen::Vector3d(0, 0, 0);
                w_r_vio = Eigen::Matrix3d::Identity();
                m_drift.lock();
                t_drift = Eigen::Vector3d(0, 0, 0);
                r_drift = Eigen::Matrix3d::Identity();
                m_drift.unlock();
            }
            
            cur_kf->getVioPose(vio_P_cur, vio_R_cur);
            vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
            vio_R_cur = w_r_vio *  vio_R_cur;
            cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
            cur_kf->index = global_index;
            global_index++;
            int loop_index = -1;
            cur_kf->computeBRIEF();
            if (flag_detect_loop_)
            {
                TicToc tmp_t;
                loop_index = detectLoop(cur_kf, cur_kf->index);
            }
            else
            {
                addKeyFrameIntoVoc(cur_kf);
            }
            if (loop_index != -1)
            {
                //std::cout << cur_kf->index << " detect loop with " << loop_index << std::endl;
                KeyFrame* old_kf = getKeyFrame(loop_index);

                if (cur_kf->findConnection(old_kf))
                {
                    if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                        earliest_loop_index = loop_index;

                    Vector3d w_P_old, w_P_cur, vio_P_cur;
                    Matrix3d w_R_old, w_R_cur, vio_R_cur;
                    old_kf->getVioPose(w_P_old, w_R_old);
                    cur_kf->getVioPose(vio_P_cur, vio_R_cur);

                    Vector3d relative_t;
                    Quaterniond relative_q;
                    relative_t = cur_kf->getLoopRelativeT();
                    relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
                    w_P_cur = w_R_old * relative_t + w_P_old;
                    w_R_cur = w_R_old * relative_q;
                    double shift_yaw;
                    Matrix3d shift_r;
                    Vector3d shift_t; 
                    if(use_imu)
                    {
                        shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
                        shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
                    }
                    else
                        shift_r = w_R_cur * vio_R_cur.transpose();
                    shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur; 
                    // shift vio pose of whole sequence to the world frame
                    if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0)
                    //if (1)
                    {  
                        w_r_vio = shift_r;
                        w_t_vio = shift_t;
                        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                        vio_R_cur = w_r_vio *  vio_R_cur;
                        cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
                        list<KeyFrame*>::iterator it = keyframelist.begin();
                        for (; it != keyframelist.end(); it++)   
                        {
                            if((*it)->sequence == cur_kf->sequence)
                            {
                                Vector3d vio_P_cur;
                                Matrix3d vio_R_cur;
                                (*it)->getVioPose(vio_P_cur, vio_R_cur);
                                vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                                vio_R_cur = w_r_vio *  vio_R_cur;
                                (*it)->updateVioPose(vio_P_cur, vio_R_cur);
                            }
                        }
                        sequence_loop[cur_kf->sequence] = 1;
                    }
                    m_optimize_buf.lock();
                    optimize_buf.push(cur_kf->index);
                    m_optimize_buf.unlock();
                }
            }
            m_keyframelist.lock();
            Vector3d P;
            Matrix3d R;
            cur_kf->getVioPose(P, R);
            P = r_drift * P + t_drift;
            R = r_drift * R;
            cur_kf->updatePose(P, R);


            /*
            if (SAVE_LOOP_PATH)
            {
                ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
                loop_path_file.setf(ios::fixed, ios::floatfield);
                loop_path_file.precision(0);
                loop_path_file << cur_kf->time_stamp * 1e9 << ",";
                loop_path_file.precision(5);
                loop_path_file  << P.x() << ","
                    << P.y() << ","
                    << P.z() << ","
                    << Q.w() << ","
                    << Q.x() << ","
                    << Q.y() << ","
                    << Q.z() << ","
                    << endl;
                loop_path_file.close();
            }
            //draw local connection
            if (SHOW_S_EDGE)
            {
                list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin();
                for (int i = 0; i < 4; i++)
                {
                    if (rit == keyframelist.rend())
                        break;
                    Vector3d conncected_P;
                    Matrix3d connected_R;
                    if((*rit)->sequence == cur_kf->sequence)
                    {
                        (*rit)->getPose(conncected_P, connected_R);
                        //posegraph_visualization->add_edge(P, conncected_P);
                    }
                    rit++;
                }
            }
            if (SHOW_L_EDGE)
            {
                if (cur_kf->has_loop)
                {
                    //std::cout << "has loop" << std::endl;
                    KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
                    Vector3d connected_P,P0;
                    Matrix3d connected_R,R0;
                    connected_KF->getPose(connected_P, connected_R);
                    //cur_kf->getVioPose(P0, R0);
                    cur_kf->getPose(P0, R0);
                    if(cur_kf->sequence > 0)
                    {
                        //std::cout << "add loop into visual" << std::endl;
                        //posegraph_visualization->add_loopedge(P0, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
                    }
                    
                }
            }
            */
            //posegraph_visualization->add_pose(P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0), Q);

            keyframelist.push_back(cur_kf);

            vector<Sophus::SE3f> traj;
            vector<Vector3f> points3d;

            if (loop_index != -1) // || true)
            {
                //traj.resize(keyframelist.size());
                for (auto it : keyframelist)
                {
                    Vector3d vio_P_cur = it->T_w_i;
                    Matrix3d vio_R_cur = it->R_w_i;
                    Eigen::Quaternionf q(vio_R_cur.cast<float>());
                    traj.push_back(Sophus::SE3<float>(q, vio_P_cur.cast<float>()));
                    for (auto p : it->point3ds)
                    {
                        points3d.push_back((vio_R_cur * p.point3d_w + vio_P_cur).cast<float>());
                    }
                }
                myViewer->publishPointPoseFrame(traj, points3d, cur_kf->image_result, true);
            }
            else
            {
                //traj.resize(1);
                Vector3d vio_P_cur = cur_kf->T_w_i;
                Matrix3d vio_R_cur = cur_kf->R_w_i;
                Eigen::Quaternionf q(vio_R_cur.cast<float>());
                traj.push_back(Sophus::SE3<float>(q, vio_P_cur.cast<float>()));
                for (auto p : cur_kf->point3ds)
                {
                    points3d.push_back((vio_R_cur * p.point3d_w + vio_P_cur).cast<float>());
                }
                myViewer->publishPointPoseFrame(traj, points3d, cur_kf->image_result, false);
            }
            
            if (0)
            {
                cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
                cv::imshow("image", cur_kf->image);
                cv::waitKey(1);
            }

            m_keyframelist.unlock();
        }

        if (!is_running && keyFrameBuf.empty())
            break;
            
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);

    }
}


void PoseGraph::loadKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
    cur_kf->index = global_index;
    global_index++;
    int loop_index = -1;
    if (flag_detect_loop)
       loop_index = detectLoop(cur_kf, cur_kf->index);
    else
    {
        addKeyFrameIntoVoc(cur_kf);
    }
    if (loop_index != -1)
    {
        std::cout << cur_kf->index << " detect loop with " << loop_index << std::endl;
        KeyFrame* old_kf = getKeyFrame(loop_index);
        if (cur_kf->findConnection(old_kf))
        {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;
            m_optimize_buf.lock();
            optimize_buf.push(cur_kf->index);
            m_optimize_buf.unlock();
        }
    }
    m_keyframelist.lock();
    Vector3d P;
    Matrix3d R;
    cur_kf->getPose(P, R);

    /*
    //draw local connection
    if (SHOW_S_EDGE)
    {
        list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin();
        for (int i = 0; i < 1; i++)
        {
            if (rit == keyframelist.rend())
                break;
            Vector3d conncected_P;
            Matrix3d connected_R;
            if((*rit)->sequence == cur_kf->sequence)
            {
                (*rit)->getPose(conncected_P, connected_R);
                //posegraph_visualization->add_edge(P, conncected_P);
            }
            rit++;
        }
    }
    */
    /*
    if (cur_kf->has_loop)
    {
        KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P,  connected_R);
        //posegraph_visualization->add_loopedge(P, connected_P, SHIFT);
    }
    */

    keyframelist.push_back(cur_kf);
    m_keyframelist.unlock();
}

KeyFrame* PoseGraph::getKeyFrame(int index)
{
//    unique_lock<mutex> lock(m_keyframelist);
    list<KeyFrame*>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)   
    {
        if((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return NULL;
}

int PoseGraph::detectLoop(KeyFrame* keyframe, int frame_index)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    //if (DEBUG_IMAGE)

    TicToc tmp_t;
    //first query; then add this frame into database!
    QueryResults ret;
    TicToc t_query;
    db.query(keyframe->brief_descriptors, ret, 4, frame_index - 50);
    //std::cout << "query time: " << t_query.toc() << std::endl;
    //std::cout << "Searching for Image " << frame_index << ". " << ret << std::endl;

    TicToc t_add;
    db.add(keyframe->brief_descriptors);
    //std::cout << "add feature time: " << t_add.toc() << std::endl;
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;
    cv::Mat loop_result;


    // a good match with its nerghbour
    if (ret.size() >= 1 &&ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {          
                find_loop = true;
            }

        }

    if (find_loop && frame_index > 50)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        return min_index;
    }
    else
        return -1;

}

void PoseGraph::addKeyFrameIntoVoc(KeyFrame* keyframe)
{
    db.add(keyframe->brief_descriptors);
}

void PoseGraph::optimize4DoF()
{
    while(true)
    {
        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while(!optimize_buf.empty())
        {
            cur_index = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();
        if (cur_index != -1)
        {
            std::cout << "optimize pose graph" << std::endl;
            TicToc tmp_t;
            m_keyframelist.lock();
            KeyFrame* cur_kf = getKeyFrame(cur_index);

            int max_length = cur_index + 1;

            // w^t_i   w^q_i
            double t_array[max_length][3];
            Quaterniond q_array[max_length];
            double euler_array[max_length][3];
            double sequence_array[max_length];

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //options.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
            //loss_function = new ceres::CauchyLoss(1.0);
            ceres::LocalParameterization* angle_local_parameterization =
                AngleLocalParameterization::Create();

            list<KeyFrame*>::iterator it;

            int i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                (*it)->local_index = i;
                Quaterniond tmp_q;
                Matrix3d tmp_r;
                Vector3d tmp_t;
                (*it)->getVioPose(tmp_t, tmp_r);
                tmp_q = tmp_r;
                t_array[i][0] = tmp_t(0);
                t_array[i][1] = tmp_t(1);
                t_array[i][2] = tmp_t(2);
                q_array[i] = tmp_q;

                Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                euler_array[i][0] = euler_angle.x();
                euler_array[i][1] = euler_angle.y();
                euler_array[i][2] = euler_angle.z();

                sequence_array[i] = (*it)->sequence;

                problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);

                if ((*it)->index == first_looped_index || (*it)->sequence == 0)
                {   
                    problem.SetParameterBlockConstant(euler_array[i]);
                    problem.SetParameterBlockConstant(t_array[i]);
                }

                //add edge
                for (int j = 1; j < 5; j++)
                {
                  if (i - j >= 0 && sequence_array[i] == sequence_array[i-j])
                  {
                    Vector3d euler_conncected = Utility::R2ypr(q_array[i-j].toRotationMatrix());
                    Vector3d relative_t(t_array[i][0] - t_array[i-j][0], t_array[i][1] - t_array[i-j][1], t_array[i][2] - t_array[i-j][2]);
                    relative_t = q_array[i-j].inverse() * relative_t;
                    double relative_yaw = euler_array[i][0] - euler_array[i-j][0];
                    ceres::CostFunction* cost_function = FourDOFError::Create( relative_t.x(), relative_t.y(), relative_t.z(),
                                                   relative_yaw, euler_conncected.y(), euler_conncected.z());
                    problem.AddResidualBlock(cost_function, NULL, euler_array[i-j], 
                                            t_array[i-j], 
                                            euler_array[i], 
                                            t_array[i]);
                  }
                }

                //add loop edge
                
                if((*it)->has_loop)
                {
                    assert((*it)->loop_index >= first_looped_index);
                    int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                    Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
                    Vector3d relative_t;
                    relative_t = (*it)->getLoopRelativeT();
                    double relative_yaw = (*it)->getLoopRelativeYaw();
                    ceres::CostFunction* cost_function = FourDOFWeightError::Create( relative_t.x(), relative_t.y(), relative_t.z(),
                                                                               relative_yaw, euler_conncected.y(), euler_conncected.z());
                    problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index], 
                                                                  t_array[connected_index], 
                                                                  euler_array[i], 
                                                                  t_array[i]);
                    
                }
                
                if ((*it)->index == cur_index)
                    break;
                i++;
            }
            m_keyframelist.unlock();

            ceres::Solve(options, &problem, &summary);
            //std::cout << summary.BriefReport() << std::endl;
            
            //std::cout << "pose optimization time: " << tmp_t.toc() << std::endl;
            /*
            for (int j = 0 ; j < i; j++)
            {
                std::cout << "optimize i: " << t_array[j][0] << " p: " << t_array[j][1] << " " << t_array[j][2] << std::endl; 
            }
            */
            m_keyframelist.lock();
            i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                Quaterniond tmp_q;
                tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
                Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                Matrix3d tmp_r = tmp_q.toRotationMatrix();
                (*it)-> updatePose(tmp_t, tmp_r);

                if ((*it)->index == cur_index)
                    break;
                i++;
            }

            Vector3d cur_t, vio_t;
            Matrix3d cur_r, vio_r;
            cur_kf->getPose(cur_t, cur_r);
            cur_kf->getVioPose(vio_t, vio_r);
            m_drift.lock();
            yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
            r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
            t_drift = cur_t - r_drift * vio_t;
            m_drift.unlock();
            //cout << "t_drift " << t_drift.transpose() << endl;
            //cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;
            //cout << "yaw drift " << yaw_drift << endl;

            it++;
            for (; it != keyframelist.end(); it++)
            {
                Vector3d P;
                Matrix3d R;
                (*it)->getVioPose(P, R);
                P = r_drift * P + t_drift;
                R = r_drift * R;
                (*it)->updatePose(P, R);
            }
            m_keyframelist.unlock();
        }

        if (!is_running && optimize_buf.empty())
            break;
        
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
    return;
}


void PoseGraph::optimize6DoF()
{
    while(true)
    {
        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while(!optimize_buf.empty())
        {
            cur_index = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();
        if (cur_index != -1)
        {
            std::cout << "optimize pose graph" << std::endl;
            TicToc tmp_t;
            m_keyframelist.lock();
            KeyFrame* cur_kf = getKeyFrame(cur_index);

            int max_length = cur_index + 1;

            // w^t_i   w^q_i
            double t_array[max_length][3];
            double q_array[max_length][4];
            double sequence_array[max_length];

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //ptions.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
            //loss_function = new ceres::CauchyLoss(1.0);
            ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

            list<KeyFrame*>::iterator it;

            int i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                (*it)->local_index = i;
                Quaterniond tmp_q;
                Matrix3d tmp_r;
                Vector3d tmp_t;
                (*it)->getVioPose(tmp_t, tmp_r);
                tmp_q = tmp_r;
                t_array[i][0] = tmp_t(0);
                t_array[i][1] = tmp_t(1);
                t_array[i][2] = tmp_t(2);
                q_array[i][0] = tmp_q.w();
                q_array[i][1] = tmp_q.x();
                q_array[i][2] = tmp_q.y();
                q_array[i][3] = tmp_q.z();

                sequence_array[i] = (*it)->sequence;

                problem.AddParameterBlock(q_array[i], 4, local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);

                if ((*it)->index == first_looped_index || (*it)->sequence == 0)
                {   
                    problem.SetParameterBlockConstant(q_array[i]);
                    problem.SetParameterBlockConstant(t_array[i]);
                }

                //add edge
                for (int j = 1; j < 5; j++)
                {
                    if (i - j >= 0 && sequence_array[i] == sequence_array[i-j])
                    {
                        Vector3d relative_t(t_array[i][0] - t_array[i-j][0], t_array[i][1] - t_array[i-j][1], t_array[i][2] - t_array[i-j][2]);
                        Quaterniond q_i_j = Quaterniond(q_array[i-j][0], q_array[i-j][1], q_array[i-j][2], q_array[i-j][3]);
                        Quaterniond q_i = Quaterniond(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
                        relative_t = q_i_j.inverse() * relative_t;
                        Quaterniond relative_q = q_i_j.inverse() * q_i;
                        ceres::CostFunction* vo_function = RelativeRTError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                                                                                0.1, 0.01);
                        problem.AddResidualBlock(vo_function, NULL, q_array[i-j], t_array[i-j], q_array[i], t_array[i]);
                    }
                }

                //add loop edge
                
                if((*it)->has_loop)
                {
                    assert((*it)->loop_index >= first_looped_index);
                    int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                    Vector3d relative_t;
                    relative_t = (*it)->getLoopRelativeT();
                    Quaterniond relative_q;
                    relative_q = (*it)->getLoopRelativeQ();
                    ceres::CostFunction* loop_function = RelativeRTError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                                                                                0.1, 0.01);
                    problem.AddResidualBlock(loop_function, loss_function, q_array[connected_index], t_array[connected_index], q_array[i], t_array[i]);                    
                }
                
                if ((*it)->index == cur_index)
                    break;
                i++;
            }
            m_keyframelist.unlock();

            ceres::Solve(options, &problem, &summary);
            //std::cout << summary.BriefReport() << "\n";
            
            //std::cout << "pose optimization time: " <<  tmp_t.toc() << std::endl;
            /*
            for (int j = 0 ; j < i; j++)
            {
                std::cout << "optimize i: " << t_array[j][0] << " p: " << t_array[j][1] << " " << t_array[j][2] << std::endl;
            }
            */
            m_keyframelist.lock();
            i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                Quaterniond tmp_q(q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]);
                Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                Matrix3d tmp_r = tmp_q.toRotationMatrix();
                (*it)-> updatePose(tmp_t, tmp_r);

                if ((*it)->index == cur_index)
                    break;
                i++;
            }

            Vector3d cur_t, vio_t;
            Matrix3d cur_r, vio_r;
            cur_kf->getPose(cur_t, cur_r);
            cur_kf->getVioPose(vio_t, vio_r);
            m_drift.lock();
            r_drift = cur_r * vio_r.transpose();
            t_drift = cur_t - r_drift * vio_t;
            m_drift.unlock();
            //cout << "t_drift " << t_drift.transpose() << endl;
            //cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;

            it++;
            for (; it != keyframelist.end(); it++)
            {
                Vector3d P;
                Matrix3d R;
                (*it)->getVioPose(P, R);
                P = r_drift * P + t_drift;
                R = r_drift * R;
                (*it)->updatePose(P, R);
            }
            m_keyframelist.unlock();
        }

        if (!is_running && optimize_buf.empty())
            break;

        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
    return;
}


void PoseGraph::savePoseGraph()
{
    m_keyframelist.lock();
    TicToc tmp_t;
    FILE *pFile;
    std::cout << "pose graph path: " << POSE_GRAPH_SAVE_PATH << std::endl;
    std::cout << "pose graph saving... " << std::endl;
    string file_path = POSE_GRAPH_SAVE_PATH + "/pose_graph.txt";
    pFile = fopen (file_path.c_str(),"w");
    //fprintf(pFile, "index time_stamp Tx Ty Tz Qw Qx Qy Qz loop_index loop_info\n");
    list<KeyFrame*>::iterator it;
    for (it = keyframelist.begin(); it != keyframelist.end(); it++)
    {
        std::string image_path, descriptor_path, brief_path, keypoints_path;

        Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i};
        Quaterniond PG_tmp_Q{(*it)->R_w_i};
        Vector3d VIO_tmp_T = (*it)->vio_T_w_i;
        Vector3d PG_tmp_T = (*it)->T_w_i;

        fprintf (pFile, " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d\n",(*it)->index, (*it)->time_stamp, 
                                    VIO_tmp_T.x(), VIO_tmp_T.y(), VIO_tmp_T.z(), 
                                    PG_tmp_T.x(), PG_tmp_T.y(), PG_tmp_T.z(), 
                                    VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(), 
                                    PG_tmp_Q.w(), PG_tmp_Q.x(), PG_tmp_Q.y(), PG_tmp_Q.z(), 
                                    (*it)->loop_index, 
                                    (*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2), (*it)->loop_info(3),
                                    (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6), (*it)->loop_info(7),
                                    (int)(*it)->keypoints.size());

        if (0)
        {
            // write keypoints, brief_descriptors   vector<cv::KeyPoint> keypoints vector<BRIEF::bitset> brief_descriptors;
            assert((*it)->keypoints.size() == (*it)->brief_descriptors.size());
            brief_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_briefdes.dat";
            std::ofstream brief_file(brief_path, std::ios::binary);
            keypoints_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_keypoints.txt";
            FILE *keypoints_file;
            keypoints_file = fopen(keypoints_path.c_str(), "w");
            for (int i = 0; i < (int)(*it)->keypoints.size(); i++)
            {
                brief_file << (*it)->brief_descriptors[i] << endl;
                fprintf(keypoints_file, "%f %f %f %f\n", (*it)->keypoints[i].pt.x, (*it)->keypoints[i].pt.y, 
                                                        (*it)->keypoints_norm[i].pt.x, (*it)->keypoints_norm[i].pt.y);
            }
            brief_file.close();
            fclose(keypoints_file);
        }
    }
    fclose(pFile);

    std::cout << "save pose graph time: " << tmp_t.toc() / 1000 << std::endl;
    m_keyframelist.unlock();
}

/*

void PoseGraph::loadPoseGraph()
{
    TicToc tmp_t;
    FILE * pFile;
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    std::cout << "lode pose graph from: " << file_path << std::endl;
    std::cout << "pose graph loading..." << std::endl;
    pFile = fopen (file_path.c_str(),"r");
    if (pFile == NULL)
    {
        std::cout << "lode previous pose graph error: wrong previous pose graph path or no previous pose graph \n the system will start with new pose graph" << std::endl;
        return;
    }
    int index;
    double time_stamp;
    double VIO_Tx, VIO_Ty, VIO_Tz;
    double PG_Tx, PG_Ty, PG_Tz;
    double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
    double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
    double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
    double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
    int loop_index;
    int keypoints_num;
    Eigen::Matrix<double, 8, 1 > loop_info;
    int cnt = 0;
    while (fscanf(pFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d", &index, &time_stamp, 
                                    &VIO_Tx, &VIO_Ty, &VIO_Tz, 
                                    &PG_Tx, &PG_Ty, &PG_Tz, 
                                    &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz, 
                                    &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz, 
                                    &loop_index,
                                    &loop_info_0, &loop_info_1, &loop_info_2, &loop_info_3, 
                                    &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7,
                                    &keypoints_num) != EOF) 
    {
        cv::Mat image;
        std::string image_path, descriptor_path;

        Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
        Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
        Quaterniond VIO_Q;
        VIO_Q.w() = VIO_Qw;
        VIO_Q.x() = VIO_Qx;
        VIO_Q.y() = VIO_Qy;
        VIO_Q.z() = VIO_Qz;
        Quaterniond PG_Q;
        PG_Q.w() = PG_Qw;
        PG_Q.x() = PG_Qx;
        PG_Q.y() = PG_Qy;
        PG_Q.z() = PG_Qz;
        Matrix3d VIO_R, PG_R;
        VIO_R = VIO_Q.toRotationMatrix();
        PG_R = PG_Q.toRotationMatrix();
        Eigen::Matrix<double, 8, 1 > loop_info;
        loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6, loop_info_7;

        if (loop_index != -1)
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
            {
                earliest_loop_index = loop_index;
            }

        // load keypoints, brief_descriptors   
        string brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_briefdes.dat";
        std::ifstream brief_file(brief_path, std::ios::binary);
        string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "r");
        vector<cv::KeyPoint> keypoints;
        vector<cv::KeyPoint> keypoints_norm;
        vector<BRIEF::bitset> brief_descriptors;
        for (int i = 0; i < keypoints_num; i++)
        {
            BRIEF::bitset tmp_des;
            brief_file >> tmp_des;
            brief_descriptors.push_back(tmp_des);
            cv::KeyPoint tmp_keypoint;
            cv::KeyPoint tmp_keypoint_norm;
            double p_x, p_y, p_x_norm, p_y_norm;
            if(!fscanf(keypoints_file,"%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm))
                std::cout <<"fail to load pose graph" << std::endl;
            tmp_keypoint.pt.x = p_x;
            tmp_keypoint.pt.y = p_y;
            tmp_keypoint_norm.pt.x = p_x_norm;
            tmp_keypoint_norm.pt.y = p_y_norm;
            keypoints.push_back(tmp_keypoint);
            keypoints_norm.push_back(tmp_keypoint_norm);
        }
        brief_file.close();
        fclose(keypoints_file);

        KeyFrame* keyframe = new KeyFrame(time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, loop_index, loop_info, keypoints, keypoints_norm, brief_descriptors);
        loadKeyFrame(keyframe, 0);
        if (cnt % 20 == 0)
        {
            publish();
        }
        cnt++;
    }
    fclose (pFile);
    std::cout << "load pose graph time: %f s\n", tmp_t.toc()/1000 << std::endl;
    base_sequence = 0;
}

*/