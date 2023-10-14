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

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include "estimator.h"
#include "viewer.h"

using namespace std;
using namespace Eigen;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);

	string config_file = argv[2];
    string vocab_path = argv[1];
    std::cout << "config_file: " << config_file << std::endl;
    std::cout << "vocabulary path: " << vocab_path << std::endl;

    Estimator estimator(vocab_path);
	readParameters(config_file);
	estimator.setParameter();

    boost::shared_ptr<PangolinDSOViewer> myViewer(new PangolinDSOViewer(COL, ROW));
    estimator.setViewer(myViewer);

	FILE* outFile;
	outFile = fopen((OUTPUT_FOLDER + "/vio.txt").c_str(),"w");
	if(outFile == NULL)
		std::cout << "Output path dosen't exist: " << OUTPUT_FOLDER << std::endl;

	cv::Mat imLeft, imRight;
	for (size_t i = 0; i < vstrImageLeft.size(); i++)
	{	

        imLeft = cv::imread(vstrImageLeft[i], CV_LOAD_IMAGE_GRAYSCALE );
        imRight = cv::imread(vstrImageRight[i], CV_LOAD_IMAGE_GRAYSCALE );

        estimator.inputImage(vTimestamps[i], imLeft, imRight);
        
        Eigen::Matrix<double, 4, 4> pose;
        estimator.getPoseInWorldFrame(pose);
        if(outFile != NULL)
            fprintf (outFile, "%f %f %f %f %f %f %f %f %f %f %f %f \n",pose(0,0), pose(0,1), pose(0,2),pose(0,3),
                                                                        pose(1,0), pose(1,1), pose(1,2),pose(1,3),
                                                                        pose(2,0), pose(2,1), pose(2,2),pose(2,3));
        
        // cv::imshow("leftImage", imLeft);
        // cv::imshow("rightImage", imRight);
        // cv::waitKey(2);

        //std::cout << "processing the " << (i+1) << " of " << vstrImageLeft.size() << std::endl;

	}
    estimator.shutdown(); 
	if(outFile != NULL)
		fclose (outFile);
    
    std::cout << "please input any to quit: ";
    std::cin.get();

	return 0;
}


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}
