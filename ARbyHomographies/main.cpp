//
//  main.cpp
//  ARbyHomographies
//
//  Created by boyang on 3/06/17.
//  Copyright © 2017 boyang. All rights reserved.
//
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

//golbal var
bool checkMatches = true;
std::vector<Point2f> obj_corners;

//Get the coordinates by mouse
void onMouse( int event, int x, int y, int, void* ) {
    if( event != CV_EVENT_LBUTTONDOWN )
        return;
    
    Point2f pt = Point2f(x,y);
    obj_corners.push_back(pt);
    std::cout<<"x="<<pt.x<<"\t y="<<pt.y<<"\n";
    
}

//Erase not interested region
void eraseContentOutOfRoi(Mat & img, Point2f topLeft, Point2f bottomRight) {
    for (int i = 0 ; i < img.rows ; i++) {
        for (int j = 0 ; j < img.cols ; j++) {
            if (j < topLeft.x || j > bottomRight.x || i < topLeft.y || i > bottomRight.y){
                img.at<Vec3b>(i,j)[0] = 0;
                img.at<Vec3b>(i, j)[1] = 0;
                img.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
}

//Check the output bounding box is reasonable or not
bool checkBoundingBox(std::vector<Point2f> scene_corners) {
    //Too small -> false
    if (scene_corners[1].x - scene_corners[0].x < 10 || scene_corners[3].y - scene_corners[0].y <10
        || scene_corners[2].x - scene_corners[3].x < 10 || scene_corners[2].y - scene_corners[1].y <10) {
        return false;
    }
    
    //Still kind of rectangle -> true
    if (scene_corners[0].x < scene_corners[1].x && scene_corners[0].y < scene_corners[3].y
        && scene_corners[2].x > scene_corners[3].x && scene_corners[2].y > scene_corners[1].y) {
        return true;
    }

    return false;
}

int main(int argc, const char * argv[]) {
    
    //read file
    std::vector<String> fileNames;
    String folder = "/Users/boyang/workspace/ARbyHomographies/src3/";
    glob(folder, fileNames);
    
    //load first frame
    Mat img_object = imread(fileNames[0], IMREAD_GRAYSCALE );
    resize(img_object, img_object, Size(640, 480));
    
    //Check load successful?
    if( !img_object.data) {
        std::cout<< " --(!) Error reading images " << std::endl; return -1;
    }
    
    //Select the bounding box in the image
    namedWindow("select ROI region");
    setMouseCallback( "select ROI region", onMouse, 0 );
    
    //Select at least 4 points and press any key
    while(obj_corners.size() < 4) {
        //Use mouse to get coordinates
        imshow( "select ROI region", img_object );
        
        waitKey(0);
    
    }
    
    
    //init detector
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    
    //adding of region of intrest
    Mat roi = img_object(Rect(obj_corners[0].x , obj_corners[0].y , obj_corners[1].x - obj_corners[0].x , obj_corners[3].y - obj_corners[0].y));
    
    //Erase not interested region
    cvtColor(img_object, img_object, COLOR_GRAY2BGR);
    eraseContentOutOfRoi(img_object, obj_corners[0], obj_corners[2]);
    cvtColor(img_object, img_object, COLOR_BGR2GRAY);
//    imshow( "temp", img_object );
    
    //detect points easy to track
    detector->detect( img_object, keypoints_object );
    
    //init extractor and compute
    Ptr<SURF> extractor = SURF::create();
    Mat descriptors_object, descriptors_scene;
    extractor->compute( img_object, keypoints_object, descriptors_object );
    
    //init matcher
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    
    //loop through all the image in the file
    for(size_t i = 1 ; i < fileNames.size() ; i++) {
        
        //load next image
        Mat img_scene = imread(fileNames[i], IMREAD_GRAYSCALE );
        resize(img_scene, img_scene, Size(640, 480));
        
        if( !img_scene.data) {
            std::cout<< " --(!) Error reading images " << std::endl; return -1;
        }
        
        // Detect the keypoints using SURF Detector
        detector->detect( img_scene, keypoints_scene );
        
        // Calculate descriptors (feature vectors)
        extractor->compute( img_scene, keypoints_scene, descriptors_scene );
        
        // Matching descriptor vectors using FLANN matcher
        //FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher->match( descriptors_object, descriptors_scene, matches );
        
        //Quick calculation of max and min distances between keypoints
        double max_dist = 0; double min_dist = 100;
        for ( int i = 0; i < matches.size(); i++ ) {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        
        //Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        std::vector< DMatch > good_matches;
        for( int i = 0; i < descriptors_object.rows; i++ ) {
            if( matches[i].distance < 3*min_dist ){
                good_matches.push_back( matches[i]);
            }
        }
        
        //Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;
        
        //Get the keypoints from the good matches
        for( int i = 0; i < good_matches.size(); i++ ) {
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }
        
        //Calculate Homagraphy
        if(good_matches.size() >= 4) {
            Mat H = findHomography( obj, scene, RANSAC );
            
            //perspectiveTransform the points
            std::vector<Point2f> scene_corners(4);
            perspectiveTransform( obj_corners, scene_corners, H);
            
            //Draw lines between the matches points
            if(checkMatches) {
                Mat img_matches;
                drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                            good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                            std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                
            //Draw tracking bounding box on the "chacking matching screen"
                line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
                line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                
                imshow( "Good Matches & Object detection", img_matches );
            }
            
            //Draw tracking bounding box
            if(checkBoundingBox(scene_corners)) {
                cvtColor(img_scene, img_scene, COLOR_GRAY2BGR);
                line( img_scene, scene_corners[0] , scene_corners[1] , Scalar(0, 255, 0), 4 );
                line( img_scene, scene_corners[1] , scene_corners[2] , Scalar( 0, 255, 0), 4 );
                line( img_scene, scene_corners[2] , scene_corners[3] , Scalar( 0, 255, 0), 4 );
                line( img_scene, scene_corners[3] , scene_corners[0] , Scalar( 0, 255, 0), 4 );
            } else {
                std::cout<<"Weird BoundingBox"<<std::endl;
            }
        
        } else {
            std::cout<< "don't have enough matches"  << std::endl;
        }
        
        //-- Show detected matches
        imshow( "After", img_scene );
        
        
        char key = waitKey(0);
        if(key == 'q'){
            return 0;
        }
    }
    
    return 0;
}
