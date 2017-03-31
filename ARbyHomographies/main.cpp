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
bool checkMatches = false;
std::vector<Point2f> obj_corners;

//Get the coordinates by mouse
void onMouse( int event, int x, int y, int, void* )
{
    if( event != CV_EVENT_LBUTTONDOWN )
        return;
    
    Point2f pt = Point2f(x,y);
    obj_corners.push_back(pt);
    std::cout<<"x="<<pt.x<<"\t y="<<pt.y<<"\n";
    
}

int main(int argc, const char * argv[]) {
    
    //read file
    std::vector<String> fileNames;
    String folder = "/Users/boyang/workspace/ARbyHomographies/src2/";
    glob(folder, fileNames);
    
    //load first frame
    Mat img_object = imread(fileNames[0], IMREAD_GRAYSCALE );
    resize(img_object, img_object, Size(640, 480));
    
    if( !img_object.data)
    {
        std::cout<< " --(!) Error reading images " << std::endl; return -1;
    }
    
    //show the bounding box in the object image
    Mat img_object_copy;
    img_object.copyTo(img_object_copy);
    cvtColor(img_object_copy, img_object_copy, COLOR_GRAY2BGR);
    
    namedWindow("ori_image_col");
    setMouseCallback( "ori_image_col", onMouse, 0 );
    
    while(obj_corners.size() < 4){
        

        //draw bounding box
//        line( img_object_copy, obj_corners[0], obj_corners[1], Scalar(0, 255, 0), 4 );
//        line( img_object_copy, obj_corners[1], obj_corners[2], Scalar( 0, 255, 0), 4 );
//        line( img_object_copy, obj_corners[2], obj_corners[3], Scalar( 0, 255, 0), 4 );
//        line( img_object_copy, obj_corners[3], obj_corners[0], Scalar( 0, 255, 0), 4 );
        
        //Use mouse to get coordinates
        imshow( "ori_image_col", img_object_copy );
        
        waitKey(0);
    
    }
    
    
    //init detector
    int minHessian = 400;
    
    Ptr<SURF> detector = SURF::create( minHessian );
    
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    
    detector->detect( img_object, keypoints_object );
    
    //init extractor
    Ptr<SURF> extractor = SURF::create();
    
    Mat descriptors_object, descriptors_scene;
    
    extractor->compute( img_object, keypoints_object, descriptors_object );
    
    
    //init matcher
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    
    
    //loop through all the image in the file
    for(size_t i = 1 ; i < fileNames.size() ; i++){
        //load next image
        Mat img_scene = imread(fileNames[i], IMREAD_GRAYSCALE );
        resize(img_scene, img_scene, Size(640, 480));
        
        if( !img_scene.data)
        {
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
        
        double max_dist = 0; double min_dist = 100;
        
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_object.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        
//        printf("-- Max dist : %f \n", max_dist );
//        printf("-- Min dist : %f \n", min_dist );
        
        //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        std::vector< DMatch > good_matches;
        
        for( int i = 0; i < descriptors_object.rows; i++ )
        {
            if( matches[i].distance < 3*min_dist ){
                good_matches.push_back( matches[i]);
            }
        }
        
        Mat img_matches;
        drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;
        
        
        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        }
        
        if(good_matches.size() >= 4){
            Mat H = findHomography( obj, scene, RANSAC );
            
            //-- Get the corners from the image_1 ( the object to be "detected" )
            
            //    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
            //    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
       
            std::vector<Point2f> scene_corners(4);
            perspectiveTransform( obj_corners, scene_corners, H);
            
            // -- Draw lines between the corners (the mapped object in the scene - image_2 )
            if(checkMatches){
            // -- Draw lines between the corners (the mapped object in the scene - image_2 )
                line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
                line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                
                imshow( "Good Matches & Object detection", img_matches );
            }

          // -- Draw lines between the corners (the mapped object in the scene - image_2 )
//            line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
//            line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//            line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//            line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//
//            imshow( "Good Matches & Object detection", img_matches );
            
            
            cvtColor(img_scene, img_scene, COLOR_GRAY2BGR);
            line( img_scene, scene_corners[0] , scene_corners[1] , Scalar(0, 255, 0), 4 );
            line( img_scene, scene_corners[1] , scene_corners[2] , Scalar( 0, 255, 0), 4 );
            line( img_scene, scene_corners[2] , scene_corners[3] , Scalar( 0, 255, 0), 4 );
            line( img_scene, scene_corners[3] , scene_corners[0] , Scalar( 0, 255, 0), 4 );
        
        }else{
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
