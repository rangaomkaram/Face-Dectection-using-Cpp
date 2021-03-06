# Face-Dectection-using-Cpp
Face detection with C++ progamming with Qtcreator
# install opencv  C++ library using Cmake tool to work with project.
 we use the Haar classifier in a cascade to detect faces in an image
 ----------------------

create new project facedetect (Qt-widget)

...
in section Sources open mainwindow.cpp

above MainWindow::MainWindow(QWidget *parent) :

delete all lines and change it to:
```

#include "mainwindow.h"
#include "ui_mainwindow.h"

//opencv
//#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>

//c++
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;
```
------------------------------------------------------------------------------------

below ui->setupUi(this); we call our new function
    facedetect();

next, we define our new function at the end of our mainwindow.cpp file:	
```
void MainWindow::facedetect() {

    //-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
    String face_cascade_name = "U://opencvProjSB//UE_8//haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "U://opencvProjSB//UE_8//haarcascade_eye_tree_eyeglasses.xml";
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    std::vector<Rect> faces;
    namedWindow("Frame1",WINDOW_AUTOSIZE);   //create GUI window

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return; };
    cout << "cascades loaded" << endl;


    /*
    VideoCapture capture;

    //if source is camera:
    capture.open(0); //>1 = external webcam

    if(!capture.isOpened()){
        cout<<"ERROR ACQUIRING VIDEO FEED\n" << endl;
        printf( "error video opening");
        return;
    }
   */
    Mat tmp, frame1;

    //check for keyboard input
    while( waitKey(10) != 27)
    {

        //read first frame
        //capture.read(tmp);  //uncomment this line if camera
        tmp = imread("U://opencvProjSB//UE_8//people1.jpg", IMREAD_COLOR);
        cv::resize(tmp, frame1, Size(640,480));
        //cv::resize(tmp, frame1,Size(),0.6,0.6,INTER_CUBIC);
        Mat frame_gray = Mat::zeros( frame1.size(), CV_8U );
        cvtColor( frame1, frame_gray, COLOR_BGR2GRAY );
        equalizeHist( frame_gray, frame_gray );
        //-- Detect faces
        face_cascade.detectMultiScale( frame_gray, faces );
        //   face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        for( size_t i = 0; i < faces.size(); i++ )
        {
            Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
            ellipse( frame1, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );

            Mat faceROI = frame_gray( faces[i] );

            //-- In each face, detect eyes
//            std::vector<Rect> eyes;

//            eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAL_CMP_GE, Size(30, 30) );
//            eyes_cascade.detectMultiScale( faceROI, eyes );

//            for( size_t j = 0; j < eyes.size(); j++ )
//            {
//                Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
//                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//                circle( frame1, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
//            }
        }
        //-- Show what we got
        imshow( "Frame1", frame1 );

    }

    //destroy GUI windows
    destroyAllWindows();
    return;
}
```
 ---------------------------------
in section Headers open mainwindow.h and declare our 2 new functions
below the line ~MainWindow(); 
    void facedetect();
    	
-----------------------------------------------------------------------------------
save all
specify the includepath of the dlls at the end of our .pro file:
```
INCLUDEPATH += U:\opencv401\include

LIBS += U:\opencv401\bin\libopencv_core401.dll
LIBS += U:\opencv401\bin\libopencv_highgui401.dll
LIBS += U:\opencv401\bin\libopencv_imgcodecs401.dll
LIBS += U:\opencv401\bin\libopencv_imgproc401.dll
LIBS += U:\opencv401\bin\libopencv_videoio401.dll
LIBS += U:\opencv401\bin\libopencv_video401.dll
LIBS += U:\opencv401\bin\libopencv_objdetect401.dll
 ----------------------
```
save all and build

if successful, open the folder build-facedetect-Desktop_Qt_5_12_1_MinGW_64_bit-Release on your USB flash drive
copy all the dlls from opencv401/include into the release folder which includes the facedetect.exe

now the program should work and indicate the face.
 ----------------
comment out the second cascade for the eyes.
build and run again 
 -------------------------------
change the loaded image name to people3.jpg
comment out 
```cv::resize(tmp, frame1, Size(640,480));```
comment in the next line as a different resize mode:
```cv::resize(tmp, frame1,Size(),0.6,0.6,INTER_CUBIC);```
build and run again


 
 





