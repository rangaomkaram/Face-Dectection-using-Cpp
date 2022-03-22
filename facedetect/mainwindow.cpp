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


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    facedetect();
}

MainWindow::~MainWindow()
{
    delete ui;
}

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



//    VideoCapture capture;

//    //if source is camera:
//    capture.open(0); //>1 = external webcam

//    if(!capture.isOpened()){
//        cout<<"ERROR ACQUIRING VIDEO FEED\n" << endl;
//        printf( "error video opening");
//        return;
//    }

    Mat tmp, frame1;

    //check for keyboard input
    while( waitKey(10) != 27)
    {

        //read first frame
       // capture.read(tmp);  //uncomment this line if camera
        tmp = imread("U://opencvProjSB//UE_8//people4.jpg", IMREAD_COLOR);
        //cv::resize(tmp, frame1, Size(640,480));
        cv::resize(tmp, frame1,Size(),1.0,1.0,INTER_CUBIC);
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
            std::vector<Rect> eyes;

            eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAL_CMP_GE, Size(30, 30) );
            eyes_cascade.detectMultiScale( faceROI, eyes );

            for( size_t j = 0; j < eyes.size(); j++ )
            {
                Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( frame1, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
            }
        }
        //-- Show what we got
        imshow( "Frame1", frame1 );

    }

    //destroy GUI windows
    destroyAllWindows();
    return;
}
