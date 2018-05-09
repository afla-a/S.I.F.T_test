//
//  main.cpp
//  
//
//  Created by afla on 5/5/18.
//
//

#include "main.h"
#include<vector>
#include<math.h>
#include<iostream>
#include<algorithm>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
#define CURVATURE_THRESHOLD 5.0
#define CONTRAST_THRESHOLD 0.002


void build_scale_space(Mat img){
    int num_octaves=4;
    int num_levels=5;
    
    vector<vector<Mat> >  octaves;
    
    
    
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	Mat scaled = gray.clone();
	

	
	for(int i=0; i<num_octaves; i++)
	{
        
		
		resize(gray, scaled, Size(), 1/(pow(2.0,i)), 1/(pow(2.0,i)));
        vector<Mat> scales;
      //  imshow( "simage",scaled);
      //  imshow( "gray",gray);
		
		for(int j=0; j<num_levels; j++)
		{
			Mat blurred = scaled.clone();
           // imshow( "bl",blurred);
			int k_size = (2*j)+1; 	// Kernel size (odd) 1, 3, 5, 7 etc.
			GaussianBlur( scaled, blurred, Size( k_size,k_size  ), 0, 0 );
			//imshow( "grayge",blurred);
            scales.push_back(blurred);
            //imshow( "grayimage",blurred);
           // imshow( "sca",scaled);
            //waitKey(30*30);
		}
        
//        for (int i=0; i<num_levels; i++) {
//            imshow("scales",scales[i]);
//        waitKey(50*50);
//        }
//        waitKey(80*80);

		octaves.push_back(scales);
    }
        

	
    
//    for (int i=0; i<num_octaves; i++) {
//        for (int j=0; j<num_levels; j++) {
//            
//        imshow("scales",octaves[i][j]);
//        waitKey(50*50);
//    }
//    }
    
////////////////////////////////////////////////////////
    
    
    vector<vector<Mat> > DOGS;
    
    for(int i=0; i<num_octaves; i++)
	{
		vector<Mat> diffs;
		for(int j=0; j<num_levels-1; j++)
		{
			Mat diff = Mat::zeros(octaves[i][j].size(),octaves[i][j].type());
			diff = octaves[i][j] - octaves[i][j+1];
			diffs.push_back(diff);
		}
		DOGS.push_back(diffs);
	}

//        for (int i=0; i<num_octaves; i++) {
//           for (int j=0; j<(num_levels)-1; j++) {
//    
//            imshow("scales",DOGS[i][j]);
//            waitKey(50*50);
//        }
//        }

	
    
    
	vector<vector<Mat> > extrema;
	
    double dxx, dyy, dxy, trH, detH;
    Mat down, middle, up;
    for(int i=0; i<num_octaves; i++)
	{
		vector<Mat> scale_extrema;
		for(int j = 1; j <= num_levels-3; j++)
		{
			
			Mat res = Mat::zeros(DOGS[i][j-1].size(), DOGS[i][j-1].type());
			down = DOGS[i][j-1];
			middle = DOGS[i][j];
			up = DOGS[i][j+1];
            
			int max=0;
			int min = INT_MAX;
		
			for(int xi=1; xi<down.rows; xi++)
			{
				for(int yi=1; yi<down.cols; yi++)
				{
					int current_point = middle.at<uchar>(xi, yi);
					bool is_local = false;
					               
					// check for first DoG image
//					for(int xi=-1; xi<1; xi++)
//					{
//						for(int yi=-1; yi<1; yi++)
//						{
							
							if(current_point > middle.at<double>(xi-1,yi) &&
                                                          current_point > middle.at<double>(xi+1,yi) &&
                                                          current_point > middle.at<double>(xi-1,yi-1) &&
                                                          current_point > middle.at<double>(xi+1,yi-1) &&
                                                          current_point > middle.at<double>(xi-1,yi+1) &&
                                                          current_point > middle.at<double>(xi+1,yi+1) &&
                                                          current_point > middle.at<double>(xi,yi-1) &&
                                                          current_point > middle.at<double>(xi,yi+1) &&
                                                          current_point > down.at<double>(xi,yi) &&
                                                          current_point > down.at<double>(xi-1,yi) &&
                                                          current_point > down.at<double>(xi+1,yi) &&
                                                          current_point > down.at<double>(xi-1,yi-1) &&
                                                          current_point > down.at<double>(xi+1,yi-1) &&
                                                          current_point > down.at<double>(xi-1,yi+1) &&
                                                          current_point > down.at<double>(xi+1,yi+1) &&
                                                          current_point > down.at<double>(xi,yi-1) &&
                                                          current_point > down.at<double>(xi,yi+1) &&
                                                          current_point > up.at<double>(xi,yi) &&
                                                          current_point > up.at<double>(xi-1,yi) &&
                                                          current_point > up.at<double>(xi+1,yi) &&
                                                          current_point > up.at<double>(xi-1,yi-1) &&
                                                          current_point > up.at<double>(xi+1,yi-1) &&
                                                          current_point > up.at<double>(xi-1,yi+1) &&
                                                          current_point > up.at<double>(xi+1,yi+1) &&
                                                          current_point > up.at<double>(xi,yi-1) &&
                                                          current_point > up.at<double>(xi,yi+1)
                                                          ){
                             //   cout<<"dads";
                                is_local = true;
                                res.at<uchar>(xi,yi) = 255;

                            }else if(	current_point < middle.at<double>(xi-1,yi) && current_point < middle.at<double>(xi+1,yi) &&current_point < middle.at<double>(xi-1,yi-1) &&current_point < middle.at<double>(xi+1,yi-1) &&current_point < middle.at<double>(xi-1,yi+1) &&current_point < middle.at<double>(xi+1,yi+1) &&current_point < middle.at<double>(xi,yi-1) &&current_point < middle.at<double>(xi,yi+1) &&current_point < down.at<double>(xi,yi) &&current_point < down.at<double>(xi-1,yi) &&current_point < down.at<double>(xi+1,yi) &&current_point < down.at<double>(xi-1,yi-1) &&current_point < down.at<double>(xi+1,yi-1) &&current_point < down.at<double>(xi-1,yi+1) &&current_point < down.at<double>(xi+1,yi+1) &&current_point < down.at<double>(xi,yi-1) &&current_point < down.at<double>(xi,yi+1) &&current_point < up.at<double>(xi,yi) &&current_point < up.at<double>(xi-1,yi) &&current_point < up.at<double>(xi+1,yi) &&current_point < up.at<double>(xi-1,yi-1) &&current_point < up.at<double>(xi+1,yi-1) &&current_point < up.at<double>(xi-1,yi+1) &&current_point < up.at<double>(xi+1,yi+1) &&current_point < up.at<double>(xi,yi-1) &&current_point < up.at<double>(xi,yi+1))
                            {
                                is_local = true;
                                res.at<uchar>(xi,yi) = 255;
                            //    cout<<"asd";
                            
                            }
                                
					
                 
                    if(is_local && fabs(current_point) < CONTRAST_THRESHOLD)
                    {
                        res.at<uchar>(xi,yi) = 0;
                       
                        is_local = false;
                                            }
                    
//                    if(is_local){
//                        
//                                                    dxx = middle.at<double>(xi,yi-1) + middle.at<double>(xi,yi+1) - 2 * middle.at<double>(xi,yi);
//                                                    dyy = middle.at<double>(xi-1,yi) + middle.at<double>(xi+1,yi) - 2 * middle.at<double>(xi,yi);
//                                                    dxy = (middle.at<double>(xi-1,yi-1) + middle.at<double>(xi+1,yi+1) - middle.at<double>(xi+1,yi-1) - middle.at<double>(xi-1,yi+1))/4;
//
//                        
//                        
//                        double dx2 = dxx * dxx;
//                        double dy2 = dyy * dyy;
//                        double R = dx2/dy2;
//                        cout<<dxx;
//                        
//                        //						// store a min/max value in the result mat
//                        						if(R > 500)
//                        						res.at<uchar>(xi, yi) = 0;
//                        					else
//                        							res.at<uchar>(xi, yi) = 255;
//                      
//
//                        
////                                                    if(detH < 0 || curv_ratio < curv_threshold){
////                                                        res.at<uchar>(xi,yi) = 0;
////                                                        kp_num--;kp_reject_num++;
////                                                        is_local=false;
////                                                    }
//                                               }
                    
                    
				}
			}
			scale_extrema.push_back(res);
		}
		extrema.push_back(scale_extrema);
	}
	
    
    
    
    
//        for (int i=0; i<num_octaves; i++) {
//            for (int j = 0; j < num_levels-3; j++) {
//    
//            imshow("scales",extrema[i][j]);
//            waitKey(50*50);
//        }
//       }
//   //////////////////////////////////////////////////////
    
    vector<vector<cv::Mat> > corners; 			// stores strong corners
	for(int i=0; i<num_octaves; i++)
	{
		vector<cv::Mat> filtered_extrema;
		for(int j=0; j<extrema[i].size(); j++)
		{
			
			Mat gray = octaves[i][j];
			Mat ext = extrema[i][j];
			
			Mat dGray = gray.clone();
			Mat res = Mat::zeros(extrema[i][j].size(), extrema[i][j].type());
            
			//imshow("gray", gray);
			//imshow("ext", ext);
		
			for(int ii=1; ii<gray.rows-1; ii++)
			{
				for(int jj=1; jj<gray.cols-1; jj++)
				{
					
					if(ext.at<uchar>(ii,jj) != 0)
					{
						
						Mat edge_kernel_x = (Mat_<int>(3,3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
						Mat edge_kernel_y = (Mat_<int>(3,3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
					
                        
						int dx = 0, dy = 0;
						
						for(int ki=0, i2=-edge_kernel_x.rows/2; ki<edge_kernel_x.rows; ki++, i2++)
						{
							for(int kj=0, j2=-edge_kernel_x.rows/2; kj<edge_kernel_x.rows; kj++, j2++)
							{
                                
								dx += (int)edge_kernel_x.at<int>(ki, kj) * (int)gray.at<uchar>(ii+i2, jj+j2);
								dy += (int)edge_kernel_y.at<int>(ki, kj) * (int)gray.at<uchar>(ii+i2, jj+j2);
							}
						}
						double dx2 = dx * dx;
						double dy2 = dy * dy;
						double R = dx2/dy2;
						
						if(R > 900)
							res.at<uchar>(ii, jj) = 255;
						else
							res.at<uchar>(ii, jj) = 0;
					}
				}
			}
            
			
			filtered_extrema.push_back(res);
           // imshow("filtered", res);
           // waitKey(60*60);
		}
        
		
		corners.push_back(filtered_extrema);
	}
	
	//imshow("filtered", corners[1][1]);
    waitKey(10*10);
    
   
    imshow("ex",extrema[0][0]);
    imshow("res",corners[0][0]);
    
    
    
    ///////////////////////////////////////////////////////////////////
    

    
}

int main()
{
    Mat src,final;
    
    src=imread("logo.jpg");
    
   // src=imread("logo.jpg");
    //create Mat from vector
//    vector<Mat> vec; // vector with your data
//    vec.push_back(src);
//    
//    imshow("dasd",vec[0]);
    
//Compute the Gaussian scale-space
    build_scale_space(src);
    if(!src.empty()){
      //  imshow("name", src);
    }
    	waitKey(100*100);
    return 0;
}
