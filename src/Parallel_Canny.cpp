#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <bits/stdc++.h>
#include <curses.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;
using namespace cv;
const double PI = 3.14159265358979323846;

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}
cv::Mat imgOriginal;        // input image
cv::Mat imgGrayscale;       // grayscale of input image
cv::Mat imgBlurred;         // intermediate blured image
cv::Mat imgCanny;           // Canny edge image

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	


	//clock_t start1, end1, start2, end2, start3, end3, start4, end4, start5, end5, start6, end6, start7, end7;
	struct timeval TimeValue_Start;
  struct timezone TimeZone_Start;
  struct timeval TimeValue_Final;
  struct timezone TimeZone_Final;
  long time_start, time_end;
  double time_overhead1,time_overhead2,time_overhead3,time_overhead4,time_overhead5,time_overhead6,time_overhead7;

	
	

	imgOriginal = cv::imread("cone.jpg");          // open image

	Mat myGreyImage(imgOriginal.rows, imgOriginal.cols, CV_8UC1, Scalar(0));  //myGreyImage

	//cout << imgOriginal.rows << "\n" << imgOriginal.cols << "\n";
	printf("%d  %d\n", imgOriginal.rows, imgOriginal.cols);
	long long pixels = imgOriginal.rows*imgOriginal.cols;

	//cout << "Number of Pixels : "<<pixels<<"\n";
	printf("Number of Pixels : %lld\n",pixels);

	int chunk = imgOriginal.rows / 8;

	if (imgOriginal.empty()) {                                  // if unable to open image
		std::cout << "error: image not read from file\n\n";     // show error message on command line
		//_getch();                                               // may have to modify this line if not using Windows
		return(0);                                              // and exit program
	}

	//cv::namedWindow("Original Image");
	//cv::imshow("Original Image", imgOriginal); 

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//  Converting to a grey scale image
    int threads;
    printf("Enter number of threads to use: ");
    scanf(" %d", &threads);
    
    omp_set_num_threads(threads);
    printf("threads = %d\n",omp_thread_count());
	gettimeofday(&TimeValue_Start, &TimeZone_Start);
    int i=0;
#pragma omp parallel for schedule(static,10)
	for (i = 0; i < imgOriginal.rows; i++) {
		//printf("thread = %d\ti = %d\n",omp_get_thread_num(),i);
		for (int j = 0; j < imgOriginal.cols; j++) {
			myGreyImage.at<uchar>(i, j) = (imgOriginal.at<Vec3b>(i, j)[0] + imgOriginal.at<Vec3b>(i, j)[1] + imgOriginal.at<Vec3b>(i, j)[2]) / 3;

		}
	}
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead1 = (time_end - time_start) / 1000000.0;

	/*cv::namedWindow("myGreyImg");
	cv::imshow("myGreyImg", myGreyImage);*/

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//Applying Gaussian Blur 

	double GausianFilter[5][5] = { {1, 4, 7, 4, 1},
								   {4, 16, 26, 16, 4},
								   {7, 26, 41, 26, 7},
								   {4, 16, 26, 16, 4},
								   {1, 4, 7, 4, 1}
	};


	Mat myGaussian1(imgOriginal.rows - 4, imgOriginal.cols - 4, CV_8UC1, Scalar(0));
	Mat myGaussian(imgOriginal.rows - 4, imgOriginal.cols - 4, CV_8UC1, Scalar(0));



	gettimeofday(&TimeValue_Start, &TimeZone_Start);

#pragma omp parallel for schedule(static,10)
	for (i = 0; i < myGaussian1.rows; i++) {
		for (int j = 0; j < myGaussian1.cols; j++) {
			double temp = 0;

			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					temp += GausianFilter[k][l] * myGreyImage.at<uchar>(k + i, l + j);
				}
			}
			myGaussian1.at<uchar>(i, j) = temp / 273;
		}
	}
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead2 = (time_end - time_start) / 1000000.0;

	gettimeofday(&TimeValue_Start, &TimeZone_Start);

	//Applied filter second time to get more smooth image

#pragma omp parallel for schedule(static,10)
	for (i = 0; i < myGaussian.rows; i++) {
		for (int j = 0; j < myGaussian.cols; j++) {
			double temp = 0;
			if (i<2 || j<2 || i> myGaussian.rows - 3 || j> myGaussian.cols - 3) {                 // to avoid reducing the size of image
				myGaussian.at<uchar>(i, j) = myGaussian1.at<uchar>(i, j);
				continue;
			}

			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					temp += GausianFilter[k][l] * myGaussian1.at<uchar>(k + i - 2, l + j - 2);
				}
			}
			myGaussian.at<uchar>(i, j) = temp / 273;
		}
	}
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead3 = (time_end - time_start) / 1000000.0;


	/*cv::GaussianBlur(myGreyImage,          // input image
		myGaussian,                         // output image
		cv::Size(5, 5),                     // smoothing window width and height in pixels
		1.5);*/

		//cv::namedWindow("myGaussian");
		//cv::imshow("myGaussian", myGaussian);


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//Applying First Difference Gradient Operator (Sobel Operator)

	double Sobel_Gx[3][3] = { {-1 , -2, -1},
							  {0, 0, 0},
							  {1, 2, 1}
	};
	double Sobel_Gy[3][3] = { {-1 , 0, 1},
							  {-2, 0, 2},
							  {-1, 0, 1}
	};

	Mat mySobelX(myGaussian.rows - 2, myGaussian.cols - 2, CV_8UC1, Scalar(0));
	Mat mySobelY(myGaussian.rows - 2, myGaussian.cols - 2, CV_8UC1, Scalar(0));
	Mat Grad_Direction(myGaussian.rows - 2, myGaussian.cols - 2, CV_8UC1, Scalar(0));



	gettimeofday(&TimeValue_Start, &TimeZone_Start);
#pragma omp parallel for schedule(static,10)
	for (i = 0; i < mySobelX.rows; i++) {
		for (int j = 0; j < mySobelX.cols; j++) {
			double temp1 = 0, temp2 = 0;

			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					temp1 += Sobel_Gx[k][l] * myGaussian.at<uchar>(k + i, l + j);
				}
			}

			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					temp2 += Sobel_Gy[k][l] * myGaussian.at<uchar>(k + i, l + j);
				}
			}

			mySobelX.at<uchar>(i, j) = sqrt(temp1*temp1 + temp2 * temp2);
			//mySobelX.at<uchar>(i, j) = temp1;
			//mySobelY.at<uchar>(i, j) = temp2;

			double dir = atan((double)temp2 / (double)temp1);
			Grad_Direction.at<uchar>(i, j) = ((int)(round(dir * (5.0 / PI)) + 5) % 5);
		}
	}
	
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead4 = (time_end - time_start) / 1000000.0;
	//cv::namedWindow("First Order Gradient");
	//cv::imshow("First Order Gradient", mySobelX);
	//cv::namedWindow("Vertical Edges");
	//cv::imshow("Vertical Edges", mySobelY);



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Non Maximal Suppression

	Mat myNonMax(myGaussian.rows - 2, myGaussian.cols - 2, CV_8UC1, Scalar(0));

	gettimeofday(&TimeValue_Start, &TimeZone_Start);
#pragma omp parallel for schedule(static,10)
	for (i = 0; i < mySobelX.rows; i++) {
		for (int j = 0; j < mySobelX.cols; j++) {

			if (i == 0 || j == 0 || i == mySobelX.rows - 1 || j == mySobelX.cols - 1)//0321
				continue;
			int direction = Grad_Direction.at<uchar>(i, j) % 4;
			if (direction == 0) {
				if (mySobelX.at<uchar>(i, j) <= mySobelX.at<uchar>(i - 1, j) || mySobelX.at<uchar>(i, j) <= mySobelX.at<uchar>(i + 1, j))
					myNonMax.at<uchar>(i, j) = 0;
				else myNonMax.at<uchar>(i, j) = mySobelX.at<uchar>(i, j);
			}
			if (direction == 3) {
				if (mySobelX.at<uchar>(i, j) <= mySobelX.at<uchar>(i + 1, j - 1) || mySobelX.at<uchar>(i, j) <= mySobelX.at<uchar>(i - 1, j + 1))
					myNonMax.at<uchar>(i, j) = 0;
				else myNonMax.at<uchar>(i, j) = mySobelX.at<uchar>(i, j);
			}
			if (direction == 2) {
				if (mySobelX.at<uchar>(i, j) <= mySobelX.at<uchar>(i, j - 1) || mySobelX.at<uchar>(i, j) <= mySobelX.at<uchar>(i, j + 1))
					myNonMax.at<uchar>(i, j) = 0;
				else myNonMax.at<uchar>(i, j) = mySobelX.at<uchar>(i, j);
			}
			if (direction == 1) {
				if (mySobelX.at<uchar>(i, j) <= mySobelX.at<uchar>(i - 1, j - 1) || mySobelX.at<uchar>(i, j) <= mySobelX.at<uchar>(i + 1, j + 1))
					myNonMax.at<uchar>(i, j) = 0;
				else myNonMax.at<uchar>(i, j) = mySobelX.at<uchar>(i, j);
			}
		}
	}

	gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead5 = (time_end - time_start) / 1000000.0;

	//cv::namedWindow("Non - Max");
	//cv::imshow("Non - Max", myNonMax);



	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Otsu's Method Thresholding 
	Mat myThres(mySobelX.rows, mySobelX.cols, CV_8UC1, Scalar(0));

	//for nonMax 
	/*double thresholdValue = threshold(myNonMax, myThres, 0, 255, THRESH_BINARY + THRESH_OTSU);

	cout << thresholdValue << "    Inbuilt\n";

	cv::namedWindow("myThres1_NonMax");
	cv::imshow("myThres1_NonMax", myThres);*/

	// Otsu Method
	int N = myNonMax.rows * myNonMax.cols;
	double threshold = 0, var_max = 0, sum = 0, sumB = 0, q1 = 0, q2 = 0, meu1 = 0, meu2 = 0;
	int max_intensity = 255;
	int histogram[256] = { 0 };

	gettimeofday(&TimeValue_Start, &TimeZone_Start);

	for (i = 0; i < myNonMax.rows; i++) {
		for (int j = 0; j < myNonMax.cols; j++) {

			histogram[(int)myNonMax.at<uchar>(i, j)] += 1;
		}
	}
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead6 = (time_end - time_start) / 1000000.0;
	//for (i = 0; i < 255; i++) cout << histogram[i] << " ";

	for (i = 0; i <= 255; i++) sum += i * histogram[i];

	for (int t = 0; t <= 255; t++) {
		q1 += histogram[t];
		if (q1 == 0) continue;
		q2 = N - q1;

		sumB += t * histogram[t];
		meu1 = sumB / q1;
		meu2 = (sum - sumB) / q2;

		double sigma = q1 * q2*(meu1 - meu2)*(meu1 - meu2);
		if (sigma > var_max) {
			threshold = t;
			var_max = sigma;

		}

	}

	//cout << threshold << "   Calculated\n";
	printf("Threshold value Calculated : %lf\n", threshold);


	gettimeofday(&TimeValue_Start, &TimeZone_Start);
#pragma omp parallel for schedule(static,10)
	for (i = 0; i < mySobelX.rows; i++) {
		for (int j = 0; j < mySobelX.cols; j++) {
			if (myNonMax.at<uchar>(i, j) > threshold) myThres.at<uchar>(i, j) = 255;
			else myThres.at<uchar>(i, j) = 0;
		}
	}

	gettimeofday(&TimeValue_Final, &TimeZone_Final);
    time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
    time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
    time_overhead7 = (time_end - time_start) / 1000000.0;
	//cv::namedWindow("CannyEdges");
	//cv::imshow("CannyEdges", myThres);
	imwrite("canny_cone.jpg", myThres);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////



	/*cv::cvtColor(imgOriginal, imgGrayscale, cv::COLOR_BGR2GRAY);       // convert to grayscale

	cv::namedWindow("imgGrayscale");
	cv::imshow("imgGrayscale", imgGrayscale);


	/*cv::GaussianBlur(imgGrayscale,          // input image
		imgBlurred,                         // output image
		cv::Size(5, 5),                     // smoothing window width and height in pixels
		1.5);                               // sigma value, determines how much the image will be blurred

	/*cv::namedWindow("imgBlurred");
	cv::imshow("imgBlurred", imgBlurred);*/

	/*cv::Canny(imgBlurred,           // input image
		imgCanny,                   // output image
		82,                         // low threshold
		164);                       // high threshold

									// declare windows
	//cv::namedWindow("imgOriginal");     // note: you can use CV_WINDOW_NORMAL which allows resizing the window
	//cv::namedWindow("imgCanny");        // or CV_WINDOW_AUTOSIZE for a fixed size window matching the resolution of the image
															// CV_WINDOW_AUTOSIZE is the default
	//cv::imshow("imgOriginal", imgOriginal);     // show windows
	//cv::imshow("imgCanny", imgCanny);
	
	*/


	//cout << "Execution Time : " << t1 + t2 + t3 + t4 + t5 + t6 + t7 << "sec \n";
	printf("Execution Time : %lf sec \n", time_overhead1+time_overhead2+time_overhead3+time_overhead4+time_overhead5+time_overhead6+time_overhead7);

	//cv::waitKey(0);                 // hold windows open until user presses a key
	//cout << "YES";*/

	//cv::waitKey(0);

	return(0);
}