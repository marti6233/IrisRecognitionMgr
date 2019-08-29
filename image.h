#ifndef IMAGE_H
#define IMAGE_H
#include <omp.h>
#include "opencv4/opencv2/opencv.hpp"
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <fstream>
#include <string>
#include <vector>
#include "eye.h"

#define MWIDTHOFNORMALIZEDIRIS 512
#define MHEIGHTOFNORMALIZEDIRIS 64
#define THREAD_NUMBER 4

#define IMAGE_TO_COMPARE 0
#define IMAGE_TO_READ_BASE 90 //line number in imgList.txt
#define IMAGE_TO_READ_SAMPLES 360 //from imgList2.txt
#define IMAGE_TO_START 0
#define PARALLEL_COMPUTING 0
#define ACCEPTANCE_THRESHOLD 0.38

extern int threadNumber;
extern int algorithmVersion;
enum version{NORMAL_VERSION, FAST_VERSION};
enum irisDatabase{IRIS_BASE_1, IRIS_SAMPLE_BASE};

class QualityStandard
{
	public:
	int tp = 0; //true positive
	int tn = 0; //true negative
	int fp = 0; //false positive
	int fn = 0; //false negative
	
	double calculateACC()
	{
		return (double) (tp+tn)/(tp+fp+tn+fn);
	}
	double calculatePPV()
	{
		return (double) tp/(tp+fp);
	}
	double calculateNPV()
	{
		return (double) tn/(tn+fn);
	}
	double calculateTPR()
	{
		return (double) tp/(tp+fn);
	}
	double calculateTNR()
	{
		return (double) tn/(tn+fp);
	}
};

class Configuration
{
public:
	~Configuration( ) {}
	std::vector<cv::Mat*> mGaborFilters;
	cv::Mat* mpApplicationPoints;
	void loadGaborFilters();
	void loadApplicationPoints();

};

class IrisUtility
{
	public:
	void runForHistograFalseCompare(Configuration& conf);
	void processMainDatabase(Configuration& conf);
	void processSingleEye(EyeBall& dstEye, int index, Configuration& conf);
	void simulateAccesRequest(Configuration& conf, int processing);
};

using namespace std;
using namespace cv;

void showImage(const cv::Mat* img, const cv::Mat* img2 = NULL,
			   const cv::Mat* img3 = NULL, const cv::Mat* img4 = NULL);

void loadImagePath(string& img, int imgNr);
#endif
