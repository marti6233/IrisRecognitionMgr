#ifndef EYE_H
#define EYE_H
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>

#define OSI_MAX_RATIO_PUPIL_IRIS 0.7f
#define OSI_MIN_RATIO_PUPIL_IRIS 0.15f
#define MIN_PUPIL_DIAMETER 45
#define MAX_PUPIL_DIAMETER 70
#define MIN_IRIS_DIAMETER 50
#define MAX_IRIS_DIAMETER 150

using namespace std;

class BAW
{
	public:
	unsigned int white;
	unsigned int black;
	float bwRatio;
	float wbRatio;
};

class EyeCircle
{
public:
	int radius;
	cv::Point center;
	void computeCircleFitting ( const vector<cv::Point> & rPoints);
};

class EyeBall
{
public:
	string name;
	cv::Mat* originalImage;
	cv::Mat* eyeMask;
	cv::Mat* irisMask;
	cv::Mat* pupilMask;
	cv::Mat* normalizedImage ;
	cv::Mat* normalizedMask = NULL;
	cv::Mat* irisCode = NULL;
	
	EyeCircle iris;
	EyeCircle pupil;
	
	vector<float> mThetaCoarsePupil;
	vector<cv::Point> mCoarsePupilContour;
	vector<float> mThetaCoarseIris;
	vector<cv::Point> mCoarseIrisContour;
	

	void detectPupil(const cv::Mat* iSrc);
	void detectPupilFasterVesrion(const cv::Mat* iSrc);
	void detectIris(const cv::Mat* iSrc);
	void normalize(int rWidthOfNormalizedIris, int rHeightOfNormalizedIris);
	void encode ( const vector<cv::Mat*> & rGaborFilters);
	float match(EyeBall& rEye , const cv::Mat* pApplicationPoints);
	
	~EyeBall()
	{

	}
};
#endif
