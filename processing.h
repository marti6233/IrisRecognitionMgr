#ifndef PROCESSING_H
#define PROCESSING_H

#define OSI_PI 3.14

using namespace std;

void computeVerticalGradients(const cv::Mat* pSrc, cv::Mat* pDst);
void runViterbi(const cv::Mat* pSrc, vector<int>& rOptimalPath);
void encode(const cv::Mat* pSrc, cv::Mat* pDst, const vector<cv::Mat*> & rFilters);
cv::Mat* addBorders ( const cv::Mat* pImage, int width);
void processAnisotropicSmoothing(const cv::Mat* pSrc, cv::Mat* pDst, int iterations, float lambda);			 
cv::Point convertPolarToCartesian(const cv::Point& rCenter, int rRadius, float rTheta);
void drawContour(cv::Mat* pImage, const vector<cv::Point>& rContour, const cv::Scalar& rColor, int thickness);
float match(const cv::Mat* image1,  const cv::Mat* image2, const cv::Mat* mask);

vector<cv::Point> findContour(const cv::Mat* pSrc,
							  const cv::Point& rCenter,
						      const vector<float>& rTheta,
							  int minRadius,
							  int maxRadius,
						      const cv::Mat* pMask = 0);								 
cv::Mat* unwrapRing(const cv::Mat* pSrc, const cv::Point& rCenter,
					int minRadius, int maxRadius,const vector<float>& rTheta);

void normalizeFromContour(const cv::Mat* pSrc,
					      cv::Mat* pDst,
						  const EyeCircle& rPupil,
					      const EyeCircle& rIris,
						  const vector<float> rThetaCoarsePupil,
						  const vector<float> rThetaCoarseIris,
						  const vector<cv::Point>& rPupilCoarseContour,
						  const vector<cv::Point>& rIrisCoarseContour);
					
cv::Point interpolate(const vector<cv::Point> coarseContour,
				      const vector<float> coarseTheta ,
				      const float theta );

#endif
