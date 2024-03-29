#include "image.h"
#include "eye.h"
#include "processing.h"

float EyeBall::match(EyeBall& rEye , const cv::Mat* pApplicationPoints)
{
    // Check that both iris codes are built
    if ( ! irisCode )
    {
        throw runtime_error("Cannot match because iris code 1 is not built (nor computed neither loaded)");
    }
    if ( ! rEye.irisCode )
    {
        throw runtime_error("Cannot match because iris code 2 is not built (nor computed neither loaded)");
    }

    // Initialize the normalized masks
    if ( !normalizedMask )
    {
        normalizedMask = new cv::Mat(pApplicationPoints->rows, pApplicationPoints->cols ,pApplicationPoints->type(),cv::Scalar(1));
        normalizedMask->setTo(cv::Scalar(255));
    }
    if ( !rEye.normalizedMask )
    {
        rEye.normalizedMask = new cv::Mat(pApplicationPoints->rows, pApplicationPoints->cols ,pApplicationPoints->type(),cv::Scalar(1));
        rEye.normalizedMask->setTo(cv::Scalar(255));
    }
    // Build the total mask = mask1 * mask2 * points    
    cv::Mat* temp = new cv::Mat(pApplicationPoints->rows, pApplicationPoints->cols, irisCode->type());
    temp->setTo(cv::Scalar(0));
    cv::multiply(*normalizedMask,*rEye.normalizedMask,*temp);
    cv::multiply(*temp,*pApplicationPoints,*temp);
    
    // Copy the mask f times, where f correspond to the number of codes (= number of filters)
    int n_codes = irisCode->rows / pApplicationPoints->rows ;
    cv::Mat total_mask(irisCode->rows, irisCode->cols, irisCode->type());
    for ( int n = 0 ; n < n_codes ; n++ )
    {
        temp->copyTo(total_mask(cv::Rect(0,n*pApplicationPoints->rows,pApplicationPoints->cols,pApplicationPoints->rows)));
    }
    // Match
    float score = ::match(irisCode,rEye.irisCode,&total_mask);
    delete temp;
    
    return score;
}


void EyeBall::encode ( const vector<cv::Mat*> & rGaborFilters)
{
	if ( ! normalizedImage )
	{
		throw runtime_error("Cannot encode because normalized image is not loaded");
	}

	// Create the image to store the iris code
	irisCode = new cv::Mat(cv::Size(normalizedImage->cols,normalizedImage->rows*rGaborFilters.size()),CV_8U,cv::Scalar(1));

	// Encode
	::encode(normalizedImage,irisCode,rGaborFilters);
}

void EyeBall::normalize(int rWidthOfNormalizedIris, int rHeightOfNormalizedIris)
{
	normalizedImage = new cv::Mat(rHeightOfNormalizedIris, rWidthOfNormalizedIris, CV_8U, cv::Scalar(0));
	normalizeFromContour(originalImage,normalizedImage,pupil,iris,mThetaCoarsePupil,mThetaCoarseIris,mCoarsePupilContour,mCoarseIrisContour);
	
	normalizedMask = new cv::Mat(rHeightOfNormalizedIris, rWidthOfNormalizedIris, CV_8U, cv::Scalar(0));
	normalizeFromContour(eyeMask,normalizedMask,pupil,iris,mThetaCoarsePupil,mThetaCoarseIris,mCoarsePupilContour,mCoarseIrisContour);
}

void EyeBall::detectIris(const cv::Mat* iSrc)
{
	eyeMask = new cv::Mat(iSrc->rows, iSrc->cols, CV_8U, cv::Scalar(0));
	pupilMask = new cv::Mat(iSrc->rows, iSrc->cols, CV_8U, cv::Scalar(0));
	irisMask = new cv::Mat(pupilMask->rows, pupilMask->cols, pupilMask->depth());
	cv::Mat* clone_src = new cv::Mat(iSrc->rows, iSrc->cols, iSrc->depth(), cv::Scalar(0));	
	iSrc->copyTo(*clone_src);
	uint8_t sX = pupil.center.x-(3.0/4.0*MAX_IRIS_DIAMETER/2.0);
	uint8_t sY = pupil.center.y-(3.0/4.0*MAX_IRIS_DIAMETER/2.0);

	// Will contain samples of angles, in radians
	vector<cv::Point> iris_coarse_contour;
	vector<float> theta;
	float theta_step = 0;

	// Pupil Accurate Contour
	/////////////////////////
	theta.clear() ;
	theta_step = 360.0 / 3.14 / pupil.radius;
	for ( float t = 0 ; t < 360 ; t += theta_step)
	{
		theta.push_back(t*3.14/180);
	}
	vector<cv::Point> pupil_accurate_contour = findContour(clone_src,
														   pupil.center,
														   theta,
														   pupil.radius-20,
														   pupil.radius+20);
														 
	mThetaCoarsePupil = theta;
	mCoarsePupilContour = pupil_accurate_contour;
	pupil.computeCircleFitting(pupil_accurate_contour);
	drawContour(pupilMask,pupil_accurate_contour,cv::Scalar(255),-1);

	// Iris Coarse Contour
	//////////////////////
	theta.clear();
	int min_radius = max<int>(pupil.radius/OSI_MAX_RATIO_PUPIL_IRIS,MIN_IRIS_DIAMETER/2);
	int max_radius = min<int>(pupil.radius/OSI_MIN_RATIO_PUPIL_IRIS,3*MAX_IRIS_DIAMETER/4);

	theta_step = 360.0 / 3.14 / min_radius ;
	for ( float t = 0 ; t < 360 ; t += theta_step )
	{
		if ( t < 180 || ( t > 225 && t < 315 ) ) t += 2*theta_step;
		theta.push_back(t*3.14/180) ;
	}

		iris_coarse_contour = findContour(clone_src,
										  pupil.center,
										  theta,
										  min_radius,
										  max_radius);

	mThetaCoarseIris = theta;
	mCoarseIrisContour = iris_coarse_contour;
	iris.computeCircleFitting(iris_coarse_contour);

	// Mask of iris
	//////////////
	pupilMask->copyTo(*irisMask);
	irisMask->setTo(cv::Scalar(0));
	drawContour(irisMask,iris_coarse_contour,cv::Scalar(255),-1) ;
	// Iris Accurate Contour
	////////////////////////
	theta_step = 0;
	// For iris accurate contour, limit the search of contour inside a mask
	// Dilate mask of iris by a disk-shape element
	cv::Mat * mask_iris2 = new cv::Mat(irisMask->rows, irisMask->cols, irisMask->depth());
	irisMask->copyTo(*mask_iris2);
	cv::Mat struct_element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11),cv::Point(5,5));
	cv::dilate(*mask_iris2, *mask_iris2, struct_element);

	// Dilate the mask of pupil by a horizontal line-shape element
	cv::Mat * mask_pupil2 = new cv::Mat(pupilMask->rows, pupilMask->cols, pupilMask->depth());
	pupilMask->copyTo(*mask_pupil2);
	struct_element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11), cv::Point(5,5));
	cv::dilate(*mask_pupil2,*mask_pupil2,struct_element);
	cv::bitwise_xor(*mask_iris2,*mask_pupil2,*mask_iris2);
	theta.clear() ;
	theta_step = 360.0 / 3.14 / iris.radius;
	for ( float t = 0 ; t < 360 ; t += theta_step )
	{
		theta.push_back(t*3.14/180) ;
	}

	vector<cv::Point> iris_accurate_contour = findContour(clone_src,
														  pupil.center,
													      theta,
														  iris.radius-50,
														  iris.radius+20,
														  mask_iris2);
			
	irisMask->setTo(cv::Scalar(0));
	drawContour(irisMask,iris_accurate_contour,cv::Scalar(255),-1);
	cv::bitwise_xor(*irisMask,*pupilMask,*eyeMask);

	delete clone_src;
	delete mask_iris2;
	delete mask_pupil2;
}

void EyeBall::detectPupil(const cv::Mat* iSrc)
{
	cv::Mat filled(iSrc->rows, iSrc->cols, iSrc->depth());
	iSrc->copyTo(filled);
	cv::Mat gh(filled.rows, filled.cols, CV_32F);
	cv::Mat mulgh(filled.rows, filled.cols, CV_32F,cv::Scalar(1.0));
	cv::Mat gv(filled.rows, filled.cols, CV_32F);
	cv::Mat mulgv(filled.rows, filled.cols, CV_32F, cv::Scalar(1.0));
	if(threadNumber > 1)
	{
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				//Horizontal Gradient
				cv::Sobel(filled,gh,gh.depth(),1,0);
				//Normalization
				cv::multiply(gh,gh,mulgh);
			}
			#pragma omp section
			{
				//Vertical Gradient
				cv::Sobel(filled,gv,gv.depth(),0,1);	
				
				cv::multiply(gv,gv,mulgv);
			}
		}
		
	}
	else
	{
		//Horizontal Gradient
		cv::Sobel(filled,gh,gh.depth(),1,0);
		//Vertical Gradient
		cv::Sobel(filled,gv,gv.depth(),0,1);

		//Normalization
		cv::multiply(gh,gh,mulgh);
		cv::multiply(gv,gv,mulgv);
	}
	cv::Mat destinationImage(filled.rows, filled.cols, CV_32F, cv::Scalar(0.0));
	cv::addWeighted(mulgh, 0.5, mulgv, 0.5, 0, destinationImage);

	//Filters
	int filterSize = MAX_PUPIL_DIAMETER;
	filterSize += (filterSize % 2) ? 0 : -1;
	cv::Mat fh(filterSize, filterSize, CV_32FC1, cv::Scalar(1.0));
	cv::Mat fv(filterSize, filterSize, CV_32FC1, cv::Scalar(1.0));

	#pragma omp parallel for num_threads(threadNumber)
	for(int i = 0; i < fh.rows; i++)
	{
		float x = i - (filterSize - 1) / 2;
		for(int j = 0; j < fh.cols; j++)
		{
			float y = j - (filterSize - 1) / 2;
			if( x != 0 || y != 0)
			{
				fh.at<float>(i*fh.cols+j) =  y / (float)sqrt(x*x+y*y);
				fv.at<float>(i*fv.cols+j) = x / (float)sqrt(x*x+y*y);
			}
			else
			{
				fh.at<float>(i*fh.cols+j) = (float)0.0;
				fv.at<float>(i*fv.cols+j) = (float)0.0;
			}
			
		}
	}

	double oldMaxVal = 0;
	double maxVal;
	if(threadNumber > 1)
	{
		#pragma omp parallel for
		for(int r = ((MIN_PUPIL_DIAMETER)/2); r < (MAX_PUPIL_DIAMETER)/2; r++)
		{
			//create copy of elements for each thread
			cv::Mat mask(filterSize, filterSize, CV_8UC1);
			cv::Mat tempFilter(filterSize, filterSize, CV_32FC1);
			cv::Mat gh2(gh.rows, gh.cols, gh.depth());
			gh.copyTo(gh2);
			cv::Mat gv2(gv.rows, gv.cols, gv.depth());
			gv.copyTo(gv2);
			cv::Mat mulgv2(mulgv.rows, mulgv.cols, mulgv.depth());
			mulgv.copyTo(mulgv2);
			cv::Mat mulgh2(mulgh.rows, mulgh.cols, mulgh.depth());
			mulgh.copyTo(mulgh2);
			cv::Mat destinationImage2(destinationImage.rows, destinationImage.cols, destinationImage.depth());
			destinationImage.copyTo(destinationImage2);

			mask.setTo(cv::Scalar(0));
			cv::circle(mask, cv::Point((filterSize-1)/2, (filterSize-1)/2), r, cv::Scalar(1), 2);
			//Fh*Gh
			tempFilter.setTo(cv::Scalar(0.0));

			fh.copyTo(tempFilter,mask);
			cv::filter2D(gh2, mulgh2, mulgh2.depth(), tempFilter);
			//Fv*Gh
			tempFilter.setTo(cv::Scalar(0.0));
			fv.copyTo(tempFilter,mask);
			cv::filter2D(gv2, mulgv2, mulgv2.depth(), tempFilter);
			
			cv::add(mulgh2, mulgv2, destinationImage2);
			destinationImage2.convertTo(destinationImage2, -1, 1.0/cv::sum(mask).val[0]);
			// Sum in the dish--shaped neighbourhood
			mask.setTo(cv::Scalar(0));
			cv::circle(mask, cv::Point((filterSize-1)/2, (filterSize-1)/2), r, cv::Scalar(1), -1);
			cv::filter2D(filled, mulgh2, mulgh2.depth(), mask); 
			mulgh2.convertTo(mulgh2, -1, -1.0/cv::sum(mask).val[0]/255.0, 1.0);

			//Add contour and darkness
			cv::add(destinationImage2, mulgh2, destinationImage2);
			//find maximum
			cv::Point maxLoc;
			cv::minMaxLoc(destinationImage2, 0, &maxVal, 0, &maxLoc);
			#pragma omp critical
			{				
				if(maxVal > oldMaxVal)
				{
					oldMaxVal = maxVal;
					pupil.center = maxLoc;
					pupil.radius = r;
				}
			}
		}
	}
	else
	{
		cv::Mat mask(filterSize, filterSize, CV_8UC1);
		//Temporary MAtrix for masking filter
		cv::Mat tempFilter(filterSize, filterSize, CV_32FC1);
		for(int r = ((MIN_PUPIL_DIAMETER)/2); r < (MAX_PUPIL_DIAMETER)/2; r++)
		{		
			mask.setTo(cv::Scalar(0));
			cv::circle(mask, cv::Point((filterSize-1)/2, (filterSize-1)/2), r, cv::Scalar(1), 2);
			//Fh*Gh
			tempFilter.setTo(cv::Scalar(0.0));
			fh.copyTo(tempFilter,mask);
			cv::filter2D(gh, mulgh, mulgh.depth(), tempFilter);

			//Fv*Gh
			tempFilter.setTo(cv::Scalar(0.0));
			fv.copyTo(tempFilter,mask);
			cv::filter2D(gv, mulgv, mulgv.depth(), tempFilter);

			cv::add(mulgh, mulgv, destinationImage);

			destinationImage.convertTo(destinationImage, -1, 1.0/cv::sum(mask).val[0]);
			// Sum in the dish--shaped neighbourhood
			mask.setTo(cv::Scalar(0));
			cv::circle(mask, cv::Point((filterSize-1)/2, (filterSize-1)/2), r, cv::Scalar(1), -1);
			cv::filter2D(filled, mulgh, mulgh.depth(), mask);
			mulgh.convertTo(mulgh, -1, -1.0/cv::sum(mask).val[0]/255.0, 1.0);

			//Add contour and darkness
			cv::add(destinationImage, mulgh, destinationImage);
			//find maximum
	
			cv::Point maxLoc;
			cv::minMaxLoc(destinationImage, 0, &maxVal, 0, &maxLoc);
			if(maxVal > oldMaxVal)
			{
				oldMaxVal = maxVal;
				pupil.center = maxLoc;
				pupil.radius = r;
			}
		}
	}
}

void EyeBall::detectPupilFasterVesrion(const cv::Mat* iSrc)
{
	cv::Mat filled(iSrc->rows, iSrc->cols, iSrc->depth());
	iSrc->copyTo(filled);

	cv::Mat destinationImage(filled.rows, filled.cols, CV_32F, cv::Scalar(0.0));
	int filterSize = MAX_PUPIL_DIAMETER;
	filterSize += (filterSize % 2) ? 0 : -1;
	double oldMaxVal = 0;
	double maxVal;

	if(threadNumber > 1)
	{
		#pragma omp parallel for
		for(int r = ((MIN_PUPIL_DIAMETER)/2); r < (MAX_PUPIL_DIAMETER)/2; r++)
		{
			//create copy of elements for each thread
			cv::Mat mask(filterSize, filterSize, CV_8UC1);
			cv::Mat destinationImage2(destinationImage.rows, destinationImage.cols, destinationImage.depth());

			// Sum in the dish--shaped neighbourhood
			mask.setTo(cv::Scalar(0));
			cv::circle(mask, cv::Point((filterSize-1)/2, (filterSize-1)/2), r, cv::Scalar(1), -1);
			//for(int i = 0; i < 200; i++)
			cv::filter2D(filled, destinationImage2, destinationImage2.depth(), mask); 
			destinationImage2.convertTo(destinationImage2, -1, -1.0/cv::sum(mask).val[0]/255.0, 1.0);
			
			//find maximum
			cv::Point maxLoc;
			cv::minMaxLoc(destinationImage2, 0, &maxVal, 0, &maxLoc);
			#pragma omp critical
			{				
				if(maxVal > oldMaxVal)
				{
					oldMaxVal = maxVal;
					pupil.center = maxLoc;
					pupil.radius = r;
				}
			}
		}
	}
	else
	{
		cv::Mat mask(filterSize, filterSize, CV_8UC1);
		for(int r = ((MIN_PUPIL_DIAMETER)/2); r < (MAX_PUPIL_DIAMETER)/2; r++)
		{		
			
			mask.setTo(cv::Scalar(0));
			cv::circle(mask, cv::Point((filterSize-1)/2, (filterSize-1)/2), r, cv::Scalar(1), -1);
			cv::filter2D(filled, destinationImage, destinationImage.depth(), mask);
			destinationImage.convertTo(destinationImage, -1, -1.0/cv::sum(mask).val[0]/255.0, 1.0);
			cv::Point maxLoc;
			cv::minMaxLoc(destinationImage, 0, &maxVal, 0, &maxLoc);
			if(maxVal > oldMaxVal)
			{
				oldMaxVal = maxVal;
				pupil.center = maxLoc;
				pupil.radius = r;
			}
		}
	}
}

void EyeCircle::computeCircleFitting ( const vector<cv::Point> & rPoints)
{
	// Compute the averages mx and my
	float mx = 0 , my = 0 ;
	for ( int p = 0 ; p < rPoints.size() ; p++ )
	{
		mx += rPoints[p].x;
		my += rPoints[p].y;
	}
	mx = mx / rPoints.size();
	my = my / rPoints.size();

	// Work in (u,v) space, with u = x-mx and v = y-my
	float u = 0 , v = 0 , suu = 0 , svv = 0 , suv = 0 , suuu = 0 , svvv = 0 , suuv = 0 , suvv = 0;

	// Build some sums
	for ( int p = 0 ; p < rPoints.size() ; p++ )
	{
		u = rPoints[p].x - mx;
		v = rPoints[p].y - my;
		suu += u * u;
		svv += v * v;
		suv += u * v;
		suuu += u * u * u;
		svvv += v * v * v;
		suuv += u * u * v;
		suvv += u * v * v;
	}

	// These equations are demonstrated in paper from R.Bullock (2006)
	float uc = 0.5 * ( suv * ( svvv + suuv ) - svv * ( suuu + suvv ) ) / ( suv * suv - suu * svv );
	float vc = 0.5 * ( suv * ( suuu + suvv ) - suu * ( svvv + suuv ) ) / ( suv * suv - suu * svv );

	// Circle parameters
	center.x = uc+mx;
	center.y = vc+my;
	radius = (int)sqrt(uc*uc+vc*vc+(suu+svv)/rPoints.size());
}
