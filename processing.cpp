#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "image.h"
#include "eye.h"
#include "processing.h"

float match(const cv::Mat* image1, const cv::Mat* image2, const cv::Mat* mask)
{    
    // Temporary matrix to store the XOR result
    cv::Mat* result = new cv::Mat(image1->rows, image1->cols, image1->type());
    result->setTo(cv::Scalar(0));
    // Add borders on the image1 in order to shift it
    int shift = 10;
    //cv::Mat * shifted = new cv::Mat(*image2);
    cv::Mat* shifted;
    shifted = addBorders(image1,shift);

    // The minimum score will be returned
    float score = 1.0;
    // Shift image1, and compare to image2
    for ( int s = -shift ; s <= shift ; s++ )
    { 
        cv::bitwise_xor((*shifted)(cv::Rect(shift+s, 0, image1->cols, image1->rows)),*image2,*result,*mask);
        float mean = (cv::sum(*result).val[0])/(cv::sum(*mask).val[0]);
        score = min(score,mean);
    }

    delete shifted;
    delete result;
    return score ;
}


cv::Mat* addBorders ( const cv::Mat* pImage, int width)
{
    // Result image
    cv::Mat* result = new cv::Mat(cv::Size(pImage->rows+2*width,pImage->cols),pImage->type());
    // Copy the image in the center
    cv::copyMakeBorder(*pImage,*result, 0, 0, width, width, cv::BORDER_REPLICATE, cv::Scalar(0));    
    // Create the borders left and right assuming wrapping
    #pragma omp parallel for
    for ( int i = 0 ; i < pImage->rows ; i++ )
    {
        for ( int j = 0 ; j < width ; j++ )
        {
            ((uchar *)(result->data + i*result->step))[j] = 
            ((uchar *)(pImage->data + i*pImage->step))[pImage->cols-width+j];
            ((uchar *)(result->data + i*result->step))[result->cols-width+j] = 
            ((uchar *)(pImage->data + i*pImage->step))[j];
        }
    }
    return result;
}


void encode(const cv::Mat* pSrc, cv::Mat* pDst, const vector<cv::Mat*> & rFilters)
{
    // Compute the maximum width of the filters        
    int max_width = 0 ;
    for ( int f = 0 ; f < rFilters.size() ; f++ )
        if (rFilters[f]->cols > max_width)
            max_width = rFilters[f]->cols ;
    max_width = (max_width-1)/2 ;
    // Add wrapping borders on the left and right of image for convolution
    cv::Mat* resized;
    resized = addBorders(pSrc,max_width) ;

    // Loop on filters
    #pragma omp parallel for
    for ( int f = 0 ; f < rFilters.size() ; f++ )
    {
	cv::Mat img1(resized->rows, resized->cols, CV_32F, cv::Scalar(1));
	cv::Mat img2(resized->rows, resized->cols, pDst->depth(), cv::Scalar(1));

        // Convolution
        cv::filter2D(*resized,img1, img1.depth(), *(rFilters[f]));

        // Threshold : above or below 0
        cv::threshold(img1, img2, 0, 255, cv::THRESH_BINARY);

        // Form the iris code
	img2(cv::Rect(max_width,0,pSrc->cols,pSrc->rows)).copyTo((*pDst)(cv::Rect(0,f*pSrc->rows,pSrc->cols,pSrc->rows)));
    }
    delete resized;
}

cv::Point interpolate (const vector<cv::Point> coarseContour, const vector<float> coarseTheta, const float theta)
{
    float interpolation;
    int i1 , i2;

    if ( theta < coarseTheta[0] )
    {
        i1 = coarseTheta.size() - 1;
        i2 = 0 ;
        interpolation = ( theta - (coarseTheta[i1]-2*OSI_PI) ) / ( coarseTheta[i2] - (coarseTheta[i1]-2*OSI_PI) );
    }
        
    else if ( theta >= coarseTheta[coarseTheta.size()-1] )
    {
        i1 = coarseTheta.size() - 1 ;
        i2 = 0 ;
        interpolation = ( theta - coarseTheta[i1] ) / ( coarseTheta[i2]+2*OSI_PI - coarseTheta[i1] );
    }
    else
    {
        int i = 0 ;
        while ( coarseTheta[i+1] <= theta ) i++;
        i1 = i ;
        i2 = i+1 ;
        interpolation = ( theta - coarseTheta[i1] ) / ( coarseTheta[i2] - coarseTheta[i1] );			
    }
    float x = (1-interpolation) * coarseContour[i1].x + interpolation * coarseContour[i2].x;
    float y = (1-interpolation) * coarseContour[i1].y + interpolation * coarseContour[i2].y;
    
    return cv::Point(x,y);
}


void normalizeFromContour(const cv::Mat* pSrc,
		          cv::Mat* pDst,
			  const EyeCircle& rPupil,
		          const EyeCircle& rIris,
			  const vector<float> rThetaCoarsePupil,
			  const vector<float> rThetaCoarseIris,
			  const vector<cv::Point>& rPupilCoarseContour,
			  const vector<cv::Point>& rIrisCoarseContour)
{
    // Set to zeros all pixels
    pDst->setTo(cv::Scalar(255));
    
    #pragma omp parallel for
    for ( int j = 0 ; j < pDst->cols ; j++ )
    {
	    cv::Point point_pupil , point_iris;
	    int x , y ;
	    float theta , radius ;
	// One column correspond to an angle teta
	theta = (float) j / pDst->cols * 2 * OSI_PI;

	// Interpolate pupil and iris radii from coarse contours
	point_pupil = interpolate(rPupilCoarseContour,rThetaCoarsePupil,theta);
	point_iris = interpolate(rIrisCoarseContour,rThetaCoarseIris,theta);

	// Loop on lines of normalized src
	for ( int i = 0 ; i < pDst->rows ; i++ )
	{   
	    // The radial parameter
	    radius = (float) i / pDst->rows;

	    // Coordinates relative to both radii : iris and pupil
	    x = (1-radius) * point_pupil.x + radius * point_iris.x;
	    y = (1-radius) * point_pupil.y + radius * point_iris.y;
;
	    // Do not exceed src size
	    if ( x>=0 && x<pSrc->cols && y>=0 && y<pSrc->rows )
	    {
		pDst->at<uchar>(i, j) = pSrc->at<uchar>(y, x);
	    }
	}
    }
}
// Draw a contour (vector of cv::Point) on an image

void drawContour(cv::Mat* pImage, const vector<cv::Point>& rContour,
		 const cv::Scalar& rColor, int thickness)
{
    // Draw INSIDE the contour if thickness is negative
    if ( thickness < 0 )
    {
	cv::Point* points = new cv::Point[rContour.size()];
	
	for ( int i = 0 ; i < rContour.size() ; i++ )
	{
	    points[i].x = rContour[i].x;
	    points[i].y = rContour[i].y;
	}
	cv::fillConvexPoly(*pImage,points,rContour.size(),rColor);
	delete [] points;
    }

    // Else draw the contour
    else
    {
	// Draw the contour on binary mask
	cv::Mat* mask = new cv::Mat(pImage->rows, pImage->cols, pImage->type());
	mask->setTo(cv::Scalar(0));
	
	for ( int i = 0 ; i < rContour.size() ; i++ )
	{
	    // Do not exceed image sizes
	    int x = min(max(0,rContour[i].x),pImage->cols);
	    int y = min(max(0,rContour[i].y),pImage->rows);

	    // Plot the point on image
	    ((uchar *)(mask->data+y*mask->step))[x] = 255;
	}
    
	// Dilate mask if user specified thickness
	if ( thickness > 1 )
	{
	    cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3), cv::Point(1,1));
	    cv::dilate(*mask, *mask, se, cv::Point(-1,-1), thickness-1);
	}

	// Color rgb
	pImage->setTo(rColor,*mask);
	delete mask;

    }
}


// Convert polar coordinates into cartesian coordinates
cv::Point convertPolarToCartesian(const cv::Point& rCenter, int rRadius, float rTheta)
{
    int x = rCenter.x + rRadius * cos(rTheta);
    int y = rCenter.y - rRadius * sin(rTheta);
    return cv::Point(x,y);
}

cv::Mat* unwrapRing(const cv::Mat* pSrc,
		     const cv::Point& rCenter,
		     int minRadius,
		     int maxRadius,
		     const vector<float>& rTheta)
{
    // Result image
    cv::Mat* result = new cv::Mat(cv::Size(rTheta.size(),maxRadius-minRadius+1),pSrc->type(),cv::Scalar(0));
    
    // Loop on columns of normalized image
    //#pragma omp parallel for
    for ( int j = 0 ; j < result->cols ; j++ )
    {
	// Loop on lines of normalized image
	for ( int i = 0 ; i < result->rows ; i++ )
	{
	    cv::Point point = convertPolarToCartesian(rCenter,minRadius+i,rTheta[j]);

	    // Do not exceed image size
	    if ( point.x >= 0 && point.x < pSrc->cols && point.y >= 0 && point.y < pSrc->rows )
		((uchar *)(result->data+i*result->step))[j] =
		((uchar *)(pSrc->data+point.y*pSrc->step))[point.x];
	}
    }
    return result;
}

// Run viterbi algorithm on gradient (or probability) image and find optimal path
void runViterbi(const cv::Mat* pSrc, vector<int>& rOptimalPath)
{
    // Initialize the output
    rOptimalPath.clear();
    rOptimalPath.resize(pSrc->cols);
    
    // Initialize cost matrix to zero
    cv::Mat* cost = new cv::Mat(pSrc->rows, pSrc->cols, CV_32F);
    cost->setTo(cv::Scalar(0));

    for ( int w = 0 ; w < pSrc->cols ; w++ )
    {
	for ( int h = 0 ; h < pSrc->rows ; h++ )
	{
	    // First column is same as source image
	    if ( w == 0 )
		((float*)(cost->data+h*cost->step))[w] =
		((uchar*)(pSrc->data+h*pSrc->step))[w];

	    else
	    {
		// First line
		if ( h == 0 )
		    ((float*)(cost->data+h*cost->step))[w] = max<float>(
		    ((float*)(cost->data+(h)*cost->step))[w-1],
		    ((float*)(cost->data+(h+1)*cost->step))[w-1]) +
		    ((uchar*)(pSrc->data+h*pSrc->step))[w];

		// Last line
		else if ( h == pSrc->rows - 1 )
		{
		    ((float*)(cost->data+h*cost->step))[w] = max<float>(
		    ((float*)(cost->data+h*cost->step))[w-1],
		    ((float*)(cost->data+(h-1)*cost->step))[w-1]) +
		    ((uchar*)(pSrc->data+h*pSrc->step))[w];
		}

		// Middle lines
		else
		    ((float*)(cost->data+h*cost->step))[w] = max<float>(
		    ((float*)(cost->data+h*cost->step))[w-1],max<float>(
		    ((float*)(cost->data+(h+1)*cost->step))[w-1],
		    ((float*)(cost->data+(h-1)*cost->step))[w-1])) +
		    ((uchar*)(pSrc->data+h*pSrc->step))[w];
	    }
	}
    }

    // Get the maximum in last column of cost matrix
    cv::Point max_loc;        
    cv::minMaxLoc((*cost)(cv::Rect(cost->cols-1,0,1,cost->rows)), NULL, NULL, NULL, &max_loc);
    int h = max_loc.y;
    int h0 = h;   

    // Store the point in output vector
    rOptimalPath[rOptimalPath.size()-1] = h0;
    float h1 , h2 , h3;

    // Backward process
    for ( int w = rOptimalPath.size() - 2 ; w >= 0 ; w-- )
    {
	// Constraint to close the contour
	if ( h - h0 > w )
	    h -- ;
	else if ( h0 - h > w )
	    h ++;

	// When no need to constraint : use the cost matrix
	else
	{
	    // h1 is the value above line h
	    h1 = ( h == 0 ) ? 0 : ((float*)(cost->data+(h-1)*cost->step))[w];

	    // h2 is the value at line h
	    h2 = ((float*)(cost->data+h*cost->step))[w];

	    // h3 is the value below line h
	    h3 = ( h == cost->cols - 1 ) ? 0 : ((float*)(cost->data+(h+1)*cost->step))[w];
	    
	    // h1 is the maximum => h decreases
	    if ( h1 > h2 && h1 > h3 )
		h--;

	    // h3 is the maximum => h increases
	    else if ( h3 > h2 && h3 > h1 )
		h++;
	}

	// Store the point in output contour
	rOptimalPath[w] = h;

    }
    delete cost;
}

// Compute vertical gradients using Sobel operator
void computeVerticalGradients(const cv::Mat* pSrc, cv::Mat* pDst)
{
    // Float values for Sobel
    cv::Mat* result_sobel = new cv::Mat(pSrc->rows, pSrc->cols ,pSrc->type());

    // Sobel filter in vertical direction
    cv::Sobel(*pSrc, *result_sobel, result_sobel->depth(), 0,1);

    // Remove negative edges, ie from white (top) to black (bottom)
    cv::threshold(*result_sobel, *result_sobel, 0, 0, cv::THRESH_TOZERO);

    // Convert into 8-bit
    double min , max;
    cv::minMaxLoc(*result_sobel, &min, &max, NULL, NULL) ;
    result_sobel->convertTo(*pDst, -1, 255/(max-min),-255*min/(max-min)) ;

    // Release memory
    delete result_sobel;
}


void processAnisotropicSmoothing(const cv::Mat* pSrc, cv::Mat* pDst,
				 int iterations, float lambda)
{
    // Temporary float images
    cv::Mat* tfs = new cv::Mat(pSrc->rows, pSrc->cols, CV_32F);

    pSrc->convertTo(*tfs, tfs->depth());
    cv::Mat* tfd = new cv::Mat(pSrc->rows, pSrc->cols, CV_32F);
    pSrc->convertTo(*tfd, tfd->depth());

    // Make borders dark
    cv::rectangle(*tfd,cv::Point(0,0),cv::Point(tfd->cols-1,tfd->rows-1),cv::Scalar(0));

    // Loop on iterations
    for ( int k = 0 ; k < iterations ; k++ )
    {
	// Odd pixels
	#pragma omp parallel for
	for ( int i = 1 ; i < tfs->rows-1 ; i++ )
	{
	    // Weber coefficients
	    float tfsc , tfsn , tfss , tfse , tfsw , tfdn , tfds , tfde , tfdw;
	    float rhon , rhos , rhoe , rhow;
	    for ( int j = 2-i%2 ; j < tfs->cols-1 ; j = j + 2 )
	    {
		// Get pixels in neighbourhood of original image
		tfsc = ((float*)(tfs->data+i*tfs->step))[j];
		tfsn = ((float*)(tfs->data+(i-1)*tfs->step))[j];
		tfss = ((float*)(tfs->data+(i+1)*tfs->step))[j];
		tfse = ((float*)(tfs->data+i*tfs->step))[j-1];
		tfsw = ((float*)(tfs->data+i*tfs->step))[j+1];                
		
		// Get pixels in neighbourhood of light image
		tfdn = ((float*)(tfd->data+(i-1)*tfd->step))[j];
		tfds = ((float*)(tfd->data+(i+1)*tfd->step))[j];
		tfde = ((float*)(tfd->data+i*tfd->step))[j-1];
		tfdw = ((float*)(tfd->data+i*tfd->step))[j+1];                    

		// Compute weber coefficients
		rhon = min(tfsn,tfsc) / max<float>(1.0,abs(tfsn-tfsc));
		rhos = min(tfss,tfsc) / max<float>(1.0,abs(tfss-tfsc));
		rhoe = min(tfse,tfsc) / max<float>(1.0,abs(tfse-tfsc));
		rhow = min(tfsw,tfsc) / max<float>(1.0,abs(tfsw-tfsc));                    

		// Compute LightImage(i,j)                    
		((float*)(tfd->data+i*tfd->step))[j] = ( ( tfsc + lambda *
		( rhon * tfdn + rhos * tfds + rhoe * tfde + rhow * tfdw ) )
		/ ( 1 + lambda * ( rhon + rhos + rhoe + rhow ) ) );
	    }
	}

	tfd->copyTo(*tfs);
	// Even pixels
	#pragma omp parallel for
	for ( int i = 1 ; i < tfs->rows-1 ; i++ )
	{
	    // Weber coefficients
	    float tfsc , tfsn , tfss , tfse , tfsw , tfdn , tfds , tfde , tfdw;
	    float rhon , rhos , rhoe , rhow;
	    for ( int j = 1+i%2 ; j < tfs->cols-1 ; j = j + 2 )
	    {
		// Get pixels in neighbourhood of original image
		tfsc = ((float*)(tfs->data+i*tfs->step))[j];
		tfsn = ((float*)(tfs->data+(i-1)*tfs->step))[j];
		tfss = ((float*)(tfs->data+(i+1)*tfs->step))[j];
		tfse = ((float*)(tfs->data+i*tfs->step))[j-1];
		tfsw = ((float*)(tfs->data+i*tfs->step))[j+1];                
		
		// Get pixels in neighbourhood of light image
		tfdn = ((float*)(tfd->data+(i-1)*tfd->step))[j];
		tfds = ((float*)(tfd->data+(i+1)*tfd->step))[j];
		tfde = ((float*)(tfd->data+i*tfd->step))[j-1];
		tfdw = ((float*)(tfd->data+i*tfd->step))[j+1];                    

		// Compute weber coefficients
		rhon = min(tfsn,tfsc) / max<float>(1.0,abs(tfsn-tfsc));
		rhos = min(tfss,tfsc) / max<float>(1.0,abs(tfss-tfsc));
		rhoe = min(tfse,tfsc) / max<float>(1.0,abs(tfse-tfsc));
		rhow = min(tfsw,tfsc) / max<float>(1.0,abs(tfsw-tfsc));                    

		// Compute LightImage(i,j)                    
		((float*)(tfd->data+i*tfd->step))[j] = ( ( tfsc + lambda *
		( rhon * tfdn + rhos * tfds + rhoe * tfde + rhow * tfdw ) )
		/ ( 1 + lambda * ( rhon + rhos + rhoe + rhow ) ) );
	    }
	}

	tfd->copyTo(*tfs);
	tfd->convertTo(*pDst, pDst->depth());
    } // end of iterations k

    // Borders of image
    #pragma omp parallel for
    for ( int i = 0 ; i < tfd->rows ; i++ )
    {
	((uchar*)(pDst->data+i*pDst->step))[0] =
	((uchar*)(pDst->data+i*pDst->step))[1] ;
	((uchar*)(pDst->data+i*pDst->step))[pDst->cols-1] =
	((uchar*)(pDst->data+i*pDst->step))[pDst->cols-2] ;
    }
    #pragma omp parallel for
    for ( int j = 0 ; j < tfd->cols ; j++ )
    {
	((uchar*)(pDst->data))[j] =
	((uchar*)(pDst->data+pDst->step))[j] ;
	((uchar*)(pDst->data+(pDst->rows-1)*pDst->step))[j] =
	((uchar*)(pDst->data+(pDst->rows-2)*pDst->step))[j] ;
    }

    delete tfs;
    delete tfd;

}

vector<cv::Point> findContour(const cv::Mat* pSrc, const cv::Point& rCenter, const vector<float>& rTheta,
			    int minRadius, int maxRadius, const cv::Mat* pMask)
{
    // Output
    vector<cv::Point> contour;
    contour.resize(rTheta.size());

    // Unwrap the image
    cv::Mat* unwrapped;
    unwrapped = unwrapRing(pSrc,rCenter,minRadius,maxRadius,rTheta);

    // Smooth image
    processAnisotropicSmoothing(unwrapped,unwrapped,100,1);	
    
    // Extract the gradients
    computeVerticalGradients(unwrapped,unwrapped);

    // Take into account the mask
    if ( pMask )
    {
	cv::Mat* mask_unwrapped;
	mask_unwrapped = unwrapRing(pMask,rCenter,minRadius,maxRadius,rTheta);
	cv::Mat* temp = new cv::Mat(unwrapped->rows, unwrapped->cols, unwrapped->depth());
	unwrapped->copyTo(*temp);
	unwrapped->setTo(cv::Scalar(0));
	temp->copyTo(*unwrapped, *mask_unwrapped);
	delete mask_unwrapped;
	delete temp;
    }

    // Find optimal path in unwrapped image
    vector<int> optimalPath;
    runViterbi(unwrapped,optimalPath);
    for ( int i = 0 ; i < optimalPath.size() ; i++ )
    {
	contour[i] = convertPolarToCartesian(rCenter,minRadius+optimalPath[i],rTheta[i]);
    }

    delete unwrapped;
    return contour;
}
