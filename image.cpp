
#include "image.h"
#include "eye.h"
#include "processing.h"

int threadNumber = 1;
int algorithmVersion = 1;
bool loadProcessedIris(EyeBall& rEye, int iterator, irisDatabase base);

void Configuration::loadGaborFilters()
{
	// Open text file containing the filters
	ifstream file("filters.txt",ios::in);
	if ( ! file )
	{
		throw runtime_error("Cannot load Gabor filters in file filters.txt");
	}

	// Get the number of filters
	int n_filters;
	file >> n_filters;
	mGaborFilters.resize(n_filters);

	// Size of filter
	int rows , cols;
	// Loop on each filter
	for ( int f = 0 ; f < n_filters ; f++ )
	{    
		// Get the size of the filter
		file >> rows;
		file >> cols;
		cout << " rows = " << to_string(rows) << " cols = " << to_string(cols) << endl;
		// Temporary filter. Will be destroyed at the end of loop

		mGaborFilters[f] = new cv::Mat(rows,cols,CV_32F);            

		// Set the value at coordinates r,c
		for ( int r = 0 ; r < rows ; r++ )
		{
			for ( int c = 0 ; c < cols ; c++ )
			{
				float value;
				file >> value;
				mGaborFilters[f]->at<float>(r*cols+c) = (float)value;

			}
		}

	} // Loop on each filter
	file.close();
}

// Load the application points (build a binary matrix) from a textfile
void Configuration::loadApplicationPoints()
{
    // Open text file containing the filters
    ifstream file("points510.txt",ios::in);
    if ( ! file )
    {
        throw runtime_error("Cannot load the application points in points.txt") ;
    }

    // Get the number of points
    int n_points = 0;
    file >> n_points;

    // Allocate memory for the matrix containing the points
    mpApplicationPoints = new cv::Mat(MHEIGHTOFNORMALIZEDIRIS,MWIDTHOFNORMALIZEDIRIS,CV_8UC1);

    // Initialize all pixels to "off"
    mpApplicationPoints->setTo(cv::Scalar(0));

    // Local variables
    int i , j;

    // Loop on each point
    for ( int p = 0 ; p < n_points ; p++ )
    {    
        // Get the coordinates
        file >> i ; file >> j;
        
        // Set pixel to "on"
        if ( i < 0 || i > mpApplicationPoints->rows-1 || j < 0 || j > mpApplicationPoints->cols-1 )
        {
            cout << "Point (" << i << "," << j << ") ";
            cout << "exceeds size of normalized image : ";
            cout << mpApplicationPoints->rows << "x" << mpApplicationPoints->cols;
            cout << " while loading application points" << endl;
        }
        else
        {
	    mpApplicationPoints->data[(i)*mpApplicationPoints->cols+j] = 255;
        }
    }
    file.close();
}


void showImage(const cv::Mat* img, const cv::Mat* img2, const cv::Mat* img3, const cv::Mat* img4)
{
	cv::namedWindow("Image", 1);
	cv::imshow("Image",*img);
	if(img2)
	{
		cv::namedWindow("Image2", 1);
		cv::imshow("Image2",*img2);
	}
	if(img3)
	{
		cv::namedWindow("Image3", 1);
		cv::imshow("Image3",*img3);
	}
	if(img4)
	{
		cv::namedWindow("Image4", 1);
		cv::imshow("Image4",*img4);
	}
	cv::waitKey(0);
}

void loadImagePath(string& img, int imgNr, irisDatabase base)
{
	string irisFile;
	if(base == IRIS_BASE_1)
	{
		irisFile = "imgList1.txt";
	}
	else if(base == IRIS_SAMPLE_BASE)
	{
		irisFile = "imgList2.txt";
	}
	ifstream imgList(irisFile,ios::in); //Open iris list
	if ( imgList.good() == false )
    {
        throw runtime_error("Cannot load the image list in loadImagePath");
    }
    imgList >> img;
    if(imgNr)
    {
	    for(int i = 0; i < imgNr; i++)
	    {
		imgList.seekg((1 + imgList.tellg()));
		imgList >> img;
	    }
    }
    imgList.close();
}

void saveProcessedImage(EyeBall& eye ,string name)
{
	if(eye.irisCode != NULL && eye.normalizedMask != NULL)
	{
		name.insert(name.find_first_of('/') + 1, "processed/");
		string dir = name;
		dir.insert(0, "mkdir -p ");
		dir.erase(dir.find_last_of('.'), dir.size()-dir.find_last_of('.'));
		system(dir.c_str());
		string tempName = name;
		tempName.insert(name.find_first_of('.'), "/irisCode");
		if ( !cv::imwrite(tempName.c_str(), *eye.irisCode) )
		{
			cout << "Cannot save irisCode" << endl ;
		}
		tempName = name;
		tempName.insert(tempName.find_first_of('.'), "/mask");
		if ( !cv::imwrite(tempName.c_str(), *eye.normalizedMask) )
		{
			cout << "Cannot save normalizedMask" << endl ;
		}
		name.erase(name.find_last_of('.'), name.size()-name.find_last_of('.'));
		name.erase(0, name.find_last_of('/')+1);
		//cout << "saving " << name << endl;
		eye.name = name;
	}
}
void IrisUtility::processMainDatabase(Configuration& conf)
{
	EyeBall eye;
	string imgName;
	cv::Mat tempImg;
	for(int i = IMAGE_TO_START; i < IMAGE_TO_READ_BASE; i++)
	{
		
		loadImagePath(imgName,i, IRIS_BASE_1);
		cout << imgName.c_str() << endl;
		tempImg = cv::imread(imgName.c_str(),IMREAD_GRAYSCALE);
		eye.originalImage = new cv::Mat(tempImg.rows, tempImg.cols, tempImg.type());
		tempImg.copyTo(*eye.originalImage);
		if(eye.originalImage == NULL)
		{
			cout << "error loading: \" " << imgName << " \" " << endl;
			continue;
		}
		if(algorithmVersion == FAST_VERSION)
			eye.detectPupilFasterVesrion(eye.originalImage);
		else
			eye.detectPupil(eye.originalImage);
		eye.detectIris(eye.originalImage);
		eye.normalize(MWIDTHOFNORMALIZEDIRIS, MHEIGHTOFNORMALIZEDIRIS);
		eye.encode(conf.mGaborFilters);
		saveProcessedImage(eye, imgName);
		
		delete eye.originalImage;
		delete eye.irisCode;
		delete eye.normalizedImage;
		delete eye.normalizedMask;
		delete eye.eyeMask;
		delete eye.irisMask;
		delete eye.pupilMask;
	}
	
}
void IrisUtility::processSingleEye(EyeBall& dstEye, int index, Configuration& conf)
{
	EyeBall eye;
	string imgName;
	cv::Mat tempImg;
chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	loadImagePath(imgName, index, IRIS_SAMPLE_BASE);
	cout << imgName.c_str() << endl;
	tempImg = cv::imread(imgName.c_str(),IMREAD_GRAYSCALE);
	eye.originalImage = new cv::Mat(tempImg.rows, tempImg.cols, tempImg.type());
	tempImg.copyTo(*eye.originalImage);
chrono::steady_clock::time_point end0= std::chrono::steady_clock::now();
	if(eye.originalImage == NULL)
	{
		cout << "error loading: \" " << imgName << " \" " << endl;
	}
	if(algorithmVersion == FAST_VERSION)
		eye.detectPupilFasterVesrion(eye.originalImage);
	else
		eye.detectPupil(eye.originalImage);
chrono::steady_clock::time_point end1= std::chrono::steady_clock::now();
	eye.detectIris(eye.originalImage);
chrono::steady_clock::time_point end2= std::chrono::steady_clock::now();
	eye.normalize(MWIDTHOFNORMALIZEDIRIS, MHEIGHTOFNORMALIZEDIRIS);
chrono::steady_clock::time_point end3= std::chrono::steady_clock::now();
	eye.encode(conf.mGaborFilters);
chrono::steady_clock::time_point end4= std::chrono::steady_clock::now();
saveProcessedImage(eye, imgName);
	dstEye.normalizedMask = new cv::Mat(eye.normalizedMask->rows, eye.normalizedMask->cols, eye.normalizedMask->depth());
	dstEye.irisCode = new cv::Mat(eye.irisCode->rows, eye.irisCode->cols, eye.irisCode->depth());
	eye.irisCode->copyTo(*dstEye.irisCode);
	eye.normalizedMask->copyTo(*dstEye.normalizedMask);
chrono::steady_clock::time_point end5= std::chrono::steady_clock::now();
cout << "Preparing phase = " << std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin).count() <<std::endl;
cout << "detectPupil time = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - end0).count() <<std::endl;
cout << "detectIris time = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count() <<std::endl;
cout << "normalize time = " << std::chrono::duration_cast<std::chrono::microseconds>(end3 - end2).count() <<std::endl;
cout << "encode time = " << std::chrono::duration_cast<std::chrono::microseconds>(end4 - end3).count() <<std::endl;
cout << "saving time = " << std::chrono::duration_cast<std::chrono::microseconds>(end5 - end4).count() <<std::endl;
cout << "total time = " << std::chrono::duration_cast<std::chrono::microseconds>(end5 - begin).count() <<std::endl;
	imgName.erase(imgName.find_last_of('.'), imgName.size()-imgName.find_last_of('.'));
	imgName.erase(0, imgName.find_last_of('/')+1);
	dstEye.name = imgName;
cv::circle(*eye.originalImage, cv::Point(eye.pupil.center.x, eye.pupil.center.y), eye.pupil.radius, cv::Scalar(255,0,255), 2, 1);
cv::circle(*eye.originalImage, cv::Point(eye.pupil.center.x, eye.pupil.center.y), eye.iris.radius, cv::Scalar(255,0,255), 2, 1);
showImage(eye.originalImage, eye.eyeMask, eye.irisCode);
	delete eye.originalImage;
	delete eye.irisCode;
	delete eye.normalizedImage;
	delete eye.normalizedMask;
	delete eye.eyeMask;
	delete eye.irisMask;
	delete eye.pupilMask;
	
}

//Run only when full database was processed
void IrisUtility::runForHistograFalseCompare(Configuration& conf)
{
	cout << "runForHistograFalseCompare" << endl;
	vector<float> scoreVect;
	string eyeName;
	float score = 0.0;
	EyeBall eye;
	EyeBall eye2;
	BAW bwBalance;
	scoreVect.clear();
	
	//#pragma omp parallel for private(eye2, score, comparision, eyeName)
	for(int imgIndex = 0; imgIndex < IMAGE_TO_READ_SAMPLES; imgIndex++)
	{
		cout << imgIndex << endl;
		loadProcessedIris(eye,imgIndex, IRIS_SAMPLE_BASE);
		#pragma omp parallel for private(eye2, score)
		for(int i = IMAGE_TO_START; i < IMAGE_TO_READ_BASE; i++)
		{
			if((int)(imgIndex/4) != i)
			{
				if(loadProcessedIris(eye2, i, IRIS_BASE_1))
				{
					
					if(eye.name != eye2.name)
					{
						score = eye.match(eye2,conf.mpApplicationPoints);
						#pragma omp critical
						{
							scoreVect.push_back(score);
						}
					}
					delete eye2.irisCode;
					delete eye2.normalizedMask;
				}
			}
		}
		delete eye.irisCode;
		delete eye.normalizedMask;
	}
fstream file("histPoints.txt", ios::out | ios::trunc);
for(auto& s : scoreVect)
{
	file << s << endl;
}
file.close();
cout << "END" << endl;
	
	
}

void IrisUtility::simulateAccesRequest(Configuration& conf, int processing)
{
	cout << "simulateAccesRequest" << endl;
	
	vector <string> badComapredEyes;
	vector<float> scoreVect;
	vector<float> comparisionVect;
	string eyeName;
	int processedImage = 0;
	int matchScore = 0;
	float score = 0.0;
	float comparision;
	EyeBall eye;
	EyeBall eye2;
	QualityStandard qt;
	BAW bwBalance;
		
	comparisionVect.clear();
	
	//#pragma omp parallel for private(eye2, score, comparision, eyeName)
	for(int imgIndex = IMAGE_TO_START; imgIndex < IMAGE_TO_READ_SAMPLES; imgIndex++)
	{
		scoreVect.clear();
		if(processing == 1)
		{
			processSingleEye(eye, imgIndex, conf);			
		}
		else
		{
			loadProcessedIris(eye,imgIndex, IRIS_SAMPLE_BASE);
		}
		comparision = 1.0;

		#pragma omp parallel for private(eye2, score)
		for(int i = IMAGE_TO_START; i < IMAGE_TO_READ_BASE; i++)
		{
			//if((int)(imgIndex/4) != i)
			//{
				if(loadProcessedIris(eye2, i, IRIS_BASE_1))
				{
					if(eye.name != eye2.name)
					{
						score = eye.match(eye2,conf.mpApplicationPoints);
						scoreVect.push_back(score);
						#pragma omp critical
						if(score < comparision)
						{
							comparision = score;
							eyeName = eye2.name;
						}
					delete eye2.irisCode;
					delete eye2.normalizedMask;
					}
				}
			//}
		}
		processedImage++;
		delete eye.irisCode;
		delete eye.normalizedMask;
		cout << eye.name << " similar with: " << eyeName << " Score: " << comparision << endl;
		if(!eye.name.compare(0,(eye.name.size()-1) ,eyeName, 0, (eyeName.size()-1) ))
		{
			if(comparision < ACCEPTANCE_THRESHOLD)
			{
				#pragma omp critical
				qt.tp++;
			}
			else
			{
				#pragma omp critical
				qt.fn++;
			}
			matchScore++;
			comparisionVect.push_back(comparision);
			cout << "############ SUCCESS "  << matchScore << "/" << IMAGE_TO_READ_SAMPLES - IMAGE_TO_START << " Tootal images: " << processedImage << endl;
		}
		else
		{
			if(comparision < ACCEPTANCE_THRESHOLD)
			{
				#pragma omp critical
				qt.fp++;
			}
			else
			{
				#pragma omp critical
				qt.tn++;
			}
			badComapredEyes.push_back(eye.name);
			badComapredEyes.push_back(eyeName);			
		}

	}
	cout << "Compared scores: " << endl;
	for(auto& c : comparisionVect)
		cout << to_string(c) << endl;
	cout << " SCORES with bad compared: " << endl;
	for(auto& v : scoreVect)
		cout << to_string(v) << endl;
	if(!badComapredEyes.empty())
	{
		cout << endl;
		cout << "Badly compared eyes: " << endl;
		for(int i = 0; i < badComapredEyes.size(); i+=2)
		{
			cout << badComapredEyes[i] << " and " << badComapredEyes[i+1] << endl;
		}
		cout << endl;
	}
	float result = matchScore * 100 / (IMAGE_TO_READ_SAMPLES - IMAGE_TO_START);
	cout << "overal Score: " << result << "%" << endl;
	cout << qt.tp << " " << qt.tn << " " << qt.fp << " " << qt.fn << endl;
	cout << "ACC: " << to_string(qt.calculateACC()) << endl;
	cout << "PPV: " << to_string(qt.calculatePPV()) << endl;
	cout << "NPV: " << to_string(qt.calculateNPV()) << endl;
	cout << "TPR: " << to_string(qt.calculateTPR()) << endl;
	cout << "TNR: " << to_string(qt.calculateTNR()) << endl;
	
	
	
}
void processDatabase(Configuration& conf)
{
	EyeBall eye;
	string imgName;
	cv::Mat tempImg;
	for(int i = IMAGE_TO_START; i < IMAGE_TO_READ_BASE; i++)
	{
		chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		
		loadImagePath(imgName,i, IRIS_BASE_1);
		cout << imgName.c_str() << endl;
		tempImg = cv::imread(imgName.c_str(),IMREAD_GRAYSCALE);
		eye.originalImage = new cv::Mat(tempImg.rows, tempImg.cols, tempImg.type());
		tempImg.copyTo(*eye.originalImage);
		//*eye.originalImage = cv::imread(imgName.c_str(),IMREAD_GRAYSCALE);
		chrono::steady_clock::time_point end0= std::chrono::steady_clock::now();
		if(eye.originalImage == NULL)
		{
			cout << "error loading: \" " << imgName << " \" " << endl;
			continue;
		}
		if(algorithmVersion == FAST_VERSION)
			eye.detectPupilFasterVesrion(eye.originalImage);
		else
			eye.detectPupil(eye.originalImage);
		chrono::steady_clock::time_point end1= std::chrono::steady_clock::now();
		eye.detectIris(eye.originalImage);
		chrono::steady_clock::time_point end2= std::chrono::steady_clock::now();
		eye.normalize(MWIDTHOFNORMALIZEDIRIS, MHEIGHTOFNORMALIZEDIRIS);
		chrono::steady_clock::time_point end3= std::chrono::steady_clock::now();
		eye.encode(conf.mGaborFilters);
		chrono::steady_clock::time_point end4= std::chrono::steady_clock::now();
		saveProcessedImage(eye, imgName);
		//cv::circle(*eye.originalImage, cv::Point(eye.pupil.center.x, eye.pupil.center.y), eye.pupil.radius, cv::Scalar(255,0,255), 2, 1);
		//cv::circle(*eye.originalImage, cv::Point(eye.pupil.center.x, eye.pupil.center.y), eye.iris.radius, cv::Scalar(255,0,255), 2, 1);
		//showImage(eye.originalImage, eye.eyeMask);
		
		chrono::steady_clock::time_point end5= std::chrono::steady_clock::now();
		cout << "Preparing phase = " << std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin).count() <<std::endl;
		cout << "detectPupil time = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - end0).count() <<std::endl;
		cout << "detectIris time = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count() <<std::endl;
		cout << "normalize time = " << std::chrono::duration_cast<std::chrono::microseconds>(end3 - end2).count() <<std::endl;
		cout << "encode time = " << std::chrono::duration_cast<std::chrono::microseconds>(end4 - end3).count() <<std::endl;
		cout << "saving time = " << std::chrono::duration_cast<std::chrono::microseconds>(end5 - end4).count() <<std::endl;
		cout << "total time = " << std::chrono::duration_cast<std::chrono::microseconds>(end5 - begin).count() <<std::endl;
		//cv::circle(*eye.originalImage, cv::Point(eye.pupil.center.x, eye.pupil.center.y), eye.pupil.radius, cv::Scalar(255,0,255), 2, 1);
		//cv::circle(*eye.originalImage, cv::Point(eye.pupil.center.x, eye.pupil.center.y), eye.iris.radius, cv::Scalar(255,0,255), 2, 1);
		//showImage(eye.originalImage);
		
		delete eye.originalImage;
		delete eye.irisCode;
		delete eye.normalizedImage;
		delete eye.normalizedMask;
		delete eye.eyeMask;
		delete eye.irisMask;
		delete eye.pupilMask;
		
	}
}

bool loadProcessedIris(EyeBall& rEye, int iterator, irisDatabase base)
{
	string imgName;
	cv::Mat tempImg;
	
	loadImagePath(imgName, iterator, base);
	imgName.insert(imgName.find_first_of('/') + 1, "processed/");
	string tempName = imgName;
	
	tempName.insert(tempName.find_first_of('.'), "/irisCode");

	tempImg = cv::imread(tempName.c_str(), IMREAD_GRAYSCALE);
	rEye.irisCode = new cv::Mat(tempImg.rows, tempImg.cols, tempImg.depth());
	tempImg.copyTo(*rEye.irisCode);
	tempName = imgName;
	tempName.insert(tempName.find_first_of('.'), "/mask");

	tempImg = cv::imread(tempName.c_str(), IMREAD_GRAYSCALE);
	rEye.normalizedMask = new cv::Mat(tempImg.rows, tempImg.cols, tempImg.depth());
	tempImg.copyTo(*rEye.normalizedMask);
	imgName.erase(imgName.find_last_of('.'), imgName.size()-imgName.find_last_of('.'));
	imgName.erase(0, imgName.find_last_of('/')+1);
	rEye.name = imgName;
	
	if ( rEye.irisCode && rEye.normalizedMask)
	{
		return true;
	}
	    
	cout << "cannot load template iris and mask from: " << imgName << endl;
    
	return false;
}

void runCompareTest(Configuration& conf)
{
	cout << "runCompareTest" << endl;
	vector <string> badComapredEyes;
	int matchScore = 0;
	EyeBall eye;
	EyeBall eye2;
	QualityStandard qt;
	int processedImage = 0;
	float score = 0.0;
	vector<float> scoreVect;
	vector<float> comparisionVect;
	comparisionVect.clear();
	float comparision;
	string eyeName;
	#pragma omp parallel for private(eye, eye2, score, comparision, eyeName)
	for(int imgIndex = IMAGE_TO_START; imgIndex < IMAGE_TO_READ_BASE; imgIndex++)
	{
		scoreVect.clear();
		loadProcessedIris(eye,imgIndex, IRIS_BASE_1);
		comparision = 1.0;
		for(int i = IMAGE_TO_START; i < IMAGE_TO_READ_BASE; i++)
		{
			if(loadProcessedIris(eye2, i, IRIS_BASE_1))
			{				
				if(eye.name != eye2.name)
				{
					score = eye.match(eye2,conf.mpApplicationPoints);
					scoreVect.push_back(score);
					if(score < comparision)
					{
						comparision = score;
						eyeName = eye2.name;
					}
				}
				delete eye2.irisCode;
				delete eye2.normalizedMask;
			}
		}
		processedImage++;
		delete eye.irisCode;
		delete eye.normalizedMask;
		cout << eye.name << " similar with: " << eyeName << " Score: " << comparision << endl;
		comparisionVect.push_back(comparision);
		if(!eye.name.compare(0,(eye.name.size()-1) ,eyeName, 0, (eyeName.size()-1) ))
		{
			if(comparision < ACCEPTANCE_THRESHOLD)
			{
				#pragma omp critical
				qt.tp++;
			}
			else
			{
				#pragma omp critical
				qt.fn++;
			}
			matchScore++;
			cout << "############ SUCCESS "  << matchScore << "/" << IMAGE_TO_READ_BASE - IMAGE_TO_START << " Tootal images: " << processedImage << endl;
		}
		else
		{
			if(comparision < ACCEPTANCE_THRESHOLD)
			{
				#pragma omp critical
				qt.fp++;
			}
			else
			{
				#pragma omp critical
				qt.tn++;
			}
			badComapredEyes.push_back(eye.name);
			badComapredEyes.push_back(eyeName);			
		}
	}
	cout << "Compared scores: " << endl;
	for(auto& c : comparisionVect)
		cout << to_string(c) << endl;
	if(!badComapredEyes.empty())
	{
		cout << endl;
		cout << "Badly compared eyes: " << endl;
		for(int i = 0; i < badComapredEyes.size(); i+=2)
		{
			cout << badComapredEyes[i] << " and " << badComapredEyes[i+1] << endl;
		}
		cout << endl;
	}
	float result = matchScore * 100 / (IMAGE_TO_READ_BASE - IMAGE_TO_START);
	cout << "overal Score: " << result << "%" << endl;
	cout << qt.tp << " " << qt.tn << " " << qt.fp << " " << qt.fn << endl;
	cout << "ACC: " << to_string(qt.calculateACC()) << endl;
	cout << "PPV: " << to_string(qt.calculatePPV()) << endl;
	cout << "NPV: " << to_string(qt.calculateNPV()) << endl;
	cout << "TPR: " << to_string(qt.calculateTPR()) << endl;
	cout << "TNR: " << to_string(qt.calculateTNR()) << endl;
}

int main(){
	printf("IrisRecognition\n");
	cout << "number of thread to execute: " ;
	cin >> threadNumber;
	cout << endl;
	cout << "Algorithm Version (1 fast, 0 normal): ";
	cin >> algorithmVersion;
	cout << endl;
	omp_set_num_threads(threadNumber);
	cout << CV_VERSION << endl;

	Configuration conf;
	conf.loadGaborFilters();
	conf.loadApplicationPoints();
	IrisUtility eyeUtil;
	int startProcessing = 0;
	cout << "type 1 to start processing" << endl;
	cin >> startProcessing;
	if(startProcessing == 1)
	{
		//processDatabase(conf);
		eyeUtil.processMainDatabase(conf);

	}
	//runCompareTest(conf);
	eyeUtil.simulateAccesRequest(conf, startProcessing);
	eyeUtil.runForHistograFalseCompare(conf);

	return 0;
}
