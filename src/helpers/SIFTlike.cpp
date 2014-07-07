/* SIFTlike.cpp
 * Author: Silvia-Laura Pintea
 */
#include "SIFTlike.h"
#include <vl/mathop.h>
#include <vl/dsift.h>
#include <math.h>
#include <assert.h>
//==============================================================================
/** Extracts a certain SIFT descriptor.
 */
void SIFTlike::getOpponent(const IplImage *iplimage,\
std::vector<IplImage*> &feat,std::vector<cv::Point2f> &points){
	clock_t begin = clock();
	cv::Mat descriptors, input;
	cv::Mat image(iplimage);
	assert(image.channels()==3 || image.channels()==1);
	image.copyTo(input);
	if(image.channels()==1){
		cv::cvtColor(image,input,CV_GRAY2BGR);
	}
	std::vector<cv::Mat> channels; // BGR
	cv::split(input,channels);
	input.release();
	// [0] Now smooth the image
	channels[0].convertTo(channels[0],CV_8UC1);
	channels[1].convertTo(channels[1],CV_8UC1);
	channels[2].convertTo(channels[2],CV_8UC1);
	cv::blur(channels[0],channels[0],cv::Size(this->patchsize_,this->patchsize_));
	cv::blur(channels[1],channels[1],cv::Size(this->patchsize_,this->patchsize_));
	cv::blur(channels[2],channels[2],cv::Size(this->patchsize_,this->patchsize_));
	channels[0].convertTo(channels[0],CV_32FC1);
	channels[1].convertTo(channels[1],CV_32FC1);
	channels[2].convertTo(channels[2],CV_32FC1);
	// [1] Extract the gray sift as the third channel.
	cv::Mat gray;
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	gray.convertTo(gray,CV_8UC1);
	cv::blur(gray,gray,cv::Size(this->patchsize_,this->patchsize_));
	gray.convertTo(gray,CV_32FC1);
	cv::Mat opp3SIFT = this->oneChannelSIFT(gray,points);
	gray.release();
	// [2] Convert image from BGR to opponent
	cv::Mat opp1 = cv::Mat::zeros(gray.size(),CV_32FC1);
	cv::Mat opp2 = cv::Mat::zeros(gray.size(),CV_32FC1);
	opp1 = (channels[2]-channels[1])*std::sqrt(2.0); // R-G
	opp2 = (channels[2]+channels[1]-2.0*channels[0])*std::sqrt(6.0); // R+G-2B
	channels[0].release(); channels[1].release(); channels[2].release();
	opp1.convertTo(opp1,CV_32FC1);
	opp2.convertTo(opp2,CV_32FC1);
	// [3] get the mini and maxi and scale it back to [0,255]
	double mini1,maxi1;
	cv::minMaxLoc(opp1,&mini1,&maxi1);
	opp1 = (opp1-mini1)*(255.0/(maxi1-mini1));
	double mini2,maxi2;
	cv::minMaxLoc(opp2,&mini2,&maxi2);
	opp2 = (opp2-mini2)*(255.0/(maxi2-mini2));
	// [4] Extract gray SIFT from first channel
 	cv::Mat opp1SIFT;
	if(mini1!=maxi1){
		points.clear();
		opp1SIFT = this->oneChannelSIFT(opp1,points);
	}else{
		opp3SIFT.copyTo(opp1SIFT);
	}
	opp1.release();
	// [5] Extract SIFT from the second channel
	cv::Mat opp2SIFT;
	if(mini2!=maxi2){
		points.clear();
		opp2SIFT = this->oneChannelSIFT(opp2,points);
	}else{
		opp3SIFT.copyTo(opp2SIFT);
	}
	opp2.release();
	// [6] Copy together all the descriptors and put them in a mat
	descriptors = cv::Mat::zeros(cv::Size(opp1SIFT.cols+opp2SIFT.cols+opp3SIFT.cols,\
		opp1SIFT.rows),opp1SIFT.type());
	assert(opp1SIFT.rows==opp2SIFT.rows && opp2SIFT.rows==opp3SIFT.rows);
	cv::Rect roi1(0,0,opp1SIFT.cols,opp1SIFT.rows);
	cv::Rect roi2(opp1SIFT.cols,0,opp2SIFT.cols,opp2SIFT.rows);
	cv::Rect roi3(opp1SIFT.cols+opp2SIFT.cols,0,opp3SIFT.cols,opp3SIFT.rows);
	opp1SIFT.copyTo(descriptors(roi1));
	opp2SIFT.copyTo(descriptors(roi2));
	opp3SIFT.copyTo(descriptors(roi3));
	descriptors.convertTo(descriptors,CV_32FC1);
	opp1SIFT.release(); opp2SIFT.release(); opp3SIFT.release();
	this->desc2iplvect(feat,descriptors,points,image.size());
	descriptors.release();
	clock_t end = clock();
	std::cout<<"[SIFT::getOpponent] time elapsed: "<<double\
		(Auxiliary<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
}
//==============================================================================
/** Puts descriptors into a vector of images.
 */
void SIFTlike::desc2iplvect(std::vector<IplImage*> &feat,const cv::Mat &descriptors,\
const std::vector<cv::Point2f> &points,const cv::Size &imsize){
	// Loop over points and reshape these points as an image
	assert(descriptors.rows==(imsize.width*imsize.height));
	assert(descriptors.rows==points.size());
	feat.resize(descriptors.cols);
	for(unsigned c=0;c<descriptors.cols;++c){
		feat[c] = cvCreateImage(cvSize(imsize.width,imsize.height),IPL_DEPTH_32F,1);
		unsigned r = 0;
		for(std::vector<cv::Point2f>::const_iterator pt=points.begin();pt!=points.end(),\
		r<descriptors.rows;++pt,++r){
			CV_IMAGE_ELEM(feat[c],float,(int)pt->y,(int)pt->x) = descriptors.at<float>(r,c);
		} // over rows
	} // over cols
}
//==============================================================================
/** Just computes SIFT over a 1 channel image.
 */
cv::Mat SIFTlike::oneChannelSIFT(const cv::Mat &originput,std::vector<cv::Point2f> \
&points,bool verbose){
	unsigned numbins = 2;
	unsigned numangl = 9;
	assert((this->patchsize_%numbins)==0);
	assert((this->patchsize_/numbins)%2==0);
	unsigned binsize  = this->patchsize_/numbins;
	float deltaCenter = 0.5F*binsize*(numbins-1); // number of bins is 4
	cv::Mat input;
	cv::copyMakeBorder(originput,input,deltaCenter,deltaCenter,deltaCenter,\
		deltaCenter,cv::BORDER_REFLECT);
	input.convertTo(input,CV_32FC1);
	// [1] Define the dense sift extractor
	int numFrames,descrSize;
	VlDsiftFilter *dsift = vl_dsift_new_basic(input.rows,input.cols,this->step_,binsize);
	if(!this->bounds_.empty()){
		assert(this->bounds_.size()==4);
		vl_dsift_set_bounds(dsift,VL_MAX(this->bounds_[1],0),VL_MAX\
			(this->bounds_[0],0),VL_MIN(this->bounds_[3],input.rows-1),\
			VL_MIN(this->bounds_[2],input.cols-1));
	}
	VlDsiftDescriptorGeometry ageom = *vl_dsift_get_geometry(dsift);
	ageom.binSizeY = binsize;
	ageom.binSizeX = binsize;
	ageom.numBinX  = numbins;
	ageom.numBinY  = numbins;
	ageom.numBinT  = numangl;
	vl_dsift_set_geometry(dsift,&ageom);
	vl_dsift_set_flat_window(dsift,VL_FALSE) ;
	numFrames = vl_dsift_get_keypoint_num(dsift);
	descrSize = vl_dsift_get_descriptor_size(dsift);
	if(verbose){
		VlDsiftDescriptorGeometry const *geom = vl_dsift_get_geometry(dsift);
		int stepX,stepY,minX,minY,maxX,maxY;
		vl_bool useFlatWindow ;
		vl_dsift_get_steps(dsift,&stepY,&stepX) ;
		vl_dsift_get_bounds(dsift,&minY,&minX,&maxY,&maxX) ;
		useFlatWindow = vl_dsift_get_flat_window(dsift) ;
		std::cout<<"[SIFTlike::oneChannelSIFT]: image size ["<<input.cols<<\
			","<<input.rows<<"]"<<std::endl;
		std::cout<<"[SIFTlike::oneChannelSIFT]: ["<<(minX+1)<<","<<(minY+1)<<\
			","<<(maxX+1)<<","<<maxY<<"]"<<std::endl;
		std::cout<<"[SIFTlike::oneChannelSIFT]: stepX="<<stepX<<", stepY="<<\
			stepY<<std::endl;
		std::cout<<"[SIFTlike::oneChannelSIFT]: [tbins:"<<geom->numBinT<<\
			",xbins:"<<geom->numBinX<<",ybins:"<<geom->numBinY<<std::endl;
		std::cout<<"[SIFTlike::oneChannelSIFT]: descriptor size: "<<descrSize<<std::endl;
		std::cout<<"[SIFTlike::oneChannelSIFT]: [binSizeX:"<<geom->binSizeX<<\
			",binSizeY:"<<geom->binSizeY<<"]"<<std::endl;
		std::cout<<"[SIFTlike::oneChannelSIFT]: flat window:"<<useFlatWindow<<std::endl;
		std::cout<<"[SIFTlike::oneChannelSIFT]: window size:"<<\
			vl_dsift_get_window_size(dsift)<<std::endl;
		std::cout<<"[SIFTlike::oneChannelSIFT]: num of features:"<<numFrames<<std::endl;
	}
	// [2] Extract the actual SIFT with all the options added.
	vl_dsift_process(dsift,reinterpret_cast<const float*>(input.data));
	VlDsiftKeypoint const *frames = vl_dsift_get_keypoints(dsift);
	float const *descrs           = vl_dsift_get_descriptors(dsift);
	input.release(); points.clear();
	// [3] Copy the points into the vector.
	cv::Mat out = cv::Mat::zeros(cv::Size(descrSize,numFrames),CV_32FC1);
	for(unsigned f=0;f<numFrames;++f){
		// Stupid vlFeat: x is y and y is x [!]
		points.push_back(cv::Point2f(frames[f].y-deltaCenter,frames[f].x-deltaCenter));
		for(unsigned d=0;d<descrSize;++d){
			out.at<float>(f,d) = VL_MIN(512.0F*descrs[f*descrSize+d],255.0F);
		}
	}
	// [4] L2 normalization
	for(unsigned r=0;r<out.rows;++r){
		float l2norm = 1e-3;
		for(unsigned c=0;c<out.cols;++c){
			l2norm += out.at<float>(r,c)*out.at<float>(r,c);
		}
		l2norm      = std::sqrt(l2norm);
		out.row(r) /= l2norm;
	}
	// [5] Release Mem and return.
	vl_dsift_delete(dsift);
  	return out;
}
//==============================================================================
/** Extracts a certain SIFT descriptor.
 */
void SIFTlike::getGray(const IplImage *iplimage,\
std::vector<IplImage*> &feat,std::vector<cv::Point2f> &points){
	clock_t begin = clock();
	cv::Mat gray;
	cv::Mat image(iplimage);
	if(image.channels()==3){
		cv::cvtColor(image,gray,CV_BGR2GRAY);
	}else{
		image.copyTo(gray);
	}
	// [0] Now smooth the image
	gray.convertTo(gray,CV_8UC1);
	cv::blur(gray,gray,cv::Size(this->patchsize_,this->patchsize_));
	gray.convertTo(gray,CV_32FC1);
	// [2] Extract gray SIFT from first channel
	points.clear();
	cv::Mat descriptors = this->oneChannelSIFT(gray,points);
	gray.release();
	this->desc2iplvect(feat,descriptors,points,image.size());
	descriptors.release();
	clock_t end = clock();
	std::cout<<"[SIFT::getGray] time elapsed: "<<double\
		(Auxiliary<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
}
//==============================================================================
//==============================================================================
//==============================================================================
















