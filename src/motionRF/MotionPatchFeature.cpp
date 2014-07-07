/* MotionPatchFeatures.cpp
 * Author: Silvia-Laura Pintea
 */
#include "MotionPatchFeature.h"
//==============================================================================
/** Gets the flow derivatives patches on x and y around the current pixel as a vector.
 */
template <class F>
void MotionPatchFeature<F>::motion(const F *feature,cv::Mat* motionXX,\
cv::Mat* motionXY,cv::Mat* motionYX,cv::Mat* motionYY) const{
	assert(feature->velXX().size()>this->imIndex_);
	assert(feature->velXY().size()>this->imIndex_);
	assert(feature->velYX().size()>this->imIndex_);
	assert(feature->velYY().size()>this->imIndex_);
	cv::Mat velXX = feature->velXX(this->imIndex_);
	cv::Mat velXY = feature->velXY(this->imIndex_);
	cv::Mat velYX = feature->velYX(this->imIndex_);
	cv::Mat velYY = feature->velYY(this->imIndex_);
	cv::Rect roi(this->point_.x-(this->motionW_)/2,this->point_.y-(this->motionH_)/2,\
		this->motionW_,this->motionH_);
	cv::Mat tmpXX  = cv::Mat(velXX,roi).clone();
	cv::Mat tmpXY  = cv::Mat(velXY,roi).clone();
	cv::Mat tmpYX  = cv::Mat(velYX,roi).clone();
	cv::Mat tmpYY  = cv::Mat(velYY,roi).clone();
	tmpXX.copyTo(*motionXX);tmpXY.copyTo(*motionXY);
	tmpYX.copyTo(*motionYX);tmpYY.copyTo(*motionYY);
	tmpXX.release();tmpXY.release();tmpYX.release();tmpYY.release();
	motionXX->convertTo(*motionXX,CV_32FC1);
	motionXY->convertTo(*motionXY,CV_32FC1);
	motionYX->convertTo(*motionYX,CV_32FC1);
	motionYY->convertTo(*motionYY,CV_32FC1);
}
//==============================================================================
/** Gets the flow patches on x and y around the current pixel as a vector.
 */
template <class F>
void MotionPatchFeature<F>::motion(const F *feature,cv::Mat* motionX,\
cv::Mat* motionY) const{
	assert(feature->velXX().size()>this->imIndex_);
	assert(feature->velYY().size()>this->imIndex_);
	cv::Mat velX = feature->velXX(this->imIndex_);
	cv::Mat velY = feature->velYY(this->imIndex_);
	cv::Rect roi(this->point_.x-(this->motionW_)/2,this->point_.y-(this->motionH_)/2,\
		this->motionW_,this->motionH_);
	cv::Mat tmpX  = cv::Mat(velX,roi).clone();
	cv::Mat tmpY  = cv::Mat(velY,roi).clone();
	tmpX.copyTo(*motionX); tmpY.copyTo(*motionY);
	tmpX.release();tmpY.release();
	motionX->convertTo(*motionX,CV_32FC1);
	motionY->convertTo(*motionY,CV_32FC1);
}
//==============================================================================
/** Gets the flow patches on x and y around the current pixel as a vector.
 */
template <class F>
void MotionPatchFeature<F>::motionCenter(const F *feature,float &mX,float &my) const{
	assert(feature->velXX().size()>this->imIndex_);
	assert(feature->velYY().size()>this->imIndex_);
	cv::Mat velX = feature->velXX(this->imIndex_);
	cv::Mat velY = feature->velYY(this->imIndex_);
	cv::Rect roi(this->point_.x-(this->motionW_)/2,this->point_.y-(this->motionH_)/2,\
		this->motionW_,this->motionH_);
	mX = velX.at<float>(this->point_);
	my = velY.at<float>(this->point_);
}
//==============================================================================
/** Gets the motion patch around the current pixel as a vector.
 */
template <class F>
cv::Mat* MotionPatchFeature<F>::motion(const F *feature) const{
	if(feature->usederivatives()){
		return this->motionDerivatives(feature);
	}else{
		return this->motionFlow(feature);
	}
}
//==============================================================================
/** Gets the flow derivatives patch around the current pixel as a vector.
 */
template <class F>
cv::Mat* MotionPatchFeature<F>::motionDerivatives(const F *feature) const{
	assert(feature->velXX().size()>this->imIndex_);
	assert(feature->velXY().size()>this->imIndex_);
	assert(feature->velYX().size()>this->imIndex_);
	assert(feature->velYY().size()>this->imIndex_);
	cv::Mat velXX   = feature->velXX(this->imIndex_);
	cv::Mat velXY   = feature->velXY(this->imIndex_);
	cv::Mat velYX   = feature->velYX(this->imIndex_);
	cv::Mat velYY   = feature->velYY(this->imIndex_);
	cv::Rect roi(this->point_.x-(this->motionW_)/2,this->point_.y-\
		(this->motionH_)/2,this->motionW_,this->motionH_);
	cv::Mat velXXroi = cv::Mat(velXX,roi).clone();
	cv::Mat velXYroi = cv::Mat(velXY,roi).clone();
	cv::Mat velYXroi = cv::Mat(velYX,roi).clone();
	cv::Mat velYYroi = cv::Mat(velYY,roi).clone();
	velXXroi.convertTo(velXXroi,CV_32FC1);
	velXYroi.convertTo(velXYroi,CV_32FC1);
	velYXroi.convertTo(velYXroi,CV_32FC1);
	velYYroi.convertTo(velYYroi,CV_32FC1);
	unsigned dataSz                 = velXXroi.cols*velXXroi.rows*4;
	cv::Mat* motion                 = new cv::Mat(1,dataSz,CV_32FC1);
	cv::MatIterator_<float> pMotion = motion->begin<float>();
	for(cv::MatConstIterator_<float> xx=velXXroi.begin<float>();xx!=\
	velXXroi.end<float>();++xx,++pMotion){
		(*pMotion) = (*xx);
	}
	for(cv::MatConstIterator_<float> xy=velXYroi.begin<float>();xy!=\
	velXYroi.end<float>();++xy,++pMotion){
		(*pMotion) = (*xy);
	}
	for(cv::MatConstIterator_<float> yx=velYXroi.begin<float>();yx!=\
	velYXroi.end<float>();++yx,++pMotion){
		(*pMotion) = (*yx);
	}
	for(cv::MatConstIterator_<float> yy=velYYroi.begin<float>();yy!=\
	velYYroi.end<float>(),pMotion!=motion->end<float>();++yy,++pMotion){
		(*pMotion) = (*yy);
	}
	velXXroi.release();velXYroi.release();velYXroi.release();velYYroi.release();
	return motion;
}
//==============================================================================
/** Gets the flow-motion patch around the current pixel as a vector.
 */
template <class F>
cv::Mat* MotionPatchFeature<F>::motionFlow(const F *feature) const{
	assert(feature->velXX().size()>this->imIndex_);
	assert(feature->velYY().size()>this->imIndex_);
	cv::Mat velX = feature->velXX(this->imIndex_);
	cv::Mat velY = feature->velYY(this->imIndex_);
	cv::Rect roi(this->point_.x-(this->motionW_)/2,this->point_.y-\
		(this->motionH_)/2,this->motionW_,this->motionH_);
	cv::Mat velXroi = cv::Mat(velX,roi).clone();
	cv::Mat velYroi = cv::Mat(velY,roi).clone();
	velXroi.convertTo(velXroi,CV_32FC1);
	velYroi.convertTo(velYroi,CV_32FC1);
	unsigned dataSz                 = velXroi.cols*velXroi.rows*2;
	cv::Mat* motion                 = new cv::Mat(1,dataSz,CV_32FC1);
	cv::MatIterator_<float> pMotion = motion->begin<float>();
	for(cv::MatConstIterator_<float> x=velXroi.begin<float>();x!=\
	velXroi.end<float>();++x,++pMotion){
		(*pMotion) = (*x);
	}
	for(cv::MatConstIterator_<float> y=velYroi.begin<float>();y!=\
	velYroi.end<float>();++y,++pMotion){
		(*pMotion) = (*y);
	}
	velXroi.release(); velYroi.release();
	return motion;
}
//==============================================================================
/** Gets the image patch around the current pixel as a matrix.
 */
template <class F>
inline cv::Mat* MotionPatchFeature<F>::image(const F *feature) const{
	assert(feature->images().size()>this->imIndex_);
	cv::Mat image = feature->images(this->imIndex_);
	cv::Rect roi(this->point_.x-(this->motionW_)/2,this->point_.y-(this->motionH_)/2,\
		this->motionW_,this->motionH_);
	cv::Mat* imageRoi = new cv::Mat(cv::Mat(image,roi).clone());
	if(imageRoi->channels()==1){
		cv::cvtColor(*imageRoi,*imageRoi,CV_GRAY2BGR);
	}else if(imageRoi->channels()==4){
		cv::cvtColor(*imageRoi,*imageRoi,CV_BGRA2BGR);
	}
	imageRoi->convertTo(*imageRoi,CV_8UC3);
	return imageRoi;
}
//==============================================================================
/** Gets the histogram at the current point position.
 */
template <class F>
std::vector<float> MotionPatchFeature<F>::histoCenter(const F *feature) const{
	assert(feature->histo().size()>this->imIndex_);
	std::vector<cv::Mat> histo = feature->histo(this->imIndex_);
	// Get the hist at the current point only
	std::vector<float> ahist;
	float sum = 0.0;
	for(unsigned b=0;b<histo.size();++b){
		ahist.push_back(histo[b].at<float>(this->point_));
		sum += histo[b].at<float>(this->point_);
	}
	assert(!std::isnan(sum) && !std::isinf(sum) && std::abs(sum-1.0)<0.1);
	return ahist;
}
//==============================================================================
/** Gets the histogram at a random point in the patch.
 */
template <class F>
cv::Mat MotionPatchFeature<F>::histo(const F *feature,cv::Point pt) const{
	assert(feature->histo().size()>this->imIndex_);
	std::vector<cv::Mat> histo = feature->histo(this->imIndex_);
	// Get the hist at the current point only
	pt.x         += (this->point_.x-(this->motionW_)/2);
	pt.y         += (this->point_.y-(this->motionH_)/2);
	cv::Mat ahist = cv::Mat::zeros(cv::Size(histo.size(),1),CV_32FC1);
	float sum     = 0.0;
	for(unsigned b=0;b<histo.size();++b){
		ahist.at<float>(0,b) = histo[b].at<float>(pt);
		sum                 += histo[b].at<float>(pt);
	}
	assert(!std::isnan(sum) && !std::isinf(sum) && std::abs(sum-1.0)<0.1);
	return ahist;
}
//==============================================================================
/** Gets the whole histogram patch.
 */
template <class F>
std::vector<cv::Mat> MotionPatchFeature<F>::histo(const F *feature) const{
	assert(feature->histo().size()>this->imIndex_);
	std::vector<cv::Mat> histo = feature->histo(this->imIndex_);
	// Get the hist at the current point only
	std::vector<cv::Mat> out;
	cv::Rect roi(this->point_.x-(this->motionW_)/2,this->point_.y-(this->motionH_)/2,\
		this->motionW_,this->motionH_);
	cv::Mat check = cv::Mat::zeros(cv::Size(roi.width,roi.height),CV_32FC1);
	for(unsigned b=0;b<histo.size();++b){
		cv::Mat histroi = histo[b](roi);
		out.push_back(histroi.clone());
		check += histroi;
	}
	cv::Scalar sum = cv::sum(check);
	sum.val[0] /= static_cast<float>(check.cols*check.rows);
	assert(!std::isnan(sum.val[0]) && !std::isinf(sum.val[0]) && std::abs\
		(sum.val[0]-1.0)<0.1);
	check.release();
	return out;
}
//==============================================================================
//==============================================================================
//==============================================================================
template class MotionPatchFeature<FeaturesMotion>;

