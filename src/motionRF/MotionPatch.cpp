/* Motion.cpp
 * Author: Silvia-Laura Pintea
 */
#include "MotionPatch.h"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <sintel/imageLib/imageLib.h>
#include <sintel/flowIO.h>
#include <SIFTlike.h>
//==============================================================================
//==============================================================================
//==============================================================================
template <class T,class F>
MotionPatch<T,F>::~MotionPatch(){
	this->imagePairs_.clear();
}
//==============================================================================
/** Resets the class members to add new patches.
 */
template <class T,class F>
void MotionPatch<T,F>::reset(){
	// [0] Clear patches
	if(!this->patches_.empty()){
		for(std::size_t l=0;l<this->patches_.size();++l){
			// [0.1] Created with: push_back(new T)
			for(std::size_t p=0;p<this->patches_[l].size();++p){
				delete this->patches_[l][p];
			}
			this->patches_[l].clear();
		}
		this->patches_.clear();
	}
	// [1] Clear image names
	this->imagePairs_.clear();
	this->imName_.clear();
	// [2] Clear the features
	if(this->features_){
		delete this->features_;
		this->features_ = new F(this->usederivatives_);
	}
}
//==============================================================================
/** Randomly picks a subset of the images names to be used for training -- pairs
 * of 2 images for OF.
 */
template <class T,class F>
void MotionPatch<T,F>::pickRandomNames(const std::string &featpath,const \
std::vector<std::string> &inifolders,const std::string &ext,const std::string \
&imgpath,std::vector<unsigned> &shuffle){
	std::vector<std::string> folders = inifolders;
	unsigned trsz = std::max(1,static_cast<int>(this->trainingSize_/folders.size()));
	// [0] First shuffle the videos.
	std::random_shuffle(folders.begin(),folders.end());
	for(std::vector<std::string>::const_iterator it=folders.begin();it!=\
	folders.end();++it){
		// [0] If the file does not exist then create it.
		Auxiliary<uchar,1>::file_exists((featpath+(*it)).c_str(),true);
		// [1] Get all the images names from the directory
		std::string path = imgpath+(*it);
		std::vector<std::string> filenames = Auxiliary<uchar,1>::listDir(path,ext);
		sort(filenames.begin(),filenames.end());
		trsz = std::min(trsz,static_cast<unsigned>(filenames.size()-1));
		if(filenames.size()<1){
			std::cerr<<"[MotionPatch::pickRandomNames]: less than 1 "<<\
				"training image"<<std::endl;
			std::exit(-1);
		}
		// [2] Pick trsz random indexes
		std::vector<unsigned> indexes;
		while(indexes.size()<trsz){
			// always drop the last image
			unsigned rndidx = std::rand()%(filenames.size()-1);
			std::vector<unsigned>::iterator isthere = std::find(indexes.begin(),\
				indexes.end(),rndidx);
			if(isthere==indexes.end()){indexes.push_back(rndidx);}
		}
		// [2] We need at least 2 images for training
		for(std::vector<unsigned>::iterator idx=indexes.begin();\
		idx!=indexes.end();++idx){
			// [3] If the vector has only 1 img, there is no flow.
			std::vector<std::string> tmpvect;
			tmpvect.push_back(std::string(imgpath+(*it)+PATH_SEP+filenames[*idx]));
			tmpvect.push_back(std::string(imgpath+(*it)+PATH_SEP+filenames[(*idx)+1]));
			this->imagePairs_.push_back(tmpvect);
			// [4] Keep the indexes to change as little code as possible
			shuffle.push_back(static_cast<unsigned>(shuffle.size()));
			this->imName_.push_back(filenames[*idx]);
		}
	}
	if(this->imName_.empty()){
		std::cerr<<"[MotionPatch::pickRandomNamesAbsolute] Empty image list "<<\
			"(file extension not correct) "<<ext<<std::endl;
		throw std::exception();
	}
	// [5] Now randomly shuffle the indices of the images
	std::random_shuffle(shuffle.begin(),shuffle.end());
	this->trainingSize_ = std::min(this->trainingSize_,static_cast<unsigned>\
		(shuffle.size()));
	shuffle.resize(this->trainingSize_);
}
//==============================================================================
/** Extracts the feature patches but also the label patches.
 * imgpath    -- path to the images
 * labpath    -- path to labels
 * ofpath     -- path to optical flow
 * featpath   -- path to features
 * vFilenames -- vector of image names
 * classinfo  -- mapping from pixel color to label ID
 * labH       -- label patch height
 * labW       -- label patch width
 */
template <class T,class F>
void MotionPatch<T,F>::extractPatches(const std::string &imgpath,const \
std::string &labpath,std::string &featpath,const std::vector<std::string> \
&vFilenames,const std::map<cv::Vec3b,unsigned,vec3bCompare> &classinfo,\
const std::string &labTerm,const std::string &ext,bool justimages,\
bool extractHisto){
	this->reset();
	// [0] Loop over all images in the folder
	std::vector<unsigned> imIndex;
	if(!justimages){
		if(this->multiclass_){
			unsigned trsz = this->trainingSize_;
			std::vector<std::string> tmpIms;
			for(std::vector<std::string>::const_iterator f=vFilenames.begin();f!=\
			vFilenames.end();++f){
				std::vector<unsigned> tmpIndex;
				std::string path               = (imgpath+(*f));
				std::vector<std::string> files = Auxiliary<char,1>::listDir(path);
				this->trainingSize_            = trsz/vFilenames.size();
				this->pickRandomNames(featpath,files,ext,path,tmpIndex);
				for(std::vector<unsigned>::iterator t=tmpIndex.begin();t!=tmpIndex.end();++t){
					unsigned index = tmpIms.size()+static_cast<int>(*t);
					imIndex.push_back(index);
				}
				tmpIms.insert(tmpIms.end(),this->imName_.begin(),this->imName_.end());
				this->imName_.clear();
			}
			this->trainingSize_ = tmpIms.size();
			this->imName_       = tmpIms;
			std::cout<<imIndex<<" \n"<<this->imName_.size()<<" "<<\
				this->imagePairs_.size()<<std::endl;
			assert(this->imName_.size()==this->imagePairs_.size());
		}else{
			this->pickRandomNames(featpath,vFilenames,ext,imgpath,imIndex);
		}
	}else{
		for(std::vector<std::string>::const_iterator f=vFilenames.begin();f!=\
		vFilenames.end()-1;++f){
			// add images one by one:
			this->imName_.push_back(*f);
			// add the pairs for flow
			std::vector<std::string> tmpvect;
			tmpvect.push_back(imgpath+(*f));
			tmpvect.push_back(imgpath+(*(f+1)));
			this->imagePairs_.push_back(tmpvect);
			// add the image indexes
			imIndex.push_back(f-vFilenames.begin());
		}
	}
	for(std::vector<unsigned>::const_iterator i=imIndex.begin();i!=imIndex.end();++i){
		if((i-imIndex.begin())%100==0){
			std::cout<<"[MotionPatch::extractPatches]: Processing image: "<<\
				(this->imagePairs_[*i][0])<<" "<<(*i)<<std::endl;
		}
		// [1] Try to load the patches if not possible then compute them
		std::string justname = this->imName_[*i].substr(0,this->imName_[*i].size()-4);
		std::string featname = std::string(featpath+justname+".bin");
		// [2] Load the original image anyway
		std::string impath   = std::string(this->imagePairs_[*i][0]);
		IplImage *tmpinit    = cvLoadImage(impath.c_str(),-1);
		IplImage *init       = cvCreateImage(cvGetSize(tmpinit),\
			IPL_DEPTH_8U,tmpinit->nChannels);
		cvConvertScale(tmpinit,init); cvReleaseImage(&tmpinit);
		float scale          = this->maximsize_/static_cast<float>(std::max\
			(init->width,init->height));
		IplImage *img;
		if(scale<1.0){
			img = cvCreateImage(cvSize((int)(init->width*scale),\
				(int)(init->height*scale)),init->depth,init->nChannels);
			cvResize(init,img);
			cvReleaseImage(&init);
		}else{
			img = cvCloneImage(init);
			cvReleaseImage(&init);
		}
		// [4] Load the actual image
		if(!img){
			cvReleaseImage(&img);
			std::cout<<"[MotionPatch::extractPatches] Could not "<<\
				"load image file: "<<this->imName_[*i]<<std::endl;
			std::exit(-1);
		}
		try{
			// [5] Labels, appearance & motion
			this->loadPatches(featname);
		}catch(std::exception &e){
			std::cout<<"[MotionPatch::extractPatches]: Cannot open file"<<\
				featname<<std::endl;
			// [6] Load optical flow matrices
			if(this->relativeOF_){
				int alright = this->extractMotionRelative(this->imagePairs_[*i],\
					this->algo_,featpath,i-imIndex.begin(),false);
				if(alright==-1){
					cvReleaseImage(&img);
					continue;
				}
			}else{
				this->extractMotionAbsolute(this->imagePairs_[*i],this->algo_,\
					featpath,i-imIndex.begin(),false);
			}
			IplImage *imgfeat = cvCloneImage(img);
			this->extractFeatures(imgfeat,featname);
			cvReleaseImage(&imgfeat);
			// [7] Save all these features/labels
			if(this->storefeat_){
				this->savePatches(featname,this->features_->size()-1);
			}
		}
		this->features_->push_backImages(cv::Mat(img,true));
		cvReleaseImage(&img);
	}
	// [8] Once all the data is there compute the histograms.
	if(extractHisto){ this->computeHistograms();}
	// [9] Resize the patches to the desired number if any
	this->resizePatches();
}
//==============================================================================
/** Get the histogram of magnitudes.
 */
template <class T,class F>
void MotionPatch<T,F>::getMagniHisto(){
	assert(this->sigmas_.size()==1);
	if(this->sigmas_[0]){
		if(this->usederivatives_){
			this->getMagniHistoDerivativesKernels();
		}else{
			this->getMagniHistoFlowKernels();
		}
	}else{
		if(this->usederivatives_){
			this->getMagniHistoDerivativesHard();
		}else{
			this->getMagniHistoFlowHard();
		}
	}
}
//==============================================================================
/** Get the histogram of magnitudes from flow derivatives.
 */
template <class T,class F>
void MotionPatch<T,F>::getMagniHistoDerivativesKernels(){
	// [0] Loop over all images one by one to get the stds
	float nopatches = 0.0,
		maxXX = -std::numeric_limits<float>::max(),
		maxXY = -std::numeric_limits<float>::max(),
		maxYX = -std::numeric_limits<float>::max(),
		maxYY = -std::numeric_limits<float>::max(),
		minXX = std::numeric_limits<float>::max(),
		minXY = std::numeric_limits<float>::max(),
		minYX = std::numeric_limits<float>::max(),
		minYY = std::numeric_limits<float>::max();
	// [1] Get the stds from the data with true means
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velXX = this->features_->velXX(i);
		cv::Mat velXY = this->features_->velXY(i);
		cv::Mat velYX = this->features_->velYX(i);
		cv::Mat velYY = this->features_->velYY(i);
		nopatches    += (velXX.cols*velXX.rows)/static_cast<float>(this->step_);
		// [1] Loop over each pixel in the mat
		cv::Mat_<float>::const_iterator xy = velXY.begin<float>();
		cv::Mat_<float>::const_iterator yx = velYX.begin<float>();
		cv::Mat_<float>::const_iterator yy = velYY.begin<float>();
		for(cv::Mat_<float>::const_iterator xx=velXX.begin<float>();xx!=velXX.end<float>(),\
		xy!=velXY.end<float>(),yx!=velYX.end<float>(),yy!=velYY.end<float>();\
		xx+=this->step_,xy+=this->step_,yx+=this->step_,yy+=this->step_){
			if(maxXX<(*xx)){maxXX = (*xx);}
			if(maxXY<(*xy)){maxXY = (*xy);}
			if(maxYX<(*yx)){maxYX = (*yx);}
			if(maxYY<(*yy)){maxYY = (*yy);}
			if(minXX>(*xx)){minXX = (*xx);}
			if(minXY>(*xy)){minXY = (*xy);}
			if(minYX>(*yx)){minYX = (*yx);}
			if(minYY>(*yy)){minYY = (*yy);}
		}
	}
	// [2] Normalize the stds properly	
	unsigned bindims = std::pow(this->bins_,1.0/4.0);
	assert(std::pow(bindims,4.0)==this->bins_);
	assert(this->sigmas_.size()==1);
	float sigma      = this->sigmas_[0];
	float constant1  = 1.0/(std::sqrt(2.0*M_PI)*sigma);
	float constant2  = (2.0*sigma*sigma);
	float stepXX     = (maxXX-minXX)/static_cast<float>(bindims);
	float stepXY     = (maxXY-minXY)/static_cast<float>(bindims);
	float stepYX     = (maxYX-minYX)/static_cast<float>(bindims);
	float stepYY     = (maxYY-minYY)/static_cast<float>(bindims);
	std::vector<float> histinfo;
	histinfo.push_back(this->bins_);
	histinfo.push_back(minXX);
	histinfo.push_back(minXY);
	histinfo.push_back(minYX);
	histinfo.push_back(minYY);
	histinfo.push_back(maxXX);
	histinfo.push_back(maxXY);
	histinfo.push_back(maxYX);
	histinfo.push_back(maxYY);
	this->features_->histinfo(histinfo);
	// [3] Now loop again and compute the histograms
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velXX = this->features_->velXX(i);
		cv::Mat velXY = this->features_->velXY(i);
		cv::Mat velYX = this->features_->velYX(i);
		cv::Mat velYY = this->features_->velYY(i);
		std::vector<cv::Mat> histo;
		for(unsigned b=0;b<this->bins_;++b){
			histo.push_back(cv::Mat::zeros(velXX.size(),CV_32FC1));
		}
		// loop over the bins (histo channels)
		cv::Mat_<float>::const_iterator xy = velXY.begin<float>();
		cv::Mat_<float>::const_iterator yx = velYX.begin<float>();
		cv::Mat_<float>::const_iterator yy = velYY.begin<float>();
		for(cv::Mat_<float>::const_iterator xx=velXX.begin<float>();xx!=velXX.end<float>(),\
		xy!=velXY.end<float>(),yx!=velYX.end<float>(),yy!=velYY.end<float>();\
		++xx,++xy,++yx,++yy){
			float norm     = 0.0;
			unsigned index = 0;
			// clip the x,y values
			float valxx    = std::max(std::min(valxx,maxXX),minXX);
			float valxy    = std::max(std::min(valxy,maxXY),minXY);
			float valyx    = std::max(std::min(valyx,maxYX),minYX);
			float valyy    = std::max(std::min(valyy,maxYY),minYY);
			cv::Point pos((xx-velXX.begin<float>())%velXX.cols,\
				(xx-velXX.begin<float>())/velXX.cols);
			for(unsigned bxx=0;bxx<bindims;++bxx){
				for(unsigned bxy=0;bxy<bindims;++bxy){
					for(unsigned byx=0;byx<bindims;++byx){
						for(unsigned byy=0;byy<bindims;++byy){
							float binXX    = (minXX+stepXX*static_cast<float>(bxx)+stepXX/2.0);
							float binXY    = (minXY+stepXY*static_cast<float>(bxy)+stepXY/2.0);
							float binYX    = (minYX+stepYX*static_cast<float>(byx)+stepYX/2.0);
							float binYY    = (minYY+stepYY*static_cast<float>(byy)+stepYY/2.0);
							float kernelXX = (binXX-valxx)*(binXX-valxx);
							float kernelXY = (binXY-valxy)*(binXY-valxy);
							float kernelYX = (binYX-valyx)*(binYX-valyx);
							float kernelYY = (binYY-valyy)*(binYY-valyy);
							float kernel   = kernelXX+kernelXY+kernelYX+kernelYY;
							float contri   = constant1*std::exp(-kernel/constant2);
							// [4.1] Now update the contribution to the current bin
							histo[index].at<float>(pos) += contri;
							norm                        += contri;
							++index;
						} // over bins for yy
					} // over bins for yx
				} // over bins for xy
			} // over bins for xx
			if(norm>SMALL){
				for(unsigned b=0;b<this->bins_;++b){
					histo[b].at<float>(pos) /= norm;
				}
			}
		} // end loop over pixels
		// [5] Save the histo in the histo vector
		this->features_->push_backHisto(histo);
		for(unsigned b=0;b<this->bins_;++b){
			histo[b].release();
		}
	}
}
//==============================================================================
/** Get the histogram of magnitudes from flow derivatives.
 */
template <class T,class F>
void MotionPatch<T,F>::getMagniHistoDerivativesHard(){
	// [0] Loop over all images one by one to get the stds
	float nopatches = 0.0,\
		minXX = std::numeric_limits<float>::max(),\
		minXY = std::numeric_limits<float>::max(),\
		minYX = std::numeric_limits<float>::max(),\
		minYY = std::numeric_limits<float>::max(),\
		maxXX = -std::numeric_limits<float>::max(),\
		maxXY = -std::numeric_limits<float>::max(),\
		maxYX = -std::numeric_limits<float>::max(),\
		maxYY = -std::numeric_limits<float>::max();
	// [1] Get the stds from the data with true means
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velXX = this->features_->velXX(i);
		cv::Mat velXY = this->features_->velXY(i);
		cv::Mat velYX = this->features_->velYX(i);
		cv::Mat velYY = this->features_->velYY(i);
		nopatches    += (velXX.cols*velXX.rows)/static_cast<float>(this->step_);
		// [1] Loop over each pixel in the mat
		cv::Mat_<float>::const_iterator xy = velXY.begin<float>();
		cv::Mat_<float>::const_iterator yx = velYX.begin<float>();
		cv::Mat_<float>::const_iterator yy = velYY.begin<float>();
		for(cv::Mat_<float>::const_iterator xx=velXX.begin<float>();xx!=velXX.end<float>(),\
		xy!=velXY.end<float>(),yx!=velYX.end<float>(),yy!=velYY.end<float>();\
		xx+=this->step_,xy+=this->step_,yx+=this->step_,yy+=this->step_){
			if(maxXX<(*xx)){maxXX = (*xx);}
			if(maxXY<(*xy)){maxXY = (*xy);}
			if(maxYX<(*yx)){maxYX = (*yx);}
			if(maxYY<(*yy)){maxYY = (*yy);}
			if(minXX>(*xx)){minXX = (*xx);}
			if(minXY>(*xy)){minXY = (*xy);}
			if(minYX>(*yx)){minYX = (*yx);}
			if(minYY>(*yy)){minYY = (*yy);}
		}
	}
	// [2] Normalize the stds properly
	unsigned bindims = std::pow(this->bins_,1.0/4.0);
	assert(std::pow(bindims,4.0)==this->bins_);
	float stepXX     = (maxXX-minXX)/static_cast<float>(bindims);
	float stepXY     = (maxXY-minXY)/static_cast<float>(bindims);
	float stepYX     = (maxYX-minYX)/static_cast<float>(bindims);
	float stepYY     = (maxYY-minYY)/static_cast<float>(bindims);
	std::vector<float> histinfo;
	histinfo.push_back(this->bins_);
	histinfo.push_back(minXX);
	histinfo.push_back(minXY);
	histinfo.push_back(minYX);
	histinfo.push_back(minYY);
	histinfo.push_back(maxXX);
	histinfo.push_back(maxXY);
	histinfo.push_back(maxYX);
	histinfo.push_back(maxYY);
	this->features_->histinfo(histinfo);
	// [3] Now loop again and compute the histograms
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velXX = this->features_->velXX(i);
		cv::Mat velXY = this->features_->velXY(i);
		cv::Mat velYX = this->features_->velYX(i);
		cv::Mat velYY = this->features_->velYY(i);
		std::vector<cv::Mat> histo;
		for(unsigned b=0;b<this->bins_;++b){
			histo.push_back(cv::Mat::zeros(velXX.size(),CV_32FC1));
		}
		// loop over the bins (histo channels)
		cv::Mat_<float>::const_iterator xy = velXY.begin<float>();
		cv::Mat_<float>::const_iterator yx = velYX.begin<float>();
		cv::Mat_<float>::const_iterator yy = velYY.begin<float>();
		for(cv::Mat_<float>::const_iterator xx=velXX.begin<float>();xx!=velXX.end<float>(),\
		xy!=velXY.end<float>(),yx!=velYX.end<float>(),yy!=velYY.end<float>();\
		++xx,++xy,++yx,++yy){
			cv::Point pos((xx-velXX.begin<float>())%velXX.cols,\
				(xx-velXX.begin<float>())/velXX.cols);
			int xxpos = std::floor(((*xx)-minXX)/stepXX);
			int xypos = std::floor(((*xy)-minXY)/stepXY);
			int yxpos = std::floor(((*yx)-minYX)/stepYX);
			int yypos = std::floor(((*yy)-minYY)/stepYY);
			if(xxpos >= bindims){xxpos = bindims-1;}
			if(xypos >= bindims){xypos = bindims-1;}
			if(yxpos >= bindims){yxpos = bindims-1;}
			if(yypos >= bindims){yypos = bindims-1;}
			// pos -- the position in the patch if !usepick, else 0
			int binid = yypos+bindims*(yxpos+bindims*(xypos+bindims*xxpos));
			assert(binid>=0 && binid<this->bins_);
			++histo[binid].at<float>(pos);
		} // end loop over pixels
		// [5] Save the histo in the histo vector
		this->features_->push_backHisto(histo);
		for(unsigned b=0;b<this->bins_;++b){
			histo[b].release();
		}
	} // over images
}
//==============================================================================
/** Get the histogram of magnitudes from flow only.
 */
template <class T,class F>
void MotionPatch<T,F>::getMagniHistoFlowKernels(){
	// [0] Loop over all images one by one to get the stds
	float nopatches = 0.0,\
		minX = std::numeric_limits<float>::max(),\
		minY = std::numeric_limits<float>::max(),\
		maxX = -std::numeric_limits<float>::max(),\
		maxY = -std::numeric_limits<float>::max();
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velX = this->features_->velXX(i);
		cv::Mat velY = this->features_->velYY(i);
		nopatches   += (velX.cols*velX.rows)/static_cast<float>(this->step_);
		// [1] Loop over each pixel in the mat
		cv::Mat_<float>::const_iterator y = velY.begin<float>();
		for(cv::Mat_<float>::const_iterator x=velX.begin<float>();x!=velX.end<float>(),\
		y!=velY.end<float>();x+=this->step_,y+=this->step_){
			if(maxX<(*x)){maxX = (*x);}
			if(maxY<(*y)){maxY = (*y);}
			if(minX>(*x)){minX = (*x);}
			if(minY>(*y)){minY = (*y);}
		}
	}
	// [2] Normalize the stds properly
	unsigned bindims = std::pow(this->bins_,1.0/2.0);
	assert(std::pow(bindims,2.0)==this->bins_);
	assert(this->sigmas_.size()==1);
	float sigma      = this->sigmas_[0];
	float constant1  = 1.0/(std::sqrt(2.0*M_PI)*sigma);
	float constant2  = (2.0*sigma*sigma);
	float stepX      = (maxX-minX)/static_cast<float>(bindims);
	float stepY      = (maxY-minY)/static_cast<float>(bindims);
	std::vector<float> histinfo;
	histinfo.push_back(this->bins_);
	histinfo.push_back(minX);
	histinfo.push_back(minY);
	histinfo.push_back(maxX);
	histinfo.push_back(maxY);
	this->features_->histinfo(histinfo);
	// [3] Now loop again and compute the histograms
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velX = this->features_->velXX(i);
		cv::Mat velY = this->features_->velYY(i);
		std::vector<cv::Mat> histo;
		for(unsigned b=0;b<this->bins_;++b){
			histo.push_back(cv::Mat::zeros(velX.size(),CV_32FC1));
		}
		// loop over the bins (histo channels)
		cv::Mat_<float>::const_iterator y = velY.begin<float>();
		for(cv::Mat_<float>::const_iterator x=velX.begin<float>();x!=velX.end<float>(),\
		y!=velY.end<float>();++x,++y){
			float norm     = 0.0;
			unsigned index = 0;
			// clip the x,y values
			float valx     = std::max(std::min((*x),maxX),minX);
			float valy     = std::max(std::min((*y),maxY),minY);
			cv::Point pos((x-velX.begin<float>())%velX.cols,\
				(x-velX.begin<float>())/velX.cols);
			for(unsigned bx=0;bx<bindims;++bx){
				for(unsigned by=0;by<bindims;++by){
					float binX    = (minX+stepX*static_cast<float>(bx)+stepX/2.0);
					float binY    = (minY+stepY*static_cast<float>(by)+stepY/2.0);
					float kernelX = (binX-valx)*(binX-valx);
					float kernelY = (binY-valy)*(binY-valy);
					float kernel  = kernelX+kernelX; 
					float contri  = constant1*std::exp(-kernel/constant2);
					// [4.1] Now update the contribution to the current bin
					histo[index].at<float>(pos) += contri;
					norm                        += contri;
					++index;
				} // bins over y
			} // bins over x
			if(norm>SMALL){
				for(unsigned b=0;b<this->bins_;++b){
					histo[b].at<float>(pos) /= norm;
				}
			}
		} // end loop over pixels
		// [5] Save the histo in the histo vector
		this->features_->push_backHisto(histo);
		for(unsigned b=0;b<this->bins_;++b){
			histo[b].release();
		}
	}
}
//==============================================================================
/** Get the histogram of magnitudes from flow only.
 */
template <class T,class F>
void MotionPatch<T,F>::getMagniHistoFlowHard(){
	// [0] Loop over all images one by one to get the stds
	float nopatches=0.0,
		minX = std::numeric_limits<float>::max(),\
		minY = std::numeric_limits<float>::max(),\
		maxX = -std::numeric_limits<float>::max(),\
		maxY = -std::numeric_limits<float>::max();
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velX = this->features_->velXX(i);
		cv::Mat velY = this->features_->velYY(i);
		nopatches   += (velX.cols*velX.rows)/static_cast<float>(this->step_);
		// [1] Loop over each pixel in the mat
		cv::Mat_<float>::iterator y = velY.begin<float>();
		for(cv::Mat_<float>::iterator x=velX.begin<float>();x!=velX.end<float>(),\
		y!=velY.end<float>();x+=this->step_,y+=this->step_){
			if(maxX<(*x)){maxX = (*x);}
			if(maxY<(*y)){maxY = (*y);}
			if(minX>(*x)){minX = (*x);}
			if(minY>(*y)){minY = (*y);}
		}
	}
	// [2] Normalize the stds properly
	unsigned bindims = std::pow(this->bins_,1.0/2.0);
	assert(std::pow(bindims,2.0)==this->bins_);
	float stepX      = (maxX-minX)/static_cast<float>(bindims);
	float stepY      = (maxY-minY)/static_cast<float>(bindims);
	std::vector<float> histinfo;
	histinfo.push_back(this->bins_);
	histinfo.push_back(minX);
	histinfo.push_back(minY);
	histinfo.push_back(maxX);
	histinfo.push_back(maxY);
	this->features_->histinfo(histinfo);
	// [3] Now loop again and compute the histograms
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velX = this->features_->velXX(i);
		cv::Mat velY = this->features_->velYY(i);
		std::vector<cv::Mat> histo;
		for(unsigned b=0;b<this->bins_;++b){
			histo.push_back(cv::Mat::zeros(velX.size(),CV_32FC1));
		}
		// loop over the bins (histo channels)
		cv::Mat_<float>::const_iterator y = velY.begin<float>();
		for(cv::Mat_<float>::const_iterator x=velX.begin<float>();x!=velX.end<float>(),\
		y!=velY.end<float>();++x,++y){
			cv::Point pos((x-velX.begin<float>())%velX.cols,\
				(x-velX.begin<float>())/velX.cols);
			int xpos = std::floor(((*x)-minX)/stepX);
			int ypos = std::floor(((*y)-minY)/stepY);
			if(xpos >= bindims){xpos = bindims-1;}
			if(ypos >= bindims){ypos = bindims-1;}
			// pos -- the position in the patch if !usepick, else 0
			int binid = ypos+bindims*xpos;
			assert(binid>0 && binid<this->bins_);
			++histo[binid].at<float>(pos);
		} // end loop over pixels
		// [5] Save the histo in the histo vector
		this->features_->push_backHisto(histo);
		for(unsigned b=0;b<this->bins_;++b){
			histo[b].release();
		}
	}//over images
}
//==============================================================================
/** Get the histogram of angles.
 */
template <class T,class F>
void MotionPatch<T,F>::getAngleHisto(){
	assert(this->sigmas_.size()==1);
	if(this->sigmas_[0]){
		if(this->usederivatives_){
			this->getAngleHistoDerivativesKernels();
		}else{
			this->getAngleHistoFlowKernels();
		}
	}else{
		if(this->usederivatives_){
			this->getAngleHistoDerivativesHard();
		}else{
			this->getAngleHistoFlowHard();
		}
	}
}
//==============================================================================
/** Get the histogram of angles.
 */
template <class T,class F>
void MotionPatch<T,F>::getAngleHistoDerivativesKernels(){
	// [0] Define all the bin information
	float step      = 2.0*M_PI/static_cast<float>(this->bins_);
	assert(this->sigmas_.size()==1);
	float sigma     = this->sigmas_[0];
	float constant1 = 1.0/(std::sqrt(2.0*M_PI)*sigma);
	float constant2 = (2.0*sigma*sigma);
	std::vector<float> histinfo;
	histinfo.push_back(this->bins_);
	histinfo.push_back(step);
	this->features_->histinfo(histinfo);
	unsigned bindims = std::pow(this->bins_,1.0/2.0);
	assert(std::pow(bindims,2.0)==this->bins_);
	// [3] Now loop again and compute the histograms
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velXX = this->features_->velXX(i);
		cv::Mat velXY = this->features_->velXY(i);
		cv::Mat velYX = this->features_->velYX(i);
		cv::Mat velYY = this->features_->velYY(i);
		std::vector<cv::Mat> histo;
		for(unsigned b=0;b<this->bins_;++b){
			histo.push_back(cv::Mat::zeros(velXX.size(),CV_32FC1));
		}
		// [4] Loop over each pixel and get the bins
		cv::Mat_<float>::const_iterator xy = velXY.begin<float>();
		cv::Mat_<float>::const_iterator yx = velYX.begin<float>();
		cv::Mat_<float>::const_iterator yy = velYY.begin<float>();
		for(cv::Mat_<float>::const_iterator xx=velXX.begin<float>();xx!=velXX.end<float>(),\
		xy!=velXY.end<float>(),yx!=velYX.end<float>(),yy!=velYY.end<float>();\
		++xx,++xy,++yx,++yy){
			// loop over the bins (histo channels)
			float norm     = 0.0;
			unsigned index = 0;
			float angleX   = std::atan2((*xy),(*xx))+M_PI;
			float angleY   = std::atan2((*yy),(*yx))+M_PI;
			cv::Point pos((xx-velXX.begin<float>())%velXX.cols,\
				(xx-velXX.begin<float>())/velXX.cols);
			// Each pixel has to contribute 1 to the overall histogram
			for(unsigned bx=0;bx<bindims;++bx){
				for(unsigned by=0;by<bindims;++by){
					float binX    = (step*static_cast<float>(bx)+step/2.0);
					float binY    = (step*static_cast<float>(by)+step/2.0);
					float diffX   = (std::abs(binX-angleX)>M_PI)?(2.0*M_PI-std::abs\
						(binX-angleX)):std::abs(binX-angleX);
					float diffY   = (std::abs(binY-angleY)>M_PI)?(2.0*M_PI-std::abs\
						(binY-angleY)):std::abs(binY-angleY);
					float contri  = constant1*std::exp(-((diffX*diffX)+(diffY*diffY))/constant2);
					histo[index].at<float>(pos) += contri;
					norm                        += contri;
					++index;
				}
			} // over bins
			if(norm>SMALL){
				for(unsigned b=0;b<histo.size();++b){
					histo[b].at<float>(pos) /= norm;
				}
			}
		} // end loop over pixels
		// [5] Save the histo in the histo vector
		this->features_->push_backHisto(histo);
		for(unsigned b=0;b<this->bins_;++b){
			histo[b].release();
		}
	} // over images
}
//==============================================================================
/** Get the histogram of angles.
 */
template <class T,class F>
void MotionPatch<T,F>::getAngleHistoDerivativesHard(){
	// [0] Define all the bin information
	float step = 2.0*M_PI/static_cast<float>(this->bins_);
	std::vector<float> histinfo;
	histinfo.push_back(this->bins_);
	histinfo.push_back(step);
	this->features_->histinfo(histinfo);
	unsigned bindims = std::pow(this->bins_,1.0/2.0);
	assert(std::pow(bindims,2.0)==this->bins_);
	// [3] Now loop again and compute the histograms
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velXX = this->features_->velXX(i);
		cv::Mat velXY = this->features_->velXY(i);
		cv::Mat velYX = this->features_->velYX(i);
		cv::Mat velYY = this->features_->velYY(i);
		std::vector<cv::Mat> histo;
		for(unsigned b=0;b<this->bins_;++b){
			histo.push_back(cv::Mat::zeros(velXX.size(),CV_32FC1));
		}
		// [4] Loop over each pixel and get the bins
		cv::Mat_<float>::const_iterator xy = velXY.begin<float>();
		cv::Mat_<float>::const_iterator yx = velYX.begin<float>();
		cv::Mat_<float>::const_iterator yy = velYY.begin<float>();
		for(cv::Mat_<float>::const_iterator xx=velXX.begin<float>();xx!=velXX.end<float>(),\
		xy!=velXY.end<float>(),yx!=velYX.end<float>(),yy!=velYY.end<float>();\
		++xx,++xy,++yx,++yy){
			// loop over the bins (histo channels)
			float angleX = std::atan2((*xy),(*xx))+M_PI;
			float angleY = std::atan2((*yy),(*yx))+M_PI;
			cv::Point pos((xx-velXX.begin<float>())%velXX.cols,\
				(xx-velXX.begin<float>())/velXX.cols);
			int posX = std::floor(angleX/step);
			int posY = std::floor(angleY/step);
			// [1] Recover the bin position on Y
			if(posX >= bindims){posX = bindims-1;}
			if(posY >= bindims){posY = bindims-1;}
			int binid = posY+bindims*posX;
			assert(binid>=0 && binid<this->bins_);
			++histo[binid].at<float>(pos);
		} // end loop over pixels
		// [5] Save the histo in the histo vector
		this->features_->push_backHisto(histo);
		for(unsigned b=0;b<this->bins_;++b){
			histo[b].release();
		}
	} // over images
}
//==============================================================================
/** Get the histogram of angles for the flow.
 */
template <class T,class F>
void MotionPatch<T,F>::getAngleHistoFlowKernels(){
	// [0] Define all the bin information
	float step      = 2.0*M_PI/static_cast<float>(this->bins_);
	assert(this->sigmas_.size()==1);
	float sigma     = this->sigmas_[0];
	float constant1 = 1.0/(std::sqrt(2.0*M_PI)*sigma);
	float constant2 = (2.0*sigma*sigma);
	std::vector<float> histinfo;
	histinfo.push_back(this->bins_);
	histinfo.push_back(step);
	this->features_->histinfo(histinfo);
	// [3] Now loop again and compute the histograms
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velX = this->features_->velXX(i);
		cv::Mat velY = this->features_->velYY(i);
		std::vector<cv::Mat> histo;
		for(unsigned b=0;b<this->bins_;++b){
			histo.push_back(cv::Mat::zeros(velX.size(),CV_32FC1));
		}
		// [4] Loop over each pixel and get the bins
		cv::Mat_<float>::const_iterator y = velY.begin<float>();
		for(cv::Mat_<float>::const_iterator x=velX.begin<float>();x!=velX.end<float>(),\
		y!=velY.end<float>();++x,++y){
			// loop over the bins (histo channels)
			float norm  = 0.0;
			float angle = std::atan2((*y),(*x))+M_PI;
			cv::Point pos((x-velX.begin<float>())%velX.cols,\
				(x-velX.begin<float>())/velX.cols);
			// Each pixel has to contribute 1 to the overall histogram
			for(unsigned b=0;b<this->bins_;++b){
				float bin    = (step*static_cast<float>(b)+step/2.0);
				float diff   = (std::abs(bin-angle)>M_PI)?(2.0*M_PI-std::abs\
					(bin-angle)):std::abs(bin-angle);
				float contri = constant1*std::exp(-(diff*diff)/constant2);
				histo[b].at<float>(pos) += contri;
				norm                    += contri;
			} // over bins
			if(norm>SMALL){
				for(unsigned b=0;b<histo.size();++b){
					histo[b].at<float>(pos) /= norm;
				}
			}
		} // end loop over pixels
		// [5] Save the histo in the histo vector
		this->features_->push_backHisto(histo);
		for(unsigned b=0;b<this->bins_;++b){
			histo[b].release();
		}
	} // over images
}
//==============================================================================
/** Get the histogram of angles for the flow.
 */
template <class T,class F>
void MotionPatch<T,F>::getAngleHistoFlowHard(){
	// [0] Define all the bin information
	float step      = 2.0*M_PI/static_cast<float>(this->bins_);
	std::vector<float> histinfo;
	histinfo.push_back(this->bins_);
	histinfo.push_back(step);
	this->features_->histinfo(histinfo);
	// [3] Now loop again and compute the histograms
	for(unsigned i=0;i<this->features_->size();++i){
		cv::Mat velX = this->features_->velXX(i);
		cv::Mat velY = this->features_->velYY(i);
		std::vector<cv::Mat> histo;
		for(unsigned b=0;b<this->bins_;++b){
			histo.push_back(cv::Mat::zeros(velX.size(),CV_32FC1));
		}
		// [4] Loop over each pixel and get the bins
		cv::Mat_<float>::const_iterator y = velY.begin<float>();
		for(cv::Mat_<float>::const_iterator x=velX.begin<float>();x!=velX.end<float>(),\
		y!=velY.end<float>();++x,++y){
			// loop over the bins (histo channels)
			float angle = std::atan2((*y),(*x))+M_PI;
			cv::Point pos((x-velX.begin<float>())%velX.cols,\
				(x-velX.begin<float>())/velX.cols);
			int binid = std::floor(angle/step);
			if(binid >= this->bins_){binid = this->bins_-1;}
			assert(binid>=0 && binid<this->bins_);
			++histo[binid].at<float>(pos);
		} // end loop over pixels
		// [5] Save the histo in the histo vector
		this->features_->push_backHisto(histo);
		for(unsigned b=0;b<this->bins_;++b){
			histo[b].release();
		}
	} // over images
}
//==============================================================================
/** Compute histograms of angle\slash magnitude.
 */
template <class T,class F>
void MotionPatch<T,F>::computeHistograms(){
	switch(this->histotype_){
		case(MotionPatch<T,F>::APPROX_MAGNI_KERNEL):
			this->getMagniHisto();
			break;
		case(MotionPatch<T,F>::APPROX_ANGLE_KERNEL):
			this->getAngleHisto();
			break;
		default:
			std::cout<<"[MotionPatch<T,F>::computeHistograms] No histogram pre-"<<\
				"computed."<<std::endl;
			this->getMagniHisto();
			break;
	}
}
//==============================================================================
/** Extracts the feature patches but also the label patches.
 * imgpath    -- path to the images
 * labpath    -- path to labels
 * ofpath     -- path to optical flow
 * featpath   -- path to features
 * vFilenames -- vector of image names
 * classinfo  -- mapping from pixel color to label ID
 * labH       -- label patch height
 * labW       -- label patch width
 */
template <class T,class F>
void MotionPatch<T,F>::extractPatchesOF(const std::string &imgpath,\
std::string &featpath,const std::vector<std::string> &vFilenames,\
const std::string &ext,const MotionPatch<T,F>::Algorithm &algo,bool justimages){
	this->reset();
	// [0] Loop over all images in the folder
	std::vector<unsigned> imIndex;
	if(!justimages){
		this->pickRandomNames(featpath,vFilenames,ext,imgpath,imIndex);
	}else{
		for(std::vector<std::string>::const_iterator f=vFilenames.begin();f!=\
		vFilenames.end()-1;++f){
			// add the images
			this->imName_.push_back(*f);
			// add the image pairs
			std::vector<std::string> tmpvect;
			tmpvect.push_back(imgpath+(*f));
			tmpvect.push_back(imgpath+(*(f+1)));
			this->imagePairs_.push_back(tmpvect);
			// add the image indexes
			imIndex.push_back(f-vFilenames.begin());
		}
	}
	for(std::vector<unsigned>::const_iterator i=imIndex.begin();i!=imIndex.end();++i){
		if((i-imIndex.begin())%100==0){
			std::cout<<"[MotionPatch::extractPatchesOF] Processing image: "<<\
				(this->imagePairs_[*i][0])<<" "<<(*i)<<std::endl;
		}
		// [1] Try to load the patches if not possible then compute them
		std::string justname = this->imName_[*i].substr(0,this->imName_[*i].size()-4);
		std::string featname = std::string(featpath+justname+".bin");
		// [2] Load the original image anyway
		std::string impath   = std::string(this->imagePairs_[*i][0]);
		IplImage *tmpinit    = cvLoadImage(impath.c_str(),-1);
		IplImage *init       = cvCreateImage(cvGetSize(tmpinit),\
			IPL_DEPTH_8U,tmpinit->nChannels);
		cvConvertScale(tmpinit,init); cvReleaseImage(&tmpinit);
		float scale          = this->maximsize_/static_cast<float>(std::max\
			(init->width,init->height));
		IplImage *img;
		if(scale<1.0){
			img = cvCreateImage(cvSize((int)(init->width*scale),\
				(int)(init->height*scale)),init->depth,init->nChannels);
			cvResize(init,img);
			cvReleaseImage(&init);
		}else{
			img = cvCreateImage(cvSize(init->width,init->height),init->depth,init->nChannels);
			img = cvCloneImage(init);
			cvReleaseImage(&init);
		}
		// [4] Load the actual image
		if(!img){
			cvReleaseImage(&img);
			std::cout<<"[MotionPatch::extractPatchesOF] Could not "<<\
				"load image file: "<<this->imName_[*i]<<std::endl;
			std::exit(-1);
		}
		// [6] Load optical flow matrices
		if(this->relativeOF_){
			switch(algo){
				case (MotionPatch<T,F>::Farneback):
					this->extractMotionRelative(this->imagePairs_[*i],\
						MotionPatch<T,F>::Farneback,featpath,0,true);
					break;
				case (MotionPatch<T,F>::LucasKanade):
					this->extractMotionRelative(this->imagePairs_[*i],\
						MotionPatch<T,F>::LucasKanade,featpath,0,true);
					break;
				case (MotionPatch<T,F>::HornSchunck):
					this->extractMotionRelative(this->imagePairs_[*i],\
						MotionPatch<T,F>::HornSchunck,featpath,0,true);
				break;
				case (MotionPatch<T,F>::Simple):
					this->extractMotionRelative(this->imagePairs_[*i],\
						MotionPatch<T,F>::Simple,featpath,0,true);
				break;
			}
		}else{
			switch(algo){
				case (MotionPatch<T,F>::Farneback):
					this->extractMotionAbsolute(this->imagePairs_[*i],\
						MotionPatch<T,F>::Farneback,featpath,0,true);
					break;
				case (MotionPatch<T,F>::LucasKanade):
					this->extractMotionAbsolute(this->imagePairs_[*i],\
						MotionPatch<T,F>::LucasKanade,featpath,0,true);
					break;
				case (MotionPatch<T,F>::HornSchunck):
					this->extractMotionAbsolute(this->imagePairs_[*i],\
						MotionPatch<T,F>::HornSchunck,featpath,0,true);
					break;
				case (MotionPatch<T,F>::Simple):
					this->extractMotionAbsolute(this->imagePairs_[*i],\
						MotionPatch<T,F>::Simple,featpath,0,true);
					break;
			}
		}
		cvReleaseImage(&img);
	}
}
//==============================================================================
/** Extract Harris interest points.
 */
template <class T,class F>
std::vector<cv::KeyPoint> MotionPatch<T,F>::getKeyPoints(const cv::Mat &image,\
unsigned step,unsigned width,unsigned height,unsigned type,bool display,float threshold){
	cv::Mat gray;
	if(image.channels()==3){
		cv::cvtColor(image,gray,CV_BGR2GRAY);
	}else{
		image.copyTo(gray);
	}
	std::vector<cv::KeyPoint> points;
	cv::Mat corners;
	unsigned winsize   = 2;
	unsigned appersize = 3;
	if(static_cast<MotionPatch<T,F>::Points>(type) == MotionPatch<T,F>::HARRIS){
		cv::cornerHarris(gray,corners,winsize,appersize,0.04,cv::BORDER_DEFAULT);
	}else if(static_cast<MotionPatch<T,F>::Points>(type) == MotionPatch<T,F>::CANNY){
		cv::blur(gray,gray,cv::Size(3,3));
		cv::Canny(gray,corners,50,150,3);
	}else if(static_cast<MotionPatch<T,F>::Points>(type) == MotionPatch<T,F>::DENSE){
		corners = cv::Mat::ones(gray.size(),CV_8UC1)*255;
	}
	if(display){Auxiliary<uchar,1>::display(corners);}
	corners.convertTo(corners,CV_32FC1);
	// [3] Loop over the responses and pick the maximum with a step of step
	cv::Mat wherepoints = image.clone();
	unsigned imH = (height%2==0?height/2:(height+1)/2);
	unsigned imW = (width%2==0?width/2:(width+1)/2);
	for(unsigned y=imH;y<corners.rows-imH;y+=step){
		for(unsigned x=imW;x<corners.cols-imW;x+=step){
			cv::Mat roi = corners(cv::Rect(x,y,step,step));
			cv::KeyPoint bestPt;
			cv::Point minPt,maxPt;
			double minVal,maxVal;
			cv::minMaxLoc(roi,&minVal,&maxVal,&minPt,&maxPt);
			if(maxVal>threshold && (maxPt.x+x)<(corners.cols-width/2) && \
			(maxPt.y+y)<(corners.rows-height/2)){
				bestPt.pt.x = maxPt.x+x;
				bestPt.pt.y = maxPt.y+y;
				bestPt.size = 2;
				points.push_back(bestPt);
				if(display){
					cv::circle(wherepoints,bestPt.pt,4,cv::Scalar(255,0,0));
				}
			}
		}
	}
	// [4] Release the corners
	corners.release(); gray.release();
	if(display){
		cv::imshow("Detected points",wherepoints);
		cv::waitKey(10);
	}
	wherepoints.release();
	return points;
}
//==============================================================================
/** Computes opponent channels.
 */
template <class T,class F>
std::vector<cv::Mat> MotionPatch<T,F>::opponent(const cv::Mat &mat){
	std::vector<cv::Mat> opponents;
	if(mat.channels()==1){
		opponents.push_back(mat.clone());
	}else{
		// [1] Extract the gray sift as the third channel.
		cv::Mat gray = cv::Mat::zeros(mat.size(),CV_8UC1);
		if(mat.channels()==3){
			cv::cvtColor(mat,gray,CV_BGR2GRAY);
		}else if(mat.channels()==4){
			cv::cvtColor(mat,gray,CV_BGRA2GRAY);
		}
		gray.convertTo(gray,CV_8UC1);
		opponents.push_back(gray.clone());
		gray.release(); 
		std::vector<cv::Mat> channels;
		cv::split(mat,channels);
		channels[0].convertTo(channels[0],CV_32FC1);
		channels[1].convertTo(channels[1],CV_32FC1);
		channels[2].convertTo(channels[2],CV_32FC1);
		// [2] Convert image from BGR to opponent
		cv::Mat opp1 = cv::Mat::zeros(mat.size(),CV_32FC1);
		cv::Mat opp2 = cv::Mat::zeros(mat.size(),CV_32FC1);
		opp1 = (channels[2]-channels[1])*std::sqrt(2.0); // R-G
		opp2 = (channels[2]+channels[1]-2.0*channels[0])*std::sqrt(6.0); // R+G-2B
		channels[0].release(); channels[1].release(); channels[2].release();
		// [3] get the mini and maxi and scale it back to [0,255]
		double mini1,maxi1;
		cv::minMaxLoc(opp1,&mini1,&maxi1);
		opp1 = (opp1-mini1)*(255.0/(maxi1-mini1));
		opp1.convertTo(opp1,CV_32FC1);
		double mini2,maxi2;
		cv::minMaxLoc(opp2,&mini2,&maxi2);
		opp2 = (opp2-mini2)*(255.0/(maxi2-mini2));
		opp2.convertTo(opp2,CV_32FC1);
		opponents.push_back(opp1.clone());
		opponents.push_back(opp2.clone());
		opp1.release(); opp2.release();
	}
	return opponents;
}
//==============================================================================
/** Computes features if not there for loading.
 */
template <class T,class F>
void MotionPatch<T,F>::extractFeatures(IplImage *img,const std::string &path2feat,\
bool showWhere){
	// [0] Extract the features:
	std::vector<IplImage*> tmpFeat;
	cv::Mat mat(img);
	std::vector<cv::Mat> channels = MotionPatch<T,F>::opponent(mat);
	// [1] Now get Opponent color space
	for(std::vector<cv::Mat>::iterator ch=channels.begin();ch!=channels.end();++ch){
		std::vector<IplImage*> chFeat;
		IplImage *chImg = new IplImage(*ch);
		if(this->hogORsift_){
			CRPatch<T>::extractFeatureChannels9(chImg,chFeat,this->featH_);
		}else{
			std::vector<cv::Point2f> points;
			SIFTlike sift(1,this->featH_);
			sift.getGray(chImg,chFeat,points);
		}
		delete chImg; ch->release();
		tmpFeat.insert(tmpFeat.end(),chFeat.begin(),chFeat.end());
	}
	this->features_->push_backImg(tmpFeat);
	// [1] We pick patches in a grid
	cv::Mat where;
	if(showWhere){
		where = cv::Mat::zeros(cv::Size(img->width,img->height),CV_8UC1);
	}
	unsigned limitW = std::max(std::max(this->labW_,this->featW_),this->motionW_);
	unsigned limitH = std::max(std::max(this->labH_,this->featH_),this->motionH_);
	// [2] Add to patches these ones --- no classes here
	if(this->patches_.empty()){
		this->patches_.resize(1);
	}
	// [3] Loop over the original cols and rows
	std::vector<cv::KeyPoint> points = MotionPatch<T,F>::getKeyPoints(cv::Mat(img),\
		this->step_,this->featW_,this->featH_,this->pttype_,this->display_);
	for(std::vector<cv::KeyPoint>::iterator pt=points.begin();pt!=points.end();++pt){
		T* patch = new T(this->featW_,this->featH_,this->labW_,this->labH_,\
			(this->features_->size()-1),pt->pt,this->motionW_,this->motionH_);
		// [4] We have 4 classes here (1 per quadrant)
		this->patches_[0].push_back(patch);
		if(showWhere){
			where.at<uchar>(pt->pt) = 255;
		}
	}
	if(showWhere){
		cv::imshow("where",where);
		cv::waitKey(10);
	}
	where.release();
}
//==============================================================================
/** Just extracts OF for a pair of images using a given algorithm (hardcoded
 * parameters).
 */
template <class T,class F>
cv::Mat MotionPatch<T,F>::justFlow(cv::Mat &origCurrent,cv::Mat &origNext,\
Algorithm algo,unsigned motionSz,float maximsize,const std::string &imName,const
std::string &featpath,bool sintel,bool store){
	if(sintel){
		return MotionPatch<T,F>::sintelFlow(featpath,maximsize,imName,store);
	}
	cv::Mat flow,status,err,point;
	std::vector<cv::Mat> in;
	// Blur the images first to remove the noise a bit
	cv::Mat current, next;
	cv::cvtColor(origCurrent,current,CV_BGR2GRAY);
	cv::cvtColor(origNext,next,CV_BGR2GRAY);
	// [0.1] Stuff for Lucas Kanade
	std::vector<cv::Point2f> pointpt;
	for(unsigned r=0;r<current.rows;++r){
		for(unsigned c=0;c<current.cols;++c){
			pointpt.push_back(cv::Point2f(c,r));
		}
	}
	std::vector<cv::Point2f> flowpt = pointpt;
	// [1] Staff for Horn Schunck
	IplImage *nextIm = new IplImage(next);
	IplImage *currIm = new IplImage(current);
	IplImage *velx   = cvCreateImage(cvSize(current.cols,current.rows),IPL_DEPTH_32F,1);
	IplImage *vely   = cvCreateImage(cvSize(current.cols,current.rows),IPL_DEPTH_32F,1);
	switch(algo){
		case(MotionPatch<T,F>::Simple):
			std::cout<<"Simple"<<std::endl;
			cv::calcOpticalFlowSF(origCurrent,origNext,flow,3,4,10);
			break;
		case(MotionPatch<T,F>::Farneback):
			// InputArray prev, InputArray next, InputOutputArray flow, \
			double pyr_scale,int levels, int winsize, int iterations, int poly_n,\
			double poly_sigma, int flags)
			std::cout<<"Farneback"<<std::endl;
			cv::calcOpticalFlowFarneback(current,next,flow,0.5,5,11,\
				100,7,1.3,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
			break;
		case(MotionPatch<T,F>::LucasKanade):
			std::cout<<"LucasKanade"<<std::endl;	
			cv::calcOpticalFlowPyrLK(current,next,pointpt,flowpt,status,err,cv::Size\
				(11,11),5,cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,\
				100,0.01),0,1e-4);
			point = cv::Mat(pointpt).t();
			flow  = cv::Mat(flowpt).t();
			flow  = flow-point;
			flow  = flow.reshape(2,current.rows);
			break;
		case(MotionPatch<T,F>::HornSchunck):
			std::cout<<"HornSchunck"<<std::endl;
			CvTermCriteria criteria;
			criteria.type     = CV_TERMCRIT_ITER+CV_TERMCRIT_EPS;
			criteria.max_iter = 100;
			criteria.epsilon  = 0.01;
			cvCalcOpticalFlowHS(currIm,nextIm,0,velx,vely,3.0,criteria);
			in.push_back(cv::Mat(velx).clone());
			in.push_back(cv::Mat(vely).clone());
			cv::merge(in,flow);
			in[0].release();in[1].release();
			break;
		default:
			// InputArray prev, InputArray next, InputOutputArray flow, \
			double pyr_scale,int levels, int winsize, int iterations, int poly_n,\
			double poly_sigma, int flags)
			std::cout<<"Farneback"<<std::endl;
			cv::calcOpticalFlowFarneback(current,next,flow,0.5,5,11,\
				100,5,1.1,cv::OPTFLOW_FARNEBACK_GAUSSIAN);
			break;
	}
	next.release(); current.release();
	// [2] Release matrices and images
	status.release(); err.release(); point.release();
	cvReleaseImage(&velx); cvReleaseImage(&vely); delete nextIm; delete currIm;
	float scale = maximsize/static_cast<float>(std::max(origCurrent.rows,origCurrent.cols));
	if(scale<1.0){
		cv::Size small = cv::Size(origCurrent.cols*scale,origCurrent.rows*scale);
		cv::resize(flow,flow,small);
	}
	return flow;
}
//==============================================================================
/** Finds matches between a set of points
 */
template <class T,class F>
void MotionPatch<T,F>::findmatches(const cv::Mat &points1,const cv::Mat &points2,\
const cv::Mat &img1,const cv::Mat &img2,double minDist,double maxDist,\
cv::Mat &outpoints1,cv::Mat &outpoints2){
	//[0] Transform points to keypoints
	std::vector<cv::KeyPoint> keypoints1,keypoints2;
	cv::KeyPoint::convert(points1,keypoints1,1,1,0,-1);
	cv::KeyPoint::convert(points2,keypoints2,1,1,0,-1);
	//[0] Step 1: Calculate descriptors (feature vectors)
	cv::SiftDescriptorExtractor extractor;
	cv::Mat descr1,descr2;
	extractor.compute(img1,keypoints1,descr1);
	extractor.compute(img2,keypoints2,descr2);
	//[1] Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> matches;
	matcher.match(descr1,descr2,matches);
	//[2] Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	for(unsigned r=0;r<descr1.rows;r++){
		double dist = matches[r].distance;
		if(dist<minDist) minDist = dist;
		if(dist>maxDist) maxDist = dist;
	}
	unsigned scale = 10;
	if(matches.size()<1e+3){scale = maxDist;}
	//[3] Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector<cv::DMatch> good;
	for(int r=0;r<descr1.rows;++r){
		if(matches[r].distance<=scale*minDist){
			good.push_back(matches[r]);
			outpoints1.push_back(points1.row(matches[r].queryIdx).clone());
			outpoints2.push_back(points2.row(matches[r].trainIdx).clone());
		}
	}
	descr1.release(); descr2.release();
	if(this->display_){
		cv::Mat immatches;
		cv::drawMatches(img1,keypoints1,img2,keypoints2,good,immatches,\
			cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),\
			cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		cv::imshow("matches",immatches);
		cv::waitKey(10);
		immatches.release();
	}
}
//==============================================================================
/** My own little sweet RANSAC.
 */
template <class T,class F>
cv::Mat MotionPatch<T,F>::getRansacAffineTransform(const cv::Mat &points1,\
const cv::Mat &points2,const cv::Mat &img1,const cv::Mat &img2,double limit,\
int &isgood){
	cv::Mat finTransform        = cv::Mat::zeros(cv::Size(3,2),CV_32FC1);
	finTransform.at<float>(0,0) = 1;
	finTransform.at<float>(1,1) = 1;
	// [1] Remove points from the roi
	cv::Mat tmppoints1,tmppoints2;
	float rangeCols = 0.4*img1.cols;
	float rangeRows = 0.4*img2.rows;
	cv::Mat_<cv::Vec2f>::const_iterator p2=points2.begin<cv::Vec2f>();
	for(cv::Mat_<cv::Vec2f>::const_iterator p1=points1.begin<cv::Vec2f>();p1!=\
	points1.end<cv::Vec2f>(),p2!=points2.end<cv::Vec2f>();++p1,++p2){
		cv::Vec2f pt1 = (*p1);
		cv::Vec2f pt2 = (*p2);
		if(pt1.val[0]<=rangeCols || pt1.val[0]>=img1.cols-rangeCols || \
		pt1.val[1]<=rangeRows ||pt1.val[1]>=img2.rows-rangeRows){
			tmppoints1.push_back(*p1);
		}
		if(pt2.val[0]<=rangeCols || pt2.val[0]>=img1.cols-rangeCols || \
		pt2.val[1]<=rangeRows ||pt2.val[1]>=img2.rows-rangeRows){
			tmppoints2.push_back(*p2);
		}
	}
	if(tmppoints1.rows<=3 || tmppoints2.rows<=3){
		tmppoints1.release();tmppoints2.release();
		isgood = 2;
		return finTransform;
	}
	cv::Mat outpoints1,outpoints2;
	this->findmatches(tmppoints1,tmppoints2,img1,img2,100,0,outpoints1,outpoints2);
	tmppoints1.release();tmppoints2.release();
	outpoints1.convertTo(outpoints1,cv::DataType<cv::Vec<float,2> >::type);
	outpoints2.convertTo(outpoints2,cv::DataType<cv::Vec<float,2> >::type);
	if(outpoints1.rows<=3 || outpoints2.rows<=3){
		outpoints1.release();outpoints2.release();
		isgood = 2;
		return finTransform;
	}
	// [2] Loop for the RANSAC
	unsigned inliersNr = 0;
	for(unsigned i=0;i<1e+3;++i){
		// [2.1] Pick 3 random points from the first image
		time_t times   = time(NULL);
		int seed       = (int)times+i;
		CvRNG cvRNG(seed);
		std::vector<unsigned> picks;
		unsigned pick = cvRandInt(&cvRNG)%(outpoints1.rows);
		cv::Mat pts1,pts2;
		for(unsigned i=0;i<3;++i){
			cv::Mat apt = outpoints1.row(pick);
			while(std::find(picks.begin(),picks.end(),pick)!=picks.end()){
				pick = cvRandInt(&cvRNG)%(outpoints1.rows);
				apt  = outpoints1.row(pick);
			}
			picks.push_back(pick);
			pts1.push_back(apt.clone());
			pts2.push_back(outpoints2.row(pick).clone());
		}
		// [2.2] Get the transformation
		unsigned inliers = 0;
		cv::Mat outpoints;
		cv::Mat transform = cv::getAffineTransform(pts1,pts2);
		cv::transform(outpoints1,outpoints,transform);
		cv::Mat_<cv::Vec2f>::iterator o2=outpoints2.begin<cv::Vec2f>();
		for(cv::Mat_<cv::Vec2f>::iterator o=outpoints.begin<cv::Vec2f>();\
		o!=outpoints.end<cv::Vec2f>(),o2!=outpoints2.end<cv::Vec2f>();++o,++o2){
			cv::Vec2f vec2 = (*o2);
			cv::Vec2f vec  = (*o);
			double dist    = std::sqrt((vec2.val[0]-vec.val[0])*(vec2.val[0]-\
				vec.val[0])+(vec2.val[1]-vec.val[1])*(vec2.val[1]-vec.val[1]));
			if(dist<=limit){
				++inliers;
			}
		}
		outpoints.release();
		// [3] Finally keep the transformation if the best
		if(inliers>inliersNr){
			finTransform.release();
			transform.copyTo(finTransform);
			inliersNr = inliers;
		}
		transform.release();
	}
	unsigned totpts = std::min(outpoints1.rows,outpoints2.rows);
	std::cout<<"[MotionPatch::getRansacAffineTransform] #inliers:"<<inliersNr<<\
		" tot #points:"<<totpts<<"  transfomr:\n"<<finTransform<<std::endl;
	if(inliersNr<totpts*0.5){isgood = -1;
	}else{isgood = 0;}
	if(std::abs(finTransform.at<float>(0,2))>50 || std::abs(finTransform.at\
	<float>(1,2))>50){isgood = -1;}
	outpoints1.release(); outpoints2.release();
	return finTransform;
}
//==============================================================================
/** Finds interest points and warps the second image to the first image.
 */
template <class T,class F>
int MotionPatch<T,F>::warpSecond2First(cv::Mat &curr,cv::Mat &next){
	// [0] First find the some interest points
	cv::Mat points1,points2;
	cv::goodFeaturesToTrack(curr,points1,1e+5,1e-1,3,cv::Mat(),3,false,0.04);
	cv::goodFeaturesToTrack(next,points2,1e+5,1e-1,3,cv::Mat(),3,false,0.04);
	cv::Mat transform;
	points1.convertTo(points1,cv::DataType<cv::Vec<float,2> >::type);
	points2.convertTo(points2,cv::DataType<cv::Vec<float,2> >::type);
	int isgood = 0;
	transform = this->getRansacAffineTransform(points1,points2,curr,next,5,isgood);
	if(isgood==-1) return -1;
	points1.release(); points2.release();
	// [2] warp the next image to the current image
	cv::Mat warped = next.clone();
	cv::warpAffine(next,warped,(transform.rowRange(0,2)).colRange(0,3),\
		curr.size(),cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT,0);
	transform.release();
	if(this->display_){
		cv::imshow("Current Image",curr);
		cv::waitKey(10);
		cv::imshow("Next Image",next);
		cv::waitKey(10);
		cv::imshow("Warped Image",warped);
		cv::waitKey(10);
	}
	next.release();
	warped.copyTo(next);
	warped.release();
	return 0;
}
//==============================================================================
/** Finds interest points and warps the second image to the first image.
 */
template <class T,class F>
int MotionPatch<T,F>::warpOpenCV(cv::Mat &curr,cv::Mat &next){
	cv::Mat transform = cv::estimateRigidTransform(curr,next,true);
	if(transform.empty()){return -1;}
	cv::Mat warped    = curr.clone();
	// [3] Find how much border we have:
	warped           += 1;
	cv::warpAffine(next,warped,transform,curr.size(),cv::WARP_INVERSE_MAP,\
		cv::BORDER_CONSTANT,0);
	cv::Mat mask;
	cv::inRange(warped,0,0,mask);
	curr.copyTo(warped,mask);
	mask.release();
	warped -= 1;
	next.release();
	warped.copyTo(next);
	warped.release();
	return 0;
}
//==============================================================================
/** Find threshold by cutting the histogram at 0.90.
 */
template <class T,class F>
cv::Mat MotionPatch<T,F>::findThresholdAngle(const cv::Mat &valuesX,const \
cv::Mat &valuesY,float &minTr,float &maxTr){
	unsigned binsz  = 9;
	float step      = (2.0*M_PI)/static_cast<float>(binsz);
	float maxbinVal = 0.0;
	std::vector<float> histo(binsz,0);
	cv::Mat angle   = cv::Mat::zeros(valuesX.size(),CV_32FC1);
	cv::Mat_<float>::iterator a       = angle.begin<float>();
	cv::Mat_<float>::const_iterator y = valuesY.begin<float>();
	for(cv::Mat_<float>::const_iterator x=valuesX.begin<float>();x!=\
	valuesX.end<float>(),y!=valuesY.end<float>(),a!=angle.end<float>();++x,++y,++a){
		(*a)         = std::atan2((*y),(*x))+M_PI;
		unsigned bin = std::floor((*a)/step);
		if(bin>=binsz){bin = binsz-1;}
		++histo[bin];
		if(histo[bin]>maxbinVal){
			maxbinVal = histo[bin];
		}
	}
	//[1] find the std to the maximum value
	minTr = std::numeric_limits<float>::max();
	maxTr = -std::numeric_limits<float>::max();
	for(std::vector<float>::iterator b=histo.begin();b!=histo.end();++b){
		if((*b)==maxbinVal){
			minTr = (step*static_cast<float>(b-(histo.begin())));
			maxTr = (step*static_cast<float>(b-(histo.begin()))+step);
			break;
		}
	}
	return angle;
}
//==============================================================================
/** Load or extract the optical flow vector from two pairs of consecutive images,
 * and then take the difference of their OFs.
 */
template <class T,class F>
int MotionPatch<T,F>::extractMotionRelative(const std::vector<std::string> &tuple,\
Algorithm algo,const std::string &featpath,unsigned offset,bool save){
	std::cout<<"[Motion::extractMotionRelatvie]: extracting OF \n"<<tuple[0]<<\
		"\n"<<tuple[1]<<std::endl;
	// [0] Read the images
	cv::Mat currImg = cv::imread(tuple[0].c_str(),1);
	assert(currImg.channels()==3);
	cv::Mat nextImg = cv::imread(tuple[1].c_str(),1);
	assert(nextImg.channels()==3);
	float scale = this->maximsize_/static_cast<float>(std::max(currImg.rows,currImg.cols));
	if(scale<1.0){
		cv::Size small = cv::Size(currImg.cols*scale,currImg.rows*scale);
		cv::resize(currImg,currImg,small);
		cv::resize(nextImg,nextImg,small);
	}
	// [1] First warp next image to current
	int alright = this->warpOpenCV(currImg,nextImg);
	if(alright == -1){ return -1;}
	clock_t begin = clock();
	// [3] curr(y,x) = next(y+flow(y,x)[1], x+flow(y,x)[0])
	cv::Mat flow = MotionPatch<T,F>::justFlow(currImg,nextImg,algo,\
		std::min(this->motionW_,this->motionH_),this->maximsize_,tuple[0],\
		featpath,this->sintel_,this->storefeat_);
	currImg.release(); nextImg.release();
	// [4] Split the flow into 2 channels
	flow.convertTo(flow,CV_32FC2);
	std::vector<cv::Mat> split;
	cv::split(flow,split);
	// [5] Find threshold to ignore background noise
	cv::Mat valuesX,valuesY;
	cv::multiply(split[0],split[0],valuesX);
	cv::multiply(split[1],split[1],valuesY);
	cv::Mat values = valuesX+valuesY;
	valuesX.release();valuesY.release();
	cv::Mat mask;
	cv::inRange(values,0,2250,mask);
	mask.convertTo(mask,CV_32FC1);
	mask /= 255.0;
	cv::multiply(split[0],mask,split[0]);
	cv::multiply(split[1],mask,split[1]);
	mask.release();
	if(this->threshold_){
		// [5.2] Now find a good threshold and use it
 		float minTr,maxTr;
 		values.convertTo(values,CV_32FC1);
		MotionPatch<T,F>::findThreshold(values,minTr,maxTr);
		cv::Mat maskTr;
		cv::inRange(values,minTr,maxTr,maskTr);
		if(this->display_){
			cv::imshow("Thresholding mask",maskTr);
			cv::waitKey(10);
		}
		maskTr.convertTo(maskTr,CV_32FC1);
		maskTr /= 255.0;maskTr -= 1.0;maskTr *= -1.0;
		cv::multiply(split[0],maskTr,split[0]);
		cv::multiply(split[1],maskTr,split[1]);
		maskTr.release();
	}
	double minX,maxX,minY,maxY;
	cv::minMaxLoc(split[0],&minX,&maxX);
	cv::minMaxLoc(split[1],&minY,&maxY);
	if(!maxX<1e-10 && maxY==1e-10){return -1;}
	values.release();
	// [6] Display the OF vectors if needed
	clock_t end = clock();
	std::cout<<"Of on 1 image time elapsed: "<<double(Auxiliary<uchar,1>::diffclock\
		(end,begin))<<" sec"<<std::endl;
	// [7] Push the flows into the feature matrix
	if(this->usederivatives_){
		cv::Mat splits [] = {split[0],split[1]};
		cv::merge(splits,2,flow);
		cv::Mat derivatives = MotionPatch<T,F>::getFlowDerivatives(flow);
		flow.release(); split[0].release(); split[1].release();
		if(save){
			std::string locally = featpath;
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			int pos = tuple[0].find_last_of(PATH_SEP);
			std::string substring = tuple[0].substr(0,pos);
			int pos2 = substring.find_last_of(PATH_SEP);
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			locally += std::string("deri_motion")+PATH_SEP;
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			locally = locally+tuple[0].substr(pos+1,tuple[0].size()-pos-5)+".bin";
			Auxiliary<float,4>::mat2bin(derivatives,locally.c_str(),false);
		}
		std::vector<cv::Mat> derivativeSplits;
		cv::split(derivatives,derivativeSplits);
		assert(derivativeSplits.size()==4);
		if(this->display_){
			cv::Mat color = cv::imread(tuple[0].c_str(),1);
			color.convertTo(color,CV_8UC3);
			cv::resize(color,color,derivativeSplits[0].size());
			std::string winname = (algo==MotionPatch::Farneback)?"Farneback":\
				((algo==MotionPatch::LucasKanade)?"LucasKanade":\
				((algo==MotionPatch::HornSchunck)?"HornSchunck":"Simple"));
			cv::Mat out = MotionPatch<T,F>::showOFderi(derivativeSplits[0],\
				derivativeSplits[1],derivativeSplits[2],derivativeSplits[3],\
				color,5,true,winname);
			color.release();
			out.release();
		}
		this->features_->push_backVelXX(derivativeSplits[0]);
		derivativeSplits[0].release();
		this->features_->push_backVelXY(derivativeSplits[1]);
		derivativeSplits[1].release();
		this->features_->push_backVelYX(derivativeSplits[2]);
		derivativeSplits[2].release();
		this->features_->push_backVelYY(derivativeSplits[3]);
		derivativeSplits[3].release();
	}else{
		if(save){
			std::string locally = featpath;
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			int pos = tuple[0].find_last_of(PATH_SEP);
			std::string substring = tuple[0].substr(0,pos);
			int pos2 = substring.find_last_of(PATH_SEP);
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			locally += std::string("flow_motion")+PATH_SEP;
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			locally = locally+tuple[0].substr(pos+1,tuple[0].size()-pos-5)+".bin";
			Auxiliary<float,2>::mat2bin(flow,locally.c_str(),false);
		}
		if(this->display_){
			cv::Mat color = cv::imread(tuple[0].c_str(),1);
			color.convertTo(color,CV_8UC3);
			cv::resize(color,color,split[0].size());
			std::string winname = (algo==MotionPatch::Farneback)?"Farneback":\
				((algo==MotionPatch::LucasKanade)?"LucasKanade":\
				((algo==MotionPatch::HornSchunck)?"HornSchunck":"Simple"));
			cv::Mat out = MotionPatch<T,F>::showOF(split[0],split[1],\
				color,5,true,winname);
			color.release();
			out.release();
		}
		this->features_->push_backVelXX(split[0]);
		this->features_->push_backVelYY(split[1]);
		flow.release(); split[0].release(); split[1].release();
	}
	return 0;
}
//==============================================================================
/** Gets the flow derivatives and merges them into a 4 channels image: xdx, xdy,
 * ydx, ydy.
 */
template <class T,class F>
cv::Mat MotionPatch<T,F>::getFlowDerivatives(const cv::Mat &flow){
	cv::Mat flowXX = cv::Mat::zeros(flow.size(), CV_32FC1);
	cv::Mat flowXY = cv::Mat::zeros(flow.size(), CV_32FC1);
	cv::Mat flowYX = cv::Mat::zeros(flow.size(), CV_32FC1);
	cv::Mat flowYY = cv::Mat::zeros(flow.size(), CV_32FC1);
	std::vector<cv::Mat> splits;
	cv::split(flow,splits);
	cv::Sobel(splits[0],flowXX,CV_32FC1, 0, 1, 1);
	cv::Sobel(splits[0],flowXY,CV_32FC1, 1, 0, 1);
	cv::Sobel(splits[1],flowYX,CV_32FC1, 0, 1, 1);
	cv::Sobel(splits[1],flowYY,CV_32FC1, 1, 0, 1);
	splits[0].release(); splits[1].release();
	cv::Mat derivatives;
	cv::Mat flows [] = {flowXX, flowXY, flowYX, flowYY};
	cv::merge(flows,4,derivatives);
	flowXX.release(); flowXY.release(); flowYX.release(); flowYY.release();
	return derivatives;
}
//==============================================================================
/** Load or extract the optical flow vector from two consecutive images.
 */
template <class T,class F>
void MotionPatch<T,F>::extractMotionAbsolute(const std::vector<std::string> &tuple,\
Algorithm algo,const std::string &featpath,unsigned offset,bool save){
	std::cout<<"[Motion::extractMotionAbsolute]: extracting OF \n"<<tuple[0]<<\
		"\n"<<tuple[1]<<std::endl;
	// [0] Read the images
	cv::Mat currImg = cv::imread(tuple[0].c_str(),1);
	assert(currImg.channels()==3);
	cv::Mat nextImg = cv::imread(tuple[1].c_str(),1);
	assert(nextImg.channels()==3);
	float scale = this->maximsize_/static_cast<float>(std::max(currImg.rows,currImg.cols));
	if(scale<1.0){
		cv::Size small = cv::Size(currImg.cols*scale,currImg.rows*scale);
		cv::resize(currImg,currImg,small);
		cv::resize(nextImg,nextImg,small);
	}
	clock_t begin = clock();
	// [3.1] curr(y,x) = next(y+flow(y,x)[1], x+flow(y,x)[0])
	cv::Mat flow  = MotionPatch<T,F>::justFlow(currImg,nextImg,algo,\
		std::min(this->motionW_,this->motionH_),this->maximsize_,tuple[0],\
		featpath,this->sintel_,this->storefeat_);
	currImg.release(); nextImg.release();
	// [4] Split the flow into 2 channels
	flow.convertTo(flow,CV_32FC2);
	std::vector<cv::Mat> split;
	cv::split(flow,split);
	assert(split.size()==2);
	// [5] Find threshold to ignore background noise
	cv::Mat valuesX; cv::multiply(split[0],split[0],valuesX);
	cv::Mat valuesY; cv::multiply(split[1],split[1],valuesY);
	cv::Mat values = valuesX+valuesY;
	valuesX.release(); valuesY.release();
	// [5.1] Now find the too big values and ignore them (errors at borders)
	cv::Mat mask;
	cv::inRange(values,0,2250,mask);
	mask.convertTo(mask,CV_32FC1);
	mask /= 255.0;
	cv::multiply(split[0],mask,split[0]);
	cv::multiply(split[1],mask,split[1]);
	mask.release();
	if(this->threshold_){
		// [5.2] Now find a good threshold and use it
		float minTr,maxTr;
		MotionPatch<T,F>::findThreshold(values,minTr,maxTr);
		cv::Mat trmask;
		cv::inRange(values,minTr,maxTr,trmask);
		trmask.convertTo(trmask,CV_32FC1);
		trmask /= 255.0;trmask -= 1.0;trmask *= -1.0;
		cv::multiply(split[0],trmask,split[0]);
		cv::multiply(split[1],trmask,split[1]);
		trmask.release();
	}
	values.release();
	// [6] Display the OF vectors if needed
	clock_t end = clock();
	std::cout<<"Of on 1 image time elapsed: "<<double(Auxiliary<uchar,1>::diffclock\
		(end,begin))<<" sec"<<std::endl;
	// [7] Push the flows into the feature matrix
	if(this->usederivatives_){
		cv::Mat splits [] = {split[0],split[1]};
		cv::merge(splits,2,flow);
		cv::Mat derivatives = MotionPatch<T,F>::getFlowDerivatives(flow);
		split[0].release(); split[1].release(); flow.release();
		// [1] Save if needed.
		if(save){
			std::string locally = featpath;
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			int pos = tuple[0].find_last_of(PATH_SEP);
			std::string substring = tuple[0].substr(0,pos);
			int pos2 = substring.find_last_of(PATH_SEP);
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			locally += std::string("deri_motion")+PATH_SEP;
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			locally = locally+tuple[0].substr(pos+1,tuple[0].size()-pos-5)+".bin";
			Auxiliary<float,4>::mat2bin(derivatives,locally.c_str(),false);
		}
		std::vector<cv::Mat> derivativeSplits;
		cv::split(derivatives,derivativeSplits);
		assert(derivativeSplits.size()==4);
		if(this->display_){
			cv::Mat color = cv::imread(tuple[0].c_str(),1);
			color.convertTo(color,CV_8UC3);
			cv::resize(color,color,derivativeSplits[0].size());
			std::string winname = (algo==MotionPatch::Farneback)?"Farneback":\
				(((algo==MotionPatch::LucasKanade)?"LucasKanade":\
				((algo==MotionPatch::HornSchunck)?"HornSchunck":"Simple")));
			cv::Mat out = MotionPatch<T,F>::showOFderi(derivativeSplits[0],\
				derivativeSplits[1],derivativeSplits[2],derivativeSplits[3],\
				color,5,true,winname);
			color.release();
			out.release();
		}
		this->features_->push_backVelXX(derivativeSplits[0]);
		derivativeSplits[0].release();
		this->features_->push_backVelXY(derivativeSplits[1]);
		derivativeSplits[1].release();
		this->features_->push_backVelYX(derivativeSplits[2]);
		derivativeSplits[2].release();
		this->features_->push_backVelYY(derivativeSplits[3]);
		derivativeSplits[3].release();
	}else{
		if(save){
			std::string locally = featpath;
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			int pos = tuple[0].find_last_of(PATH_SEP);
			std::string substring = tuple[0].substr(0,pos);
			int pos2 = substring.find_last_of(PATH_SEP);
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			locally += std::string("flow_motion")+PATH_SEP;
			Auxiliary<char,1>::fixPath(locally);
			Auxiliary<char,1>::file_exists(locally.c_str(),true);
			locally = locally+tuple[0].substr(pos+1,tuple[0].size()-pos-5)+".bin";
			Auxiliary<float,2>::mat2bin(flow,locally.c_str(),false);
		}
		if(this->display_){
			cv::Mat color = cv::imread(tuple[0].c_str(),1);
			color.convertTo(color,CV_8UC3);
			cv::resize(color,color,split[0].size());
			std::string winname = (algo==MotionPatch::Farneback)?"Farneback":\
				(((algo==MotionPatch::LucasKanade)?"LucasKanade":\
				((algo==MotionPatch::HornSchunck)?"HornSchunck":"Simple")));
			cv::Mat out = MotionPatch<T,F>::showOF(split[0],split[1],\
				color,5,true,winname);
			color.release();
			out.release();
		}
		this->features_->push_backVelXX(split[0]);
		this->features_->push_backVelYY(split[1]);
		flow.release(); split[0].release(); split[1].release();
	}
}
//==============================================================================
/** Find threshold by cutting the histogram at 0.90.
 */
template <class T,class F>
void MotionPatch<T,F>::findThreshold(const cv::Mat &values,float &minTr,float &maxTr){
	unsigned binsz  = 101;
	std::vector<float> histo(binsz,0.0);
	double mini,maxi;
	cv::minMaxLoc(values,&mini,&maxi);
	float maxbinVal = 0.0;
	float step      = (maxi-mini)/static_cast<float>(binsz);
	for(cv::Mat_<float>::const_iterator m=values.begin<float>();m!=\
	values.end<float>();++m){
		unsigned bin = std::floor((*m)/step);
		if(bin>=binsz){bin = binsz-1;}
		++histo[bin];
		if(histo[bin]>maxbinVal){
			maxbinVal = histo[bin];
		}
	}
	//[1] find the std to the maximum value
	float stdbin = 0.0;
	for(std::vector<float>::iterator b=histo.begin();b!=histo.end();++b){
		stdbin += ((*b)-maxbinVal)*((*b)-maxbinVal);
	}
	float devbin = std::sqrt(stdbin/static_cast<float>(histo.size()));
	minTr        = std::numeric_limits<float>::max();
	maxTr        = -std::numeric_limits<float>::max();
	for(std::vector<float>::iterator b=histo.begin();b!=histo.end();++b){
		if((*b)>(maxbinVal-devbin/3)){
			float minbound = (mini+step*static_cast<float>(b-(histo.begin())));
			float maxbound = (mini+step*static_cast<float>(b-(histo.begin()))+\
				step);
			if(minTr>=minbound){
				minTr = minbound;
			}
			if(maxTr<=maxbound){
				maxTr = maxbound;
			}
		}
	}
}
//==============================================================================
/** Warp the image with the flow in 10 steps.
 */
template <class T,class F>
void MotionPatch<T,F>::warpInter(const cv::Mat &motionX,const cv::Mat &motionY,\
const cv::Mat &origim,unsigned offset,bool display){
	cv::Mat bigim,bigX,bigY;
	cv::resize(origim,bigim,cv::Size(origim.cols*2,origim.rows*2),cv::INTER_CUBIC);
	cv::resize(motionX,bigX,cv::Size(motionX.cols*2,motionX.rows*2),cv::INTER_CUBIC);
	cv::resize(motionY,bigY,cv::Size(motionY.cols*2,motionY.rows*2),cv::INTER_CUBIC);
	unsigned index = 1;
	for(float i=0.01;i<=0.9;i+=0.03){
		cv::Mat inter; bigim.copyTo(inter);
		for(int r=0;r<bigim.rows;++r){
			for(int c=0;c<bigim.cols;++c){
				float newx     = static_cast<float>(c)-bigX.at<float>(r,c)*i;
				float newy     = static_cast<float>(r)-bigY.at<float>(r,c)*i;		
				cv::Vec3b val1 = bigim.at<cv::Vec3b>(static_cast<int>(newy),\
					static_cast<int>(newx));
				inter.at<cv::Vec3b>(r,c) = val1;
			}
		}
		cv::Mat intersmall;
		cv::resize(inter,intersmall,cv::Size(inter.cols/2,inter.rows/2),cv::INTER_CUBIC);
		if(display){
			Auxiliary<uchar,3>::display(intersmall);
		}
		char videoname[100];
		sprintf(videoname,"video%06d",offset);
		char imname[100];
		sprintf(imname,"im%06d",index);
		std::string filename = "inter_"+std::string(videoname)+std::string(imname)+".jpg";
		while(Auxiliary<char,1>::file_exists(filename.c_str(),false)){
			++offset;
			sprintf(videoname,"video%06d",offset);
			filename = "inter_"+std::string(videoname)+std::string(imname)+".jpg";
		}
		cv::imwrite(filename,intersmall);
		inter.release();intersmall.release();
		++index;
	}
	bigim.release();bigX.release();bigY.release();
}
//==============================================================================
/** Showing OF vectors (for check only).
 */
template <class T,class F>
cv::Mat MotionPatch<T,F>::showOF(const cv::Mat &velX,const cv::Mat &velY,const \
cv::Mat &image,unsigned step,bool display,const std::string &winname,\
const cv::Rect &roi){
	if(image.channels()!=3){
		std::cerr<<"[MotionPatch<T,F>::showOF] image should be colorful."<<std::endl;
	}
	cv::Mat toDisplay;
	if(!roi.width || !roi.height){
		assert(velX.size()==velY.size());
		assert(velX.size()==image.size());
		toDisplay = image.clone();
	}else{
		toDisplay = image(roi).clone();
		cv::resize(toDisplay,toDisplay,image.size());
	}
	toDisplay.convertTo(toDisplay,CV_8UC3);
	for(unsigned y=0;y<image.rows;y+=step){
		for(unsigned x=0;x<image.cols;x+=step){
			if(!velY.at<float>(y,x) && !velX.at<float>(y,x)){continue;}
			// [0] Point in next image image (end-point)
			// curr_pt = next_pt+flow_pt <=> next_pt = curr_pt-flow_pt
			cv::circle(toDisplay,cv::Point(x+velX.at<float>(y,x),\
				y+velY.at<float>(y,x)),1.0,CV_RGB(50,50,50),-1,1,0);
			// [1] Line between end point and start point
			float angle = std::atan2(velY.at<float>(y,x),velX.at<float>(y,x))+M_PI;
			cv::Scalar color;
			if(angle<=M_PI/2.0){
				color = CV_RGB(0,255,0);
			}else if(angle<=M_PI){
				color = CV_RGB(255,0,0);
			}else if(angle<=3.0*M_PI/2.0){
				color = CV_RGB(0,0,255);
			}else{
				color = CV_RGB(255,255,0);
			}
			cv::line(toDisplay,cv::Point(x,y),cv::Point(x+velX.at<float>(y,x),\
				y+velY.at<float>(y,x)),color,1,CV_AA,0);
			// [2] Point in current image (start-point)
			cv::Scalar darker(color.val[0]+20,color.val[1]+20,color.val[2]+20);
			cv::circle(toDisplay,cv::Point(x,y),1,darker,1,1,0);
		}
	}
	if(display){
		cv::imshow(winname,toDisplay);
		cv::waitKey(10);
	}
	return toDisplay;
}
//==============================================================================
/** Showing OF derivatives back as vectors in the image (for check only).
 */
template <class T,class F>
cv::Mat MotionPatch<T,F>::showOFderi(const cv::Mat &velXX,const cv::Mat &velXY,\
const cv::Mat &velYX,const cv::Mat &velYY,const cv::Mat &image,unsigned step,\
bool display,const std::string &winname,const cv::Rect &roi){
	cv::Rect newroi = roi;
	if(!roi.width || !roi.height){
		assert(velXX.size()==velXY.size());
		assert(velXX.size()==velYX.size());
		assert(velYX.size()==velYY.size());
		assert(velXX.size()==image.size());
		newroi = cv::Rect(0,0,image.cols,image.rows);
	}
	cv::Mat tmpVelXX = velXX(newroi),
			tmpVelXY = velXY(newroi),
			tmpVelYX = velYX(newroi),
			tmpVelYY = velYY(newroi);
	cv::Mat magniX   = cv::Mat::zeros(tmpVelXX.size(),CV_32FC1),\
			magniY   = cv::Mat::zeros(tmpVelXX.size(),CV_32FC1);
	cv::Mat_<float>::iterator xy = tmpVelXY.begin<float>();
	cv::Mat_<float>::iterator yx = tmpVelYX.begin<float>();
	cv::Mat_<float>::iterator yy = tmpVelYY.begin<float>();
	cv::Mat_<float>::iterator mx = magniX.begin<float>();
	cv::Mat_<float>::iterator my = magniY.begin<float>();
	for(cv::Mat_<float>::iterator xx=tmpVelXX.begin<float>();xx!=tmpVelXX.end<float>(),\
	xy!=tmpVelXY.end<float>(),yx!=tmpVelYX.end<float>(),yy!=tmpVelYY.end<float>(),\
	mx!=magniX.end<float>(),my!=magniY.end<float>();++xx,++xy,++yx,++yy,++mx,++my){
		(*mx) = std::sqrt((*xx)*(*xx)+(*xy)*(*xy));
		(*my) = std::sqrt((*yx)*(*yx)+(*yy)*(*yy));
	}
	cv::Mat magnitudes = magniX+magniY;
	magniX.release(); magniY.release();
	double minVal, maxVal;
	cv::minMaxLoc(magnitudes,&minVal,&maxVal);
	cv::Mat draw;
	magnitudes.convertTo(draw,CV_8UC3,255.0/(maxVal-minVal),-minVal*255.0/(maxVal-minVal));
	magnitudes.release();
	// [3] Resize it to a visible size
	if(display){
		cv::imshow("image",image);
		cv::waitKey(10);
		cv::imshow(winname,draw);
		cv::waitKey(10);
	}
	return draw;
}
//==============================================================================
/** Saves the labels and the image features --- for each image make one file.
 */
template <class T,class F>
void MotionPatch<T,F>::savePatches(const std::string &path2feat,unsigned pos){
	std::ofstream pFile;
	try{
		pFile.open(path2feat.c_str(),std::ios::out|std::ios::binary);
	}catch(std::exception &e){
		std::cerr<<"[MotionPatch<T>::savePatches]: Cannot open file: %s"<<\
			e.what()<<std::endl;
		std::exit(-1);
	}
	pFile.precision(std::numeric_limits<double>::digits10);
	pFile.precision(std::numeric_limits<float>::digits10);
	// [0] Write down the features
	std::vector<IplImage*> tmpImg = this->features_->vImg(pos);
	unsigned vsize = static_cast<unsigned>(tmpImg.size());
	pFile.write(reinterpret_cast<char*>(&vsize),sizeof(unsigned));
	if(vsize>0){
		// [0.1] Write each channel
		for(std::vector<IplImage*>::const_iterator it=tmpImg.begin();\
		it!=tmpImg.end();++it){
			pFile.write(reinterpret_cast<char*>(&tmpImg[0]->nChannels),sizeof(int));
			pFile.write(reinterpret_cast<char*>(&tmpImg[0]->depth),sizeof(int));
			pFile.write(reinterpret_cast<char*>(&tmpImg[0]->width),sizeof(int));
			pFile.write(reinterpret_cast<char*>(&tmpImg[0]->height),sizeof(int));
			for(int y=0;y<(*it)->height;++y){
				for(int x=0;x<(*it)->width;++x){
					uchar val = (*it)->imageData[y*((*it)->width)+x];
					pFile.write(reinterpret_cast<char*>(&val),sizeof(uchar));
				}
			}
		}
	}
	// [1] Write the interest point at which to extract the features
	unsigned nopoints = this->patches_[0].size();
	pFile.write(reinterpret_cast<char*>(&nopoints),sizeof(unsigned));
	for(typename std::vector<const T*>::const_iterator pa=this->patches_[0].begin();\
	pa!=this->patches_[0].end();++pa){
		unsigned ptsize = (*pa)->featH();
		cv::Point point = (*pa)->point();
		pFile.write(reinterpret_cast<char*>(&ptsize),sizeof(unsigned));
		pFile.write(reinterpret_cast<char*>(&point.x),sizeof(unsigned));
		pFile.write(reinterpret_cast<char*>(&point.y),sizeof(unsigned));
	}
	// [2] Write down the x-velocity & y-velocity
	if(this->usederivatives_){
		cv::Mat tmpVelXX = this->features_->velXX(pos),\
				tmpVelXY = this->features_->velXY(pos),\
				tmpVelYX = this->features_->velYX(pos),\
				tmpVelYY = this->features_->velYY(pos);
		int rows         = tmpVelXX.rows;
		int cols         = tmpVelXX.cols;
		pFile.write(reinterpret_cast<char*>(&rows),sizeof(int));
		pFile.write(reinterpret_cast<char*>(&cols),sizeof(int));
		// [1.1] Loop over the pixels and write them
		for(int y=0;y<rows;++y){
			for(int x=0;x<cols;++x){
				float valXX = tmpVelXX.at<float>(y,x);
				pFile.write(reinterpret_cast<char*>(&valXX),sizeof(float));
				float valXY = tmpVelXY.at<float>(y,x);
				pFile.write(reinterpret_cast<char*>(&valXY),sizeof(float));
				float valYX = tmpVelYX.at<float>(y,x);
				pFile.write(reinterpret_cast<char*>(&valYX),sizeof(float));
				float valYY = tmpVelYY.at<float>(y,x);
				pFile.write(reinterpret_cast<char*>(&valYY),sizeof(float));
			}
		}
	}else{
		cv::Mat tmpVelX = this->features_->velXX(pos),\
				tmpVelY = this->features_->velYY(pos);
		int rows        = tmpVelX.rows;
		int cols        = tmpVelX.cols;
		pFile.write(reinterpret_cast<char*>(&rows),sizeof(int));
		pFile.write(reinterpret_cast<char*>(&cols),sizeof(int));
		// [1.1] Loop over the pixels and write them
		for(int y=0;y<rows;++y){
			for(int x=0;x<cols;++x){
				float valX = tmpVelX.at<float>(y,x);
				pFile.write(reinterpret_cast<char*>(&valX),sizeof(float));
				float valY = tmpVelY.at<float>(y,x);
				pFile.write(reinterpret_cast<char*>(&valY),sizeof(float));
			}
		}
	}
	pFile.close();
}
//==============================================================================
/** Loads the labels and the image features --- 1 file per image.
 */
template <class T,class F>
void MotionPatch<T,F>::loadPatches(const std::string &path2feat,bool showWhere){
	if(!Auxiliary<uchar,1>::file_exists(path2feat.c_str())){
		std::cerr<<"[LabelPatchFeature::loadPatches]: Error opening the file: "<<\
			path2feat<<std::endl;
		std::exception e;
		throw(e);
	}
	std::ifstream pFile;
	pFile.open(path2feat.c_str(),std::ios::in | std::ios::binary);
	if(pFile.is_open()){
		pFile.seekg (0,ios::beg);
		// [0] Read the features matrices (one per channel)
		unsigned vsize;
		pFile.read(reinterpret_cast<char*>(&vsize),sizeof(unsigned));
		std::vector<IplImage*> img;
		if(vsize>0){
			for(unsigned s=0;s<vsize;++s){
				int width,height,nChannels,depth;
				pFile.read(reinterpret_cast<char*>(&nChannels),sizeof(int));
				pFile.read(reinterpret_cast<char*>(&depth),sizeof(int));
				pFile.read(reinterpret_cast<char*>(&width),sizeof(int));
				pFile.read(reinterpret_cast<char*>(&height),sizeof(int));
				IplImage* im = cvCreateImage(cvSize(width,height),depth,nChannels);
				for(int y=0;y<height;++y){
					for(int x=0;x<width;++x){
						uchar val;
						pFile.read(reinterpret_cast<char*>(&val),sizeof(uchar));
						im->imageData[y*width+x] = val;
					}
				}
				img.push_back(cvCloneImage(im));
			}
		}
		this->features_->push_backImg(img);
		// [1] Add the patches (we have no classes now)
		if(this->patches_.empty()){
			this->patches_.resize(1);
		}
		// [1] Write the interest point at which to extract the features
		cv::Mat where;
		if(showWhere){
			where =	cv::Mat::zeros(cv::Size(img[0]->width,img[0]->height),CV_8UC1);
		}
		unsigned nopoints;
		pFile.read(reinterpret_cast<char*>(&nopoints),sizeof(unsigned));
		for(unsigned pt=0;pt<nopoints;++pt){
			unsigned ptsize, pointX, pointY;
			pFile.read(reinterpret_cast<char*>(&ptsize),sizeof(unsigned));
			pFile.read(reinterpret_cast<char*>(&pointX),sizeof(unsigned));
			pFile.read(reinterpret_cast<char*>(&pointY),sizeof(unsigned));
			T* patch = new T(this->featW_,this->featH_,this->labW_,this->labH_,\
				this->features_->size()-1,cv::Point(pointX,pointY),\
				this->motionW_,this->motionH_);
			// [5.1] We have only 4 classes here
			this->patches_[0].push_back(patch);
			if(showWhere){
				where.at<uchar>(pointX,pointY) = 255;
			}

		}
		// [6] If we want to see where the patches are
		if(showWhere){
			cv::imshow("where",where);
			cv::waitKey(10);
		}
		where.release();
		// [2] Read the OF matrices
		if(this->usederivatives_){
			int cols, rows;
			pFile.read(reinterpret_cast<char*>(&rows),sizeof(int));
			pFile.read(reinterpret_cast<char*>(&cols),sizeof(int));
			cv::Mat veloXX = cv::Mat(cv::Size(cols,rows),CV_32FC1);
			cv::Mat veloXY = cv::Mat(cv::Size(cols,rows),CV_32FC1);
			cv::Mat veloYX = cv::Mat(cv::Size(cols,rows),CV_32FC1);
			cv::Mat veloYY = cv::Mat(cv::Size(cols,rows),CV_32FC1);
			// [3] If we want to see where are the patches
			unsigned limitW = std::max(std::max(this->labW_,this->featW_),this->motionW_);
			unsigned limitH = std::max(std::max(this->labH_,this->featH_),this->motionH_);
			// [4] Loop over the cols and rows
			for(int y=0;y<rows;++y){
				for(int x=0;x<cols;++x){
					pFile.read(reinterpret_cast<char*>(&(veloXX.at<float>(y,x))),\
						sizeof(float));
					pFile.read(reinterpret_cast<char*>(&(veloXY.at<float>(y,x))),\
						sizeof(float));
					pFile.read(reinterpret_cast<char*>(&(veloYX.at<float>(y,x))),\
						sizeof(float));
					pFile.read(reinterpret_cast<char*>(&(veloYY.at<float>(y,x))),\
						sizeof(float));
				}
			}
			this->features_->push_backVelXX(veloXX);
			this->features_->push_backVelXY(veloXY);
			this->features_->push_backVelYX(veloYX);
			this->features_->push_backVelYY(veloYY);
			veloXX.release(); veloXY.release(); veloYX.release(); veloYY.release();
		}else{
			int cols, rows;
			pFile.read(reinterpret_cast<char*>(&rows),sizeof(int));
			pFile.read(reinterpret_cast<char*>(&cols),sizeof(int));
			cv::Mat veloX = cv::Mat(cv::Size(cols,rows),CV_32FC1);
			cv::Mat veloY = cv::Mat(cv::Size(cols,rows),CV_32FC1);
			// [3] If we want to see where are the patches
			unsigned limitW = std::max(std::max(this->labW_,this->featW_),this->motionW_);
			unsigned limitH = std::max(std::max(this->labH_,this->featH_),this->motionH_);
			// [4] Loop over the cols and rows
			for(int y=0;y<rows;++y){
				for(int x=0;x<cols;++x){
					pFile.read(reinterpret_cast<char*>(&(veloX.at<float>(y,x))),\
						sizeof(float));
					pFile.read(reinterpret_cast<char*>(&(veloY.at<float>(y,x))),\
						sizeof(float));
				}
			}
			this->features_->push_backVelXX(veloX);
			this->features_->push_backVelYY(veloY);
			veloX.release(); veloY.release();
		}
		pFile.close();
	}
}
//==============================================================================
/** Just reads the Sintel flow for local files.
 */
template <class T,class F>
cv::Mat MotionPatch<T,F>::sintelFlow(const std::string &featpath,unsigned maxisize,\
const std::string &imName,bool store){
	CFloatImage floimg;
	std::string justdir  = Auxiliary<char,1>::getStringSplit(imName,PATH_SEP,-2);
	std::string justname = Auxiliary<char,1>::getStringSplit(imName,PATH_SEP,-1);
	std::string filename = featpath+justdir+PATH_SEP+justname.substr\
		(0,justname.size()-4)+".flo";
	if(!Auxiliary<char,1>::file_exists(filename.c_str(),false)){
		std::cerr<<"[MotionPatch::sintelFlow] 404: "<<filename<<std::endl;
		std::exit(-1);
	}
	ReadFlowFile(floimg,filename.c_str());
	CShape sh = floimg.Shape();
	int width = sh.width, height = sh.height, nBands = sh.nBands;



	assert(nBands==2);
	cv::Mat cvflow = cv::Mat::zeros(cv::Size(width,height),CV_32FC2);
	for(unsigned row=0;row<height;++row){
		for(unsigned col=0;col<width;++col){
			cv::Vec2f value;
			value.val[0] = floimg.Pixel(col,row,0);
			value.val[1] = floimg.Pixel(col,row,1);
			cvflow.at<cv::Vec2f>(row,col) = value;
		}
	}
	float scale = maxisize/static_cast<float>(std::max(height,width));
	if(scale<1.0){
		cv::Size small = cv::Size(width*scale,height*scale);
		cv::resize(cvflow,cvflow,small);
	}
	if(store){
		std::string outfilename = featpath+justdir+PATH_SEP+justname.substr\
			(0,justname.size()-4)+".bin";
		Auxiliary<float,2>::mat2bin(cvflow,outfilename.c_str());
	}
	return cvflow;
}
//==============================================================================
//==============================================================================
template class MotionPatch<MotionPatchFeature<FeaturesMotion>,FeaturesMotion>;





