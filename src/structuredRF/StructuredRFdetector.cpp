/* StructuredRFdetector.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef STRUCTUREDRFDETECTOR_CPP_
#define STRUCTUREDRFDETECTOR_CPP_
#include "StructuredRFdetector.h"
#include <map>
#include <limits.h>
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
StructuredRFdetector<L,M,T,F,N,U>::StructuredRFdetector(StructuredRF<L,M,T,F,N,U>* pRF,\
int w,int h,unsigned cls,unsigned labW,unsigned labH,Puzzle<PuzzlePatch>::METHOD \
method,unsigned step):noCls_(cls),labW_(labW),labH_(labH),method_(method),\
width_(w),height_(h),step_(step){
	this->forest_    = pRF;
	this->maxsize_   = 300;
	this->storefeat_ = false;
}
//==============================================================================
/** Loads the test features from file for the current test image.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
std::vector<std::vector<IplImage*> > StructuredRFdetector<L,M,T,F,N,U>::loadFeatures\
(const std::vector<std::string> &path2feat,std::vector<std::vector<const T*> > &patches,\
bool showWhere) const{
	std::cout<<"[StructuredRFdetector::loadFeatures] loading test features"<<std::endl;
	std::vector<std::vector<IplImage*> > vImg;
	// [0] One vector of patches per scale
	if(patches.empty()){
		patches.resize(path2feat.size());
	}
	// [1] Loop over the pyramid scales
	for(std::vector<std::string>::const_iterator it=path2feat.begin();it!=\
	path2feat.end();++it){
		unsigned pos = it-path2feat.begin();
		if(!Auxiliary<uchar,1>::file_exists((*it).c_str())){
			std::cerr<<"[StructuredRFdetector::loadFeatures]: Error opening the file: "<<\
				(*it)<<std::endl;
			std::exception e;
			throw(e);
		}
		std::ifstream pFile;
		pFile.open((*it).c_str(),std::ios::in | std::ios::binary);
		// [2] Read the features matrix
		if(pFile.is_open()){
			pFile.seekg (0,ios::beg);
			unsigned vsize;
			pFile.read(reinterpret_cast<char*>(&vsize),sizeof(unsigned));
			std::vector<IplImage*> tmp;
			if(vsize>0){
				// [2.1] Loop over channels in feature matrix
				for(unsigned s=0;s<vsize;++s){
					int width,height,nChannels,depth;
					pFile.read(reinterpret_cast<char*>(&nChannels),sizeof(int));
					pFile.read(reinterpret_cast<char*>(&depth),sizeof(int));
					pFile.read(reinterpret_cast<char*>(&width),sizeof(int));
					pFile.read(reinterpret_cast<char*>(&height),sizeof(int));
					IplImage* im = cvCreateImage(cvSize(width,height),depth,nChannels);
					// [2.2] If we want to see where the patches are
					cv::Mat where;
					if(showWhere && s==0){
						where = cv::Mat::zeros(cv::Size(im->width,im->height),CV_8UC1);
					}
					for(int y=0;y<height;++y){
						for(int x=0;x<width;++x){
							uchar val;
							pFile.read(reinterpret_cast<char*>(&val),sizeof(uchar));
							im->imageData[y*width+x] = val;
							// [2.3] Read the patches only once
							if(s==0 && (x<width-(this->width_/2)) && \
							x>=(this->width_/2) && (y<height-(this->height_/2)) && \
							y>=(this->height_/2) && ((x-(this->width_/2))%\
							(this->step_))==0 && ((y-(this->height_/2))%(this->step_))==0){
								T* patch = new T(this->width_,this->height_,\
									this->labW_,this->labH_,pos,cv::Point(x,y));
								patches[pos].push_back(patch);
								if(showWhere){
									where.at<uchar>(y,x) = 255;
								}
							}
						}
					}
					// [3] Show where the patches are centered
					if(showWhere && s==0){
						cv::imshow("where",where);
						cv::waitKey(0);
					}
					where.release();
					tmp.push_back(cvCloneImage(im));
				}
			}
			vImg.push_back(tmp);
			pFile.close();
		}
	}
	return vImg;
}
//==============================================================================
/** Loads the test features from file for the current test image.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRFdetector<L,M,T,F,N,U>::saveFeatures(const std::vector<std::string> \
&path2feat,const std::vector<std::vector<IplImage*> > &vImg,const std::vector\
<std::vector<const T*> > &patches) const {
	std::cout<<"[StructuredRFdetector::saveFeatures] saving test features"<<std::endl;
	// [0] Loop over pyramid scales
	for(std::vector<std::string>::const_iterator it=path2feat.begin();it!=\
	path2feat.end();++it){
		std::ofstream pFile;
		try{
			pFile.open((*it).c_str(),std::ios::out|std::ios::binary);
		}catch(std::exception &e){
			std::cerr<<"[StructuredRFdetector::saveFeatures]: Cannot open file: %s"<<\
				e.what()<<std::endl;
			std::exit(-1);
		}
		pFile.precision(std::numeric_limits<double>::digits10);
		pFile.precision(std::numeric_limits<float>::digits10);
		// [1] Write down the features
		unsigned pos   = it-path2feat.begin();
		unsigned vsize = static_cast<unsigned>(vImg[pos].size());
		pFile.write(reinterpret_cast<char*>(&vsize),sizeof(unsigned));
		if(vsize>0){
			// [1.1] Loop over feature channels
			for(std::vector<IplImage*>::const_iterator v=vImg[pos].begin();\
			v!=vImg[pos].end();++v){
				pFile.write(reinterpret_cast<char*>(&vImg[pos][0]->nChannels),sizeof(int));
				pFile.write(reinterpret_cast<char*>(&vImg[pos][0]->depth),sizeof(int));
				pFile.write(reinterpret_cast<char*>(&vImg[pos][0]->width),sizeof(int));
				pFile.write(reinterpret_cast<char*>(&vImg[pos][0]->height),sizeof(int));
				for(int y=0;y<(*v)->height;++y){
					for(int x=0;x<(*v)->width;++x){
						uchar val = (*v)->imageData[y*((*v)->width)+x];
						pFile.write(reinterpret_cast<char*>(&val),sizeof(uchar));
					}
				}
			}
		}
		pFile.close();
	}
}
//==============================================================================
/** Extracts or loads the test features for the current test image.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
std::vector<std::vector<IplImage*> > StructuredRFdetector<L,M,T,F,N,U>::getFeatures\
(const std::string &path2img,const std::vector<std::string> &path2feat,\
const std::vector<float> &pyr,std::vector<std::vector<const T*> > &patches,\
cv::Size &origsize,bool showWhere) const{
	std::cout<<"[StructuredRFdetector::getFeatures] extracting/loading test "<<\
		"features "<<path2feat<<std::endl;
	std::vector<std::vector<IplImage*> > vImg;
	// [0] Try to load the feature matrices
	try{
		vImg            = this->loadFeatures(path2feat,patches,showWhere);
		unsigned cls    = 0;
		while(vImg[cls].empty()){++cls;}
		origsize.height = vImg[cls][0]->height-(this->height_);
		origsize.width  = vImg[cls][0]->width-(this->width_);
	// [1] If the features are not there, create them
	}catch(std::exception &e){
		// [4] Also add one vector of patches per scale
		if(patches.empty()){
			patches.resize(pyr.size());
		}
		// [5] Loop over all scales
		for(std::size_t i=0;i<pyr.size();++i){
			// [5.1] Resize to desired image scale
			IplImage* img = NULL;
			if(pyr[i]==1){
				img             = cvLoadImage(path2img.c_str(),-1);
				origsize.height = img->height;
				origsize.width  = img->width;
			}else{
				IplImage* tmpimg = cvLoadImage(path2img.c_str(),-1);
				origsize.height  = tmpimg->height;
				origsize.width   = tmpimg->width;
				CvSize imSize    = cvSize(static_cast<int>(tmpimg->width*pyr[i]),\
					static_cast<int>(tmpimg->height*pyr[i]));
				img              = cvCreateImage(imSize,IPL_DEPTH_8U,tmpimg->nChannels);
				cvResize(tmpimg,img,CV_INTER_LINEAR);
				cvReleaseImage(&tmpimg);
			}
			if(!img){
				std::cerr<<"[StructuredRFdetector::getFeatures]: Could not load "<<\
					"image file: "<<path2img<<std::endl;
				std::exit(-1);
			}
			// [3] Make border (if the patch size changes better re-extract features)
			CvSize borderSize = cvSize(static_cast<int>(img->width+this->width_),\
				static_cast<int>(img->height+this->height_));
			IplImage* cLevel  = cvCreateImage(borderSize,IPL_DEPTH_8U,img->nChannels);
			cvCopyMakeBorder(img,cLevel,cvPoint(this->width_/2,this->height_/2),\
				IPL_BORDER_REPLICATE);
			cvReleaseImage(&img);
			// [5.2] Extract HoG-like features:
			std::vector<IplImage*> tmpFeat;
			CRPatch<PatchFeature>::extractFeatureChannels32(cLevel,tmpFeat);
			vImg.push_back(tmpFeat);
			// [5.3] write down the patches also
			cv::Mat where;
			if(showWhere){
				where = cv::Mat::zeros(cv::Size(cLevel->width,cLevel->height),CV_8UC1);
			}
			// [5.4] We get a vector of patches for each scale
			for(unsigned y=(this->height_/2);y<(cLevel->height-(this->height_)/2);\
			y+=this->step_){
				for(unsigned x=(this->width_/2);x<(cLevel->width-(this->width_)/2);\
				x+=this->step_){
					T* patch  = new T(this->width_,this->height_,this->labW_,\
						this->labH_,i,cv::Point(x,y));
					patches[i].push_back(patch);
					if(showWhere){
						where.at<uchar>(y,x) = 255;
					}
				}
			}
			cvReleaseImage(&cLevel);
			// [5.5] If we want to see where the patches are
			if(showWhere){
				cv::imshow("where",where);
				cv::waitKey(0);
			}
			where.release();
		}
		// [6] Store everything in a feature matrix for each image
		if(this->storefeat_){
			this->saveFeatures(path2feat,vImg,patches);
		}
	}
	return vImg;
}
//==============================================================================
/** Gets an input image and returns a detection image (pixel labels by RF regression).
 * Given a set of predicted leafs for current pixel, get the final label:
 * Simple: [1] Just get the most voted pixel label per position.
 * Puzzle: [2] Optimized the patch selection label \cite{kontschider}.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRFdetector<L,M,T,F,N,U>::detectColor(const F *features,cv::Mat \
&imgDetect,const std::vector<const T*> &patches,const cv::Size &imsize) const {
	std::vector<std::vector<PuzzlePatch> > puzzleLabels;
	std::cout<<"[StructuredRFdetector::detectColor] predicting on 1 image"<<std::endl;
	clock_t begin = clock();
	// [0] Loop over each image position
	for(typename std::vector<const T*>::const_iterator p=patches.begin();p!=\
	patches.end();++p){
		// [1] Access patch at positions saved in tests (go through tree, return leaf)
		std::vector<const U*> result; // over all trees
		this->forest_->regression(result,(*p),features);
		std::vector<PuzzlePatch> tmpPuzzle;
		// [2] Loop over all trees at this position
		for(typename std::vector<const U*>::const_iterator it=result.begin();\
		it!=result.end();++it){
			tmpPuzzle.push_back(PuzzlePatch((*p)->point(),(*it)->vLabels(),\
				(*it)->labelProb()));
			delete (*it);
		}
		puzzleLabels.push_back(tmpPuzzle);
	}
	clock_t end = clock();
	std::cout<<"Regression 1 img time elapsed: "<<double(Auxiliary<uchar,1>::diffclock\
		(end,begin))<<" sec"<<std::endl;
	// [3] For each tree get class frequencies and class variances
	std::vector<std::vector<float> > clsFreq = this->forest_->treeClsFreq();
	std::cout<<"[StructuredRFdetector::detectColor] optimizing final labels"<<std::endl;
	// [4] For each point in the image we have a prediction-patch => optimal labels
	cv::Mat result = Puzzle<PuzzlePatch>::solve(puzzleLabels,imsize,this->labW_,\
		this->labH_,this->noCls_,clsFreq,this->method_);
	unsigned limitW = std::max(this->labW_,static_cast<unsigned>(this->width_));
	unsigned limitH = std::max(this->labH_,static_cast<unsigned>(this->height_));
	cv::Rect imRoi(limitW/2,limitH/2,imsize.width-limitW,imsize.height-limitH);
	imgDetect = result(imRoi).clone();
	result.release();
}
//==============================================================================
/** Scales the image at a number of sizes and it labels each scale [?].
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRFdetector<L,M,T,F,N,U>::detectPyramid(const std::string &imname,\
const std::string &path2img,const std::string &path2feat,const std::string &ext,\
const std::vector<float> &pyramid,std::vector<cv::Mat> &vImgDetect) const {
	std::vector<std::string> featPath;
	for(std::vector<float>::const_iterator it=pyramid.begin();it!=pyramid.end();++it){
		featPath.push_back(path2feat+Auxiliary<int,1>::number2string(*it)+"_"+imname+".bin");
	}
	// [0] Get the features and patches for the current image
	std::vector<std::vector<const T*> > patches;
	cv::Size origsize;
	std::vector<std::vector<IplImage*> > vImg = this->getFeatures((path2img+\
		imname+ext),featPath,pyramid,patches,origsize);
	// [1] Loop over all scales
	for(std::vector<std::string>::iterator it=featPath.begin();it!=\
	featPath.end();++it){
		unsigned pos    = it-featPath.begin();
		// [1.1] Get the output image size
		cv::Size featsize((vImg[pos][0]->width)*pyramid[pos],\
			(vImg[pos][0]->height)*pyramid[pos]);
		vImgDetect[pos] = cv::Mat::zeros(featsize,CV_8UC1);
		F* features     = new F();
		features->vImg(vImg);
		// [1.2] Get the prediction image at this scale
		this->detectColor(features,vImgDetect[pos],patches[pos],featsize);
		delete features;
		// [1.3] Release patches at this size
		for(unsigned p=0;p<patches[pos].size();++p){
			delete patches[pos][p];
		}
		patches[pos].clear();
	}
	patches.clear();
	// [2] Release the features
	for(std::vector<std::vector<IplImage*> >::iterator it=vImg.begin();it!=\
		vImg.end();++it){
		for(unsigned int c=0;c<(*it).size();++c){
			cvReleaseImage(&((*it)[c]));
		}
		it->clear();
	}
	vImg.clear();
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif // STRUCTUREDRFDETECTOR_CPP_









































