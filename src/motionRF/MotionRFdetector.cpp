/* MotionRFdetector.cpp
 * Author: Silvia-Laura Pintea
 */
#include "MotionRFdetector.h"
#include "MotionPuzzle.h"
#include "SIFTlike.h"
//==============================================================================
//==============================================================================
//==============================================================================
/** Loads the test features from file for the current test image.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
std::vector<std::vector<IplImage*> > MotionRFdetector<L,M,T,F,N,U>::loadFeatures\
(const std::vector<std::string> &path2feat,std::vector<std::vector<const T*> > &patches,\
bool showWhere) const{
	std::cout<<"[MotionRFdetector::loadFeatures] loading test features"<<std::endl;
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
			std::cerr<<"[MotionRFdetector::loadFeatures]: Error opening the file: "<<\
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
					for(int y=0;y<height;++y){
						for(int x=0;x<width;++x){
							uchar val;
							pFile.read(reinterpret_cast<char*>(&val),sizeof(uchar));
							im->imageData[y*width+x] = val;
						}
					}
					tmp.push_back(cvCloneImage(im));
				}
			}
			vImg.push_back(tmp);
			// [2] Read the points
			cv::Mat where;
			if(showWhere){
				where = cv::Mat::zeros(cv::Size(tmp[0]->width,tmp[0]->height),CV_8UC1);
			}
			unsigned nopoints;
			pFile.read(reinterpret_cast<char*>(&nopoints),sizeof(unsigned));
			for(unsigned pt=0;pt<nopoints;++pt){
				unsigned ptsize,ptx,pty;
				pFile.read(reinterpret_cast<char*>(&ptsize),sizeof(unsigned));
				pFile.read(reinterpret_cast<char*>(&ptx),sizeof(unsigned));
				pFile.read(reinterpret_cast<char*>(&pty),sizeof(unsigned));
				T* patch = new T(this->width_,this->height_,this->labW_,this->labH_,\
					pos,cv::Point(ptx,pty));
				patches[pos].push_back(patch);
				if(showWhere){
					where.at<uchar>(pty,ptx) = 255;
				}
			}
			// [3] Show where the patches are centered
			if(showWhere){
				cv::imshow("where",where);
				cv::waitKey(0);
			}
			where.release();
			pFile.close();
		}
	}
	return vImg;
}
//==============================================================================
/** Loads the test features from file for the current test image.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void MotionRFdetector<L,M,T,F,N,U>::saveFeatures(const std::vector<std::string> \
&path2feat,const std::vector<std::vector<IplImage*> > &vImg,const std::vector\
<std::vector<const T*> > &patches) const {
	std::cout<<"[MotionRFdetector::saveFeatures] saving test features"<<std::endl;
	// [0] Loop over pyramid scales
	for(std::vector<std::string>::const_iterator it=path2feat.begin();it!=\
	path2feat.end();++it){
		std::ofstream pFile;
		try{
			pFile.open((*it).c_str(),std::ios::out|std::ios::binary);
		}catch(std::exception &e){
			std::cerr<<"[MotionRFdetector::saveFeatures]: Cannot open file: %s"<<\
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
		unsigned nopoints = patches[0].size();
		pFile.write(reinterpret_cast<char*>(&nopoints),sizeof(unsigned));
		for(typename std::vector<const T*>::const_iterator pa=patches[0].begin();\
		pa!=patches[0].end();++pa){
			unsigned patchsize = (*pa)->featH();
			cv::Point center   = (*pa)->point();
			pFile.write(reinterpret_cast<char*>(&patchsize),sizeof(unsigned));
			pFile.write(reinterpret_cast<char*>(&center.x),sizeof(unsigned));
			pFile.write(reinterpret_cast<char*>(&center.y),sizeof(unsigned));
		}
		pFile.close();
	}
}
//==============================================================================
/** Extracts or loads the test features for the current test image.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
std::vector<std::vector<IplImage*> > MotionRFdetector<L,M,T,F,N,U>::getFeatures\
(const std::string &path2img,const std::vector<std::string> &path2feat,\
const std::vector<float> &pyr,std::vector<std::vector<const T*> > &patches,\
cv::Size &origsize,F *features,bool showWhere) const{
	std::cout<<"[MotionRFdetector::getFeatures] extracting/loading test "<<\
		"features "<<path2feat<<std::endl;
	std::vector<std::vector<IplImage*> > vImg;
	// [0] Try to load the feature matrices
	try{
		vImg         = this->loadFeatures(path2feat,patches,showWhere);
		unsigned cls = 0;
		while(vImg[cls].empty()){++cls;}
		for(std::size_t i=0;i<pyr.size();++i){
			// [1] Resize to desired image scale
			cv::Mat img     = cv::imread(path2img.c_str(),1);
			origsize.height = img.rows;
			origsize.width  = img.cols;
			float scale     = static_cast<float>(this->maxsize_)/\
				std::max(origsize.height,origsize.width);
			if(scale<1){
				cv::Mat tmpimg   = cv::imread(path2img.c_str(),1);
				origsize.height  = tmpimg.rows*scale;
				origsize.width   = tmpimg.cols*scale;
				cv::Size imSize  = cv::Size(static_cast<int>(tmpimg.cols*scale),\
					static_cast<int>(tmpimg.rows*scale));
				img.release();
				cv::resize(tmpimg,img,imSize); tmpimg.release();
			}else if(pyr[i]!=1){ 
				cv::Mat tmpimg   = cv::imread(path2img.c_str(),1);
				origsize.height  = tmpimg.rows*pyr[i];
				origsize.width   = tmpimg.cols*pyr[i];
				cv::Size imSize  = cv::Size(static_cast<int>(tmpimg.cols*pyr[i]),\
					static_cast<int>(tmpimg.rows*pyr[i]));
				img.release();
				cv::resize(tmpimg,img,imSize); tmpimg.release();
			}
			if(img.empty()){
				std::cerr<<"[MotionRFdetector::getFeatures]: Could not load "<<\
					"image file: "<<path2img<<std::endl;
				std::exit(-1);
			}
			// [3] Make border (if the patch size changes better re-extract features)
			cv::Size borderSize = cv::Size(static_cast<int>(img.cols+this->overallW_),\
				static_cast<int>(img.rows+this->overallH_));
			cv::Mat cLevel = cv::Mat::zeros(borderSize,img.type());
			cv::copyMakeBorder(img,cLevel,this->overallH_/2,this->overallH_/2,\
				this->overallW_/2,this->overallW_/2,cv::BORDER_REPLICATE);
			img.release();
			features->push_backImages(cLevel.clone());
			cLevel.release();
		}
	// [1] If the features are not there, create them
	}catch(std::exception &e){
		// [4] Also add one vector of patches per scale
		if(patches.empty()){
			patches.resize(pyr.size());
		}
		// [5] Loop over all scales
		for(std::size_t i=0;i<pyr.size();++i){
			// [5.1] Resize to desired image scale
			cv::Mat img     = cv::imread(path2img.c_str(),1);
			origsize.height = img.rows;
			origsize.width  = img.cols;
			float scale     = static_cast<float>(this->maxsize_)/\
				static_cast<float>(std::max(origsize.height,origsize.width));
			if(scale<1){
				cv::Mat tmpimg   = cv::imread(path2img.c_str(),1);
				origsize.height  = tmpimg.rows*scale;
				origsize.width   = tmpimg.cols*scale;
				cv::Size imSize  = cv::Size(static_cast<int>(tmpimg.cols*scale),\
					static_cast<int>(tmpimg.rows*scale));
				img.release();
				cv::resize(tmpimg,img,imSize);
				tmpimg.release();
			}else if(pyr[i]!=1){
				cv::Mat tmpimg   = cv::imread(path2img.c_str(),1);
				origsize.height  = tmpimg.rows*pyr[i];
				origsize.width   = tmpimg.cols*pyr[i];
				cv::Size imSize  = cv::Size(static_cast<int>(tmpimg.cols*pyr[i]),\
					static_cast<int>(tmpimg.rows*pyr[i]));
				img.release();
				cv::resize(tmpimg,img,imSize); tmpimg.release();
			}
			if(img.empty()){
				std::cerr<<"[MotionRFdetector::getFeatures]: Could not load "<<\
					"image file: "<<path2img<<std::endl;
				std::exit(-1);
			}
			// [3] Make border (if the patch size changes better re-extract features)
			cv::Size borderSize = cv::Size(static_cast<int>(img.cols+this->overallW_),\
				static_cast<int>(img.rows+this->overallH_));
			cv::Mat cLevel = cv::Mat::zeros(borderSize,img.type());
			cv::copyMakeBorder(img,cLevel,this->overallH_/2,this->overallH_/2,\
				this->overallW_/2,this->overallW_/2,cv::BORDER_REPLICATE);
			img.release();
			features->push_backImages(cLevel.clone());
			// [5.2] Extract HoG-like features:
			std::vector<IplImage*> tmpFeat;
			std::vector<cv::Mat> channels = MotionPatch<T,F>::opponent(cLevel);
			img.release();
			for(std::vector<cv::Mat>::iterator ch=channels.begin();ch!=channels.end();++ch){
				std::vector<IplImage*> chFeat;
				IplImage *chImg = new IplImage(*ch);
				if(this->hogORsift_){
					CRPatch<T>::extractFeatureChannels9(chImg,chFeat,this->height_);
				}else{
					std::vector<cv::Point2f> points;
					SIFTlike sift(1,this->height_);
					sift.getGray(chImg,chFeat,points);
				}
				delete chImg; ch->release();
				tmpFeat.insert(tmpFeat.end(),chFeat.begin(),chFeat.end());
			}
			vImg.push_back(tmpFeat);
			// [5.3] write down the patches also
			cv::Mat where;
			if(showWhere){
				where = cv::Mat::zeros(cLevel.size(),CV_8UC1);
			}
			// [5.4] We get a vector of patches for each scale
			std::vector<cv::KeyPoint> points = MotionPatch<T,F>::getKeyPoints\
				(cLevel,this->step_,this->width_,this->height_,\
				this->pttype_,showWhere);
			for(std::vector<cv::KeyPoint>::iterator pt=points.begin();pt!=points.end();++pt){
				T* patch = new T(this->width_,this->height_,this->labW_,\
					this->labH_,i,pt->pt,this->motionW_,this->motionH_);
				patches[i].push_back(patch);
				if(showWhere){
					where.at<uchar>(pt->pt) = 255;
				}
			}
			cLevel.release();
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
	features->vImg(vImg);
	return vImg;
}
//==============================================================================
/** Gets an input image and returns a detection image (pixel labels by RF regression).
 * Given a set of predicted leafs for current pixel, get the final label:
 * Simple: [1] Just get the most voted pixel label per position.
 * Puzzle: [2] Optimized the patch selection label \cite{kontschider}.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void MotionRFdetector<L,M,T,F,N,U>::detectColor(const F *features,cv::Mat \
&motionDetect,cv::Mat &arrowsDetect,cv::Mat &appearDetect,const std::vector\
<const T*> &patches,const cv::Size &imsize,const std::string &imname,\
const std::string &path2model,unsigned offset,bool display){
	std::cout<<"[MotionRFdetector::detectColor] predicting on 1 image"<<std::endl;
	clock_t begin = clock();
	std::vector<std::vector<MotionPuzzlePatch*> > predictions;
	// [0] For each original image pixel
	for(typename std::vector<const T*>::const_iterator p=patches.begin();p!=\
	patches.end();++p){
		// [1] Access patch at positions saved in tests (go through tree, return leaf)
		std::vector<const U*> result;
		std::vector<const U*> resultFlip;
		this->forest_->regression(result,(*p),features);
		if(display){
			cv::imshow("Input Appearance",*((*p)->image(features)));
			cv::waitKey(10);
		}
		// [2] Flip the image and predict on it again.
		typename std::vector<const U*>::iterator itf;
		if(this->flip_){
			F* featFlip  = features->flip(this->usederivatives_);
			T* patchFlip = new T(*(*p));
			patchFlip->point(cv::Point(imsize.width-(patchFlip->point().x),\
				patchFlip->point().y));
			this->forest_->regression(result,(*p),features);
			delete patchFlip; delete featFlip;
			itf=resultFlip.begin();
		}
		// [3] Loop over all tree-predictions at this position
		std::vector<MotionPuzzlePatch*> pixelPredictions;
		for(typename std::vector<const U*>::iterator it=result.begin();it!=\
		result.end();++it){
			if(display){
				cv::imshow("Output Appearance",*(*it)->vAppearance());
				cv::waitKey(10);
			}
			MotionRF<L,M,T,F,N,U> *motionTree = dynamic_cast<MotionRF<L,M,T,F,N,U>* >(this->forest_);
			std::vector<float> histinfo = motionTree->histinfo(it-result.begin());
			pixelPredictions.push_back(new MotionPuzzlePatch((*p)->\
				point(),(*it)->vLabels(),*((*it)->vMotion()),*((*it)->vAppearance()),\
				((*it)->vHistos()),(*it)->labelProb(),(*it)->motionProb(),\
				(*it)->appearanceProb(),histinfo));
			// [3.1] Flip the predictions
			if(this->flip_){
				cv::Mat motionFlip;
				cv::Mat* vMotion = (*itf)->vMotion();
				cv::flip(*vMotion,motionFlip,1);
				cv::Mat appearanceFlip;
				cv::Mat* vAppearance = (*itf)->vAppearance();
				cv::flip(*vAppearance,appearanceFlip,1);
				std::vector<cv::Mat> vHisto = (*itf)->vHistos();
				std::vector<cv::Mat> histoFlip(vHisto.size(),cv::Mat());
				for(unsigned b=0;b<vHisto.size();++b){
					cv::flip(vHisto[b],histoFlip[b],1);
				}
				pixelPredictions.push_back(new MotionPuzzlePatch((*p)->\
					point(),(*it)->vLabels(),motionFlip,appearanceFlip,histoFlip,\
					(*itf)->labelProb(),(*itf)->motionProb(),(*itf)->appearanceProb(),\
					histinfo));
				motionFlip.release(); appearanceFlip.release();
				for(unsigned b=0;b<vHisto.size();++b){histoFlip[b].release();}
				delete (*itf); ++itf;
			}
			delete (*it);
		} // over trees
		result.clear(); resultFlip.clear();
		predictions.push_back(pixelPredictions);
		if(display){
			MotionTree<M,T,F,N,U>::showSamplesFlows(result,this->motionW_,\
				this->motionH_,(*p)->point());
		}
	} // over pixels
	clock_t end = clock();
	std::cout<<"Regression 1 img time elapsed: "<<double\
		(Auxiliary<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
	// [3] Propose motion on the image
	if(this->usederivatives_){
		cv::Mat tmpMotionXX,tmpMotionXY,tmpMotionYX,tmpMotionYY,tmpAppear;
		MotionPuzzle<MotionPuzzlePatch>::solve(predictions,imsize,\
			this->motionW_,this->motionH_,tmpMotionXX,tmpMotionXY,tmpMotionYX,\
			tmpMotionYY,tmpAppear,this->entropy_,this->method_,this->step_,imname,\
			this->path2results_);
		// [4] Reset the previous motion
		std::cout<<"[MotionRFdetector::detectColor] optimizing final labels"<<std::endl;
		// [5] Cut out the borders for everybody
		cv::Rect imRoi  = cv::Rect(this->overallW_/2,this->overallH_/2,\
			(imsize.width-this->overallW_),(imsize.height-this->overallH_));
		// [6] Get the arrow pictures to look at
		cv::Mat origim = arrowsDetect(imRoi).clone();
		arrowsDetect   = MotionPatch<T,F>::showOFderi(tmpMotionXX(imRoi),\
			tmpMotionXY(imRoi),tmpMotionYX(imRoi),tmpMotionYY(imRoi),origim,5,false);
		origim.release();
		tmpAppear(imRoi).copyTo(appearDetect);
		tmpAppear.release();
		// [7] Get the motion matrices to save.
		cv::Mat in[] = {tmpMotionXX(imRoi),tmpMotionXY(imRoi),\
			tmpMotionYX(imRoi),tmpMotionYY(imRoi)};
		cv::Mat mergedmotion;
		cv::merge(in,4,mergedmotion);
		mergedmotion.copyTo(motionDetect);
		mergedmotion.release();
		tmpMotionXX.release(); tmpMotionXY.release();
		tmpMotionYX.release(); tmpMotionYY.release();
	}else{
		cv::Mat tmpMotionX,tmpMotionY,tmpAppear;
		MotionPuzzle<MotionPuzzlePatch>::solve(predictions,imsize,\
			this->motionW_,this->motionH_,tmpMotionX,tmpMotionY,tmpAppear,\
			this->entropy_,this->method_,this->step_,imname,this->path2results_);
		// [4] Reset the previous motion
		std::cout<<"[MotionRFdetector::detectColor] optimizing final labels"<<std::endl;
		// [5] Cut out the borders for everybody
		unsigned limitW = std::max(this->motionW_,static_cast<unsigned>(this->overallW_));
		unsigned limitH = std::max(this->motionH_,static_cast<unsigned>(this->overallH_));
		cv::Rect imRoi  = cv::Rect(limitW/2,limitH/2,(imsize.width-limitW),\
			(imsize.height-limitH));
		// [6] Get the arrow pictures to look at
		cv::Mat origim    = arrowsDetect(imRoi).clone();
		cv::Mat origimbig = arrowsDetect.clone();
		arrowsDetect      = MotionPatch<T,F>::showOF(7*tmpMotionX(imRoi),7*tmpMotionY\
			(imRoi),origim,5,false);
		//MotionPatch<T,F>::warpInter(7*tmpMotionX,7*tmpMotionY,origimbig,offset,false);
		arrowsDetect = MotionPatch<T,F>::showOF(7*tmpMotionX(imRoi),7*tmpMotionY\
			(imRoi),origim,5,false);
		origim.release();
		origimbig.release();
		tmpAppear(imRoi).copyTo(appearDetect);
		tmpAppear.release();
		// [7] Get the motion matrices to save.
		cv::Mat in[] = {tmpMotionX(imRoi),tmpMotionY(imRoi)};
		cv::Mat mergedmotion;
		cv::merge(in,2,mergedmotion);
		mergedmotion.copyTo(motionDetect);
		mergedmotion.release();
		tmpMotionX.release(); tmpMotionY.release();
	}
	for(std::vector<std::vector<MotionPuzzlePatch*> >::iterator p=predictions.begin();\
	p!=predictions.end();++p){
		for(std::vector<MotionPuzzlePatch*>::iterator pr=p->begin();pr!=p->end();++pr){
			delete (*pr); (*pr) = NULL;
		}
		p->clear();
	}
	predictions.clear();
}
//==============================================================================
/** Scales the image at a number of sizes and it labels each scale [?].
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
std::vector<std::vector<const T*> > MotionRFdetector<L,M,T,F,N,U>::justfeatures\
(const std::string &imname,const std::string &path2img,const std::string &ext,\
const std::string &path2feat,const std::vector<float> &pyramid,F* features,\
cv::Size &origsize){
	std::vector<std::string> featPath;
	for(std::vector<float>::const_iterator it=pyramid.begin();it!=pyramid.end();++it){
		featPath.push_back(path2feat+imname+"_"+Auxiliary<int,1>::number2string\
			(it-pyramid.begin())+".bin");
	}
	std::vector<std::vector<const T*> > patches;
	std::string path2images = (path2img+imname+ext);
	// [2] Loop over pyramid scales (none used for now)
	std::vector<std::vector<IplImage*> > vImg = this->getFeatures(path2images,\
		featPath,pyramid,patches,origsize,features);
	return patches;
}
//==============================================================================
/** Scales the image at a number of sizes and it labels each scale [?].
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void MotionRFdetector<L,M,T,F,N,U>::detectPyramid(const std::string &imname,\
const std::string &path2img,const std::string &path2feat,const std::string &ext,\
const std::vector<float> &pyramid,std::vector<cv::Mat> &vMotionDetect,\
std::vector<cv::Mat> &vArrowsDetect,std::vector<cv::Mat> &vAppearDetect,\
const std::string &path2model,unsigned offset){
	// [0] Get the image name for each pyramid scale
	std::vector<std::string> featPath;
	for(std::vector<float>::const_iterator it=pyramid.begin();it!=pyramid.end();++it){
		featPath.push_back(path2feat+imname+"_"+Auxiliary<int,1>::number2string\
			(it-pyramid.begin())+".bin");
	}
	// [1] Get the features for the current image for all scales
	std::vector<std::vector<const T*> > patches;
	std::string path2images = (path2img+imname+ext);
	cv::Size origsize;
	// [2] Loop over pyramid scales (none used for now)
	F* features = new F(this->usederivatives_);
	std::vector<std::vector<IplImage*> > vImg = this->getFeatures(path2images,\
		featPath,pyramid,patches,origsize,features);
	for(std::vector<std::string>::iterator it=featPath.begin();it!=\
	featPath.end();++it){
		// [3] Define the result size
		unsigned pos = it-featPath.begin();
		cv::Size featsize(vImg[pos][0]->width,vImg[pos][0]->height);
		// [4] Write the initial image in there to plot it back
		cv::Mat original = cv::imread(path2img+imname+ext,-1);
		original.copyTo(vArrowsDetect[pos]);
		original.release();
		cv::resize(vArrowsDetect[pos],vArrowsDetect[pos],featsize,0,0,cv::INTER_LINEAR);
		// [5] Predict on the current image scale
		this->detectColor(features,vMotionDetect[pos],vArrowsDetect[pos],\
			vAppearDetect[pos],patches[pos],featsize,imname,path2model,offset);
		// [6] Resize the predictions to the original size
		cv::resize(vArrowsDetect[pos],vArrowsDetect[pos],origsize,0,0,cv::INTER_LINEAR);
		cv::resize(vMotionDetect[pos],vMotionDetect[pos],origsize,0,0,cv::INTER_LINEAR);
		// [7] Release patches at this size
		for(unsigned p=0;p<patches[pos].size();++p){
			delete patches[pos][p];
		}
		patches[pos].clear();
	}
	delete features;
	patches.clear();
	// [8] Release the features
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
template class MotionRFdetector<MotionTree,MotionPatch<MotionPatchFeature<FeaturesMotion>,\
	FeaturesMotion>,MotionPatchFeature<FeaturesMotion>,FeaturesMotion,MotionTreeNode\
	<MotionLeafNode,MotionPatchFeature<FeaturesMotion> >,MotionLeafNode>;























