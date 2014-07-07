/* StructuredPatch.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef StructuredPatch_CPP_
#define StructuredPatch_CPP_
#include <fstream>
#include <stdio.h>
#include <string.h>
#include "StructuredPatch.h"
#include <CRPatch.h>
//==============================================================================
template <class T,class F>
StructuredPatch<T,F>::StructuredPatch(CvRNG* pRNG,unsigned patchW,unsigned patchH,\
unsigned noCls,unsigned labW,unsigned labH,unsigned trainSize,unsigned noPatches,\
unsigned consideredCls,bool balance,unsigned step):noCls_(noCls),\
labW_(labW),labH_(labH),featW_(patchW),featH_(patchH),cvRNG_(pRNG),consideredCls_\
(consideredCls),balance_(balance),trainingSize_(trainSize),noPatches_(noPatches),\
step_(step){
	this->features_ = new F();
};
//==============================================================================
template <class T,class F>
StructuredPatch<T,F>::~StructuredPatch(){
	// [0] Clear all patches
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
	// [1] Clear the features
	if(this->features_){
		delete this->features_;
		this->features_ = NULL;
	}
}
//==============================================================================
/** Get image labels. Try loading it, if not there compute it (very slow).
 */
template <class T,class F>
cv::Mat* StructuredPatch<T,F>::loadLabels(const std::string &justname,const \
std::string &ext,const std::string &term,const std::string &labpath,const std::map\
<cv::Vec3b,unsigned,vec3bCompare> &classinfo,const std::string &featpath){
	cv::Mat* labels;
	std::string path = std::string(featpath+justname+"_labels"+ext);
	// [0] If the file exists then load it, if not then make it
	if(Auxiliary<uchar,3>::file_exists(path.c_str(),false)){
		// [1] Load the greyscale image
		cv::Mat tmp = cv::imread(path.c_str(),0);
		labels      = new cv::Mat(tmp);
		tmp.release();
	}else{ // [2] The image is not there so make it
		std::cout<<"[StructuredPatch::loadLabels]: Creating label matrix"<<std::endl;
		std::string labname = std::string(labpath+justname+term+ext);
		cv::Mat labimg      = cv::imread(labname.c_str(),1);
		if(labimg.empty()){
			std::cerr<<"[StructuredPatch::loadLabels]: No labels found!"<<std::endl;
			std::exception e;
			throw(e);
		}
		labimg.convertTo(labimg,CV_8UC3);
		// [3] For each Nth pixel store its label id
		labels = new cv::Mat(cv::Mat::zeros(cv::Size((int)(labimg.cols),(int)\
			(labimg.rows)),CV_8UC1));
		cv::MatConstIterator_<cv::Vec3b> i2 = labimg.begin<cv::Vec3b>();
		for(cv::MatIterator_<uchar> i1=labels->begin<uchar>();\
		i1!=labels->end<uchar>(),i2!=labimg.end<cv::Vec3b>();++i1,++i2){
			std::map<cv::Vec3b,unsigned,vec3bCompare>::const_iterator it;
			bool found = false;
			// [4] Convert the color coding to label id
			for(std::map<cv::Vec3b,unsigned,vec3bCompare>::const_iterator itC=\
			classinfo.begin();itC!=classinfo.end();++itC){
				if((itC->first).val[0]==(*i2).val[0] && (itC->first).val[1]==(*i2).val[1] \
				&& (itC->first).val[2]==(*i2).val[2]){
					it    = itC;
					found = true;
					break;
				}
			}
			assert(found);
			(*i1)        = it->second;
			unsigned pos = (i2-labimg.begin<cv::Vec3b>());
		}
		labimg.release();
		// [5] Save the label image locally so we don't have to re-compute this
		std::vector<int> params;
		if(strcmp(ext.c_str(),".jpg")){
			params.push_back(CV_IMWRITE_JPEG_QUALITY);
			params.push_back(100);
			cv::imwrite(path,(*labels),params);
		}else if(strcmp(ext.c_str(),".png")){
			params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			params.push_back(4);
			cv::imwrite(path,(*labels),params);
		}else{
			cv::imwrite(path,(*labels));
		}

	}
	return labels;
}
//==============================================================================
/** Randomly picks a subset of the images names to be used for training.
 */
template <class T,class F>
void StructuredPatch<T,F>::pickRandomNames(const std::vector<std::string> &filenames){
	// [0] Randomly shuffle the vector
	this->imName_       = filenames;
	std::random_shuffle(this->imName_.begin(),this->imName_.end());
	// [1] Keep trainingSize images out of these
	this->trainingSize_ = std::min(this->trainingSize_,static_cast<unsigned>\
		(this->imName_.size()));
	this->imName_.resize(this->trainingSize_);
}
//==============================================================================
/** Resets the class members to add new patches.
 */
template <class T,class F>
void StructuredPatch<T,F>::reset(){
	// [0] Clear all patches
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
	// [1] Clear features and labels
	if(this->features_){
		delete this->features_;
		this->features_ = new F();
	}
	// [2] Clear image names
	this->imName_.clear();
}
//==============================================================================
/** Extracts the feature patches but also the label patches.
 * imgpath    -- path to the images
 * labpath    -- path to labels
 * featpath   -- path to features
 * vFilenames -- vector of image names
 * classinfo  -- mapping from pixel color to label ID
 * labH       -- label patch height
 * labW       -- label patch width
 */
template <class T,class F>
void StructuredPatch<T,F>::extractPatches(const std::string &imgpath,const \
std::string &labpath,const std::string &featpath,const std::vector<std::string>\
&vFilenames,const std::map<cv::Vec3b,unsigned,vec3bCompare> &classinfo,\
const std::string &term,const std::string &ext){
	// [0] First randomly shuffle the images
	this->reset();
	this->pickRandomNames(vFilenames);
	// [1] Loop over all images kept for training
	for(unsigned i=0;i<this->imName_.size();++i){
		if(i%100==0){
			std::cout<<"[StructuredPatch::extractPatches]: Processing image: "<<\
				this->imName_[i]<<" "<<i<<std::endl;
		}
		// [2] Try to load the patches if not possible then compute them
		std::string justname = this->imName_[i].substr(0,this->imName_[i].size()-4);
		std::string featname = std::string(featpath+justname+".bin");
		try{
			this->loadPatches(featname);
		}catch(std::exception &e){
			std::cout<<"[StructuredPatch::extractPatches]: Cannot open file"<<\
				featname<<std::endl;
			// [2] Load the actual image
			std::string impath = std::string(imgpath+(this->imName_[i]));
			IplImage *img      = cvLoadImage(impath.c_str(),-1);
			if(!img){
				cvReleaseImage(&img);
				std::cout<<"[StructuredPatch::extractPatches] Could not load image file: "\
					<<this->imName_[i]<<std::endl;
				std::exit(-1);
			}
			cv::Mat* labels  = this->loadLabels(justname,ext,term,labpath,classinfo,featpath);
			// [4] Extract features & patches at each Nth pixel in the image
			this->extractFeatures(labels,img,featname);
			// [5] Store everything in a feature matrix for each image
			this->savePatches(featname,this->features_->size()-1);
			cvReleaseImage(&img);
			labels->release();
			delete labels;
		}
	}
	this->resizePatches();
}
//==============================================================================
/** Resize patches to the number of patches to be used for training.
 */
template <class T,class F>
void StructuredPatch<T,F>::resizePatches(){
	// [0] Find the number of patches per class
	unsigned allsize = 0;
	unsigned minSize = std::numeric_limits<unsigned>::max();
	for(typename std::vector<std::vector<const T*> >::iterator i=\
	this->patches_.begin();i!=this->patches_.end();++i){
		if(this->balance_ && minSize>(i->size()) && (i->size())){
			minSize = i->size();
		}
		allsize += i->size();
		std::cout<<"[StructuredPatch<T>::extractPatches]: Patches for class "<<\
			(i-(this->patches_.begin()))<<" => "<<(i->size())<<std::endl;
	}
	// [1] If the classes should be balanced to the minSize
	if(this->balance_){
		allsize = minSize*this->consideredCls_;
		std::cout<<"[StructuredPatch<T>::extractPatches]: All resized to "<<\
			allsize<<std::endl;
		for(typename std::vector<std::vector<const T*> >::iterator i=\
		this->patches_.begin();i!=this->patches_.end();++i){
			if(!i->empty()){
				std::random_shuffle(i->begin(),i->end());
				for(typename std::vector<const T*>::iterator j=i->begin()+minSize;\
				j!=i->end();++j){delete *j;}
				i->resize(minSize);
			}
		}
	}
	std::cout<<std::endl;
	// [2] If there is a desired number of patches to keep
	if(this->noPatches_){
		// [2.1] If less patches then the number to retain, then add for each cls
		if(this->noPatches_>allsize){ // add random patches for each clss
			unsigned howmany = std::ceil(static_cast<float>(this->noPatches_-allsize)/\
				static_cast<float>(this->consideredCls_));
			for(typename std::vector<std::vector<const T*> >::iterator i=\
			this->patches_.begin();i!=this->patches_.end();++i){ // over patches
				if(!i->empty()){
					for(unsigned h=0;h<howmany;++h){ // howmany times push back a random val
						unsigned pick = cvRandInt(this->cvRNG_)%(i->size());
						i->push_back(new T(*(i->at(pick))));
					}
				}
				std::cout<<"[StructuredPatch<T>::extractPatches]: Patches for class "<<\
					(i-(this->patches_.begin()))<<" => "<<(i->size())<<std::endl;
			}
		}
		// [2.2] If less patches then the number to retain ==> bad luck
	}
}
//==============================================================================
/** Computes features if not there for loading.
 */
template <class T,class F>
void StructuredPatch<T,F>::extractFeatures(const cv::Mat *labimg,IplImage *img,\
const std::string &path2feat,bool showWhere){
	// [0] No borders. We update the class frequencies with new counts.
	for(cv::MatConstIterator_<uchar> l=labimg->begin<uchar>();l!=labimg->end\
	<uchar>();++l){
		this->features_->updateClsFreq(*l,this->noCls_);
	}
	this->features_->push_backLab(labimg->clone());
	// [1] Extract the HoG-like features:
	std::vector<IplImage*> tmpFeat;
	CRPatch<T>::extractFeatureChannels32(img,tmpFeat);
	this->features_->push_backImg(tmpFeat);
	// [2] Verify that everything is OK
	if(this->features_->vImg().size()!=this->features_->labelImg().size()){
		std::cerr<<"[StructuredPatch::extractFeatures]: Each image should have a "<<
			"label image and a feature vector of images."<<std::endl;
		std::exit(-1);
	}
	// [3] Add to patches these ones
	if(this->patches_.empty()){
		this->patches_.resize(this->noCls_);
	}
	// [4] We pick patches in a grid
	cv::Mat where;
	if(showWhere){
		where = cv::Mat::zeros(labimg->size(),labimg->type());
	}
	unsigned limitW = std::max(this->labW_,this->featW_);
	unsigned limitH = std::max(this->labH_,this->featH_);
	// [5] Loop over the image rows and cols
	for(unsigned r=limitH/2;r<(labimg->rows-limitH/2);r+=this->step_){
		for(unsigned c=limitW/2;c<(labimg->cols-limitW/2);c+=this->step_){
			unsigned label = static_cast<unsigned>(labimg->at<uchar>(r,c));
			if(label<this->consideredCls_){
				T* patch = new T(this->featW_,this->featH_,this->labW_,this->labH_,\
					(this->features_->size()-1),cv::Point(c,r));
				// [5.1] Get the pixel label at which this patch is centered
				this->patches_[label].push_back(patch);
				if(showWhere){
					where.at<uchar>(r,c) = 255;
				}
				// [5.2] Update class co-frequencies
				std::vector<unsigned> labels = patch->label(this->features_);
				this->features_->updateClsCoFreq(labels,label,this->noCls_);
			}
		}
	}
	// [6] If we want to see where the patches are
	if(showWhere){
		cv::imshow("where",where);
		cv::waitKey(10);
	}
	where.release();
}
//==============================================================================
/** Saves the labels and the image features --- for each image make one file.
 */
template <class T,class F>
void StructuredPatch<T,F>::savePatches(const std::string &path2feat,unsigned pos){
	std::ofstream pFile;
	try{
		pFile.open(path2feat.c_str(),std::ios::out|std::ios::binary);
	}catch(std::exception &e){
		std::cerr<<"[StructuredPatch<T>::savePatches]: Cannot open file: %s"<<\
			e.what()<<std::endl;
		std::exit(-1);
	}
	pFile.precision(std::numeric_limits<double>::digits10);
	pFile.precision(std::numeric_limits<float>::digits10);
	// [0] Write down the label-image
	cv::Mat tmpLab = this->features_->labelImg(pos);
	int cols = tmpLab.cols,rows = tmpLab.rows;
	pFile.write(reinterpret_cast<char*>(&cols),sizeof(int));
	pFile.write(reinterpret_cast<char*>(&rows),sizeof(int));
	// [0.1] Loop over its rows and cols
	for(int y=0;y<tmpLab.rows;++y){
		for(int x=0;x<tmpLab.cols;++x){
			uchar val = tmpLab.at<uchar>(y,x);
			pFile.write(reinterpret_cast<char*>(&val),sizeof(uchar));
		}
	}
	// [1] Write down the Hog-like features
	std::vector<IplImage*> tmpImg = this->features_->vImg(pos);
	unsigned vsize                = static_cast<unsigned>(tmpImg.size());
	pFile.write(reinterpret_cast<char*>(&vsize),sizeof(unsigned));
	if(vsize>0){
		// [1.1] Write data to file for each channel
		for(std::vector<IplImage*>::const_iterator it=tmpImg.begin();\
		it!=tmpImg.end();++it){
			pFile.write(reinterpret_cast<char*>(&tmpImg[0]->nChannels),sizeof(int));
			pFile.write(reinterpret_cast<char*>(&tmpImg[0]->depth),sizeof(int));
			pFile.write(reinterpret_cast<char*>(&tmpImg[0]->width),sizeof(int));
			pFile.write(reinterpret_cast<char*>(&tmpImg[0]->height),sizeof(int));
			for(int y=0;y<(*it)->height;++y){
				for(int x=0;x<(*it)->width;++x){
					uchar val = (*it)->imageData[y*(*it)->width+x];
					pFile.write(reinterpret_cast<char*>(&val),sizeof(uchar));
				}
			}
		}
	}
	pFile.close();
}
//==============================================================================
/** Loads the labels and the image features --- 1 file per image.
 */
template <class T,class F>
void StructuredPatch<T,F>::loadPatches(const std::string &path2feat,bool showWhere){
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
		// [0] Read the label image
		int rows,cols;
		pFile.read(reinterpret_cast<char*>(&cols),sizeof(int));
		pFile.read(reinterpret_cast<char*>(&rows),sizeof(int));
		// [1] Add to also the patches
		if(this->patches_.empty()){
			this->patches_.resize(this->noCls_);
		}
		// [2] Read the data of the matrix
		cv::Mat tmp     = cv::Mat::zeros(cv::Size(cols,rows),CV_8UC1);
		unsigned limitW = std::max(this->labW_,this->featW_);
		unsigned limitH = std::max(this->labH_,this->featH_);
		// [3] Loop over the label cols and rows
		for(int y=0;y<tmp.rows;++y){
			for(int x=0;x<tmp.cols;++x){
				// [3.1] Read the label image value
				pFile.read(reinterpret_cast<char*>(&(tmp.at<uchar>(y,x))),\
					sizeof(uchar));
				// [3.2] Update class frequencies
				unsigned label = static_cast<unsigned>(tmp.at<uchar>(y,x));
				this->features_->updateClsFreq(label,this->noCls_);
			}
		}
		this->features_->push_backLab(tmp);
		cv::Mat where;
		if(showWhere){
			where =	cv::Mat::zeros(tmp.size(),tmp.type());
		}
		// [5] Read the features matrix
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
						if(s==0){
							// [5.1] If on the grid and a considered class, add patch
							unsigned label = static_cast<unsigned>(tmp.at<uchar>(y,x));
							if(y>=(limitH/2) && (y+limitH/2)<tmp.rows && x>=(limitW/2) \
							&& (x+limitW/2)<tmp.cols && label<this->consideredCls_ && \
							((y-limitH/2)%(this->step_))==0 && ((x-limitW/2)%\
							(this->step_))==0){
								// [5.2] The feature index is not updates
								T* patch = new T(this->featW_,this->featH_,\
									this->labW_,this->labH_,(this->features_->size()-1),\
									cv::Point(x,y));
								this->patches_[label].push_back(patch);
								if(showWhere){
									where.at<uchar>(y,x) = 255;
								}
								// [5.3] Update class co-frequencies
								std::vector<unsigned> labels = patch->label(this->features_);
								this->features_->updateClsCoFreq(labels,label,this->noCls_);
							}
						}
					}
				}
				img.push_back(cvCloneImage(im));
			}
		}
		tmp.release();
		// [6] If we want to see where the patches are
		if(showWhere){
			cv::imshow("where",where);
			cv::waitKey(0);
		}
		where.release();
		this->features_->push_backImg(img);
		pFile.close();
	}
}
//==============================================================================
/** Gets the number of feature channels.
 */
template <class T,class F>
unsigned StructuredPatch<T,F>::getPatchChannels() const{
	if(this->features_->size()>0){
		return this->features_->vImg(0).size();
	}
	return 0;
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif // StructuredPatch_CPP_



















