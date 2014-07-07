/* StructuredPatch.h
 * Author: Silvia-Laura Pintea
 */
#ifndef StructuredPatch_H_
#define StructuredPatch_H_
#pragma once
#include <Auxiliary.h>
//==============================================================================
/** To keep inside all the vectors we need.
 */
class Features{
	public:
		Features(){this->size_ = 0;}
		Features(const std::vector<cv::Mat> &labelImg,const std::vector<std::vector\
		<IplImage*> > &vImg){
			this->vImg(vImg);
			this->labelImg(labelImg);
			this->size_ = this->labelImg_.size();
		}
		virtual ~Features(){
			// Clear feature-images
			for(unsigned i=0;i<this->vImg_.size();++i){ // over images
				for(unsigned c=0;c<this->vImg_[i].size();++c){ // over channels
					if(this->vImg_[i][c]){
						// Created with: cvCloneImage
						cvReleaseImage(&this->vImg_[i][c]);
						this->vImg_[i][c] = NULL;
					}
				}
				this->vImg_[i].clear();
			}
			this->vImg_.clear();
			// Created with: new cv::Mat();
			for(unsigned i=0;i<this->labelImg_.size();++i){ // over images
				if(!this->labelImg_[i].empty()){
					this->labelImg_[i].release();
				}
			}
			this->labelImg_.clear();
		}
		//----------------------------------------------------------------------
		/** Returns the size of the label image.
		 */
		cv::Size labelSize(){
			if(this->labelImg_.empty()){
				return cv::Size(0,0);
			}else{
				return cv::Size(this->labelImg_[0].cols,this->labelImg_[0].rows);
			}
		}
		/** Adds a vector of images to the vector-of-vector of images at the end.
		 */
		void push_backImg(std::vector<IplImage*> &vImg){
			this->vImg_.push_back(vImg);
			this->size_ = std::max(this->vImg_.size(),this->labelImg_.size());
		}
		/** Adds a vector of images to the vector-of-vector of images at the end.
		 */
		void push_backLab(cv::Mat lab){
			this->labelImg_.push_back(lab.clone());
			this->size_ = std::max(this->vImg_.size(),this->labelImg_.size());
		}
		/** Updates the co-frequencies of classes inside patches.
		 */
		void updateClsCoFreq(const std::vector<unsigned> &labels,unsigned center,\
		unsigned nocls){
			if(this->coFreq_.empty()){
				this->coFreq_.resize(nocls);
				for(unsigned c=0;c<nocls;++c){
					this->coFreq_[c] = std::vector<unsigned>(nocls,0);
				}
			}
			for(std::vector<unsigned>::const_iterator l=labels.begin();l!=\
			labels.end();++l){
				if(l!=(labels.begin()+(labels.size()-1)/2)){
					++this->coFreq_[center][*l];
				}
			}
		}
		/** Updates the frequencies of classes in the training data.
		 */
		void updateClsFreq(unsigned label,unsigned nocls){
			if(this->clsFreq_.empty()){
				this->clsFreq_ = std::vector<unsigned>(nocls,0);
			}
			++this->clsFreq_[label];
		}
		/** Computes and overwrites the co-frequencies with inverse co-frequencies.
		 */
		std::vector<std::vector<float> > invCoFreq() const{
			unsigned sumCoFreq = 0;
			for(std::vector<std::vector<unsigned> >::const_iterator c1=this->coFreq_.\
			begin();c1!=this->coFreq_.end();++c1){
				for(std::vector<unsigned>::const_iterator c2=c1->begin();c2!=\
				c1->end();++c2){
					sumCoFreq += (*c2);
				}
			}
			std::vector<std::vector<float> > invcofreq(this->coFreq_.size(),\
				std::vector<float>());
			for(std::vector<std::vector<unsigned> >::const_iterator c1=this->coFreq_.\
			begin();c1!=this->coFreq_.end();++c1){
				for(std::vector<unsigned>::const_iterator c2=c1->begin();c2!=\
				c1->end();++c2){
					if(*c2){
						invcofreq[c1-(this->coFreq_.begin())].push_back\
							(static_cast<float>(sumCoFreq)/static_cast<float>(*c2));
					}else{
						invcofreq[c1-(this->coFreq_.begin())].push_back(0.0);
					}
				}
			}
			return invcofreq;
		}
		/** Computes and overwrites the frequencies with inverse frequencies.
		 */
		std::vector<float> invClsFreq() const{
			unsigned sumFreq = 0;
			for(std::vector<unsigned>::const_iterator c=this->clsFreq_.begin();c!=\
			this->clsFreq_.end();++c){
				sumFreq += (*c);
			}
			std::vector<float> invfreq;
			for(std::vector<unsigned>::const_iterator c=this->clsFreq_.begin();c!=\
			this->clsFreq_.end();++c){
				if(*c){
					invfreq.push_back(static_cast<float>(sumFreq)/static_cast<float>(*c));
				}else{
					invfreq.push_back(0.0);
				}
			}
			return invfreq;
		}
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		virtual unsigned size() const {
			return std::max(this->vImg_.size(),this->labelImg_.size());
		}
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		std::vector<cv::Mat> labelImg() const {return this->labelImg_;}
		cv::Mat labelImg(unsigned pos) const {return this->labelImg_[pos];}
		std::vector<std::vector<IplImage*> > vImg() const {return this->vImg_;}
		std::vector<IplImage*> vImg(unsigned pos) const {return this->vImg_[pos];}
		IplImage* vImg(unsigned pos,unsigned ch) const {return this->vImg_[pos][ch];}
		std::vector<unsigned> clsFreq() const {return this->clsFreq_;}
		unsigned clsFreq(unsigned pos) const {return this->clsFreq_[pos];}
		std::vector<std::vector<unsigned> > coFreq() const {return this->coFreq_;}
		std::vector<unsigned> coFreq(unsigned pos) const {return this->coFreq_[pos];}
		unsigned coFreq(unsigned ctr,unsigned rnd) const {return this->coFreq_[ctr][rnd];}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void size(unsigned size){this->size_ = size;}
		void labelImg(const std::vector<cv::Mat> &labelImg){
			for(unsigned s=0;s<this->labelImg_.size();++s){
				// created with: new cv::Mat
				if(!this->labelImg_[s].empty()){
					this->labelImg_[s].release();
				}
			}
			this->labelImg_.clear();
			this->labelImg_.resize(labelImg.size(),cv::Mat());
			for(unsigned s=0;s<labelImg.size();++s){
				labelImg[s].copyTo(this->labelImg_[s]);
			}
		}
		void labelImg(unsigned pos,const cv::Mat labelImg){
			if(pos<this->labelImg_.size()){
				this->labelImg_[pos].release();
			}
			labelImg.copyTo(this->labelImg_[pos]);
		}
		void vImg(const std::vector<std::vector<IplImage*> > &vImg){
			for(unsigned s=0;s<this->vImg_.size();++s){
				// created with: cvCloneImage
				for(unsigned c=0;c<this->vImg_[s].size();++c){
					if(this->vImg_[s][c]){
						cvReleaseImage(&this->vImg_[s][c]);
						this->vImg_[s][c] = NULL;
					}
				}
				this->vImg_[s].clear();
			}
			this->vImg_.clear();
			for(std::vector<std::vector<IplImage*> >::const_iterator s=vImg.begin();\
			s!=vImg.end();++s){
				std::vector<IplImage*> tmp;
				for(std::vector<IplImage*>::const_iterator c=s->begin();\
				c!=s->end();++c){
					tmp.push_back(cvCloneImage(*c));
				}
				this->vImg_.push_back(tmp);
			}
		}
		void vImg(unsigned pos,const std::vector<IplImage*> &vImg){
			if(!this->vImg_[pos].empty()){
				// created with: cvCloneImage
				for(unsigned c=0;c<this->vImg_[pos].size();++c){
					if(this->vImg_[pos][c]){
						cvReleaseImage(&this->vImg_[pos][c]);
						this->vImg_[pos][c] = NULL;
					}
				}
				this->vImg_[pos].clear();
			}
			for(std::vector<IplImage*>::const_iterator c=vImg.begin();\
			c!=vImg.end();++c){
				this->vImg_[pos].push_back(cvCloneImage(*c));
			}
		}
		void vImg(unsigned pos,unsigned ch,const IplImage *vImg){
			if(this->vImg_[pos][ch]){
				// created with: cvCloneImage
				cvReleaseImage(&this->vImg_[pos][ch]);
				this->vImg_[pos][ch] = NULL;
			}
			this->vImg_[pos][ch] = cvCloneImage(vImg);
		}
		void clsFreq(const std::vector<unsigned> &clsFreq){
			this->clsFreq_.clear();
			this->clsFreq_ = clsFreq;
		}
		void coFreq(const std::vector<std::vector<unsigned> > &coFreq){
			this->coFreq_.clear();
			for(std::vector<std::vector<unsigned> >::const_iterator it=coFreq.begin();\
			it!=coFreq.end();++it){
				this->coFreq_.push_back(*it);
			}
		}
		//----------------------------------------------------------------------
	protected:
		/** @var labelImg_
		 * The grayscale images containing the labels.
		 */
		std::vector<cv::Mat> labelImg_;
		/** @var vImg_
		 * The vectors of feature matrices -- one per image.
		 */
		std::vector<std::vector<IplImage*> > vImg_;
		/** @var size_
		 * The size of the data (number of images).
		 */
		unsigned size_;
		/** @var clsFreq_
		 * The class frequencies in the original training data for unbalanced data.
		 */
		std::vector<unsigned> clsFreq_;
		/** @var coFreq_
		 * The class co-frequencies in the original training data for unbalanced data.
		 */
		std::vector<std::vector<unsigned> > coFreq_;
	private:
		DISALLOW_COPY_AND_ASSIGN(Features);
};
//==============================================================================
//==============================================================================
//==============================================================================
/** Patches are always relative to corner: top-left.
 */
template <class F>
class LabelPatchFeature{
	public:
		LabelPatchFeature(unsigned featW,unsigned featH,unsigned labW,unsigned labH,\
		unsigned imIndex,const cv::Point &point):featW_(featW),featH_(featH),\
		labW_(labW),labH_(labH),imIndex_(imIndex),point_(point){}
		virtual ~LabelPatchFeature(){}
		//----------------------------------------------------------------------
		/** Gets the label patch around the current pixel as a vector.
		 */
		std::vector<unsigned> label(const F *features) const;
		/** Gets the feature around the current pixel as a matrix.
		 */
		std::vector<CvMat*> feat(const F *features) const;
		/** Gets the feature around the current pixel as a matrix for the give
		 * channel value.
		 */
		CvMat* feat(const F *features,int channel) const;
		/** Gets the feature around the current pixel as a row-matrix.
		 */
		cv::Mat featRow(const F *features) const;
		/** Gets the feature around the current pixel as a row-matrix.
		 */
		cv::Mat featRowHack(const F *features) const;
		/** Gets the feature around the current pixel as a vector of row-matrix.
		 */
		std::vector<cv::Mat> vecFeatRow(const F *features) const;
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned labH() const {return this->labH_;}
		unsigned labW() const {return this->labW_;}
		unsigned featH() const {return this->featH_;}
		unsigned featW() const {return this->featW_;};
		unsigned imIndex() const {return this->imIndex_;};
		cv::Point point() const {return this->point_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void labH(unsigned labH){this->labH_ = labH;}
		void labW(unsigned labW){this->labW_ = labW;}
		void featH(unsigned featH){this->featH_= featH_;}
		void featW(unsigned featW){this->featW_ = featW;};
		void imIndex(unsigned imIndex){this->imIndex_ = imIndex;};
		void point(const cv::Point &point){
			this->point_.x = point.x;
			this->point_.y = point.y;
		}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors the default ones are not good with IplImages
		 */
		LabelPatchFeature(LabelPatchFeature const &rhs){
			this->featH_    = rhs.featH();
			this->featW_    = rhs.featW();
			this->labH_     = rhs.labH();
			this->labW_     = rhs.labW();
			this->imIndex_  = rhs.imIndex();
			this->point(rhs.point());
		}
		LabelPatchFeature& operator=(LabelPatchFeature const &rhs){
			if(this == &rhs) return *this;
			if(this){delete this;}
			this->featH_   = rhs.featH();
			this->featW_   = rhs.featW();
			this->labH_    = rhs.labH();
			this->labW_    = rhs.labW();
			this->imIndex_ = rhs.imIndex();
			this->point(rhs.point());
			return *this;
		}
		//----------------------------------------------------------------------
	protected:
		/** @var featH_
		 * The height of the feature patch.
		 */
		unsigned featH_;
		/** @var featW_
		 * The width of the feature patch.
		 */
		unsigned featW_;
		/** @var labH_
		 * The height of the label patch.
		 */
		unsigned labH_;
		/** @var labW_
		 * The width of the label patch.
		 */
		unsigned labW_;
		/** @var imIndex_
		 * Index in the image-matrix and feature-matrix .
		 */
		unsigned imIndex_;
		/** @var labPt_
		 * The actual position in the original image.
		 */
		cv::Point point_;
};
//==============================================================================
/** Gets the label patch around the current pixel as a vector.
 */
template <class F>
inline std::vector<unsigned> LabelPatchFeature<F>::label(const F *features) const{
	if(features->labelImg().size()<=this->imIndex_){
		std::cerr<<"[LabelPatchFeature<F>::label] No labels are loaded."<<std::endl;
		std::exit(-1);
	}
	std::vector<unsigned> label;
	cv::Mat tmp = features->labelImg(this->imIndex_);
	if(this->labH_*(this->labW_)>1){
		cv::Rect roi(this->point_.x-(this->labW_/2),this->point_.y-(this->labH_/2),\
			this->labW_,this->labH_);
		cv::Mat labelRoi = cv::Mat(tmp,roi).clone();
		labelRoi         = labelRoi.reshape(1,1); // 1 channel, 1 row
		uchar* rowpt     = labelRoi.ptr(0);
		label            = std::vector<unsigned>(rowpt,rowpt+labelRoi.cols);
		labelRoi.release();
	}else{
		label.push_back(tmp.at<uchar>(this->point_));
	}
	return label;
}
//==============================================================================
/** Gets the feature around the current pixel as a matrix.
 */
template <class F>
inline std::vector<CvMat*> LabelPatchFeature<F>::feat(const F *features) const{
	std::vector<CvMat*> vPatch;
	std::vector<IplImage*> tmpImg = features->vImg(this->imIndex_);
	vPatch.resize(tmpImg.size());
	cv::Rect roi(this->point_.x-(this->featW_/2),this->point_.y-(this->featH_/2),\
		this->featW_,this->featH_);
	for(unsigned int c=0;c<tmpImg.size();++c){
		CvMat tmp;
		cvGetSubRect(tmpImg[c],&tmp,roi);
		vPatch[c] = cvCloneMat(&tmp);
	}
	return vPatch;
}
//==============================================================================
/** Gets the feature around the current pixel as a row-matrix.
 */
template <class F>
inline cv::Mat LabelPatchFeature<F>::featRow(const F *features) const{
	cv::Mat vPatch;
	cv::Rect roi(this->point_.x-(this->featW_/2),this->point_.y-(this->featH_/2),\
		this->featW_,this->featH_);
	std::vector<IplImage*> tmpImg = features->vImg(this->imIndex_);
	for(unsigned int c=0;c<tmpImg.size();++c){
		CvMat tmp;
		cvGetSubRect(tmpImg[c],&tmp,roi);
		cv::Mat onecol(&tmp,true);
		vPatch.push_back(onecol.reshape(1,tmp.rows*tmp.cols).clone());
		onecol.release();
	}
	return vPatch.t();
}
//==============================================================================
/** Gets the feature around the current pixel as a row-matrix.
 */
template <class F>
inline cv::Mat LabelPatchFeature<F>::featRowHack(const F *features) const{
	cv::Mat vPatch;
	assert((this->featW_-2)%3==0 && (this->featH_-2)%3==0);
	int stepW = (this->featW_-2)/3;
	int stepH = (this->featH_-2)/3;
	std::vector<cv::Point> points;
	points.push_back(cv::Point(stepW,stepH));
	points.push_back(cv::Point(stepW,this->featH_-stepH));
	points.push_back(cv::Point(this->featW_-stepW,stepH));
	points.push_back(cv::Point(this->featW_-stepW,this->featH_-stepH));
	cv::Rect roi(this->point_.x-(this->featW_/2),this->point_.y-(this->featH_/2),1,1);
	std::vector<IplImage*> tmpImg = features->vImg(this->imIndex_);
	for(unsigned int c=0;c<tmpImg.size();++c){
		cv::Mat mat = cv::Mat(tmpImg[c]);
		for(std::vector<cv::Point>::iterator pt=points.begin();pt!=points.end();++pt){
			cv::Mat oneval = cv::Mat::ones(1,1,CV_32FC1)*mat.at<float>(*pt);
			vPatch.push_back(oneval.clone());
			oneval.release();
		}
	}
	return vPatch.t();
}
//==============================================================================
/** Gets the feature around the current pixel as a vector of row-matrix.
 */
template <class F>
inline std::vector<cv::Mat> LabelPatchFeature<F>::vecFeatRow(const F *features) \
const{
	std::vector<IplImage*> tmpImg = features->vImg(this->imIndex_);
	std::vector<cv::Mat> vPatch(tmpImg.size());
	cv::Rect roi(this->point_.x-(this->featW_/2),this->point_.y-(this->featH_/2),\
		this->featW_,this->featH_);
	for(unsigned int c=0;c<tmpImg.size();++c){
		CvMat tmp;
		cvGetSubRect(tmpImg[c],&tmp,roi);
		cv::Mat onecol(&tmp,true);
		vPatch[c].push_back(onecol.reshape(1,1).clone());
		onecol.release();
	}
	return vPatch;
}
//==============================================================================
/** Gets the feature around the current pixel as a matrix for the give
 * channel value.
 */
template <class F>
inline CvMat* LabelPatchFeature<F>::feat(const F *features,int channel) const{
	std::vector<IplImage*> tmpImg = features->vImg(this->imIndex_);
	cv::Rect roi(this->point_.x-(this->featW_/2),this->point_.y-(this->featH_/2),\
		this->featW_,this->featH_);
	assert(tmpImg.size()>channel);
	CvMat tmp;
	cvGetSubRect(tmpImg[channel],&tmp,roi);
	CvMat* vPatch = cvCloneMat(&tmp);
	return vPatch;
}
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
template <class T,class F>
class StructuredPatch{
	public:
		StructuredPatch(CvRNG *pRNG,unsigned patchW,unsigned patchH,unsigned noCls,\
			unsigned labW,unsigned labH,unsigned trainSize,unsigned noPatches,\
			unsigned consideredCls,bool balance,unsigned step);
		virtual ~StructuredPatch();
		//----------------------------------------------------------------------
		/** Get image labels. Try loading it, if not there compute it (very slow).
		 */
		cv::Mat* loadLabels(const std::string &justname,const std::string &ext,\
			const std::string &term,const std::string &labpath,const std::map\
			<cv::Vec3b,unsigned,vec3bCompare> &classinfo,const std::string \
			&featpath);
		/** Resize patches to the number of patches to be used for training.
		 */
		void resizePatches();
		/** Gets the number of feature channels.
		 */
		unsigned getPatchChannels() const;
		/** Get the total number of training images.
		 */
		unsigned noImages() const{return this->features_->size_;}
		/** Gets the label-patch size.
		 */
		unsigned getLabelSize() const{
			return (this->labH_)*(this->labW_);
		}
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Resets the class members to add new patches.
		 */
		virtual void reset();
		/** Randomly picks a subset of the images names to be used for training.
		 */
		virtual void pickRandomNames(const std::vector<std::string> &filenames);
		/** Extracts the feature patches but also the label patches.
		 * imgpath    -- path to the images
		 * labpath    -- path to labels
		 * featpath   -- path to features
		 * vFilenames -- vector of image names
		 * classinfo  -- mapping from pixel color to label ID
		 * labH       -- label patch height
		 * labW       -- label patch width
		 */
		virtual void extractPatches(const std::string &imgpath,const std::string &labpath,\
			const std::string &featpath,const std::vector<std::string> &vFilenames,\
			const std::map<cv::Vec3b,unsigned,vec3bCompare> &classinfo,\
			const std::string &term,const std::string &ext);
		/** Computes features if not there for loading.
		 */
		virtual void extractFeatures(const cv::Mat *labimg,IplImage *img,\
			const std::string &path2feat,bool showWhere=false);
		/** Saves the labels and the image features --- for each image make one file.
		 */
		virtual void savePatches(const std::string &path2feat,unsigned pos);
		/** Loads the labels and the image features --- 1 file per image.
		 */
		virtual void loadPatches(const std::string &path2feat,bool showWhere=false);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned noPatches() const {return this->noPatches_;}
		bool balance() const {return this->balance_;}
		F *features() const {return this->features_;}
		CvRNG *cvRNG() const {return this->cvRNG_;};
		unsigned step() const {return this->step_;}
		unsigned trainingSize() const {return this->trainingSize_;}
		unsigned labW() const {return this->labW_;}
		unsigned labH() const {return this->labH_;}
		unsigned featW() const {return this->featW_;}
		unsigned featH() const {return this->featH_;}
		unsigned noCls() const {return this->noCls_;}
		std::vector<std::string> imName() const {return this->imName_;}
		std::string imName(unsigned pos) const {return this->imName_[pos];}
		std::vector<std::vector<const T*> > patches() const {return this->patches_;}
		std::vector<const T*> patches(unsigned cls) const {return this->patches_[cls];}
		const T* patches(unsigned cls,unsigned pt) const {return this->patches_[cls][pt];}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void noPatches(unsigned noPatches){this->noPatches_ = noPatches;}
		void balance(bool balance){this->balance_ = balance;}
		void features(F *features){
			if(this->features_){
				delete this->features_;
				this->features_ = NULL;
			}
			this->features_ = new F(*features);
		}
		void cvRNG(CvRNG *cvRNG){this->cvRNG_ = cvRNG;};
		void step(unsigned step){this->step_ = step;}
		void trainingSize(unsigned trainingSize){this->trainingSize_ = trainingSize;}
		void noCls(unsigned noCls){this->noCls_ = noCls;}
		void labW(unsigned labW){this->labW_ = labW;}
		void labH(unsigned labH){this->labH_ = labH;}
		void featW(unsigned featW){this->featW_ = featW;}
		void featH(unsigned featH){this->featH_ = featH;}
		void imName(const std::vector<std::string> &imName){
			this->imName_.clear();
			this->imName_ = imName;
		}
		void imName(unsigned pos,const std::string &imName){
			this->imName_[pos] = imName;
		}
		void patches(const std::vector<std::vector<const T*> > &patches){
			for(unsigned c=0;c<this->patches_.size();++c){
				// created with: push_back(new T)
				for(unsigned s=0;s<this->patches_[c].size();++s){
					delete this->patches_[c][s];
				}
				this->patches_[c].clear();
			}
			this->patches_.clear();
			for(typename std::vector<std::vector<const T*> >::const_iterator c=\
			patches.begin();c!=patches.end();++c){
				std::vector<const T*> tmp;
				for(typename std::vector<const T*>::const_iterator s=c->begin();\
				s!=c->end();++s){
					tmp.push_back(new T(*(*s)));
				}
				this->patches_.push_back(tmp);
			}
		}
		void patches(unsigned cls,const std::vector<const T*> &patches){
			if(!this->patches_[cls].empty()){
				// created with: psuh_back(new T)
				for(unsigned s=0;s<this->patches_[cls].size();++s){
					delete this->patches_[cls][s];
				}
				this->patches_[cls].clear();
			}
			for(typename std::vector<const T*>::const_iterator s=patches.begin();\
			s!=patches.end();++s){
				this->patches_[cls].push_back(new T(*(*s)));
			}
		}
		void patches(unsigned cls,unsigned pt,const T *patches){
			if(!this->patches_[cls][pt]){
				// created with: new T
				delete this->patches_[cls][pt];
			}
			this->patches_[cls][pt] = new T(*patches);
		};
		//----------------------------------------------------------------------
	protected:
		/** @var step_
		 * Lattice step for picking up features.
		 */
		unsigned step_;
		/** @var trainingSize_
		 * Number of images to sample for training one tree.
		 */
		unsigned trainingSize_;
		/** @var imName_
		 * The original image name.
		 */
		std::vector<std::string> imName_;
		/** @var patches_
		 * The vector of LabelPatchFeatures per classes.
		 */
		std::vector<std::vector<const T*> > patches_;
		/** @var noCls_
		 * Number of classes.
		 */
		unsigned noCls_;
		/** @var labW_
		 * Label patch width (redundant but how else?).
		 */
		unsigned labW_;
		/** @var labH_
		 * Label patch height (redundant but how else?).
		 */
		unsigned labH_;
		/** @var cvRNG_
		 * For picking random stuff.
		 */
		CvRNG *cvRNG_;
		/** @var featW_
		 * The width of the feature patch.
		 */
		unsigned featW_;
		/** @var featH_
		 * The height of the feature patch.
		 */
		unsigned featH_;
		/** @var consideredCls_
		 * Number of classes we consider.
		 */
		unsigned consideredCls_;
		/** @var feature_
		 * An instance of the class feature.
		 */
		F* features_;
		/** @var balance_
		 * To balance the class rations or not.
		 */
		bool balance_;
		/** @var noPatches_
		 * Regardless of the number of features we can have a desired number of
		 * patches for training.
		 */
		unsigned noPatches_;
	private:
		DISALLOW_COPY_AND_ASSIGN(StructuredPatch);
};
//==============================================================================
#endif /* StructuredPatch_H_ */
#include "StructuredPatch.cpp"
