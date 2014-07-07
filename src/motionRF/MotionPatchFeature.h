/* MotionPatchFeatures.h
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONPATCHFEATURE_H_
#define MOTIONPATCHFEATURE_H_
#include <Auxiliary.h>
#include <StructuredPatch.h>
//==============================================================================
/** To keep inside all the vectors we need.
 */
class FeaturesMotion:public Features{
	public:
		FeaturesMotion(bool usederivatives=true):usederivatives_(usederivatives){};
		FeaturesMotion(const std::vector<cv::Mat> &velXX,const std::vector\
		<cv::Mat> &velXY,const std::vector<cv::Mat> &velYX,const std::vector\
		<cv::Mat> &velYY,const std::vector<cv::Mat> &labelImg,const std::vector\
		<std::vector<IplImage*> > &vImg,const std::vector<cv::Mat> &images,\
		const std::vector<std::vector<cv::Mat> > &histo,bool usederivatives):\
		Features(labelImg,vImg),usederivatives_(usederivatives){
			this->velXX(velXX);
			this->velXY(velXY);
			this->velYX(velYX);
			this->velYY(velYY);
			this->images(images);
			this->histo(histo);
		}
		virtual ~FeaturesMotion(){
			assert(this->velXX_.size()==this->velYY_.size());
			if(this->usederivatives_){
				assert(this->velXX_.size()==this->velXY_.size());
				assert(this->velYX_.size()==this->velYY_.size());
			}else{
				assert(this->velYX_.empty());
				assert(this->velXY_.empty());
			}
			for(unsigned s=0;s<this->images_.size();++s){
				if(!this->velXX_.empty()){
					this->velXX_[s].release();
					this->velYY_[s].release();
					if(this->usederivatives_){
						this->velYX_[s].release();
						this->velXY_[s].release();
					}
					if(!this->histo_.empty()){
						for(unsigned b=0;b<this->histo_[s].size();++b){
							this->histo_[s][b].release();
						}
						this->histo_[s].clear();
					}
				}
				this->images_[s].release();
			}
			this->velXX_.clear();
			this->velXY_.clear();
			this->velYX_.clear();
			this->velYY_.clear();
			this->images_.clear();
			this->histo_.clear();
		}
		//----------------------------------------------------------------------
		/** Flips the motion patch.
		 */
		FeaturesMotion* flip(bool usederivatives) const{
			FeaturesMotion *flip = new FeaturesMotion(usederivatives);
			// [0] Flip the motion labels
			if(this->usederivatives_){
				std::vector<cv::Mat>::const_iterator xy=this->velXY_.begin();
				std::vector<cv::Mat>::const_iterator yx=this->velYX_.begin();
				std::vector<cv::Mat>::const_iterator yy=this->velYY_.begin();
				for(std::vector<cv::Mat>::const_iterator xx=this->velXX_.begin();\
				xx!=this->velXX_.end(),xy!=this->velXY_.end(),yx!=this->velYX_.end(),\
				yy!=this->velYY_.end();++xx,++xy,++yx,++yy){
					cv::Mat flipXX; cv::flip((*xx),flipXX,1);
					flip->velXX_.push_back(flipXX.clone());
					flipXX.release();
					cv::Mat flipXY; cv::flip((*xy),flipXY,1);
					flip->velXY_.push_back(flipXY.clone());
					flipXY.release();
					cv::Mat flipYX; cv::flip((*yx),flipYX,1);
					flip->velYX_.push_back(flipYX.clone());
					flipYX.release();
					cv::Mat flipYY; cv::flip((*yy),flipYY,1);
					flip->velYY_.push_back(flipYY.clone());
					flipYY.release();
				}
			}else{
				std::vector<cv::Mat>::const_iterator yy=this->velYY_.begin();
				for(std::vector<cv::Mat>::const_iterator xx=this->velXX_.begin();\
				xx!=this->velXX_.end(),yy!=this->velYY_.end();++xx,++yy){
					cv::Mat flipXX; cv::flip((*xx),flipXX,1);
					flip->velXX_.push_back(flipXX.clone());
					flipXX.release();
					cv::Mat flipYY; cv::flip((*yy),flipYY,1);
					flip->velYY_.push_back(flipYY.clone());
					flipYY.release();
				}
			}
			// [1] Flip the images if any
			for(std::vector<cv::Mat>::const_iterator i=this->images_.begin();\
			i!=this->images_.end();++i){
				cv::Mat flipI; cv::flip((*i),flipI,1);
				flip->images_.push_back(flipI.clone());
				flipI.release();
			}
			// [1] Flip the features if any (don't care for static labels)
			for(std::vector<std::vector<IplImage*> >::const_iterator feat=\
			this->vImg_.begin();feat!=this->vImg_.end();++feat){
				std::vector<IplImage*> tmp;
				for(std::vector<IplImage*>::const_iterator f=feat->begin();f!=\
				feat->end();++f){
					IplImage* flipF = cvCreateImage(cvSize((*f)->width,\
						(*f)->height),(*f)->depth,(*f)->nChannels);
					cvFlip((*f),flipF,1);
					tmp.push_back(cvCloneImage(flipF));
				}
				flip->vImg_.push_back(tmp);
			}
			return flip;
		}
		/** Adds a matrix to the vector of x-derivatives of velocity matrices on X.
		 */
		void push_backVelXX(const cv::Mat &velXX){this->velXX_.push_back(velXX.clone());}
		/** Adds a matrix to the vector of y-derivatives of velocity matrices on X.
		 */
		void push_backVelXY(const cv::Mat &velXY){this->velXY_.push_back(velXY.clone());}
		/** Adds a matrix to the vector of y-derivatives of velocity matrices on Y.
		 */
		void push_backVelYX(const cv::Mat &velYX){this->velYX_.push_back(velYX.clone());}
		/** Adds a matrix to the vector of y-derivatives of velocity matrices on Y.
		 */
		void push_backVelYY(const cv::Mat &velYY){this->velYY_.push_back(velYY.clone());}
		/** Adds a matrix to the vector of images.
		 */
		void push_backImages(const cv::Mat &images){this->images_.push_back(images.clone());}
		/** Adds a matrix to the histogram of matrices for angle.
		 */
		void push_backHisto(const std::vector<cv::Mat> &hist){
			this->histo_.push_back(hist);
		}
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		virtual unsigned size() const {
			return std::max(this->images_.size(),std::max(this->velXX_.size(),\
				std::max(this->vImg_.size(),this->labelImg_.size())));
		}
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		std::vector<cv::Mat> velXX() const {return this->velXX_;}
		std::vector<cv::Mat> velXY() const {return this->velXY_;}
		std::vector<cv::Mat> velYX() const {return this->velYX_;}
		std::vector<cv::Mat> velYY() const {return this->velYY_;}
		cv::Mat velXX(unsigned pos) const {return this->velXX_[pos];}
		cv::Mat velXY(unsigned pos) const {return this->velXY_[pos];}
		cv::Mat velYX(unsigned pos) const {return this->velYX_[pos];}
		cv::Mat velYY(unsigned pos) const {return this->velYY_[pos];}
		std::vector<cv::Mat> images() const {return this->images_;}
		std::vector<std::vector<cv::Mat> > histo() const {return this->histo_;}
		std::vector<cv::Mat> histo(unsigned pos) const {return this->histo_[pos];}
		cv::Mat images(unsigned pos) const {return this->images_[pos];}
		std::vector<float> histinfo() const {return this->histinfo_;}
		bool usederivatives() const {return this->usederivatives_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void usederivatives(bool usederivatives){this->usederivatives_ = usederivatives;}
		void velXX(const std::vector<cv::Mat> &velXX){
			for(unsigned s=0;s<this->velXX_.size();++s){
				// Created with: new cv::Mat
				this->velXX_[s].release();
			}
			this->velXX_.clear();
			this->velXX_.resize(velXX.size(),cv::Mat());
			for(unsigned s=0;s<velXX.size();++s){
				velXX[s].copyTo(this->velXX_[s]);
			}
		}
		void velXX(unsigned pos,const cv::Mat velXX){
			this->velXX_[pos].release();
			velXX.copyTo(this->velXX_[pos]);
		}
		void velXY(const std::vector<cv::Mat> &velXY){
			for(unsigned s=0;s<this->velXY_.size();++s){
				// created with: new cv::Mat
				this->velXY_[s].release();
			}
			this->velXY_.clear();
			this->velXY_.resize(velXY.size());
			for(unsigned s=0;s<velXY.size();++s){
				velXY[s].copyTo(this->velXY_[s]);
			}
		}
		void velXY(unsigned pos,const cv::Mat velXY){
			this->velXY_[pos].release();
			velXY.copyTo(this->velXY_[pos]);
		}
		void velYX(const std::vector<cv::Mat> &velYX){
			for(unsigned s=0;s<this->velYX_.size();++s){
				// Created with: new cv::Mat
				this->velYX_[s].release();
			}
			this->velYX_.clear();
			this->velYX_.resize(velYX.size(),cv::Mat());
			for(unsigned s=0;s<velYX.size();++s){
				velYX[s].copyTo(this->velYX_[s]);
			}
		}
		void velYX(unsigned pos,const cv::Mat velYX){
			this->velYX_[pos].release();
			velYX.copyTo(this->velYX_[pos]);
		}
		void velYY(const std::vector<cv::Mat> &velYY){
			for(unsigned s=0;s<this->velYY_.size();++s){
				// created with: new cv::Mat
				this->velYY_[s].release();
			}
			this->velYY_.clear();
			this->velYY_.resize(velYY.size());
			for(unsigned s=0;s<velYY.size();++s){
				velYY[s].copyTo(this->velYY_[s]);
			}
		}
		void velYY(unsigned pos,const cv::Mat velYY){
			this->velYY_[pos].release();
			velYY.copyTo(this->velYY_[pos]);
		}
		void images(const std::vector<cv::Mat> &images){
			for(unsigned s=0;s<this->images_.size();++s){
				// Created with: new cv::Mat
				this->images_[s].release();
			}
			this->images_.clear();
			this->images_.resize(images.size());
			for(unsigned s=0;s<images.size();++s){
				images[s].copyTo(this->images_[s]);
			}
		}
		void images(unsigned pos,const cv::Mat images){
			this->images_[pos].release();
			images.copyTo(this->images_[pos]);
		}
		void histinfo(std::vector<float> &histinfo){
			this->histinfo_ = histinfo;
		}
		void histo(const std::vector<std::vector<cv::Mat> > &histo){
			for(unsigned s=0;s<this->histo_.size();++s){
				for(unsigned b=0;b<this->histo_[s].size();++b){
					this->histo_[s][b].release();
				}
			}
			this->histo_.clear();
			this->histo_.resize(histo.size(),cv::Mat());
			for(unsigned s=0;s<this->histo_.size();++s){
				for(unsigned b=0;b<this->histo_[s].size();++b){
					histo[s][b].copyTo(this->histo_[s][b]);
				}
			}
		}
		//----------------------------------------------------------------------
	private:
		/** @var histo_
		 * Vector of histogram matrices (a bin for each dimension).
		 */
		std::vector<std::vector<cv::Mat> > histo_;
		/** @var velXX_
		 * The optical flow velocity x-derivative on the x direction.
		 */
		std::vector<cv::Mat> velXX_;
		/** @var velXY_
		 * The optical flow velocity y-derivative on the x direction.
		 */
		std::vector<cv::Mat> velXY_;
		/** @var velXY_
		 * The optical flow velocity x-derivative on the y direction.
		 */
		std::vector<cv::Mat> velYX_;
		/** @var velYY_
		 * The optical flow velocity y-derivative on the y direction.
		 */
		std::vector<cv::Mat> velYY_;
		/** @var images_
		 * The original images to see what kind of patches are put together.
		 */
		std::vector<cv::Mat> images_;
		/** @var histinfo_
		 * The info about the histograms: #bins, boundaries, step
		 */
		std::vector<float> histinfo_;
		/** @var usederivatives_
		 * If we use derivatives of flow or just the flow.
		 */
		bool usederivatives_;
	private:
		DISALLOW_COPY_AND_ASSIGN(FeaturesMotion);
};
//==============================================================================
//==============================================================================
//==============================================================================
/** Patches are always relative to corner: top-left.
 */
template <class F>
class MotionPatchFeature:public LabelPatchFeature<F>{
	public:
		enum HistoType {CENTER,RANDOM,CENTER_RANDOM,MEAN_DIFF,APPROX_MAGNI_KERNEL,\
			APPROX_ANGLE_KERNEL};
		MotionPatchFeature(unsigned featW,unsigned featH,unsigned labW,unsigned labH,\
		unsigned imIndex,const cv::Point &point,unsigned motionW=0,\
		unsigned motionH=0):LabelPatchFeature<F>(featW,featH,\
		labW,labH,imIndex,point),motionW_(motionW),motionH_(motionH){}
		virtual ~MotionPatchFeature(){}
		//----------------------------------------------------------------------
		/** Gets the flow derivatives patches on x and y around the current pixel as a vector.
		 */
		void motion(const F *feature,cv::Mat* motionXX,cv::Mat* motionXY,\
			cv::Mat* motionYX,cv::Mat* motionYY) const;
		/** Gets the flow patches on x and y around the current pixel as a vector.
		 */
		void motion(const F *feature,cv::Mat* motionX,cv::Mat* motionY) const;
		/** Gets the motion patch around the current pixel as a vector.
		 */
		cv::Mat* motion(const F *feature) const;
		/** Gets the flow derivatives patch around the current pixel as a vector.
		 */
		cv::Mat* motionDerivatives(const F *feature) const;
		/** Gets the flow-motion patch around the current pixel as a vector.
		 */
		cv::Mat* motionFlow(const F *feature) const;
		/** Gets the image patch around the current pixel as a matrix.
		 */
		cv::Mat* image(const F *feature) const;
		/** Gets the histogram at the current point position.
		 */
		std::vector<float> histoCenter(const F *feature) const;
		/** Gets the histogram at a random point in the patch.
		 */
		cv::Mat histo(const F *feature,const cv::Point pt) const;
		/** Gets the patch histogram at current point.
		 */
		std::vector<cv::Mat> histo(const F *feature) const;
		/** Gets the flow patches on x and y around the current pixel as a vector.
		 */
		void motionCenter(const F *feature,float &mX,float &my) const;
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned motionW() const {return this->motionW_;}
		unsigned motionH() const {return this->motionH_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void motionW(unsigned motionW){this->motionW_ = motionW;}
		void motionH(unsigned motionH){this->motionH_ = motionH;}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors the default ones are not good with IplImages
		 */
		MotionPatchFeature(MotionPatchFeature<F> const &rhs):LabelPatchFeature<F>(rhs){
			this->featH_   = rhs.featH();
			this->featW_   = rhs.featW();
			this->labH_    = rhs.labH();
			this->labW_    = rhs.labW();
			this->imIndex_ = rhs.imIndex();
			this->point(rhs.point());
			this->motionW_ = rhs.motionW();
			this->motionH_ = rhs.motionH();
		}
		MotionPatchFeature& operator=(MotionPatchFeature<F> const &rhs){
			if(this == &rhs) return *this;
			if(this){delete this;}
			this->featH_   = rhs.featH();
			this->featW_   = rhs.featW();
			this->labH_    = rhs.labH();
			this->labW_    = rhs.labW();
			this->imIndex_ = rhs.imIndex();
			this->point(rhs.point());
			this->motionW_ = rhs.motionW();
			this->motionH_ = rhs.motionH();
			return *this;
		}
		//----------------------------------------------------------------------
	private:
		/** @var motionW_
		 * The width of the motion patch.
		 */
		unsigned motionW_;
		/** @var motionH_
		 * The height of the motion patch.
		 */
		unsigned motionH_;
};
//==============================================================================
//==============================================================================
//==============================================================================
#endif /* MOTIONPATCHFEATURE_H_ */
