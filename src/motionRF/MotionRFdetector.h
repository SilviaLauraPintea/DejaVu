/* MotionRFdetector.h
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONRFDETECTOR_H_
#define MOTIONRFDETECTOR_H_
#include <StructuredRFdetector.h>
#include <MotionTree.h>
#include <MotionRF.h>
//==============================================================================
/** Class performing the test-time prediction.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
class MotionRFdetector:public StructuredRFdetector<L,M,T,F,N,U>{
	public:
		MotionRFdetector(MotionRF<L,M,T,F,N,U> *pRF,int w,int h,\
		unsigned cls,unsigned labW,unsigned labH,unsigned motionW,unsigned \
		motionH,typename Puzzle<PuzzlePatch>::METHOD method,\
		unsigned step,typename MotionTree<M,T,F,N,U>::ENTROPY entropy,\
		bool usederivatives=true,bool hogORsift=true,unsigned pttype=0,const \
		std::string &path2results="",unsigned maxsize=160):StructuredRFdetector\
		<L,M,T,F,N,U>(pRF,w,h,cls,labW,labH,method,step),motionW_(motionW),\
		motionH_(motionH),entropy_(entropy),usederivatives_(usederivatives),\
		hogORsift_(hogORsift),pttype_(pttype),path2results_(path2results){
			this->forest_   = dynamic_cast<MotionRF<L,M,T,F,N,U>* >(pRF);
			this->flip_     = false;
			this->overallW_ = std::max((int)this->motionW_,this->width_);
			this->overallH_ = std::max((int)this->motionH_,this->height_);
			this->maxsize_  = maxsize;
		};
		virtual ~MotionRFdetector(){};
		//----------------------------------------------------------------------
		/** Extracts or loads the test features for the current test image.
		 */
		std::vector<std::vector<IplImage*> > getFeatures(const std::string &path2img,\
			const std::vector<std::string> &path2feat,const std::vector<float> &pyr,\
			std::vector<std::vector<const T*> > &patches,cv::Size &origsize,\
			F *features,bool showWhere=false) const;
		/** Scales the image at a number of sizes and it labels each scale [?].
		 */
		std::vector<std::vector<const T*> > justfeatures(const std::string &imname,\
			const std::string &path2img,const std::string &ext,const std::string \
			&path2feat,const std::vector<float> &pyramid,F* features,cv::Size &imsize);
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Gets an input image and returns a detection image (pixel labels by RF regression).
		 * Given a set of predicted leafs for current pixel, get the final label:
		 * Simple: [1] Just get the most voted pixel label per position.
		 * Puzzle: [2] Optimized the patch selection label \cite{kontschider}.
		 */
		virtual void detectColor(const F *features,cv::Mat &motionDetect,\
			cv::Mat &arrowsDetect,cv::Mat &appearDetect,const std::vector\
			<const T*> &patches,const cv::Size &imsize,const std::string &imname,\
			const std::string &path2model,unsigned offset,bool display=false);
		/** Scales the image at a number of sizes and it labels each scale [?].
		 */
		virtual void detectPyramid(const std::string &imname,const std::string \
			&path2img,const std::string &path2feat,const std::string &ext,\
			const std::vector<float> &pyramid,std::vector<cv::Mat> &vMotionDetect,\
			std::vector<cv::Mat> &vArrowsDetect,std::vector<cv::Mat> &vAppearDetect,\
			const std::string &path2model,unsigned offset);
		/** Loads the test features from file for the current test image.
		 */
		virtual std::vector<std::vector<IplImage*> > loadFeatures(const std::vector\
			<std::string> &path2feat,std::vector<std::vector<const T*> > &patches,\
			bool showWhere) const;
		/** Loads the test features from file for the current test image.
		 */
		virtual void saveFeatures(const std::vector<std::string> &path2feat,\
			const std::vector<std::vector<IplImage*> > &vImg,const std::vector\
			<std::vector<const T*> > &patches) const;
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned motionW() const {return this->motionW_;}
		unsigned motionH() const {return this->motionH_;}
		typename MotionTree<M,T,F,N,U>::ENTROPY entropy() const {return this->entropy_;}
		bool usederivatives() const {return this->usederivatives_;}
		bool hogORsift() const {return this->hogORsift_;}
		unsigned pttype() const {return this->pttype_;}
		std::string path2resultS() const {return this->path2results_;}
		unsigned maxsize() const {return this->maxsize_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void motionW(unsigned motionW){this->motionW_ = motionW;}
		void motionH(unsigned motionH){this->motionH_ = motionH;}
		void entropy(typename MotionTree<M,T,F,N,U>::ENTROPY entropy){
			this->entropy_ = entropy;
		}
		void usederivatives(bool usederivatives){this->usederivatives_ = usederivatives;}
		void hogORsift(bool hogORsift){this->hogORsift_ = hogORsift;}
		void pttype(unsigned pttype){this->pttype_ = pttype;}
		void path2resultS(const std::string &path2results){
			this->path2results_ = path2results;
		}
		void maxsize(unsigned maxsize){this->maxsize_ = maxsize;}
		//----------------------------------------------------------------------
	private:
		/** @var motionW_
		 * Width of the motion patch.
		 */
		unsigned motionW_;
		/** @var motionH_
		 * Width of the motion patch.
		 */
		unsigned motionH_;
		/** @var overallW_
		 * Maximum Width of the patch.
		 */
		unsigned overallW_;
		/** @var overallH_
		 * Maximum Height of the patch.
		 */
		unsigned overallH_;
		/** @var entropy_;
		 * Entropy method used so we know what to choose for prediction
		 */
		typename MotionTree<M,T,F,N,U>::ENTROPY entropy_;
		/** @var flip_
		 * If we should also flip the predictions.
		 */
		bool flip_;
		/** @var usederivatives_
		 * If the predictions are derivatives or just flows.
		 */
		bool usederivatives_;
		/** @var hogORsift_
		 * HOG - 1 and SIFT - 0.
		 */
		bool hogORsift_;
		/** @var pttype_
		 * Points type for patches.
		 */
		unsigned pttype_;
		/** @var path2results_
		 * Path to where prediction results are stored.
		 */
		std::string path2results_;
		/** @var maxsize_
		 * The maximum image size.
		 */
		unsigned maxsize_;
	private:
		DISALLOW_COPY_AND_ASSIGN(MotionRFdetector);
};
//==============================================================================
#endif /* MOTIONRFDETECTOR_H_ */
