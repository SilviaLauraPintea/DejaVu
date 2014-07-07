/* Motion.h
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONPATCH_H_
#define MOTIONPATCH_H_
#pragma once
#include <Auxiliary.h>
#include <MotionPatchFeature.h>
//==============================================================================
//==============================================================================
//==============================================================================
template <class T,class F>
class MotionPatch:public StructuredPatch<T,F>{
	public:
		/** On what do we evaluate the entropy for pre-computing histos.
		 */
		enum HistoType {CENTER,RANDOM,CENTER_RANDOM,MEAN_DIFF,APPROX_MAGNI_KERNEL,\
			APPROX_ANGLE_KERNEL};
		enum Points {HARRIS, CANNY, DENSE};
		enum Algorithm {Farneback,LucasKanade,HornSchunck, Simple};
		MotionPatch(CvRNG* pRNG,unsigned patchW,unsigned patchH,\
		unsigned noCls,unsigned labW,unsigned labH,unsigned trainSize,unsigned \
		noPatches,unsigned consideredCls,unsigned balance,unsigned step,\
		unsigned motionW=0,unsigned motionH=0,bool warpping=false,\
		bool ofThresh=true,unsigned histotype=4,const std::vector<float> &sigmas=\
		std::vector<float>(4,1.0),unsigned bins=25,bool multicls=false,\
		bool usederivatives=true,bool hogORsift=true,unsigned pttype=0,unsigned maxsize=160):\
		StructuredPatch<T,F>(pRNG,patchW,patchH,noCls,labW,labH,trainSize,noPatches,\
		consideredCls,balance,step),motionW_(motionW),motionH_(motionH),bins_(bins),\
		multiclass_(multicls),usederivatives_(usederivatives),hogORsift_(hogORsift),\
		pttype_(pttype){
			// initialize the transitions:
			this->sintel_     = false;
			this->display_    = false;
			this->threshold_  = ofThresh;
			this->storefeat_  = false;
			this->relativeOF_ = warpping;
			this->maximsize_  = maxsize;
			this->sigmas_     = sigmas;
			this->histotype_  = static_cast<HistoType>(histotype);
			this->algo_       = MotionPatch<T,F>::Simple;
			this->features_->usederivatives(usederivatives_);
		};
		virtual ~MotionPatch();
		//----------------------------------------------------------------------
		/** Finds interest points and warps the second image to the first image.
		 */
		int warpOpenCV(cv::Mat &curr,cv::Mat &next);
		/** Randomly picks a subset of the images names to be used for training -- pairs
		 * of 2 images for OF.
		 */
		void pickRandomNames(const std::string &featpath,const std::vector\
			<std::string> &folders,const std::string &ext,const std::string &imgpath,\
			std::vector<unsigned> &shuffle);
		/** Find threshold by cutting the histogram at 0.90.
		 */
		static cv::Mat findThresholdAngle(const cv::Mat &valuesX,const cv::Mat \
			&valuesY,float &minTr,float &maxTr);
		/** Find threshold by cutting the histogram at 0.90.
		 */
		static void findThreshold(const cv::Mat &values,float &minTr,float &maxTr);
		/** Load or extract the optical flow vector from two pairs of consecutive images,
		 * and then take the difference of their OFs.
		 */
		int extractMotionRelative(const std::vector<std::string> &tuples,\
			Algorithm algo,const std::string &featpath,unsigned offset,bool save=false);
		/** Compute histograms of angle\slash magnitude.
		 */
		void computeHistograms(unsigned pos);
		/** Load or extract the optical flow vector from two consecutive images.
		 */
		void extractMotionAbsolute(const std::vector<std::string> &tuples,\
			Algorithm algo,const std::string &featpath,unsigned offset,bool save=false);
		/** Showing OF vectors (for check only).
		 */
		static cv::Mat showOF(const cv::Mat &velX,const cv::Mat &velY,const \
			cv::Mat &image,unsigned step=1,bool display=false,const std::string \
			&winname="Of",const cv::Rect &roi = cv::Rect(0,0,0,0));
		/** Showing OF derivatives back as vectors in the image (for check only).
		 */
		static cv::Mat showOFderi(const cv::Mat &velXX,const cv::Mat &velXY,\
			const cv::Mat &velYX,const cv::Mat &velYY,const cv::Mat &image,\
			unsigned step=1,bool display=false,const std::string &winname="OF deri",\
			const cv::Rect &roi=cv::Rect(0,0,0,0));
		/** Extracts the feature patches but also the label patches.
		 */
		void extractPatchesOF(const std::string &imgpath,std::string &featpath,\
			const std::vector<std::string> &vFilenames,const std::string &ext,\
			const MotionPatch<T,F>::Algorithm &algo,bool justimages=false);
		/** Just extracts OF for a pair of images using a given algorithm (hardcoded
		 * parameters).
		 */
		static cv::Mat justFlow(cv::Mat &current,cv::Mat &next,Algorithm algo,\
			unsigned motionSz,float maximsize,const std::string &imName,\
			const std::string &featpath,bool sintel,bool store);
		/** Finds interest points and warps the second image to the first image.
		 */
		int warpSecond2First(cv::Mat &curr,cv::Mat &next);
		/** Finds matches between a set of points
		 */
		/** Finds matches between a set of points
		 */
		void findmatches(const cv::Mat &points1,const cv::Mat &points2,\
			const cv::Mat &img1,const cv::Mat &img2,double minDist,double maxDist,\
			cv::Mat &outpoints1,cv::Mat &outpoints2);
		/** My own little sweet RANSAC.
		 */
		cv::Mat getRansacAffineTransform(const cv::Mat &points1,const cv::Mat &points2,\
			const cv::Mat &img1,const cv::Mat &img2,double limit,int &isgood);
		/** Gets the flow derivatives and merges them into a 4 channels image: xdx, xdy,
		 * ydx, ydy.
		 */
		static cv::Mat getFlowDerivatives(const cv::Mat &flow);
		/** Extract Harris interest points.
		 */
		static std::vector<cv::KeyPoint> getKeyPoints(const cv::Mat &image,\
			unsigned step,unsigned width,unsigned height,unsigned pttype,\
			bool display=false,float threshold=1e-5);
		/** Computes opponent channels.
		 */
		static std::vector<cv::Mat> opponent(const cv::Mat &mat);
		/** Get the histogram of angles.
		 */
		void getAngleHisto();
		/** Get the histogram of angles for the flow.
		 */
		void getAngleHistoFlowKernels();
		/** Get the histogram of angles.
		 */
		void getAngleHistoDerivativesKernels();
		/** Get the histogram of angles for the flow.
		 */
		void getAngleHistoFlowHard();
		/** Get the histogram of angles.
		 */
		void getAngleHistoDerivativesHard();
		/** Get the histogram of magnitudes.
		 */
		void getMagniHisto();
		/** Get the histogram of magnitudes from flow derivatives.
		 */
		void getMagniHistoDerivativesKernels();
		/** Get the histogram of magnitudes from flow only.
		 */
		void getMagniHistoFlowKernels();
		/** Get the histogram of magnitudes from flow derivatives.
		 */
		void getMagniHistoDerivativesHard();
		/** Get the histogram of magnitudes from flow only.
		 */
		void getMagniHistoFlowHard();
		/** Compute histograms of angle\slash magnitude.
		 */
		void computeHistograms();
		/** Warp the image with the flow in 10 steps.
 		 */
		static void warpInter(const cv::Mat &motionX,const cv::Mat &motionY,\
			const cv::Mat &origin,unsigned offset,bool display);
		/** Just reads the sintel flow for local files.
		 */
		static cv::Mat sintelFlow(const std::string &featpath,unsigned maxisize,\
			const std::string &imName,bool store);
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Resets the class members to add new patches.
		 */
		virtual void reset();
		/** Saves the labels and the image features --- for each image make one file.
		 */
		virtual void savePatches(const std::string &path2feat,unsigned pos);
		/** Loads the labels and the image features --- 1 file per image.
		 */
		virtual void loadPatches(const std::string &path2feat,bool showWhere=false);
		/** Computes features if not there for loading.
		 */
		virtual void extractFeatures(IplImage *img,const std::string &path2feat,\
			bool showWhere=false);
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
		virtual void extractPatches(const std::string &imgpath,\
			const std::string &labpath,std::string &featpath,const std::vector\
			<std::string> &vFilenames,const std::map<cv::Vec3b,unsigned,vec3bCompare> \
			&classinfo,const std::string &labTerm,const std::string &ext,\
			bool justimages=false,bool extractHisto=true);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned motionW() const {return this->motionW_;}
		unsigned motionH() const {return this->motionH_;}
		float maximsize() const {return this->maximsize_;}
		std::vector<std::vector<std::string> > imagePairs() const{
			return this->imagePairs_;
		}
		bool relativeOF() const {return this->relativeOF_;}
		bool display() const {return this->display_;}
		bool storefeat() const {return this->storefeat_;}
		unsigned bins() const {return this->bins_;}
		bool usederivatives() const {return this->usederivatives_;}
		std::vector<float> sigmas() const {return this->sigmas_;}
		bool hogORsift() const {return this->hogORsift_;}
		unsigned pttype() const {return this->pttype_;}
		MotionPatch<T,F>::Algorithm algo() const{return this->algo_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void imagePairs(const std::vector<std::vector<std::string> > &imagePairs){
			this->imagePairs_ = imagePairs;
		}
		void motionW(unsigned motionW){this->motionW_ = motionW;}
		void motionH(unsigned motionH){this->motionH_ = motionH;}
		void relativeOF(bool relativeOF){this->relativeOF_ = relativeOF;}
		void display(bool display){this->display_ = display;}
		void maximsize(float maximsize){this->maximsize_ = maximsize;}
		void storefeat(bool storefeat){this->storefeat_ = storefeat;}
		void bins(unsigned bins){this->bins_ = bins;}
		void usederivatives(bool usederivatives){
			this->usederivatives_ = usederivatives;
		}
		void sigmas(const std::vector<float> &sigmas){this->sigmas_ = sigmas;}
		void hogORsift(bool hogORsift){this->hogORsift_ = hogORsift;}
		void pttype(unsigned pttype){this->pttype_ = pttype;}
		void algo(MotionPatch<T,F>::Algorithm algo){this->algo_ = algo;}
		//----------------------------------------------------------------------
	private:
		/** @var multiclass_
		 * If multiclass, make sure there are training samples from all classes.
		 */
		bool multiclass_;
		/** @var sigmas_
		 * The sigmas for the histograms in KDE.
		 */
		std::vector<float> sigmas_;
		/** @var histotype_
		 * The type of the histograms to pre-compute.
		 */
		HistoType histotype_;
		/** @var maximsize_
		 * For big data, do not store features.
		 */
		float maximsize_;
		/** @var storefeat_
		 * For big data, do not store features.
		 */
		bool storefeat_;
		/** @var threshold_
		 * If we want to threshold the OF arrows
		 */
		bool threshold_;
		/** @var display_
		 * To display images or not.
		 */
		bool display_;
		/** @var motionW_
		 * The width of the motion patch.
		 */
		unsigned motionW_;
		/** @var motionH_
		 * The height of the motion patch.
		 */
		unsigned motionH_;
		/** @var imagePairs_
		 * The vector containing the pairs of images for OF.
		 */
		std::vector<std::vector<std::string> > imagePairs_;
		/** @var relativeOF_
		 * If the OF should be relative or absolute.
		 */
		bool relativeOF_;
		/** @var bins_
		 * The number of bins in the precomputed histograms.
		 */
		unsigned bins_;
		/** @var usederivatives_
		 * If we predict the flow derivatives or the actual flow.
		 */
		bool usederivatives_;
		/** @var hogORsift_
		 * Hog - 1 and SIFT - 0.
		 */
		bool hogORsift_;
		/** @var pttype_
		 * The point types: Harris or Canny.
		 */
		unsigned pttype_;
		/** @var algo_
		 * The algorithm to be used for extracting OF.
		 */
		MotionPatch<T,F>::Algorithm algo_;
		/** @var sintel_
		 * If we have Sintel features then we need to load them and use them.
		 */
		bool sintel_;
	private:
		DISALLOW_COPY_AND_ASSIGN(MotionPatch);
};
//==============================================================================
#endif /* MOTIONPATCH_H_ */
