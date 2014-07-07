/** Puzzle.h
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONPUZZLE_H_
#define MOTIONPUZZLE_H_
#pragma once
#include <Puzzle.h>
#include "MotionTree.h"
#include "MotionTreeNode.h"
typedef MotionTree<MotionPatch<MotionPatchFeature<FeaturesMotion>,\
	FeaturesMotion>,MotionPatchFeature<FeaturesMotion>,FeaturesMotion,\
	MotionTreeNode<MotionLeafNode,MotionPatchFeature<FeaturesMotion> >,\
	MotionLeafNode> MotionTreeClass;
typedef MotionPatch<MotionPatchFeature<FeaturesMotion>,\
	FeaturesMotion> MotionPatchClass;
//==============================================================================
/** For storing the possible label-ings.
 */
struct MotionPuzzlePatch:public PuzzlePatch{
	public:
		MotionPuzzlePatch(){
			this->motionProb_     = 0.0;
			this->appearanceProb_ = 0.0;
			this->motion_         = cv::Mat();
			this->appearance_     = cv::Mat();
		}
		MotionPuzzlePatch(const cv::Point &center,const std::vector<unsigned> &piece,\
		const cv::Mat &motion,const cv::Mat &appearance,const std::vector\
		<cv::Mat> &histo,float logProb,float motionProb,float appProb,const \
		std::vector<float> &histinfo):PuzzlePatch(center,piece,logProb){
			this->motionProb_     = motionProb;
			this->appearanceProb_ = appProb;
			motion.copyTo(this->motion_);
			appearance.copyTo(this->appearance_);
			this->histo(histo);
			this->histinfo(histinfo);
		}
		~MotionPuzzlePatch(){
			this->motion_.release();
			this->appearance_.release();
			for(unsigned b=0;b<this->histo_.size();++b){
				this->histo_[b].release();
			}
		}
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Finds the corresponding position in the original (small) image.
		 */
		virtual cv::Point pos2pt(unsigned pos,unsigned motionW,unsigned motionH) const{
			// [0] Formula: mat.cols*r+c (r=y,c=x)
			cv::Point inMotion(0,0);
			if(this->motion_.cols>1){
				inMotion = cv::Point(static_cast<unsigned>(pos % motionW),\
					std::floor(static_cast<float>(pos)/static_cast<float>(motionW)));
			}
			return cv::Point(inMotion.x+(this->center_.x-(motionW/2)),\
				inMotion.y+(this->center_.y-(motionH/2)));
		}
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		float appearanceProb() const {return this->appearanceProb_;}
		float motionProb() const {return this->motionProb_;}
		std::vector<float> histinfo() const {return this->histinfo_;}
		cv::Mat motion() const {return this->motion_;}
		void motion(cv::Mat &motionXX,cv::Mat &motionXY,cv::Mat &motionYX,\
		cv::Mat &motionYY,unsigned motionW,unsigned motionH) const {
			assert(this->motion_.cols==4*motionW*motionH);
			cv::Mat roixx = this->motion_.colRange(0,motionW*motionH);
			roixx.reshape(1,motionH);
			roixx.copyTo(motionXX);
			cv::Mat roixy = this->motion_.colRange(motionW*motionH,2*motionW*motionH);
			roixy.reshape(1,motionH);
			roixy.copyTo(motionXY);
			cv::Mat roiyx = this->motion_.colRange(2*motionW*motionH,3*motionW*motionH);
			roiyx.reshape(1,motionH);
			roiyx.copyTo(motionYX);
			cv::Mat roiyy = this->motion_.colRange(3*motionW*motionH,4*motionW*motionH);
			roiyy.reshape(1,motionH);
			roiyy.copyTo(motionYY);
		}
		void motion(cv::Mat &motionX,cv::Mat &motionY,unsigned motionW,\
		unsigned motionH) const {
			assert(this->motion_.cols==2*motionW*motionH);
			cv::Mat roix = this->motion_.colRange(0,motionW*motionH);
			roix.reshape(1,motionH);
			roix.copyTo(motionX);
			cv::Mat roiy = this->motion_.colRange(motionW*motionH,2*motionW*motionH);
			roiy.reshape(1,motionH);
			roiy.copyTo(motionY);
		}
		cv::Mat appearance() const {return this->appearance_;}
		std::vector<cv::Mat> histo() const {return this->histo_;}
		cv::Mat histo(const cv::Point &pt) const{
			// Get the hist at the current point only
			cv::Mat ahist = cv::Mat::zeros(cv::Size(this->histo_.size(),1),CV_32FC1);
			float sum     = 0.0;
			for(unsigned b=0;b<this->histo_.size();++b){
				ahist.at<float>(0,b) = this->histo_[b].at<float>(pt);
				sum += this->histo_[b].at<float>(pt);
			}
			assert(!std::isnan(sum)); assert(!std::isinf(sum));
			assert(std::abs(sum-1.0)<0.1);
			return ahist;
		}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void appearanceProb(float appearanceProb){
			this->appearanceProb_ = appearanceProb;
		}
		void histinfo(const std::vector<float> &histinfo){
			this->histinfo_ = histinfo;
		}
		void motionProb(float motionProb){this->motionProb_ = motionProb;}
		void motion(const cv::Mat motion){
			this->motion_.release();
			motion.copyTo(this->motion_);
		}
		void appearance(const cv::Mat appearance){
			this->appearance_.release();
			appearance.copyTo(this->appearance_);
		}
		void histo(const std::vector<cv::Mat> &histo){
			for(unsigned b=0;b<this->histo_.size();++b){
				this->histo_[b].release();
			}
			this->histo_.clear();
			for(std::vector<cv::Mat>::const_iterator h=histo.begin();h!=histo.end();++h){
				this->histo_.push_back(h->clone());
			}
		}
		float motionAgreement(const cv::Mat &motion,bool usederivatives) const{
			if(motion.cols!=this->motion_.cols || motion.rows!=this->motion_.rows){
				std::cerr<<"[MotionPuzzlePatch::motionAgreement] sizes to not match"<<std::endl;
			}
			float dist = 0.0;
			unsigned cols;
			if(usederivatives){
				cols = motion.cols/4;
				cv::Mat_<float>::const_iterator t=motion.begin<float>();
				for(cv::Mat_<float>::const_iterator c=(this->motion_).begin<float>();\
				c!=(this->motion_).end<float>()-cols,t!=motion.end<float>()-cols;++c,++t){
					float tXX = (*t),          tXY = (*(t+cols)),
						  tYX = (*(t+2*cols)), tYY = (*(t+3*cols));
					float cXX = (*c),          cXY = (*(c+cols)),
						  cYX = (*(c+2*cols)), cYY = (*(c+3*cols));
					dist += std::sqrt((tXX-cXX)*(tXX-cXX)+(tXY-cXY)*(tXY-cXY)+\
						(tYX-cYX)*(tYX-cYX)+(tYY-cYY)*(tYY-cYY));
				}
			}else{
				cols = motion.cols/2;
				cv::Mat_<float>::const_iterator t=motion.begin<float>();
				for(cv::Mat_<float>::const_iterator c=(this->motion_).begin<float>();\
				c!=(this->motion_).end<float>()-cols,t!=motion.end<float>()-cols;++c,++t){
					float tX = (*t),\
						  tY = (*(t+cols));
					float cX = (*c),\
						  cY = (*(c+cols));
					dist += std::sqrt((tX-cX)*(tX-cX)+(tY-cY)*(tY-cY));
				}
			}
			return dist/static_cast<float>(cols);
		}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors the default ones are not good with IplImages
		 */
		MotionPuzzlePatch(MotionPuzzlePatch const &rhs):PuzzlePatch(rhs){
			this->center_         = rhs.center();
			this->logProb_        = rhs.logProb();
			this->piece(rhs.piece());
			this->motionProb_     = rhs.motionProb();
			this->appearanceProb_ = rhs.appearanceProb();
			this->motion(rhs.motion());
			this->appearance(rhs.appearance());
			this->histinfo(rhs.histinfo());
			this->histo(rhs.histo());
		}
		MotionPuzzlePatch& operator=(MotionPuzzlePatch const &rhs){
			if(this == &rhs) return *this;
			if(this){delete this;}
			this->center_         = rhs.center();
			this->logProb_        = rhs.logProb();
			this->piece(rhs.piece());
			this->motionProb_     = rhs.motionProb();
			this->appearanceProb_ = rhs.appearanceProb();
			this->motion(rhs.motion());
			this->appearance(rhs.appearance());
			this->histo(rhs.histo());
			this->histinfo(rhs.histinfo());
			return *this;
		}
		//----------------------------------------------------------------------
	private:
		/** @var motionProb_
		 * The probability of the motion patch.
		 */
		float motionProb_;
		/** @var appearanceProb_
		 * The probability of the feature patch.
		 */
		float appearanceProb_;
		/** @var motion_
		 * The motion patch.
		 */
		cv::Mat motion_;
		/** @var appearance_
		 * The appearance patch.
		 */
		cv::Mat appearance_;
		/** @var histo_
		 * The histogram of a patch.
		 */
		std::vector<cv::Mat> histo_;
		/** @var histinfo_
		 * The histogram information for recovering the bins.
		 */
		std::vector<float> histinfo_;
};
//==============================================================================
template <class P>
class MotionPuzzle:public Puzzle<P> {
	public:
		MotionPuzzle():Puzzle<P>(){};
		virtual ~MotionPuzzle(){};
		//----------------------------------------------------------------------
		/** Solves choosing the final motion prediction problem:
		 * - choosing among the trees
		 * - choosing in the neighborhood.
		 */
		static void solve(const std::vector<std::vector<P*> > &patches,\
			const cv::Size &featsize,unsigned motionW,unsigned motionH,\
			cv::Mat &motionXX,cv::Mat &motionXY,cv::Mat &motionYX,cv::Mat &motionYY,\
			cv::Mat &appear,MotionTreeClass::ENTROPY entropy,Puzzle<PuzzlePatch>::\
			METHOD method,unsigned step,const std::string &imname,\
			const std::string &path2results,unsigned maxIter=75,bool display=false,\
			bool pertree=false);
		/** Solves choosing the final motion prediction problem:
		 * - choosing among the trees
		 * - choosing in the neighborhood.
		 */
		static void solve(const std::vector<std::vector<P*> > &patches,\
			const cv::Size &featsize,unsigned motionW,unsigned motionH,\
			cv::Mat &motionX,cv::Mat &motionY,cv::Mat &appear,MotionTreeClass::\
			ENTROPY entropy,Puzzle<PuzzlePatch>::METHOD method,unsigned step,\
			const std::string &imname,const std::string &path2results,\
			unsigned maxIter=75,bool display=false,bool pertree=false);
		/** Generate a set of initial candidates (picks the most likely patches among
		 * the trees).
		 */
		static std::vector<P*> initialPick(const std::vector<std::vector<P*> > \
			&candidates,unsigned motionW,unsigned motionH,MotionTreeClass::\
			ENTROPY entropy,bool display,bool usederivatives);
		/** Displays the set of predicted leaves.
		 */
		static void showSamples(const P &leaf,unsigned sampleW,unsigned sampleH,\
			bool usederivatives);
		/** Displays the set of predicted leaves with flow derivatives.
		 */
		static void showSamplesDerivatives(const P &leaf,unsigned sampleW,\
			unsigned sampleH);
		/** Displays the set of predicted leaves with flow.
		 */
		static void showSamplesFlow(const P &leaf,unsigned sampleW,unsigned sampleH);
		/** Returns proposals for motions on X and Y.
		 */
		static cv::Mat proposePredictionSum(const std::vector<P*> &candidates,\
			const cv::Size &featsize,unsigned motionW,unsigned motionH,\
			cv::Mat &appearance,bool usederivatives);
		/** Returns proposals for motion derivatives on X and Y.
		 */
		static cv::Mat proposePredictionSumDerivatives(const std::vector<P*> \
			&candidates,const cv::Size &featsize,unsigned motionW,unsigned motionH,\
			cv::Mat &appearance);
		/** Returns proposals for motion flows on X and Y.
		 */
		static cv::Mat proposePredictionSumFlows(const std::vector<P*> &candidates,\
			const cv::Size &featsize,unsigned motionW,unsigned motionH,\
			cv::Mat &appearance);
		/** Propose the final prediction over overlapping neighborhoods.
		 */
		static cv::Mat proposePredictionOverlap(const std::vector<P*> &candidates,\
			const cv::Size &featsize,unsigned motionW,unsigned motionH,unsigned step,\
			MotionTreeClass::ENTROPY entropy,cv::Mat &appearance,bool usederivatives);
		/** Propose the final motion derivative prediction over overlapping neighborhoods.
		 */
		static cv::Mat proposePredictionOverlapDerivatives(const std::vector<P*> \
			&candidates,const cv::Size &featsize,unsigned motionW,unsigned motionH,\
			unsigned step,MotionTreeClass::ENTROPY entropy,cv::Mat &appearance);
		/** Propose the final flow prediction over overlapping neighborhoods.
		 */
		static cv::Mat proposePredictionOverlapFlows(const std::vector<P*> \
			&candidates,const cv::Size &featsize,unsigned motionW,unsigned motionH,\
			unsigned step,MotionTreeClass::ENTROPY entropy,cv::Mat &appearance);
		/** Selects the patches that agree the most with the previous prediction.
		 */
		static std::vector<P*> selectPatches(const cv::Mat &motion,const std::vector\
			<std::vector<P*> > &candidates,unsigned motionW,unsigned motionH,\
			bool usederivatives);
		/** Checks to see how much the prediction has changed between iterations.
		 */
		static bool checkConvergence(const cv::Mat &motion,const cv::Mat &prevMotion,\
			bool usederivatives);
		/** Just get the the binning information.
		 */
		static std::vector<float> histInfo(const std::vector<std::vector<P*> > \
			&candidates,unsigned motionW,unsigned motionH,MotionTreeClass::ENTROPY entropy);
		/** Generates the best patch per position among trees as the mean of all patches.
		 */
		static std::vector<P*> pickMean(const std::vector<std::vector<P*> > &candidates,\
			unsigned motionW,unsigned motionH,bool display,bool usederivatives);
		/** Picks the best patch per position among trees based on probability
		 * approximated using kernel density estimation.
		 */
		static std::vector<P*> pickApproxKernel(const std::vector<std::vector<P*> > \
			&candidates,unsigned motionW,unsigned motionH,MotionTreeClass::ENTROPY \
			entropy,bool display,bool usederivatives);
		/** Given a vector of candidate patches compute their probabilities.
		 */
		static std::vector<float> approxKernel(const std::vector<P*> &candidates,\
			unsigned motionW,unsigned motionH,unsigned &bestId,\
			MotionTreeClass::ENTROPY entropy,bool usederivatives);
		/** Gets the patch probabilities as sum_px log p(px)
		 */
		static float patchProb(const std::vector<cv::Mat> &probs,const P &patch,\
			unsigned motionW,unsigned motionH,MotionTreeClass::ENTROPY entropy,\
			bool usederivatives);
		/** Show per tree predictions and appearance.
		 */
		static void perTreePredictions(const std::vector<std::vector<P*> > &candidates,\
			unsigned motionW,unsigned motionH,const cv::Size &featsize,\
			unsigned step,MotionTreeClass::ENTROPY entropy,const std::string \
			&imname,const std::string &path2results,bool display,bool usederivatives);
	private:
		DISALLOW_COPY_AND_ASSIGN(MotionPuzzle);
};
//==============================================================================
#endif /* MOTIONPUZZLE_H_ */
