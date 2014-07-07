/* RunRF.h
 * Author: Silvia-Laura Pintea
 */
#ifndef RUNRF_H_
#define RUNRF_H_
#pragma once
#include "StructuredRFdetector.h"
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
class RunRF {
	public:
	 	 /** Modes of running the RF code
	 	  */
		enum MODE {TRAIN_RF,TEST_RF,TRAIN_TEST_RF,EXTRACT};
		RunRF(){
			this->ext_           = "";
			this->patchWidth_    = 0;
			this->patchHeight_   = 0;
			this->labWidth_      = 0;
			this->labHeight_     = 0;
			this->path2train_    = "";
			this->path2labs_     = "";
			this->path2test_     = "";
			this->path2results_  = "";
			this->path2model_    = "";
			this->noTrees_       = 0;
			this->path2feat_     = "";
			this->runName_       = "";
			this->consideredCls_ = 0;
			this->balance_       = false;
			this->trainSize_     = 0;
			this->noPatches_     = 0;
			this->iterPerNode_   = 0;
			this->entropy_       = StructuredTree<M,T,F,N,U>::CENTER;
			this->predMethod_    = Puzzle<PuzzlePatch>::SIMPLE;
			this->step_          = 0;
			this->binary_        = true;
		};
		RunRF(const char* config);
		virtual ~RunRF();
		/** Gets the color labels for the image.
		 */
		static cv::Mat getColorLabels(const cv::Mat &output,const \
			std::map<cv::Vec3b,unsigned,vec3bCompare> &classinfo);
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Initialize and start training.
		 */
		virtual void run(RunRF::MODE mode);
		/** Initialize and start detector on test set.
		 */
		virtual void runDetect();
		/** Extracts feature/label patches from all the images.
		 */
		virtual void runExtract();
		/** Initialize and start training.
		 */
		virtual void runTrain();
		/** Performs the RF detection on test images.
		 */
		virtual void detect(StructuredRFdetector<L,M,T,F,N,U> &crDetect);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		bool binary() const {return this->binary_;}
		unsigned step() const {return this->step_;}
		Puzzle<PuzzlePatch>::METHOD predMethod() const {return this->predMethod_;}
		bool balance() const {return this->balance_;};
		unsigned trainSize() const {return this->trainSize_;};
		unsigned noPatches() const {return this->noPatches_;};
		unsigned iterPerNode() const {return this->iterPerNode_;};
		typename StructuredTree<M,T,F,N,U>::ENTROPY entropy() const {return this->entropy_;};
		unsigned consideredCls() const{return this->consideredCls_;}
		std::string ext() const {return this->ext_;}
		std::string labTerm() const {return this->labTerm_;}
		unsigned patchWidth() const {return this->patchWidth_;}
		unsigned patchHeight() const {return this->patchHeight_;}
		unsigned labWidth() const {return this->labWidth_;}
		unsigned labHeight() const {return this->labHeight_;}
		std::string path2train() const {return this->path2train_;}
		std::string path2labs() const {return this->path2labs_;}
		std::string path2test() const {return this->path2test_;}
		std::string path2results() const {return this->path2results_;}
		std::string path2model() const {return this->path2model_;}
		std::string path2feat() const {return this->path2feat_;}
		unsigned noTrees() const {return this->noTrees_;}
		std::vector<float> pyrScales() const {return this->pyrScales_;}
		unsigned pyrScales(unsigned pos) const {return this->pyrScales_[pos];}
		std::map<cv::Vec3b,unsigned,vec3bCompare> classInfo(){return this->classInfo_;}
		unsigned classInfo(const cv::Vec3b &color){return this->classInfo_[color];}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void binary(bool binary){this->binary_ = binary;}
		void step(unsigned step){this->step_ = step;}
		void predMethod(Puzzle<PuzzlePatch>::METHOD method){
			this->predMethod_ = method;
		}
		void balance(bool balance){this->balance_ = balance;};
		void trainSize(unsigned trainSize){this->trainSize_ = trainSize;};
		void noPatches(unsigned noPatches){this->noPatches_ = noPatches;};
		void iterPerNode(unsigned iterPerNode){this->iterPerNode_ = iterPerNode;};
		void entropy(typename StructuredTree<M,T,F,N,U>::ENTROPY entropy){
			this->entropy_ = entropy;
		}
		void consideredCls(unsigned consideredCls){this->consideredCls_=consideredCls;}
		void ext(const std::string &ext){this->ext_ = ext;}
		void labTerm(const std::string &labTerm){this->labTerm_ = labTerm;}
		void patchWidth(unsigned patchWidth){this->patchWidth_ = patchWidth;}
		void patchHeight(unsigned patchHeight){this->patchHeight_ = patchHeight;}
		void labWidth(unsigned labWidth){this->labWidth_ = labWidth;}
		void labHeight(unsigned labHeight){this->labHeight_ = labHeight;}
		void path2train(const std::string &path2train){this->path2train_ = path2train;}
		void path2labs(const std::string &path2labs){this->path2labs_ = path2labs;}
		void path2test(const std::string &path2test){this->path2test_ = path2test;}
		void path2results(const std::string &path2results){
			this->path2results_ = path2results;
		}
		void path2model(const std::string &path2model){this->path2model_ = path2model;}
		void path2feat(const std::string &path2feat){this->path2feat_ = path2feat;}
		void noTrees(unsigned noTrees){this->noTrees_=noTrees;}
		void pyrScales(const std::vector<float> &pyrScales){
			this->pyrScales_ = pyrScales;
		}
		void pyrScales(unsigned pos,float pyrScales){
			this->pyrScales_[pos] = pyrScales;
		}
		void classInfo(const std::map<cv::Vec3b,unsigned,vec3bCompare> &classInfo){
			this->classInfo_ = classInfo;
		}
		void classInfo(const cv::Vec3b &color,unsigned classInfo){
			this->classInfo_[color] = classInfo;
		}
		//----------------------------------------------------------------------
	protected:
		/** @var ext_
		 * Image extension used for reading images from dir.
		 */
		std::string ext_;
		/** @var ext_
		 * Extra termination concatenated at the end of the label names.
		 */
		std::string labTerm_;
		/** @var patchWidth_
		 * The width of our appearance/label patches.
		 */
		unsigned patchWidth_;
		/** @var patchHeight_
		 * The height of our appearance/label patches.
		 */
		unsigned patchHeight_;
		/** @var labWidth_
		 * The width of our appearance/label patches.
		 */
		unsigned labWidth_;
		/** @var labHeight_
		 * The height of our appearance/label patches.
		 */
		unsigned labHeight_;
		/** @var path2train_
		 * Path to train images.
		 */
		std::string path2train_;
		/** @var path2labs_
		 * Path to labeled images.
		 */
		std::string path2labs_;
		/** @var path2test_
		 * Path to test images.
		 */
		std::string path2test_;
		/** @var path2results_
		 * Path results images (labeled).
		 */
		std::string path2results_;
		/** @var path2model_
		 * Path to RF model.
		 */
		std::string path2model_;
		/** @var noTrees_
		 * Number of trees.
		 */
		unsigned noTrees_;
		/** @var pyrScale_
		 * Scales for the image pyramid.
		 */
		std::vector<float> pyrScales_;
		/** @var path2feat_
		 * Path to features (if none, then extract).
		 */
		std::string path2feat_;
		/** @var classInfo_
		 * Maps class names to color to ids.
		 */
		std::map<cv::Vec3b,unsigned,vec3bCompare> classInfo_;
		/** @var runName_
		 * The name of the current run for the log-files.
		 */
		std::string runName_;
		/** @var consideredCls_
		 * The first x considered classes (they are taken in the order from the
		 * config file).
		 */
		unsigned consideredCls_;
		/** @var balance_
		 * To balance the class data or not.
		 */
		bool balance_;
		/** @var trainSize_
		 * Number of images to be used for training 1 tree.
		 */
		unsigned trainSize_;
		/** @var noPatches_
		 * Number of patches to use for training (repeat some).
		 */
		unsigned noPatches_;
		/** @var iterPerNode_
		 * Number of iterations per node for tree-training.
		 */
		unsigned iterPerNode_;
		/** @var entropy_
		 * The entropy type to be used (CENTER, RANDOM, ..)
		 */
		typename StructuredTree<M,T,F,N,U>::ENTROPY entropy_;
		/** @var predMethod_
		 * Puzzle or Simple prediction method.
		 */
		typename Puzzle<PuzzlePatch>::METHOD predMethod_;
		/** @var step_
		 * The step on the grid for sampling patches.
		 */
		unsigned step_;
		/** @var binary_
		 * If the tree is in a binary file or not.
		 */
		bool binary_;
	private:
		DISALLOW_COPY_AND_ASSIGN(RunRF);
};
//==============================================================================
#endif /* RUNRF_H_ */
#include "RunRF.cpp"
