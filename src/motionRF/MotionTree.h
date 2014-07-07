/* MotionTree.h
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONTREE_H_
#define MOTIONTREE_H_
#include <StructuredTree.h>
#include "MotionPatch.h"
//==============================================================================
/** One Structured Tree for the motion prediction task.
 */
template <class M,class T,class F,class N,class U>
class MotionTree:public StructuredTree<M,T,F,N,U>{
	public:
		typedef typename std::vector<std::vector<const T*> >::const_iterator vectConstIterT;
		typedef typename std::vector<const T*>::const_iterator constIterT;
		typedef typename std::vector<std::vector<const T*> >::iterator vectIterT;
		typedef typename std::vector<const T*>::iterator IterT;
		MotionTree(const char* filename,unsigned treeid,bool binary);
		MotionTree(unsigned minS,unsigned maxD,CvRNG* pRNG,unsigned labSz,\
		unsigned patchW,unsigned patchH,unsigned patchCh,unsigned treeId,const \
		char *path2models,const std::string &runName,typename StructuredTree\
		<M,T,F,N,U>::ENTROPY entropy,unsigned consideredCls,bool binary,\
		bool leafavg=false,bool parentfreq=true,bool leafParentFreq=true,\
		const std::string &runname="",float entropythresh=1e-1,bool usepick=false,\
		bool hogOrSift=true,unsigned growthtype=1):StructuredTree<M,T,F,N,U>\
		(minS,maxD,pRNG,labSz,patchW,patchH,patchCh,treeId,path2models,runName,\
		entropy,consideredCls,binary){
			this->motionW_        = 0;
			this->motionH_        = 0;
			this->parentFreq_     = parentfreq;
			this->leafParentFreq_ = leafParentFreq;
			this->leafavg_        = leafavg;
			this->path2models_    = path2models;
			this->treeId_         = treeId;
			this->binary_         = binary;
			this->runname_        = runname;		
			this->entropythresh_  = entropythresh;
			this->clockbegin_     = clock();
			this->usepick_        = usepick;
			this->hogOrSift_      = hogOrSift;
			this->growthtype_     = growthtype;
		};
		virtual ~MotionTree(){};
		//----------------------------------------------------------------------
		/** Reads the tree from a binary file.
		 */
		void readTreeBin();
		/** Reads the tree from a regular text file.
		 */
		void readTree();
		/** Decides for a split based on the cosine similarity.
		 */
		float stopCosSimilarity(const std::vector<std::vector<const T*> > &SetA,\
			const F *features);
		/** Check if all patches have converged to a single pattern by looking as MSE.
		 */
		float stopCosSimilarityDerivatives(const std::vector<std::vector<const T*> > \
			&trainSet,const F *features);
		/** Check if all patches have converged to a single pattern by looking as MSE.
		 */
		float stopCosSimilarityFlows(const std::vector<std::vector<const T*> > \
			&trainSet,const F *features);
		/** Check if all patches have converged to a single pattern by looking as MSE.
		 */
		float stopEuclDist(const std::vector<std::vector<const T*> > &trainSet,\
			const F *features);
		/** Check if all patches have converged to a single pattern by looking as MSE.
		 */
		float stopEuclDistFlows(const std::vector<std::vector<const T*> > \
			&trainSet,const F *features);
		/** Check if all patches have converged to a single pattern by looking as MSE.
		 */
		float stopEuclDistDerivaties(const std::vector<std::vector<const T*> > \
			&trainSet,const F *features);
		/** In split: sum-squared-distance to the mean of the samples at the
		 * picked position.
		 */
		float splitDistance2mean(const std::vector<std::vector<const T*> > &SetA,\
			const F *features,float &sizeA);
		/** Sum-Squared-Distance to the mean of the samples at the picked position.
		 */
		float splitDistance2meanDerivatives(const std::vector<std::vector\
			<const T*> > &SetA,const F *features,float &sizeA);
		/** Sum-Squared-Distance to the mean of the samples at the picked position.
		 */
		float splitDistance2meanFlows(const std::vector<std::vector<const T*> > \
			&SetA,const F *features,float &sizeA);
		/** Approximating continuous entropy with sum over sample probability, in turn
		 * approximated the density kernel estimation with pixel-wise kernels.
		 */
		float splitApproxKernel(const std::vector<std::vector<const T*> > \
			&SetA,const F *features,float &sizeA,std::vector<cv::Mat> &prevfreq);
		/** Approximating continuous entropy with sum over sample probability, in turn
		 * approximated the density kernel estimation with pixel-wise kernels over
		 * complete patch.
		 */
		float splitApproxKernelPatch(const std::vector<std::vector<const T*> > \
			&SetA,const F *features,float &sizeA,std::vector<cv::Mat> &prevfreq);
		/** show the mean to the samples for the picked best test.
		 */
		float showPickedSplitDerivatives(const std::vector<std::vector<const T*> > \
			&SetA,const std::vector<std::vector<const T*> > &SetB,\
			const F *features,long unsigned nodeid);
		/** show the mean to the samples for the picked best test.
		 */
		float showPickedSplitFlow(const std::vector<std::vector<const T*> > &SetA,\
			const std::vector<std::vector<const T*> > &SetB,\
			const F *features,long unsigned nodeid);
		/** show the mean to the samples for the picked best test.
		 */
		float showPickedSplit(const std::vector<std::vector<const T*> > &SetA,\
			const std::vector<std::vector<const T*> > &SetB,const F *features,\
			long unsigned nodeid);
		/** Take the mean of all patches arriving to the leaf.
		 */
		void leafMean(const F* features,const std::vector<std::vector<const T*> > \
			&trainSet,int first,unsigned totsize,float &bestAppProb,float \
			&bestMotionProb,cv::Mat *bestMotion,cv::Mat *bestApp);
		/** Keeps the most likely patch in the leaf given the approximation of kernel
		 * density estimation for the patch probability.
		 */
		void leafApprox(const F* features,const std::vector<std::vector\
			<const T*> > &trainSet,int first,unsigned totsize,float &bestAppProb,\
			float &bestMotionProb,cv::Mat *bestApp,cv::Mat *bestMotion,\
			std::vector<cv::Mat> &bestHisto,const std::vector<cv::Mat> &prevfreq,\
			long unsigned nodeid,bool writeprobs=true);
		/** Writes down the probability for each leaf. As a check.
		 */
		void writeprobs(const std::vector<std::vector<float> > &mProb,\
			long unsigned nodeid,unsigned bestPatchId);
		/** Gets the appearance probabilities in the leaf based on similarity.
		 */
		std::vector<std::vector<float> > patchAppearanceSim(const std::vector\
			<std::vector<const T*> > &trainSet,const F* features,unsigned totPatches);
		/** Gets the closest patch to the mean-motion in the leaf (euclidian distance).
		 */
		std::vector<std::vector<float> > patchDist2Mean(const std::vector\
			<std::vector<const T*> > &trainSet,const F* features,unsigned totsize,\
			const std::vector<float> &prevfreq);
		/** Gets the closest patch to the mean-motion in the leaf (euclidian distance).
		 */
		std::vector<std::vector<float> > patchDist2MeanDerivatives(const \
			std::vector<std::vector<const T*> > &trainSet,const F* features,\
			unsigned totsize,const std::vector<float> &prevfreq);
		/** Gets the closest patch to the mean-motion in the leaf (euclidian distance).
		 */
		std::vector<std::vector<float> > patchDist2MeanFlows(const \
			std::vector<std::vector<const T*> > &trainSet,const F* features,\
			unsigned totsize,const std::vector<float> &prevfreq);
		/** For each patch finds it probability as 1/#bins sum_bins k(sample-bin).
		 */
		std::vector<std::vector<float> > patchApprox(const F* features,const \
			std::vector<std::vector<const T*> > &trainSet,unsigned totPatches,\
			const std::vector<cv::Mat> &prevfreq);
		/** Given and input sample, find its corresponding inverse frequency.
		 */
		static float getProbMagni(const std::vector<cv::Mat> &probs,const std::vector\
			<float> &bininfo,const std::vector<float> &values,const cv::Point &pos);
		/** Given and input sample, find its corresponding inverse frequency.
		 */
		static float getProbAngle(const std::vector<cv::Mat> &probs,const std::vector\
		<float> &bininfo,const std::vector<float> &values,const cv::Point &pos);
		/** Displays the set of predicted leaves.
			 */
		static void showSamplesDerivatives(const std::vector<const U*> &leaves,\
			unsigned sampleW,unsigned sampleH,const cv::Point &point);
		/** Displays the set of predicted leaves.
			 */
		static void showSamplesFlows(const std::vector<const U*> &leaves,\
			unsigned sampleW,unsigned sampleH,const cv::Point &point);
		/** Displays the samples among which we need to choose to make a leaf
		 */
		void showSamplesFlows(const std::vector<std::vector<const T*> > &trainSet,\
			const F* features,long unsigned nodeid,float entropy,const cv::Mat* bestMotion,\
			const cv::Mat *bestApp,bool justdisplay);
		/** Just dot product between vectors.
		 */
		static float dotProd(const std::vector<float> &asmpl,const \
			std::vector<float> &dimprobs);
		/** Gets the patch probabilities as sum_px log p(px)
		 */
		float patchProb(const std::vector<cv::Mat> &probs,const T *patch,\
			const F *features);
		/** Get "class inverse frequencies" --- inverse priors for reweighting.
		 */
		std::vector<cv::Mat> setFreq(const F* features,const std::vector\
			<std::vector<const T*> > &allTrainSet);
		/** Get the node info. Does all administrative bits to get the info be saved in the node.
		 */
		bool addNodeInfo(N*current,N *parent,const typename Tree<N,U>::SIDE side,\
			const F *features,unsigned &countA,unsigned &countB,float &entropyA,\
			float &entropyB,unsigned nodeiters,bool showsplits);
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Recursively read tree from binary file.
		 */
		virtual void readNodeBin(N *parent,std::ifstream &in,typename \
			Tree<N,U>::SIDE side);
	 	/** Recursively read tree from file.
		 */
		virtual void readNode(N *parent,std::ifstream &in,typename \
			Tree<N,U>::SIDE side);
		/** Writes the current tree into a given binary file.
		 */
		virtual bool saveTree();
		/** Writes the current tree into a given binary file.
		 */
		virtual bool saveTreeBin();
		/** Writes the current tree into a given file.
		 */
		virtual bool saveTreeTxt();
		/** Initializes the size of the labels, number of channels, etc.
		 */
		virtual void initDataSizes(const M& trData);
		/** Implementing the <<growTee>> with multiple labels.
		 */
		virtual void growTree(const M& trData,unsigned nodeiters,long unsigned maxleaves=5e+3);
		/** Creates the actual tree from the samples.
		 */
		virtual void grow(const std::vector<std::vector<const T*> > &trainSet,\
			const F *features,long unsigned &nodeid,unsigned int depth,unsigned nodeiters,\
			N* parent,typename Tree<N,U>::SIDE side,std::vector<cv::Mat> &prevfreq,\
			std::vector<cv::Mat> &prevprevfreq,long unsigned maxleaves,bool showSplits=false);
		/** Grow the tree on the depth.
		 */
		virtual void growDepth(const std::vector<std::vector<const T*> > &trainSet,\
			const F *features,long unsigned &nodeid,unsigned int depth,unsigned nodeiters,\
			N* parent,typename Tree<N,U>::SIDE side,std::vector<cv::Mat> &prevfreq,\
			std::vector<cv::Mat> &prevprevfreq,bool showSplits=false);
		/** Grows the tree either breath-first or worst-first until a leaf is reached.
 		 */
		virtual void growLimit(const std::vector<std::vector<const T*> > &trainSet,\
			const F *features,const std::vector<cv::Mat> &prevfreq,long unsigned \
			maxleaves,unsigned nodeiters,bool showSplits=false);
		/** Optimizes tests and thresholds.
		 * [1] Generate a 5 random values (for x1 y1 x2 y2 channel) in the <<test>> vector.
		 * [2] Evaluates the thresholds and finds the minimum and maximum index value [?].
		 * [3] Iteratively generate random thresholds to split the index values
		 * [4] Split the data according to each threshold.
		 * [5] Find the best threshold and store it on the 6th position in <<test>>
		 */
		virtual bool optimizeTest(std::vector<std::vector<const T*> >& SetA,\
			std::vector<std::vector<const T*> >& SetB,const std::vector\
			<std::vector<const T*> >& TrainSet,const F* features,long double* test,\
			unsigned int iter,unsigned pick,std::vector<cv::Mat> &freqA,\
			std::vector<cv::Mat> &freqB,float &best,float &entropyA,float &entropyB);
		/** Just splits the data into subsets and makes sure the subsets are not empty
		 */
		virtual float performSplit(std::vector<std::vector<const T*> >& tmpA,\
			std::vector<std::vector<const T*> >& tmpB,const std::vector\
			<std::vector<const T*> >& TrainSet,const F* features,const \
			std::vector<std::vector<Index> > &valSet,unsigned pick,\
			long double threshold,unsigned &sizeA,unsigned &sizeB,std::vector\
			<cv::Mat> &parentfreqA,std::vector<cv::Mat> &parentfreqB,\
			float &entropyA,float &entropyB);
		/** Overloading the function to carry around the labels matrices.
		 */
		virtual float measureSet(const std::vector<std::vector<const T*> > &SetA,\
			const std::vector<std::vector<const T*> > &SetB,const F *features,\
			unsigned pick,std::vector<cv::Mat> &parentfreqA,std::vector<cv::Mat> \
			&parentfreqB,float &motionA,float &motionB);
		/** Create leaf node from all patches.
		 */
		virtual void makeLeaf(const F* features,const std::vector<std::vector<const T*> >\
			&trainSet,long unsigned nodeid,N* parent,typename Tree<N,U>::SIDE side,\
			unsigned nopatches,const std::vector<cv::Mat> &prevfreq,float entropy,\
			bool showLeaves=false);
		/** Displays the samples among which we need to choose to make a leaf
		 */
		virtual void showSamples(const std::vector<std::vector<const T*> > &trainSet,\
			const F* features,long unsigned nodeid,float entropy=0,\
			const cv::Mat* bestMotion=NULL,const cv::Mat *bestApp=NULL,\
			bool justdisplay=false);
		/** Displays the samples among which we need to choose to make a leaf
		 */
		virtual void showSamplesDerivatives(const std::vector<std::vector\
			<const T*> > &trainSet,const F* features,long unsigned nodeid,float entropy=0,\
			const cv::Mat* bestMotion=NULL,const cv::Mat *bestApp=NULL,\
			bool justdisplay=false);
		/** Applied the test on a feature patch. The center is fixed and we look at the
		 * sift dimensions/channels.
		 */
		virtual bool siftapplyTest(const long double *test,const T* testPatch,\
			const F* features) const;
		/** Generates a random test of a random type.
		 */
		virtual void siftgenerateTest(long double* test,unsigned int max_w,\
			unsigned int max_h,unsigned int max_c);
		/** Evaluates 1 test (given by 5 numbers: x1, y1, x2, y2, channel).
		 * It gets the feature channel and then it accesses it at the 2 randomly selected
		 * points and gets the difference between them.
		 */
		virtual void siftevaluateTest(std::vector<std::vector<Index> >& valSet,\
			const long double* test,const std::vector<std::vector<const T*> > \
			&TrainSet,const F *features);
		/** Predicts on a one single test patch.
		 * A node contains: [0] -- node type (0,1,-1),[1] -- x1,[2] -- y1,[3] -- x2,
		 * 					[4] -- y2,[5] -- channel,[6] -- threshold, [7] -- test type,
		 * 					[8] -- node ID
		 */
		virtual const U* siftregression(const T* testPatch,const F* features,\
			N* node,unsigned treeId);
		/** Adds a node to the tree given the parent node and the side.
 		 */
		virtual N* addNode(N *current,N *parent,typename Tree<N,U>::SIDE side);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		cv::Point mpick() const {return this->mpick_;}
		unsigned motionW() const {return this->motionW_;}
		unsigned motionH() const {return this->motionH_;}
		bool parentFreq() const {return this->parentFreq_;}
		bool leafParentFreq() const {return this->leafParentFreq_;}
		bool leafavg() const {return this->leafavg_;}
		std::string runname() const {return this->runname_;}
		float entropythresh() const {return this->entropythresh_;}
		float usepick() const {return this->usepick_;}
		clock_t clockbegin() const {return this->clockbegin_;}
		std::vector<float> histinfo() const {return this->histinfo_;}
		bool hogOrSift() const {return this->hogOrSift_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void mpick(const cv::Point &mpick){this->mpick_ = mpick;}
		void hogOrSift(bool hogOrSift){this->hogOrSift_ = hogOrSift;}
		void motionW(unsigned motionW){this->motionW_ = motionW;};
		void motionH(unsigned motionH){this->motionH_ = motionH;};
		void parentFreq(bool parentFreq){this->parentFreq_ = parentFreq;}
		void leafParentFreq(bool leafParentFreq){this->leafParentFreq_ = leafParentFreq;}
		void leafavg(bool leafavg){this->leafavg_ = leafavg;}
		void runname(const std::string &runname){this->runname_ = runname;}
		void entropythresh(float entropythresh){this->entropythresh_ = entropythresh_;}
		void usepick(float usepick){this->usepick_ = usepick_;}
		void clockbegin(const clock_t &clockbegin){this->clockbegin_ = clockbegin;}
		void histinfo(const std::vector<float> &histinfo){this->histinfo_ = histinfo;}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors for trees (to put them in the forest).
		 */
		MotionTree(MotionTree const &rhs):StructuredTree<M,T,F,N,U>(rhs){
			this->labSz_          = rhs.labSz();
			this->patchW_         = rhs.patchW();
			this->patchH_         = rhs.patchH();
			this->patchCh_        = rhs.patchCh();
			this->nodeSize_       = rhs.nodeSize();
			this->minSamples_     = rhs.minSamples();
			this->cvRNG_          = rhs.cvRNG();
			this->maxDepth_       = rhs.maxDepth();
			this->entropy_        = rhs.entropy();
			this->consideredCls_  = rhs.consideredCls();
			this->motionW_        = rhs.motionW();
			this->motionH_        = rhs.motionH();
			this->parentFreq_     = rhs.parentFreq();
			this->leafParentFreq_ = rhs.leafParentFreq();
			this->leafavg_        = rhs.leafavg();
			this->runname_        = rhs.runname();
			this->entropythresh_  = rhs.entropythresh();
			this->clockbegin_     = rhs.clockbegin();
			this->histinfo(rhs.histinfo());
			this->usepick(rhs.usepick());
			this->mpick(rhs.mpick());
			this->hogOrSift(rhs.hogOrSift());
			this->clsFreq(rhs.clsFreq());
		}
		MotionTree& operator=(MotionTree const &rhs){
			if(this == &rhs) return *this;
			if(this){delete this;}
			this->labSz_         = rhs.labSz();
			this->patchW_        = rhs.patchW();
			this->patchH_        = rhs.patchH();
			this->patchCh_       = rhs.patchCh();
			this->nodeSize_      = rhs.nodeSize();
			this->minSamples_    = rhs.minSamples();
			this->cvRNG_         = rhs.cvRNG();
			this->maxDepth_      = rhs.maxDepth();
			this->entropy_       = rhs.entropy();
			this->consideredCls_ = rhs.consideredCls();
			this->motionW_       = rhs.motionW();
			this->motionH_       = rhs.motionH();
			this->parentFreq_     = rhs.parentFreq();
			this->leafParentFreq_ = rhs.leafParentFreq();
			this->leafavg_        = rhs.leafavg();
			this->runname_        = rhs.runname();
			this->entropythresh_  = rhs.entropythresh();
			this->clockbegin_     = rhs.clockbegin();
			this->histinfo(rhs.histinfo());
			this->usepick(rhs.usepick());
			this->mpick(rhs.mpick());
			this->hogOrSift(rhs.hogOrSift());
			this->clsFreq(rhs.clsFreq());
			return *this;
		}
		//----------------------------------------------------------------------
	private:
		/** @var parentFreq_
		 * Whether we weight by parent frequencies or not.
		 */
		bool parentFreq_;
		/** @var leafParentFreq_
		 * Whether we weight by parent frequencies or not in the leaves.
		 */
		bool leafParentFreq_;
		/** @var mpick_
		 * The picked position from the motion patch.
		 */
		cv::Point mpick_;
		/** @var motionW_
		 * The width of the motion patch.
		 */
		unsigned motionW_;
		/** @var motionH_
		 * The height of the motion patch.
		 */
		unsigned motionH_;
		/** @var leafavg_
		 * Weighting the leaves by parent frequencies.
		 */
		bool leafavg_;
		/** @var runname_
		 * The name of the run for logging.
		 */
		std::string runname_;
		/** @var entropythresh_
		 * The entropy ratio for leaf making.
		 */
		float entropythresh_;
		/** @var usepick_
		 * If we pick a random position or assume independence.
		 */
		bool usepick_;
		/** @var clockbegin_
		 * To check how long it takes to train 1 tree.
		 */
		clock_t clockbegin_;
		/** @var histinfo_
		 * The information about the histograms in the RF.
		 */
		std::vector<float> histinfo_;
		/** @var hogOrSift_
		 * Hog - 1, sift - 0
		 */
		bool hogOrSift_;
		/** @var queueinfo_
		 * The queue in which to store the nodes to be processed.
		 */
		std::vector<std::pair<unsigned,N*> > queueinfo_;
};
//==============================================================================
#endif /* MOTIONTREE_H_ */
































