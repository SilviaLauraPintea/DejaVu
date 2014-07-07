/* StructuredTree.h
 * Author: Silvia-Laura Pintea
 */
#ifndef STRUCTUREDTREE_H_
#define STRUCTUREDTREE_H_
#pragma once
#include "StructuredPatch.h"
#include "StructuredTreeNode.h"
#include "Puzzle.h"
#include "Tree.h"
//==============================================================================
template <class M,class T,class F,class N,class U>
class StructuredTree:public Tree<N,U> {
	public:
		/** On what do we evaluate the entropy.
		 */
		enum ENTROPY {CENTER,RANDOM,CENTER_RANDOM,MEAN_DIFF,APPROX_MAGNI_KERNEL,\
			APPROX_ANGLE_KERNEL};
		typedef typename std::vector<std::vector<const T*> >::const_iterator vectConstIterT;
		typedef typename std::vector<const T*>::const_iterator constIterT;
		typedef typename std::vector<std::vector<const T*> >::iterator vectIterT;
		typedef typename std::vector<const T*>::iterator IterT;
		//----------------------------------------------------------------------
		StructuredTree(){};
		StructuredTree(const char *filename,unsigned treeid,bool binary);
		StructuredTree(unsigned minS,unsigned maxD,CvRNG *pRNG,unsigned labSz,\
		unsigned patchW,unsigned patchH,unsigned patchCh,unsigned treeId,const \
		char *path2models,const std::string &runName,typename StructuredTree\
		<M,T,F,N,U>::ENTROPY entropy,unsigned consideredCls,bool binary):\
		minSamples_(minS),maxDepth_(maxD),labSz_(labSz),patchW_(patchW),\
		patchH_(patchH),patchCh_(patchCh),cvRNG_(pRNG),entropy_(entropy),\
		consideredCls_(consideredCls){
			this->binary_      = binary;
			this->treeId_      = treeId;
			this->path2models_ = path2models;
			this->nodeSize_    = 8;
			this->root_        = NULL;
			this->log_.open(("log_"+runName+Auxiliary<int,1>::number2string\
				(this->treeId_)+".txt").c_str());
			if(!this->log_.is_open()){
				std::cerr<<"[StructuredTree::StructuredTree]: could not open log file"<<std::endl;
			}
		};
		virtual ~StructuredTree();
		/** Reads the tree from a binary file.
		 */
		void readTreeBin();
		/** Reads the tree from a text file.
		 */
		void readTreeTxt();
		/** Generates a random test of a random type.
		 */
		void generateTest(long double *test,unsigned int max_w,unsigned int max_h,\
			unsigned int max_c);
		/** Generates a random test of a random type -- hack for bins of 2x2 in features.
		 */
		void generateTestHack(long double* test,unsigned int max_w,unsigned int max_h,\
			unsigned int max_c);
		/** Just gets total number of patches regardless of class
		 */
		unsigned getNoPatches(const std::vector<std::vector<const T*> > &trainSet);
		/** Computes the negative entropy for 1 set wrt to the central pixel of the
		 * label patch.
		 */
		float nEntropy1Cls(const std::vector<std::vector<const T*> >& SetA,\
			float &sizeA);
		/** Computes the negative entropy for 1 set wrt to a random pixel of the
		 * label patch.
		 */
		float nEntropy1ClsRnd(const std::vector<std::vector<const T*> >& SetA,\
			const F* features,float &sizeA,unsigned pick);
		/** Computes the negative entropy for 1 set wrt to the central pixel of the
		 * label patch and a randomly picked pixel.
		 */
		float nEntropy2Cls(const std::vector<std::vector<const T*> >& SetA,\
			const F* features,unsigned pick,float &sizeA);
		/** Just splits the data into subsets and makes sure the subsets are not empty
		 */
		float performSplit(std::vector<std::vector<const T*> >& tmpA,std::vector\
			<std::vector<const T*> >& tmpB,const std::vector<std::vector<const T*> >& \
			TrainSet,const F* features,const std::vector<std::vector<Index> > \
			&valSet,unsigned pick,long double threshold,unsigned &sizeA,unsigned &sizeB);
		/** Optimizes tests and thresholds.
		 * [1] Generate a 5 random values (for x1 y1 x2 y2 channel) in the <<test>> vector.
		 * [2] Evaluates the thresholds and finds the minimum and maximum index value [?].
		 * [3] Iteratively generate random thresholds to split the index values
		 * [4] Split the data according to each threshold.
		 * [5] Find the best threshold and store it on the 6th position in <<test>>
		 */
		bool optimizeTest(std::vector<std::vector<const T*> > &SetA,\
			std::vector<std::vector<const T*> > &SetB,const std::vector<std::vector\
			<const T*> > &TrainSet,const F* features,long double *test,\
			unsigned int iter,unsigned pick,float &best);
		/** Evaluates 1 test (given by 5 numbers: x1, y1, x2, y2, channel).
		 * It gets the feature channel and then it accesses it at the 2 randomly selected
		 * points and gets the difference between them.
		 */
		void evaluateTest(std::vector<std::vector<Index> > &valSet,\
			const long double* test,const std::vector<std::vector<const T*> > &TrainSet,\
			const F* features);
		/** Evaluates 1 test (given by 5 numbers: x1, y1, x2, y2, channel).
		 * It gets the feature channel and then it accesses it at the 2 randomly selected
		 * points and gets the difference between them.
		 */
		void evaluateTestHack(std::vector<std::vector<Index> >& valSet,const long \
			double* test,const std::vector<std::vector<const T*> >& TrainSet,\
			const F *features);
		/** Applies a test to a patch.
		 */
		bool applyTest(const long double *test,const T* testPatch,const F* features) const;
		/** Applies a test to a patch --- hack to get features on 2x2 bins.
		 */
		bool applyTestHack(const long double *test,const T* testPatch,const F* features) const;
		/** Splits the training samples into a left set and a right set.
		 */
		void split(std::vector<std::vector<const T*> >& SetA,std::vector\
			<std::vector<const T*> >& SetB,const std::vector<std::vector<const T*> >& \
			TrainSet,const std::vector<std::vector<Index> >& valSet,long double t);
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Predicts on a one single test patch.
		 * A node contains: [0] -- node type (0,1,-1),[1] -- x1,[2] -- y1,[3] -- x2,
		 * 					[4] -- y2,[5] -- channel,[6] -- threshold,
		 */
		virtual const U* regression(const T* testPatch,const F* features,\
			N* node,unsigned treeid);
		/** Displays the leaves of the tree.
		 */
		virtual void showLeaves(unsigned labWidth,unsigned labHeight,\
			const std::map<cv::Vec3b,unsigned,vec3bCompare> &classinfo);
		/** Writes the current tree into a given file.
		 */
		virtual bool saveTree() const;
		/** Writes the current tree into a given file.
		 */
		virtual bool saveTreeBin() const;
		/** Writes the current tree into a given file.
		 */
		virtual bool saveTreeTxt() const;
		/** Initializes the size of the labels, number of channels, etc.
		 */
		virtual void initDataSizes(const M& trData);
		/** Implementing the <<growTee>> with multiple labels.
		 */
		virtual void growTree(const M &trData,int samples);
		/** Creates the actual tree from the samples.
		 */
		virtual void grow(const std::vector<std::vector<const T*> > &trainSet,const F* \
			features,long unsigned &nodeid,unsigned int depth,int samples,N* parent,\
			typename Tree<N,U>::SIDE side,float &prevInfGain);
		/** Create leaf node from all patches corresponding to a class.
		 */
		virtual void makeLeaf(const F* features,const std::vector<std::vector<const T*> > \
			&trainSet,long unsigned nodeid,N* parent,typename Tree<N,U>::SIDE side);
		/** Computes the probabilities of each label-patch given the complete of patches.
		 * [1] For each label-patch get its prob as prod of pixel-label probs.
		 * [2] Find the label-patch with the maximum prob.
		 */
		virtual std::vector<std::vector<float> > getPatchProb(const std::vector<std::vector\
			<const T*> > &trainSet,const F* features);
		/** Overloading the function to carry around the labels matrices.
		 */
		virtual float measureSet(const std::vector<std::vector<const T*> > &SetA,\
			const std::vector<std::vector<const T*> > &SetB,const F* features,\
			unsigned pick);
		/** Classification information gain check.
		 * [1] Associate each app-patch with a random label we pick from the label-patch
		 * [2] Compute the negative entropy as: sum_c p(c) log p(c)
		 * [3] return: (size(A)entropy(A)+size(B)entropy(B)) / (size(A)+size(B)).
		 */
		virtual float InfGain(const std::vector<std::vector<const T*> > &SetA,const \
			std::vector<std::vector<const T*> > &SetB,const F* features,unsigned pick);
		/** Adds a node to the tree given the parent node and the side.
		 */
		virtual N* addNode(N *parent,typename Tree<N,U>::SIDE side,const long double *test,unsigned nodeSize,\
			long unsigned nodeid,const U *leaf);
		/** Recursively read tree from binary file.
		 */
		virtual void readNodeBin(N *parent,std::ifstream &in,typename Tree<N,U>::SIDE side);
		/** Recursively read tree from file.
		 */
		virtual void readNodeTxt(N *parent,std::ifstream &in,typename Tree<N,U>::SIDE side);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned consideredCls() const {return this->consideredCls_;}
		ENTROPY entropy() const {return this->entropy_;}
		unsigned maxDepth() const {return this->maxDepth_;}
		CvRNG* cvRNG() const {return this->cvRNG_;}
		unsigned minSamples() const {return this->minSamples_;}
		unsigned nodeSize() const {return this->nodeSize_;}
		unsigned labSz() const {return this->labSz_;}
		unsigned patchW() const {return this->patchW_;}
		unsigned patchH() const {return this->patchH_;}
		unsigned patchCh() const {return this->patchCh_;}
		std::ofstream log() const {return this->log_;}
		std::vector<float> clsFreq() const {return this->clsFreq_;}
		float clsFreq(unsigned pos) const {return this->clsFreq_[pos];}
		std::vector<std::vector<float> > coFreq() const {return this->coFreq_;}
		std::vector<float> coFreq(unsigned pos) const {return this->coFreq_[pos];}
		float coFreq(unsigned ctr,unsigned rnd) const {return this->coFreq_[ctr][rnd];}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void consideredCls(unsigned consideredCls){this->consideredCls_ = consideredCls;}
		void entropy(ENTROPY entropy){this->entropy_ = entropy;}
		void maxDepth(unsigned maxDepth){this->maxDepth_ = maxDepth;}
		void cvRNG(CvRNG* cvRNG){this->cvRNG_ = cvRNG;}
		void minSamples(unsigned minSamples){this->minSamples_ = minSamples;}
		void nodeSize(unsigned nodeSize){this->nodeSize_ = nodeSize;}
		void labSz(unsigned labSz){this->labSz_ = labSz;}
		void patchW(unsigned patchW){this->patchW_ = patchW;}
		void patchH(unsigned patchH){this->patchH_ = patchH;}
		void patchCh(unsigned patchCh){this->patchCh_ = patchCh;}
		void clsFreq(const std::vector<float> &clsFreq){
			this->clsFreq_.clear();
			this->clsFreq_ = clsFreq;
		}
		void coFreq(const std::vector<std::vector<float> > &coFreq){
			this->coFreq_.clear();
			for(std::vector<std::vector<float> >::const_iterator it=coFreq.begin();\
			it!=coFreq.end();++it){
				this->coFreq_.push_back(*it);
			}
		}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors for trees (to put them in the forest).
		 */
		StructuredTree(StructuredTree const &rhs){
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
			this->binary(rhs.binary());
			this->clsFreq(rhs.clsFreq());
			this->coFreq(rhs.coFreq());
		}
		StructuredTree& operator=(StructuredTree const &rhs){
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
			this->clsFreq(rhs.clsFreq());
			this->coFreq(rhs.coFreq());
			this->binary(rhs.binary());
			return *this;
		}
		//----------------------------------------------------------------------
	protected:
		/** @var log_
		 * For output logging during training.
		 */
		std::ofstream log_;
		/** @var labS_
		 * The size of the label patch.
		 */
		unsigned labSz_;
		/** @var patchW_
		 * The patch width.
		 */
		unsigned patchW_;
		/** @var patchH_
		 * The patch height.
		 */
		unsigned patchH_;
		/** @var patchCh_
		 * The number of channels in the feature matrix.
		 */
		unsigned patchCh_;
		/** @var nodeSize_
		 * The size of the test-nodes.
		 */
		unsigned nodeSize_;
		/** @var minSamples_
		 * Minimum samples to build a test node (else leaf).
		 */
		unsigned minSamples_;
		/** @var cvRNG_
		 * Pointer to a cv random number.
		 */
		CvRNG* cvRNG_;
		/** @var maxDepth_
		 * Maximum tree depth.
		 */
		unsigned maxDepth_;
		/** @var entropy_
		 * How to evaluate the entropy: central label, random label, both.
		 */
		ENTROPY entropy_;
		/** @var consideredCls_
		 * Considered classes (taken in the id order).
		 */
		unsigned consideredCls_;
		/** @var clsFreq_
		 * The class frequencies in the original training data for unbalanced data.
		 */
		std::vector<float> clsFreq_;
		/** @var coFreq_
		 * The class co-frequencies in the original training data for unbalanced data.
		 */
		std::vector<std::vector<float> > coFreq_;
		/** @var binary_
		 * If it is binary tree than save it.
		 */
		bool binary_;
};
//==============================================================================
#endif /* STRUCTUREDTREED_H_ */
#include "StructuredTree.cpp"


