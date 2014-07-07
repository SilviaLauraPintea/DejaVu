/* MotionTreeNode.h
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONTREENODE_H_
#define MOTIONTREENODE_H_
#include <Auxiliary.h>
#include <Tree.h>
#include <StructuredTreeNode.h>
//==============================================================================
/** Class for storing the leaf nodes of the structured tree.
 */
class MotionLeafNode:public LabelLeafNode{
	public:
		MotionLeafNode();
		MotionLeafNode(const char *path2models,long unsigned leafid,\
			unsigned treeid,bool binary);
		virtual ~MotionLeafNode(){
			if(this->vMotion_){
				this->vMotion_->release();
				delete this->vMotion_;
				this->vMotion_ = NULL;
			}
			for(unsigned b=0;b<this->vHistos_.size();++b){
				this->vHistos_[b].release();
			}
			if(this->vAppearance_){
				this->vAppearance_->release();
				delete this->vAppearance_;
				this->vAppearance_ = NULL;
			}
		};
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Reads the leaf from a regular file.
		 */
		virtual void readLeafTxt(const char *path2models,long unsigned leafid,\
			unsigned treeid);
		/** Reads the leaf from a binary file.
		 */
		virtual void readLeafBin(const char *path2models,long unsigned leafid,\
		unsigned treeid);
		/** Writes the leaf info into an opened file.
		 */
		virtual void showLeafTxt(const char *path2models,long unsigned leafid,\
			unsigned treeid);
		/** Writes the leaf info into an opened binary file.
		 */
		virtual void showLeafBin(const char *path2models,long unsigned leafid,\
			unsigned treeid);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		bool isempty() const {return this->isempty_;}
		float motionProb() const {return this->motionProb_;}
		float appearanceProb() const {return this->appearanceProb_;}
		cv::Mat* vMotion() const {return this->vMotion_;}
		cv::Mat* vAppearance() const {return this->vAppearance_;}
		float vMotion(cv::Point &pt) const {
			return this->vMotion_->at<float>(pt);
		}
		std::vector<cv::Mat> vHistos() const {return this->vHistos_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void isempty(bool isempty){this->isempty_ = isempty;}
		void appearanceProb(float appearanceProb){
			this->appearanceProb_ = appearanceProb;
		}
		void motionProb(float motionProb){
			this->motionProb_ = motionProb;
		}
		void vMotion(const cv::Mat *vMotion){
			if(this->vMotion_){
				this->vMotion_->release();
				delete this->vMotion_;
				this->vMotion_ = NULL;
			}
			if(vMotion){
				this->vMotion_ = new cv::Mat(*vMotion);
				this->isempty_ = false;
			}
		}
		void vAppearance(const cv::Mat *vAppearance){
			if(this->vAppearance_){
				this->vAppearance_->release();
				delete this->vAppearance_;
				this->vAppearance_ = NULL;
			}
			if(vAppearance){
				this->vAppearance_ = new cv::Mat(*vAppearance);
				this->isempty_     = false;
			}
		}
		void vHistos(const std::vector<cv::Mat> &vHistos){
			for(unsigned b=0;b<this->vHistos_.size();++b){
				this->vHistos_[b].release();
			}
			this->vHistos_.clear();
			this->vHistos_.resize(vHistos.size(),cv::Mat());
			for(unsigned b=0;b<this->vHistos_.size();++b){
				vHistos[b].copyTo(this->vHistos_[b]);
			}
		}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors for trees (to put them in the forest). The assignment
		 * operator is done.
		 */
		MotionLeafNode(MotionLeafNode const &rhs):LabelLeafNode(rhs){
			this->vMotion_     = NULL;
			this->vAppearance_ = NULL;
			this->vHistos(rhs.vHistos());
			this->vMotion(rhs.vMotion());
			this->vAppearance(rhs.vAppearance());
			this->motionProb_     = rhs.motionProb();
			this->appearanceProb_ = rhs.appearanceProb();
			this->isempty_        = rhs.isempty();
		}
		//----------------------------------------------------------------------
	private:
		MotionLeafNode& operator=(const MotionLeafNode&);
	private:
		/** @var isempty_
		 * If the leaf is empty or not.
		 */
		bool isempty_;
		/** @var appearanceProb_
		 * Appearance (feature) probability.
		 */
		float appearanceProb_;
		/** @var motionProb_
		 * Motion probability.
		 */
		float motionProb_;
		/** @var vMotion_
		 * For each pixel we have a motion-patch.
		 */
		cv::Mat* vMotion_;
		/** @var vAppearance_
		 * For each pixel we have an appearance-patch.
		 */
		cv::Mat* vAppearance_;
		/** @var vHistos_
		 * For each pixel we have a patch of histograms.
		 */
		std::vector<cv::Mat> vHistos_;
};
//==============================================================================
//==============================================================================
//==============================================================================
/** Class for accessing the nodes of the structured tree.
 */
template <class U,class T>
class MotionTreeNode:public StructuredTreeNode<U> {
	public:
		MotionTreeNode();
		MotionTreeNode(long unsigned nodeid,const U* leaf=NULL,const long double* test=NULL,\
			unsigned nodeSize=0,const std::vector<cv::Mat> &nodefreq=std::vector<cv::Mat>(),\
			const std::vector<cv::Mat> &freqA=std::vector<cv::Mat>(),const std::vector\
			<cv::Mat> &freqB=std::vector<cv::Mat>(),const typename std::vector<std::vector\
			<const T*> > &setA=std::vector<std::vector<const T*> >(),const \
			typename std::vector<std::vector<const T*> > &setB=std::vector\
			<std::vector<const T*> >());
		virtual ~MotionTreeNode(){
			for(unsigned b=0;b<this->freqA_.size();++b){
				this->freqA_[b].release();
			}
			this->freqA_.clear();
			for(unsigned b=0;b<this->freqB_.size();++b){
				this->freqB_[b].release();
			}
			this->freqB_.clear();
			for(unsigned b=0;b<this->nodefreq_.size();++b){
				this->nodefreq_[b].release();
			}
			this->nodefreq_.clear();
		}
		//----------------------------------------------------------------------
		//----VIRTUAL METHODS---------------------------------------------------
		//----------------------------------------------------------------------
		/** Use default Copy and assignment constructors.
		 */
		virtual MotionTreeNode<U,T>& clone(MotionTreeNode<U,T> const &rhs);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		MotionTreeNode<U,T> *left() const {return this->left_;};
		MotionTreeNode<U,T> *right() const {return this->right_;};
		std::vector<cv::Mat> freqA() const {return this->freqA_;}
		std::vector<cv::Mat> freqB() const {return this->freqB_;}
		std::vector<cv::Mat> nodefreq() const {return this->nodefreq_;}
		std::vector<std::vector<const T*> > setA() const {return this->setA_;};
		std::vector<std::vector<const T*> > setB() const {return this->setB_;};
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void freqA(const std::vector<cv::Mat> &freqA){
			for(unsigned b=0;b<this->freqA_.size();++b){
				this->freqA_[b].release();
			}
			this->freqA_.clear();
			this->freqA_.resize(freqA.size(),cv::Mat());
			for(unsigned b=0;b<this->freqA_.size();++b){
				freqA[b].copyTo(this->freqA_[b]);
			}
		}
		void freqB(const std::vector<cv::Mat> &freqB){
			for(unsigned b=0;b<this->freqB_.size();++b){
				this->freqB_[b].release();
			}
			this->freqB_.clear();
			this->freqB_.resize(freqB.size(),cv::Mat());
			for(unsigned b=0;b<this->freqB_.size();++b){
				freqB[b].copyTo(this->freqB_[b]);
			}
		}
		void nodefreq(const std::vector<cv::Mat> &nodefreq){
			for(unsigned b=0;b<this->nodefreq_.size();++b){
				this->nodefreq_[b].release();
			}
			this->nodefreq_.clear();
			this->nodefreq_.resize(nodefreq.size(),cv::Mat());
			for(unsigned b=0;b<this->nodefreq_.size();++b){
				nodefreq[b].copyTo(this->nodefreq_[b]);
			}
		}
		void setA(const std::vector<std::vector<const T*> > &setA){
			this->setA_ = setA;
		}
		void setB(const std::vector<std::vector<const T*> > &setB){
			this->setB_ = setB;
		}
		void left(MotionTreeNode<U,T> *left){this->left_ = left;};
		void right(MotionTreeNode<U,T> *right){this->right_ = right;};
		//----------------------------------------------------------------------
	protected:
		/** @var nodefreq_
		 * The node frequency to be used in generating child splits.
		 */
		std::vector<cv::Mat> nodefreq_;
		/** @var freqA_
		 * The node frequency to be used in generating child splits.
		 */
		std::vector<cv::Mat> freqA_;
		/** @var freqB_
		 * The node frequency to be used in generating child splits.
		 */
		std::vector<cv::Mat> freqB_;
		/** @var setA_
		 * The left set split by the node test.
		 */
		std::vector<std::vector<const T*> > setA_;
		/** @var setB_
		 * The right set split by the node test.
		 */
		std::vector<std::vector<const T*> > setB_;
	private:
		DISALLOW_COPY_AND_ASSIGN(MotionTreeNode);
		/** @var left_
		 * Left child of the tree.
		 */
		MotionTreeNode<U,T> *left_;
		/** @var right_
		 * Right child of the tree.
		 */
		MotionTreeNode<U,T> *right_;
};
//==============================================================================
#endif /* MOTIONTREENODE_H_ */
#include <MotionTreeNode.cpp>
