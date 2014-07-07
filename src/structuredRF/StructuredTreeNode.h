/* StructuredTreeNode.h
 * Author: Silvia-Laura Pintea
 */
#ifndef STRUCTUREDTREENODE_H_
#define STRUCTUREDTREENODE_H_
#include <Auxiliary.h>
#include <Tree.h>
//==============================================================================
class LabelLeafNode{
	public:
		LabelLeafNode(const char* path2models,long unsigned leafid,\
			unsigned treeid,bool binary);
		LabelLeafNode(){
			this->labelProb_ = 0.0;
			this->vLabels_   = std::vector<unsigned>();
			this->isempty_   = true;
		};
		virtual ~LabelLeafNode(){
			this->vLabels_.clear();
		};
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Reads the leaf from a regular file.
		 */
		virtual void readLeafTxt(const char *path2models,long unsigned leafid,\
			unsigned treeid);
		/** Reads the leaf from a regular file.
		 */
		virtual void readLeafBin(const char *path2models,long unsigned leafid,\
			unsigned treeid);
		/** Writes the leaf info into an opened file.
		 */
		virtual void showLeafTxt(const char *path2models,long unsigned leafid,\
			unsigned treeid);
		/** Writes the leaf info into an opened file.
		 */
		virtual void showLeafBin(const char *path2models,long unsigned leafid,\
			unsigned treeid);
		/** Prints the leaf values to the screen.
		 */
		virtual void print() const{
			std::cout <<"Leaf "<<this->vLabels_.size()<<" "<<this->labelProb_<<std::endl;
		};
		/** Shows a leaf nicely with colors.
		 */
		virtual void display(unsigned labW,unsigned labH,const std::map<cv::Vec3b,\
			unsigned,vec3bCompare> &classinfo,const std::string &path2model) const;
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		float labelProb() const {return this->labelProb_;};
		std::vector<unsigned> vLabels() const {return this->vLabels_;};
		unsigned vLabels(unsigned pos) const {return this->vLabels_[pos];};
		bool isempty() const {return this->isempty_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void labelProb(float labelProb){this->labelProb_ = labelProb;};
		void vLabelsResize(unsigned val){
			this->vLabels_.clear();
			this->vLabels_.resize(val,0);
		}
		void vLabels(const std::vector<unsigned> &vLabels){
			this->vLabels_=vLabels;
			if(!this->vLabels_.empty()){this->isempty_ = false;}
		};
		void vLabels(unsigned pos,unsigned vLabels){this->vLabels_[pos]=vLabels;};
		void isempty(bool isempty){this->isempty_ = isempty;}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors for trees (to put them in the forest). The assignment
		 * operator is done.
		 */
		LabelLeafNode(LabelLeafNode const &rhs){
			this->labelProb(rhs.labelProb());
			this->vLabels(rhs.vLabels());
			this->isempty_ = rhs.isempty();
		}
		//----------------------------------------------------------------------
	private:
		LabelLeafNode& operator=(const LabelLeafNode&);
	protected:
		/** @var isempty_
		 * If the leaf is empty or not.
		 */
		bool isempty_;
		/** @var fgProb_
		* Probability of foreground.
		*/
		float labelProb_;
		/** @var vLabel_
		* For each pixel we have a label-patch.
		*/
		std::vector<unsigned> vLabels_;
};
//==============================================================================
//==============================================================================
//==============================================================================
/** The tree node structure to store the info in it.
 */
template <class U>
class StructuredTreeNode:public node<U> {
	public:
		StructuredTreeNode(){
			this->leaf_     = NULL;
			this->test_     = NULL;
			this->nodeSize_ = 0;
			this->left_     = NULL;
			this->right_    = NULL;
		}
		StructuredTreeNode(long unsigned nodeid,const U* leaf=NULL,const long \
			double* test=NULL,unsigned nodeSize=0);
		virtual ~StructuredTreeNode(){
			if(this->test_){
				delete [] this->test_;
				this->test_ = NULL;
			}
			if(this->leaf_){
				delete this->leaf_;
				this->leaf_ = NULL;
			}
		};
		//----------------------------------------------------------------------
		//----VIRTUAL METHODS---------------------------------------------------
		//----------------------------------------------------------------------
		virtual void showNode(std::ofstream &out,const char *path2model,\
		unsigned treeid,bool binary){
			if(binary){
				this->showNodeBin(out,path2model,treeid);
			}else{
				this->showNodeTxt(out,path2model,treeid);
			}
		}
		/** Write the node to a text output stream.
		 */
		virtual void showNodeTxt(std::ofstream &out,const char* path2model,\
			unsigned treeid);
		/** Write the node to a binary output stream.
		 */
		virtual void showNodeBin(std::ofstream &out,const char *path2model,\
			unsigned treeid);
		/** Use default Copy and assignment constructors.
		 */
		virtual StructuredTreeNode& clone(StructuredTreeNode const &rhs);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		U* leaf() const {return this->leaf_;};
		long double *test() const {return this->test_;};
		unsigned nodeSize() const {return this->nodeSize_;};
		StructuredTreeNode<U> *left() const {return this->left_;};
		StructuredTreeNode<U> *right() const {return this->right_;};
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void leaf(const U* leaf){
			if(this->leaf_){
				delete this->leaf_;
				this->leaf_ = NULL;
			}
			this->leaf_ = new U(*leaf);
		};
		void test(const long double *test){
			if(this->test_){
				delete [] this->test_;
				this->test_ = NULL;
 			}
			if(test){
				this->test_ = new long double[this->nodeSize_]();
				std::fill_n(this->test_,this->nodeSize_,0);
				long double *ptT1       = this->test_;
				const long double *ptT2 = test;
				for(unsigned i=0;i<this->nodeSize_;++ptT1,++ptT2,++i){
					(*ptT1) = (*ptT2);
				}
			}
		};
		void nodeSize(unsigned nodeSize){
			this->nodeSize_ = nodeSize;
		};
		void left(StructuredTreeNode<U> *left){
			this->left_ = left; // no deep copy here, just switch pointers
		};
		void right(StructuredTreeNode<U> *right){
			this->right_ = right; // no deep copy here, just switch pointers
		};
		//----------------------------------------------------------------------
	protected:
		/** @var test_
		 * The test node that keeps the information.
		 */
		long double *test_;
		/** @var nodeSize_
		 */
		unsigned nodeSize_;
		DISALLOW_COPY_AND_ASSIGN(StructuredTreeNode);
		/** @var left_
		 * Left child of the tree.
		 */
		StructuredTreeNode<U> *left_;
		/** @var right_
		 * Right child of the tree.
		 */
		StructuredTreeNode<U> *right_;
};
//==============================================================================
#endif /* STRUCTUREDTREENODE_H_ */
#include "StructuredTreeNode.cpp"
