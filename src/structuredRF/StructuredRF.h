/* StructuredRF.h
 * Author: Silvia-Laura Pintea
 */
#ifndef StructuredRF_H_
#define StructuredRF_H_
#pragma once
#include "StructuredTree.h"
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
class StructuredRF{
	public:
		virtual ~StructuredRF(){
			for(unsigned i=0;i<this->vTrees_.size();++i){
				if(this->vTrees_[i]){
					delete this->vTrees_[i];
					this->vTrees_[i] = NULL;
				}
			}
			this->vTrees_.clear();
			this->noTrees_ = 0;
		};
		StructuredRF(int trees = 0){
			this->noTrees_ = trees;
			this->vTrees_.resize(this->noTrees_,NULL);
		}
		//----------------------------------------------------------------------
		/** Loads all the trees from directory.
		 */
		void loadForest(std::string &filename,bool binary);
		/** Trains the forest on the given patches.
		 */
		void trainForest(int min_s,int max_d,CvRNG *pRNG,const M &TrData,\
			int samples,const char* path2models,const std::string &runName,\
			typename StructuredTree<M,T,F,N,U>::ENTROPY entropy,\
			unsigned consideredCls,bool binary);
		/** Trains a specified tree in the forest on the given patches.
		 */
		void trainForestTree(unsigned min_s,unsigned max_d,CvRNG *pRNG,\
			const M &TrData,unsigned samples,unsigned treeId,const char*\
			path2models,const std::string &runName,typename StructuredTree\
			<M,T,F,N,U>::ENTROPY entropy,unsigned consideredCls,bool binary);
		/** Returns a vector of class prior frequencies (one for each tree).
		 */
		std::vector<std::vector<float> > treeClsFreq() const;
		/** Writes the trees in the forest to files.
		 */
		void saveForest(const char *filename);
		/** Writes the trees in the forest to files.
		 */
		void saveTree(const char *filename,unsigned treeId);
		/** Writes the trees in the forest to binary files.
		 */
		void saveForestBin(const char *filename);
		/** Writes the trees in the forest to binary files.
		 */
		void saveTreeBin(const char *filename,unsigned treeId);
		/** Loads all the trees from directory.
		 */
		void loadForestBin(std::string &filename,bool binary);
		/** Loads a trees from directory.
		 */
		void loadTreeBin(const char *filename,unsigned treeId,bool binary);
		/** Loads a trees from directory.
		 */
		void loadTree(const char *filename,unsigned treeId,bool binary);
		/** Predicts on 1 single test patch.
		 */
		void regressionPerTree(std::vector<const U*> &result,const T *testPatch,\
			const F *features,const char* filename,unsigned treeId) const;
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Predicts on 1 single test patch.
		 */
		virtual void regression(std::vector<const U*> &result,const T *testPatch,\
			const F *features) const;
		virtual void regressionPerTree(const U *result,const T *testPatch,\
			const F *features,const char* filename,unsigned treeId);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned noTrees() const {return this->noTrees_;}
		std::vector<L<M,T,F,N,U>*> vTrees() const {return this->vTrees_;};
		L<M,T,F,N,U>* vTrees(unsigned pos) const {return this->vTrees_[pos];};
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void noTrees(unsigned noTrees){this->noTrees_ = noTrees;}
		void vTrees(const std::vector<L<M,T,F,N,U>*> &vTrees){
			for(unsigned i=0;i<this->vTrees_.size();++i){
				if(this->vTrees_[i]){
					delete this->vTrees_[i];
					 this->vTrees_[i] = NULL;
				}
			}
			vTrees_.clear();
			for(typename std::vector<L<M,T,F,N,U>*>::const_iterator i=vTrees.begin();\
			i!=vTrees.end();++i){
				this->vTrees_.push_back(new L<M,T,F,N,U>(*(*i)));
			}
		};
		void vTrees(unsigned pos,const L<M,T,F,N,U> *tree){
			if(this->vTrees_[pos]){
				delete this->vTrees_[pos];
				this->vTrees_[pos] = NULL;
			}
			this->vTrees_.push_back(new L<M,T,F,N,U>(*tree));
		};
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors because we use vectors of pointers to trees.
		 */
		StructuredRF(const StructuredRF &rhs){
			this->noTrees_ = rhs.noTrees();
			this->vTrees(rhs.vTrees());
		}
		StructuredRF& operator=(const StructuredRF &rhs){
			if(this == &rhs) return *this;
			if(this){delete this;}
			this->vTrees(rhs.vTrees());
			return *this;
		}
		//----------------------------------------------------------------------
	protected:
		/** @var noTrees_
		 * The number of trees to use.
		 */
		unsigned noTrees_;
		/** @var vTrees_
		 * The vector of pointer to trees.
		 */
		std::vector<L<M,T,F,N,U>*> vTrees_;
	private:
};
//==============================================================================
#endif /* StructuredRF_H_ */
#include "StructuredRF.cpp"









