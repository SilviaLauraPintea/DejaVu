/* StructuredRF.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef StructuredRF_CPP_
#define StructuredRF_CPP_
#include "StructuredRF.h"
//==============================================================================
/** Forest training: training each tree.
 * min_s   -- minimum samples
 * max_d   -- maximum depth
 * samples -- total samples
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::trainForest(int min_s,int max_d,CvRNG* pRNG,\
const M& TrData,int samples,const char* path2models,const std::string &runName,\
typename StructuredTree<M,T,F,N,U>::ENTROPY entropy,unsigned consideredCls,bool binary){
	for(int i=0;i<this->noTrees_;++i){
		std::cout<<"Creating tree "<<i<<" on images"<<std::endl;
		this->vTrees_[i] = new L<M,T,F,N,U>(min_s,max_d,pRNG,TrData.getLabelSize(),\
			TrData.featW(),TrData.featH(),TrData.getPatchChannels(),i,path2models,\
			runName,entropy,consideredCls,binary);
		this->vTrees_[i]->growTree(TrData,samples);
	}
}
//==============================================================================
/** Writes the trees in the forest to files.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::saveForest(const char *filename){
	for(unsigned int i=0; i<this->vTrees_.size(); ++i) {
		this->vTrees_[i]->saveTree();
	}
}
//==============================================================================
/** Writes the trees in the forest to files.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::saveTree(const char *filename,unsigned treeId){
	assert(treeId<this->vTrees_.size());
	this->vTrees_[treeId]->saveTree();
}
//==============================================================================
/** Predicts on 1 single test patch.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::regressionPerTree(const U *result,const T *testPatch,\
const F *features,const char* filename,unsigned treeId){
	this->vTrees_[0] = new L<M,T,F,N,U>(filename,treeId,true);
	result           = this->vTrees_[0]->regression(testPatch,features,\
		this->vTrees_[0]->root(),treeId);
	delete this->vTrees_[0]; this->vTrees_[0] = NULL;
}
//==============================================================================
/** Trains a specified tree in the forest on the given patches.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::trainForestTree(unsigned min_s,unsigned max_d,CvRNG* pRNG,\
const M& TrData,unsigned samples,unsigned treeId,const char* path2models,\
const std::string &runName,typename StructuredTree<M,T,F,N,U>::ENTROPY entropy,\
unsigned consideredCls,bool binary){
	std::cout<<"Creating tree "<<treeId<<" on images"<<std::endl;
	this->vTrees_[treeId] = new L<M,T,F,N,U>(min_s,max_d,pRNG,TrData.getLabelSize(),\
		TrData.featW(),TrData.featH(),TrData.getPatchChannels(),treeId,\
		path2models,runName,entropy,consideredCls,binary);
	this->vTrees_[treeId]->growTree(TrData,samples);
}
//==============================================================================
/** Loads all the trees from directory.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::loadForest(std::string &filename,bool binary){
	std::vector<std::string> trees = Auxiliary<uchar,1>::listDir(filename,".txt");
	for(unsigned int i=0;i<this->noTrees_;++i){
		this->vTrees_[i] = new L<M,T,F,N,U>(filename.c_str(),i,binary);
	}
}
//==============================================================================
/** Writes the trees in the forest to binary files.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::saveForestBin(const char *filename){
	for(unsigned int i=0; i<this->vTrees_.size(); ++i) {
		this->vTrees_[i]->saveTreeBin();
	}
}
//==============================================================================
/** Writes the trees in the forest to binary files.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::saveTreeBin(const char *filename,unsigned treeId){
	assert(treeId<this->vTrees_.size());
	this->vTrees_[treeId]->saveTreeBin();
}
//==============================================================================
/** Loads a trees from directory.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::loadTreeBin(const char *filename,unsigned treeId,\
bool binary){
	if(treeId>=this->vTrees_.size()){
		this->vTrees_.resize(treeId+1);
	}
	this->vTrees_[treeId] = new L<M,T,F,N,U>(filename,treeId,binary);
}
//==============================================================================
/** Loads a trees from directory.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::loadTree(const char *filename,unsigned treeId,\
bool binary){
	if(treeId>=this->vTrees_.size()){
		this->vTrees_.resize(treeId+1);
	}
	this->vTrees_[treeId] = new L<M,T,F,N,U>(filename,treeId,binary);
}
//==============================================================================
/** Loads all the trees from directory.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::loadForestBin(std::string &filename,bool binary){
	std::vector<std::string> trees = Auxiliary<uchar,1>::listDir(filename,".bin");
	for(unsigned int i=0;i<this->noTrees_;++i){
		this->vTrees_[i] = new L<M,T,F,N,U>(filename.c_str(),i,binary);
	}
}
//==============================================================================
/** Predicts on 1 single test patch.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::regression(std::vector<const U*> &result,\
const T *testPatch,const F *features) const{
	result.resize(this->noTrees_);
	for(int i=0;i<(int)this->vTrees_.size();++i){
		result[i] = this->vTrees_[i]->regression(testPatch,features,\
			this->vTrees_[i]->root(),i);
	}
}
//==============================================================================
/** Predicts on 1 single test patch.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void StructuredRF<L,M,T,F,N,U>::regressionPerTree(std::vector<const U*> &result,\
const T *testPatch,const F *features,const char* filename,unsigned treeId) const{
	this->vTrees_.push_back(new L<M,T,F,N,U>(filename,treeId,false));
	result.push_back(this->vTrees_[0]->regression(testPatch,features,\
		this->vTrees_[0]->root()),treeId);
	delete this->vTrees_[0]; this->vTrees_[0] = NULL;
}
//==============================================================================
/** Returns a vector of class prior frequencies (one for each tree).
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
std::vector<std::vector<float> > StructuredRF<L,M,T,F,N,U>::treeClsFreq() const{
	std::vector<std::vector<float> > clsFreq;
	for(int i=0;i<(int)this->vTrees_.size();++i){
		clsFreq.push_back(this->vTrees_[i]->clsFreq());
	}
	return clsFreq;
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif // StructuredRF_CPP_
