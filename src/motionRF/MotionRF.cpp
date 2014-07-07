/* MotionRF.cpp
 * Author: Silvia-Laura Pintea
 */
#include "MotionRF.h"
#include "MotionTree.h"
#include "MotionTreeNode.h"
//==============================================================================
/** Trains a specified tree in the forest on the given patches.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void MotionRF<L,M,T,F,N,U>::trainForestTree(unsigned min_s,unsigned max_d,CvRNG* pRNG,\
const M& TrData,unsigned samples,unsigned treeId,const char *path2models,\
const std::string &runName,typename StructuredTree<M,T,F,N,U>::ENTROPY entropy,\
unsigned consideredCls,bool binary,bool leafavg,bool parentfreq,bool leafparentfreq,\
const std::string &runname,float entropysigma,bool usepick,bool hogOrSift,\
unsigned growthtype,long unsigned maxleaves){
	std::cout<<"Creating tree "<<treeId<<" on images"<<std::endl;
	this->vTrees_[treeId] = new L<M,T,F,N,U>(min_s,max_d,pRNG,TrData.getLabelSize(),\
		TrData.featW(),TrData.featH(),TrData.getPatchChannels(),treeId,path2models,\
		runName,entropy,consideredCls,binary,leafavg,parentfreq,leafparentfreq,\
		runname,entropysigma,usepick,hogOrSift,growthtype);
	this->vTrees_[treeId]->growTree(TrData,samples,maxleaves);
}
//==============================================================================
/** Gets the histogram information for one tree.
 */
template <template <class M,class T,class F,class N, class U> class L,class M,class T,\
class F,class N,class U>
std::vector<float> MotionRF<L,M,T,F,N,U>::histinfo(unsigned treeId){
	assert(this->vTrees_.size()>treeId);
	return this->vTrees_[treeId]->histinfo();
}
//==============================================================================
/** Predicts on 1 single test patch.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void MotionRF<L,M,T,F,N,U>::regression(std::vector<const U*> &result,\
const T *testPatch,const F *features) const{
	result.resize(this->noTrees_);
	for(int i=0;i<(int)this->vTrees_.size();++i){
		assert(features->size()!=0);
		if(this->hogOrSift_){
			result[i] = this->vTrees_[i]->regression(testPatch,features,\
				this->vTrees_[i]->root(),i);
		}else{
			result[i] = this->vTrees_[i]->siftregression(testPatch,features,\
				this->vTrees_[i]->root(),i);
		}
	}
}
//==============================================================================
template class MotionRF<MotionTree,MotionPatch<MotionPatchFeature<FeaturesMotion>,\
FeaturesMotion>,MotionPatchFeature<FeaturesMotion>,FeaturesMotion,MotionTreeNode\
<MotionLeafNode,MotionPatchFeature<FeaturesMotion> >,MotionLeafNode>;
