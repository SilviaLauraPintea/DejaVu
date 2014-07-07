/* StructuredTree.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef STRUCTUREDTREE_CPP_
#define STRUCTUREDTREE_CPP_
#include "StructuredTree.h"
#include <math.h>
//==============================================================================
//TODO: Parent frequencies no initial data frequencies!!!
//==============================================================================
//==============================================================================
/** Reads the tree from file.
 */
template <class M,class T,class F,class N,class U>
StructuredTree<M,T,F,N,U>::StructuredTree(const char* filename,unsigned treeId,bool binary){
	// [0] Initialize to default the variables from CRTree (useless stuff).
	this->minSamples_  = 0;
	this->root_        = NULL;
	this->path2models_ = filename;
	this->treeId_      = treeId;
	this->binary_      = binary;
	this->log_.open(("log_predict"+Auxiliary<int,1>::number2string\
		(this->treeId_)+".txt").c_str());
	if(!this->log_.is_open()){
		std::cerr<<"[StructuredTree::StructuredTree]: could not open log file"<<std::endl;
	}
	std::cout<<"[StructuredTree<evalFct,T,U>::Tree] Load Tree "<<filename<<std::endl;
	if(this->binary_){
		this->readTreeBin();
	}else{
		this->readTreeTxt();
	}
}
//==============================================================================
/** Reads the tree from a text file.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::readTreeTxt(){
	std::string dummy(this->path2models_);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%s%03d.txt",this->path2models_,this->treeId_);
	std::ifstream in(filename);
	if(in.is_open()){
		// [1] Read general tree info.
		in>>this->treeId_;
		in>>this->labSz_;
		in>>this->patchW_;
		in>>this->patchH_;
		in>>this->patchCh_;
		in>>this->maxDepth_;
		// [2] Read the class frequencies
		unsigned nocls;in>>nocls;
		this->clsFreq_ = std::vector<float>(nocls,0);
		for(std::vector<float>::iterator ptF=this->clsFreq_.begin();\
		ptF!=this->clsFreq_.end();++ptF){
			in>>*ptF;
		}
		// [3] Read the co-frequencies from the data
		unsigned cosize;in>>cosize;
		if(cosize){
			this->coFreq_ = std::vector<std::vector<float> >(nocls,std::vector<float>());
			for(std::vector<std::vector<float> >::iterator ptC=this->coFreq_.begin();\
			ptC!=this->coFreq_.end();++ptC){
				ptC->resize(nocls);
				for(std::vector<float>::iterator pc=ptC->begin();pc!=ptC->end();++pc){
					in>>(*pc);
				}
			}
		}
		// [4] Read the actual tree nodes
		this->readNodeTxt(NULL,in,Tree<N,U>::ROOT);
		// [5] Get the used node size
		this->nodeSize_ = this->root_->nodeSize();
	}else{
		std::cerr<<"Could not read tree: "<<filename<<std::endl;
	}
	delete [] filename;
	in.close();
}
//==============================================================================
/** Reads the tree from a binary file.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::readTreeBin(){
	std::ifstream in;
	std::string dummy(this->path2models_);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%s%03d.bin",this->path2models_,this->treeId_);
	in.open(filename,std::ios::in | std::ios::binary);
	if(in.is_open()){
		// [1] Read general tree info.
		unsigned treeId,labSz,patchW,patchH,patchCh,maxDepth;
		in.read(reinterpret_cast<char*>(&treeId),sizeof(unsigned));
		this->treeId_ = treeId;
		in.read(reinterpret_cast<char*>(&labSz),sizeof(unsigned));
		this->labSz_ = labSz;
		in.read(reinterpret_cast<char*>(&patchW),sizeof(unsigned));
		this->patchW_ = patchW;
		in.read(reinterpret_cast<char*>(&patchH),sizeof(unsigned));
		this->patchH_ = patchH;
		in.read(reinterpret_cast<char*>(&patchCh),sizeof(unsigned));
		this->patchCh_ = patchCh;
		in.read(reinterpret_cast<char*>(&maxDepth),sizeof(unsigned));
		this->maxDepth_ = maxDepth;
		// [2] Read the class frequencies
		unsigned nocls;
		in.read(reinterpret_cast<char*>(&nocls),sizeof(unsigned));
		this->clsFreq_ = std::vector<float>(nocls,0);
		for(std::vector<float>::iterator ptF=this->clsFreq_.begin();\
		ptF!=this->clsFreq_.end();++ptF){
			float afreq;
			in.read(reinterpret_cast<char*>(&afreq),sizeof(float));
			(*ptF) = afreq;
		}
		// [3] Read the co-frequencies from the data
		unsigned cosize;
		in.read(reinterpret_cast<char*>(&cosize),sizeof(unsigned));
		if(cosize){
			this->coFreq_ = std::vector<std::vector<float> >(nocls,std::vector<float>());
			for(std::vector<std::vector<float> >::iterator ptC=this->coFreq_.begin();\
			ptC!=this->coFreq_.end();++ptC){
				ptC->resize(nocls);
				for(std::vector<float>::iterator pc=ptC->begin();pc!=ptC->end();++pc){
					float acofreq;
					in.read(reinterpret_cast<char*>(&acofreq),sizeof(float));
					(*pc) = acofreq;
				}
			}
		}
		// [4] Read the actual tree nodes
		this->readNodeBin(NULL,in,Tree<N,U>::ROOT);
		// [5] Get the used node size
		this->nodeSize_ = this->root_->nodeSize();
	}else{
		std::cerr<<"Could not read tree: "<<filename<<std::endl;
	}
	delete [] filename;
	in.close();
}
//==============================================================================
template <class M,class T,class F,class N,class U>
StructuredTree<M,T,F,N,U>::~StructuredTree(){
	this->clsFreq_.clear();
	this->coFreq_.clear();
	if(this->log_.is_open()){
		this->log_.close();
	}
}
//==============================================================================
/** Just gets total number of patches regardless of class.
 */
template <class M,class T,class F,class N,class U>
unsigned StructuredTree<M,T,F,N,U>::getNoPatches(const std::vector<std::vector\
<const T*> > &trainSet){
	unsigned nopatches = 0;
	for(vectConstIterT it=trainSet.begin();it!=trainSet.end();++it){
		nopatches += it->size();
	}
	return nopatches;
}
//==============================================================================
/** Initializes the size of the labels, number of channels, etc.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::initDataSizes(const M& trData){
	if(!this->labSz_){
		this->labSz_ = trData.getLabelSize();
	}
	if(!this->patchH_){
		this->patchH_ = trData.featH();
	}
	if(!this->patchW_){
		this->patchW_ = trData.featW();
	}
	if(!this->patchCh_){
		this->patchCh_ = trData.getPatchChannels();
	}
}
//==============================================================================
/** Implementing the <<growTee>> with multiple labels.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::growTree(const M& trData,int samples){
	// [0] Get ratio positive patches/negative patches
	this->initDataSizes(trData);
	int pos = 0;
	// [1] Copy trData into trainSet because we will change it
	std::vector<std::vector<const T*> > trainSet = trData.patches();
	// [2] Find class frequencies and probability
	const F* features  = trData.features();
	if(trData.balance()){
		this->clsFreq_ = std::vector<float>(trainSet.size(),1.0);
		this->coFreq_.resize(trainSet.size());
		for(unsigned c=0;c<trainSet.size();++c){
			this->coFreq_[c] = std::vector<float>(trainSet.size(),1.0);
		}
	}else{
		this->clsFreq_ = features->invClsFreq();
		this->coFreq_  = features->invCoFreq();
	}
	// [3] Grow the tree recursively from the root
	long unsigned nodeid = 0.0;
	float prevInfGain    = std::numeric_limits<float>::max();
	this->grow(trainSet,features,nodeid,0,samples,NULL,Tree<N,U>::ROOT,prevInfGain);
}
//==============================================================================
/** Creates the actual tree from the samples.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::grow(const std::vector<std::vector<const T*> > &trainSet,\
const F* features,long unsigned &nodeid,unsigned int depth,int samples,N* parent,\
typename Tree<N,U>::SIDE side,float &prevInfGain){
	// [0] Checks if at least 1 class is not empty, else make a leaf
	unsigned howmany = 0;
	for(vectConstIterT i=trainSet.begin();i!=trainSet.end();++i){ // for each class
		if(!i->empty()){++howmany;}
		if(howmany>1){break;}
	}
	// [1] If at least 1 class and not maxDepth
	if(depth<this->maxDepth_ && howmany>1){
		std::vector<std::vector<const T*> > SetA;
		std::vector<std::vector<const T*> > SetB;
		SetA.resize(trainSet.size());
		SetB.resize(trainSet.size());
		// [2] Node: leaf-index x1 y1 x2 y2 channel threshold testType
		long double *test = new long double[this->nodeSize_]();
		std::fill_n(test,this->nodeSize_,0);
		test[0] = 1;
		test[6] = 0;
		// [3] Check if the pick is not the center (for the full entropy only)
		unsigned pick = cvRandInt(this->cvRNG_)%(this->labSz_);
		if(this->entropy_==CENTER_RANDOM && this->labSz_>1){
			while(pick == (this->labSz_-1)/2){
				pick = cvRandInt(this->cvRNG_)%(this->labSz_);
			}
		}
		if(static_cast<int>(nodeid)%100==0){
			std::cout<<"["<<this->treeId_<<"] [StructuredTree<M,T,U>::grow]: "<<\
				"optimize node: "<<nodeid<<" depth: "<<depth<<std::endl;
		}
		this->log_<<"["<<this->treeId_<<"] [StructuredTree<M,T,U>::grow]: "<<\
			"optimize node: "<<nodeid<<" depth: "<<depth<<std::endl;
		// [5] Find optimal test. Pick one random position per node
		if(this->optimizeTest(SetA,SetB,trainSet,features,(test+1),\
		samples,pick,prevInfGain)){
			// [6] Add a new test node to the tree
			N* newnode = this->addNode(parent,side,test,this->nodeSize_,\
				nodeid,NULL);
			delete [] test;
			// [7] Split the data according to the new test-node
			unsigned countA = 0;unsigned countB = 0;
			vectIterT a=SetA.begin();
			for(vectIterT b=SetB.begin();b!=SetB.end(),a!=SetA.end();++a,++b){
				countA += a->size();
				countB += b->size();
			}
			if(static_cast<int>(nodeid)%100==0){
				std::cout<<"["<<this->treeId_<<"] #A:"<<countA<<" #B:"<<countB<<std::endl;
			}
			this->log_<<"["<<this->treeId_<<"] #A:"<<countA<<" #B:"<<countB<<std::endl;
			// [7.1] Go left. If enough patches are left continue growing else stop
			if(countA>this->minSamples_){
				this->grow(SetA,features,++nodeid,depth+1,samples,\
					newnode,Tree<N,U>::LEFT,prevInfGain);
			}else{
				this->makeLeaf(features,SetA,++nodeid,newnode,Tree<N,U>::LEFT);
			}
			// [7.2] Go right. If enough patches are left continue growing else stop
			if(countB>this->minSamples_){
				this->grow(SetB,features,++nodeid,depth+1,samples,\
					newnode,Tree<N,U>::RIGHT,prevInfGain);
			}else{
				this->makeLeaf(features,SetB,++nodeid,newnode,Tree<N,U>::RIGHT);
			}
		}else{
			// [8] Could not find split (only invalid one leave split)
			this->makeLeaf(features,trainSet,nodeid,parent,side);
			delete [] test;
		}
	}else{
		std::cout<<"["<<this->treeId_<<"] [StructuredTree<M,T,U>::grow]: "<<\
			"Forced leaf (depth>max depth); "<<"left samples: "<<\
			this->getNoPatches(trainSet)<<std::endl;
		this->log_<<"["<<this->treeId_<<"] [StructuredTree<M,T,U>::grow]: "<<\
			"Forced leaf (depth>max depth); "<<"left samples: "<<\
			this->getNoPatches(trainSet)<<std::endl;
		// [9] Only one-class patches are left or maximum depth is reached
		this->makeLeaf(features,trainSet,nodeid,parent,side);
	}
}
//==============================================================================
/** Computes the probabilities of each label-patch given the complete of patches.
 * [1] For each label-patch get its prob as prod of pixel-label probs.
 * [2] Find the label-patch with the maximum prob.
 */
template <class M,class T,class F,class N,class U>
std::vector<std::vector<float> > StructuredTree<M,T,F,N,U>::getPatchProb\
(const std::vector<std::vector<const T*> > &trainSet,const F* features){
	// [0] Get total number of patches
	unsigned allPatches = 0;
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		allPatches += l->size();
	}
	// [1] If only 1 patch then the log-probability is 0
	if(allPatches == 1){
		std::vector<std::vector<float> > probs(trainSet.size(),std::vector<float>());
		for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
			probs[l-trainSet.begin()] = std::vector<float>(l->size(),0.0);
		}
		return probs;
	}
	// [2] Else we count occurrences of class label at each position
	std::vector<float> toNormalize(this->labSz_,0.0);
	std::vector<std::vector<float> > classMarginals(trainSet.size(),\
		std::vector<float>(this->labSz_,0.0));
	// [3] Loop over classes and patches
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
		for(constIterT p=l->begin();p!=l->end();++p){
			std::vector<unsigned> patchLabel = (*p)->label(features);
			// [3.1] Loop over each position in the label patch
			for(std::vector<unsigned>::const_iterator v=patchLabel.begin();\
			v!=patchLabel.end();++v){
				// [3.2] Count class occurrences (*v) at the this position in label patch
				classMarginals[*v][v-(patchLabel.begin())] += this->clsFreq_[*v];
				toNormalize[v-patchLabel.begin()] += this->clsFreq_[*v];
			}
		}
	}
	bool there = false;
	std::vector<std::vector<float> > probs(trainSet.size(),std::vector<float>());
	// [4] Loop over classes and patches
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
		probs[l-trainSet.begin()] = std::vector<float>(l->size(),0.0);
		for(constIterT p=l->begin();p!=l->end();++p){
			std::vector<unsigned> patchLabel = (*p)->label(features);
			// [4.1] Loop over each position in the patch
			for(std::vector<unsigned>::const_iterator v=patchLabel.begin();\
			v!=patchLabel.end();++v){
				// [4.2] If we have a patch then use log-probabilities
				if(this->labSz_>1 && toNormalize[v-patchLabel.begin()]){
					there = true;
					if(!classMarginals[*v][v-(patchLabel.begin())]){
						classMarginals[*v][v-(patchLabel.begin())] = 1.0e-30;
					}
					// [4.3] Normalize the counts to get probabilities
					probs[l-trainSet.begin()][p-(l->begin())] += \
						std::log(classMarginals[*v][v-(patchLabel.begin())]/\
						toNormalize[v-patchLabel.begin()]);
				// [4.4] If only 1 label (no patch), don't use log probabilities
				}else if(toNormalize[v-patchLabel.begin()]){
					probs[l-trainSet.begin()][p-(l->begin())] = \
						classMarginals[*v][v-(patchLabel.begin())]/\
							toNormalize[v-patchLabel.begin()];
				}
			}
			// [5] For multiple probabilities we work with logs<0 (no 0 default)
			if(this->labSz_>1 && !there){
				probs[l-trainSet.begin()][p-(l->begin())] = -std::numeric_limits\
					<float>::max();
			}
		}
	}
	return probs;
}
//==============================================================================
/** Create leaf node from all patches corresponding to a class.
 * [1] For each label-patch get its prob as prod of pixel-label probs.
 * [2] Find the label-patch with the maximum prob
 * [3] Add the best patch in the leaf.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::makeLeaf(const F* features,const std::vector\
<std::vector<const T*> > &trainSet,long unsigned nodeid,N* parent,\
typename Tree<N,U>::SIDE side){
	clock_t begin = clock();
	if(static_cast<int>(nodeid)%100==0){
		std::cout<<"["<<this->treeId_<<"] [StructuredTree<M,T,U>::makeLeaf]: "<<\
			"make a leaf node..."<<std::endl;
	}
	// [0] For each patch compute its probability as prod over pixel probs \
	 	   p(p | Pt) = prod_ij p_ij(c=c_ij | Pt)
	std::vector<std::vector<float> > probs = this->getPatchProb(trainSet,features);
	// [1] Find the first non-empty class
	int first = -1;
	for(std::vector<std::vector<float> >::iterator it=probs.begin();\
	it!=probs.end();++it){
		if(!it->empty()){
			first = it-probs.begin();
			break;
		}
	}
	// [2] Find the best patch probability
	float bestPatchProb             = probs[first][0];
	std::vector<unsigned> bestPatch = trainSet[first][0]->label(features);
	for(std::vector<std::vector<float> >::iterator l=probs.begin();l!=probs.end();++l){
		for(std::vector<float>::iterator p=l->begin();p!=l->end();++p){
			if(*p>bestPatchProb){
				bestPatchProb = *p;
				bestPatch     = trainSet[l-probs.begin()][p-(l->begin())]->\
					label(features);
			}
		}
	}
	// [3] Add leaf: Now add the label patch and its probability in the tree.
	long double* test = new long double[this->nodeSize_]();
	std::fill_n(test,this->nodeSize_,0);
	U* aleaf = new U();
	aleaf->labelProb(bestPatchProb);
	aleaf->vLabels(bestPatch);
	N* leaf     = this->addNode(parent,side,test,this->nodeSize_,nodeid,aleaf);
	delete [] test;
	delete aleaf;
	clock_t end = clock();
	if(static_cast<int>(nodeid)%100==0){
		std::cout<<"["<<this->treeId_<<"] Make 1 leaf time elapsed: "<<\
			double(Auxiliary<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
	}
}
//==============================================================================
/** Generates a random test of a random type -- hack for bins of 2x2 in features.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::generateTestHack(long double* test,unsigned int max_w,\
unsigned int max_h,unsigned int max_c){
	// test[0] - leaf-index; test[1] - x1;      test[2] - y1;        test[3] - x2;
	// test[4] - y2;         test[5] - channel; test[6] - threshold; test[7] - testType
	// test[8] - nodeID
	assert(this->nodeSize_>6);
	/* Here we start with position 1 in real test set:
	 * test[7] - generate a random test-type:
	 * 			 [0] I(x1,y1) > t
	 * 			 [1] I(x1,y1) - I(x2,y2) > t
	 * 			 [2] I(x1,y1) + I(x2,y2) > t
	 * 			 [3] |I(x1,y1) - I(x2,y2)| > t
	 */
	test[6] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % 4);
	/* Here we start with position 1 in real test set:
	 * test[1] - x1, test[2] - y1
	 * test[3] - x2, test[4] - y2
	 */
	test[0] = 0;
	test[1] = 0;
	test[2] = 0;
	test[3] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % (max_c*4));
	/* Here we start with position 1 in real test set:
	 * test[5] - channel
	 */
	test[4] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % (max_c*4));
	while(test[4]==test[3]){
		test[4] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % (max_c*4));
	}
	test[5] = 0;
}
//==============================================================================
/** Generates a random test of a random type.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::generateTest(long double* test,unsigned int max_w,\
unsigned int max_h,unsigned int max_c){
	// test[0] - leaf-index; test[1] - x1;      test[2] - y1;        test[3] - x2;
	// test[4] - y2;         test[5] - channel; test[6] - threshold; test[7] - testType
	// test[8] - nodeID
	assert(this->nodeSize_>6);
	/* Here we start with position 1 in real test set:
	 * test[7] - generate a random test-type:
	 * 			 [0] I(x1,y1) > t
	 * 			 [1] I(x1,y1) - I(x2,y2) > t
	 * 			 [2] I(x1,y1) + I(x2,y2) > t
	 * 			 [3] |I(x1,y1) - I(x2,y2)| > t
	 */
	test[6] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % 4);
	/* Here we start with position 1 in real test set:
	 * test[1] - x1, test[2] - y1
	 * test[3] - x2, test[4] - y2
	 */
	test[0] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_w); // x1
	while(static_cast<int>(test[0])%2!=0){
		test[0] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_w); // x1
	}
	test[1] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_h); // y1
	while(static_cast<int>(test[1])%2!=0){
		test[1] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_h); // y1
	}
	test[2] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_w); // x2
	while(static_cast<int>(test[2])%2!=0){
		test[2] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_w); // x2
	}
	test[3] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_h); // y2
	while(static_cast<int>(test[3])%2!=0){
		test[3] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_h); // y2
	}
	/* Here we start with position 1 in real test set:
	 * test[5] - channel
	 */
	test[4] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % (max_c*4));
	while(test[4]==test[3]){
		test[4] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % (max_c*4));
	}
	test[5] = 0;
}
//==============================================================================
/** Just splits the data into subsets and makes sure the subsets are not empty
 */
template <class M,class T,class F,class N,class U>
float StructuredTree<M,T,F,N,U>::performSplit(std::vector<std::vector<const T*> >& tmpA,\
std::vector<std::vector<const T*> >& tmpB,const std::vector<std::vector<const T*> >& \
TrainSet,const F* features,const std::vector<std::vector<Index> > &valSet,\
unsigned pick,long double threshold,unsigned &sizeA,unsigned &sizeB){
	float gain = std::numeric_limits<float>::max();
	// [0] Do the actual split
	sizeA = 0; sizeB = 0;
	this->split(tmpA,tmpB,TrainSet,valSet,threshold);
	// [1] Do not allow empty set split (all patches end up in set A or B)
	for(vectIterT itA=tmpA.begin();itA!=tmpA.end();++itA){
		sizeA += itA->size();
	}
	for(vectIterT itB=tmpB.begin();itB!=tmpB.end();++itB){
		sizeB += itB->size();
	}
	// [2] Split: 0 - classification (infGain),1 - regression (meanDist)
	if(sizeA>0 && sizeB>0){
		gain = this->measureSet(tmpA,tmpB,features,pick);
	}
	return gain;
}
//==============================================================================
/** Optimizes tests and thresholds.
 * [1] Generate a 5 random values (for x1 y1 x2 y2 channel) in the <<test>> vector.
 * [2] Evaluates the thresholds and finds the minimum and maximum index value [?].
 * [3] Iteratively generate random thresholds to split the index values
 * [4] Split the data according to each threshold.
 * [5] Find the best threshold and store it on the 6th position in <<test>>
 */
template <class M,class T,class F,class N,class U>
bool StructuredTree<M,T,F,N,U>::optimizeTest(std::vector<std::vector<const T*> >& SetA,\
std::vector<std::vector<const T*> >& SetB,const std::vector<std::vector<const T*> >& \
TrainSet,const F* features,long double* test,unsigned int iter,unsigned pick,\
float &best){
	// [0] Temporary data for finding best test
	std::vector<std::vector<Index> > valSet(TrainSet.size());
	best       = std::numeric_limits<float>::max();
	bool found = false;
	long double tmpTest[this->nodeSize_-1];
	std::fill_n(test,this->nodeSize_-1,0);
	// [1] Find best test of ITER iterations
	for(unsigned int i=0;i<iter;++i){
		// [2] Generate binary test without threshold
		this->generateTestHack(&tmpTest[0],this->patchW_,this->patchH_,this->patchCh_);
		// [3] Compute value for each patch
		this->evaluateTestHack(valSet,&tmpTest[0],TrainSet,features);
		// [4] Find min/max values for threshold
		long double vmin = std::numeric_limits<long double>::max();
		long double vmax = -std::numeric_limits<long double>::max();
		for(unsigned int l=0;l<TrainSet.size();++l) {
			if(valSet[l].size()>0) {
				if(vmin>valSet[l].front().val()){
					vmin = valSet[l].front().val();
				}
				if(vmax<valSet[l].back().val()){
					vmax = valSet[l].back().val();
				}
			}
		}
		long double maxDist = vmax-vmin;
		maxDist            *= 1000.0;
		vmin               *= 1000.0;
		vmax               *= 1000.0;
		if(maxDist>1){
			// [5] Generate all 10 thresholds
			std::vector<std::vector<const T*> > tmpA;
			std::vector<std::vector<const T*> > tmpB;
			for(unsigned int j=0;j<10;++j){
				long double thresh = static_cast<long double>(cvRandInt(this->cvRNG_) %\
					static_cast<int>(maxDist))+vmin;
				thresh /= 1000.0;
				// [6] Evaluate how well this threshold splits the data
				unsigned sizeA=0, sizeB=0;
				float gain = this->performSplit(tmpA,tmpB,TrainSet,features,\
					valSet,pick,thresh,sizeA,sizeB);
				// [7] If this is best, update the best test with the current test
				if(gain<best && sizeA>0 && sizeB>0){
					for(int t=0;t<this->nodeSize_-1;++t){
						test[t] = tmpTest[t];
					}
					test[5] = static_cast<long double>(thresh);
					best    = gain;
					SetA    = tmpA;
					SetB    = tmpB;
					found   = true;
				}
			}
		}
	}
	return found;
}
//==============================================================================
/** Overloading the function to carry around the labels matrices.
 */
template <class M,class T,class F,class N,class U>
float StructuredTree<M,T,F,N,U>::measureSet(const std::vector<std::vector<const T*> > \
&SetA,const std::vector<std::vector<const T*> > &SetB,const F* features,unsigned pick){
	return this->InfGain(SetA,SetB,features,pick);
}
//==============================================================================
/** Computes the negative entropy for 1 set wrt to the central pixel of the
 * label patch.
 */
template <class M,class T,class F,class N,class U>
float StructuredTree<M,T,F,N,U>::nEntropy1Cls(const std::vector<std::vector\
<const T*> >& SetA,float &totalFreqA){
	// [0] Get size(A) & center frequencies for p(p00)
	totalFreqA = 0.0;
	std::vector<float> labelFreqA = std::vector<float>(SetA.size(),0.0);
	// [0.1] Loop over classes
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		float freq = this->clsFreq_[l-SetA.begin()];
		// [0.1] Count the frequency of the center class p00
		labelFreqA[l-SetA.begin()] += freq*static_cast<float>(l->size());
		totalFreqA                 += freq*static_cast<float>(l->size());
	}
	// [1] Negative entropy for SetA: sum_i p_i*log(p_i)
	float n_entropyA = 0.0;
	for(std::vector<float>::iterator i=labelFreqA.begin();i!=labelFreqA.end();++i){
		// [1.1] - E(A) = sum_c p(c) log p(c)
		if(*i){
			float prob  = (*i)/totalFreqA;
			n_entropyA += prob*log2(prob);
		}
	}
	return n_entropyA;
}
//==============================================================================
/** Computes the negative entropy for 1 set wrt to a random pixel of the
 * label patch.
 */
template <class M,class T,class F,class N,class U>
float StructuredTree<M,T,F,N,U>::nEntropy1ClsRnd(const std::vector<std::vector\
<const T*> >& SetA,const F* features,float &totalFreqA,unsigned pick){
	// [0] Get size(A) & random position frequency for p(pij)
	totalFreqA = 0.0;
	std::vector<float> labelFreqA = std::vector<float>(SetA.size(),0.0);
	// [0.1] Loop over classes and patches
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		for(constIterT p=l->begin();p!=l->end();++p){
			unsigned newL = (*p)->label(features)[pick];
			float freq;
			if(newL>=this->consideredCls_){
				freq = 0.0;
			}else{
				freq = this->clsFreq_[newL];
			}
			// [0.2] Count the frequency of random pij
			labelFreqA[newL] += freq;
			totalFreqA       += freq;
		}
	}
	// [1] Negative entropy for SetA: sum_i p_i*log(p_i)
	float n_entropyA = 0.0;
	for(std::vector<float>::iterator i=labelFreqA.begin();i!=labelFreqA.end();++i){
		// [1.1] - E(A) = sum_c p(c) log p(c)
		if(*i){
			float prob  = (*i)/totalFreqA;
			n_entropyA += prob*log2(prob);
		}
	}
	return n_entropyA;
}
//==============================================================================
/** Computes the negative entropy for 1 set wrt to the central pixel of the
 * label patch and a randomly picked pixel.
 */
template <class M,class T,class F,class N,class U>
float StructuredTree<M,T,F,N,U>::nEntropy2Cls(const std::vector<std::vector\
<const T*> >& SetA,const F* features,unsigned pick,float &totalFreqA){
	// [0] Get size of set A & joint p(p00,pij)
	totalFreqA = 0.0;
	std::vector<std::vector<float> > labelFreqA = std::vector<std::vector\
		<float> >(SetA.size(),std::vector<float>(SetA.size(),0.0));
	// [0.1] Loop over classes and patches
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		for(constIterT p=l->begin();p!=l->end();++p){
			unsigned newL = (*p)->label(features)[pick];
			float freq;
			if(newL>=this->consideredCls_){
				freq = 0.0;
			}else{
				freq = this->coFreq_[l-SetA.begin()][newL];
			}
			// [0.2] Count the frequency of [p00,pij]
			totalFreqA                       += freq;
			labelFreqA[l-SetA.begin()][newL] += freq;
		}
	}
	// [1] Negative JOINT entropy for SetA: sum_p00 sum_pij p(pij,p00)*log(pij,p00)
	float n_entropyA = 0.0;
	// [1.1] Loop over center classes and random classes
	for(std::vector<std::vector<float> >::iterator i=labelFreqA.begin();\
	i!=labelFreqA.end();++i){
		for(std::vector<float>::iterator j=i->begin();j!=i->end();++j){
			// [1.2] - E(A) = sum_pij sum_p00 p(p00,pij) log p(p00,pij)
			if(*j){
				float prob  = (*j)/totalFreqA;
				n_entropyA += prob*log2(prob);
			}
		}
	}
	return n_entropyA;
}
//==============================================================================
/** Classification information gain check.
 * [1] Associate each app-patch with a random label we pick from the label-patch
 * [2] Compute the negative entropy as: sum_c p(c) log p(c)
 * [3] return: (size(A)entropy(A)+size(B)entropy(B)) / (size(A)+size(B)).
 */
template <class M,class T,class F,class N,class U>
float StructuredTree<M,T,F,N,U>::InfGain(const std::vector<std::vector<const T*> >& SetA,\
const std::vector<std::vector<const T*> >& SetB,const F* features,unsigned pick){
	float n_entropyA = 0.0, n_entropyB = 0.0;
	float sizeA      = 0.0, sizeB      = 0.0;
	// [0] Compute one of the negative entropies
	switch(this->entropy_){
		case(StructuredTree<M,T,F,N,U>::CENTER_RANDOM):
			n_entropyA = this->nEntropy2Cls(SetA,features,pick,sizeA);
			n_entropyB = this->nEntropy2Cls(SetB,features,pick,sizeB);
			break;
		case(StructuredTree<M,T,F,N,U>::RANDOM):
			n_entropyA = this->nEntropy1ClsRnd(SetA,features,sizeA,pick);
			n_entropyB = this->nEntropy1ClsRnd(SetB,features,sizeB,pick);
			break;
		case(StructuredTree<M,T,F,N,U>::CENTER):
			n_entropyA = this->nEntropy1Cls(SetA,sizeA);
			n_entropyB = this->nEntropy1Cls(SetB,sizeB);
			break;
		default:
			n_entropyA = this->nEntropy1ClsRnd(SetA,features,sizeA,pick);
			n_entropyB = this->nEntropy1ClsRnd(SetB,features,sizeB,pick);
			break;
	}
	// [1] Negative entropy add - and take the maximum afterwards
	float entropy = (sizeA*(-n_entropyA)+sizeB*(-n_entropyB))/(sizeA+sizeB);
	return entropy;
}
//==============================================================================
/** Evaluates 1 test (given by 5 numbers: x1, y1, x2, y2, channel).
 * It gets the feature channel and then it accesses it at the 2 randomly selected
 * points and gets the difference between them. Hack for bins of 2x2 in features.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::evaluateTestHack(std::vector<std::vector<Index> >& valSet,\
const long double* test,const std::vector<std::vector<const T*> >& TrainSet,\
const F *features){
	int ch1 = static_cast<int>(*(test+4))%this->patchCh_;
	int pt1 = (*(test+4))/this->patchCh_;
	int ch2 = static_cast<int>(*(test+3))%this->patchCh_;
	int pt2 = (*(test+3))/this->patchCh_;
	assert((this->patchW_-2)%3==0 && (this->patchH_-2)%3==0);
	int stepW = (this->patchW_-2)/3;
	int stepH = (this->patchH_-2)/3;
	std::vector<cv::Point> points;
	points.push_back(cv::Point(stepW,stepH));
	points.push_back(cv::Point(stepW,this->patchH_-stepH));
	points.push_back(cv::Point(this->patchW_-stepW,stepH));
	points.push_back(cv::Point(this->patchW_-stepW,this->patchH_-stepH));
	cv::Point point1 = points[pt1];
	cv::Point point2 = points[pt2];
	// [0] Loop over classes and patches
	for(unsigned int l=0;l<TrainSet.size();++l){
		valSet[l].resize(TrainSet[l].size());
		for(unsigned int i=0;i<TrainSet[l].size();++i){
			// [1] Get a pointer to the test channel
			CvMat* ptC1 = TrainSet[l][i]->feat(features,ch1);
			CvMat* ptC2 = TrainSet[l][i]->feat(features,ch2);
			cv::Mat channel1(ptC1,true); channel1.convertTo(channel1,CV_32FC1);
			cv::Mat channel2(ptC2,true); channel2.convertTo(channel2,CV_32FC1);
			long double p1 = static_cast<long double>(channel1.at<float>(point1.x,point1.y));
			long double p2 = static_cast<long double>(channel2.at<float>(point2.x,point2.y));
			// [2] Get pixel values:  cvPtr2D(img, y, x, NULL);
			switch(static_cast<unsigned>(test[6])){
				case 0: // I(x1,y1) > t
					valSet[l][i].val(p1);
					break;
				case 1: // I(x1,y1) - I (x2,y2) > t
					valSet[l][i].val(p1 - p2);
					break;
				case 2: // I(x1,y1) + I (x2,y2) > t
					valSet[l][i].val(p1 + p2);
					break;
				case 3: // |I(x1,y1) - I (x2,y2)| > t
					valSet[l][i].val(std::abs(p1 - p2));
					break;
			}
			valSet[l][i].index(i);
			channel1.release(); channel2.release();
			cvReleaseMat(&ptC1); cvReleaseMat(&ptC2);
		}
		std::sort(valSet[l].begin(),valSet[l].end());
	}
}
//==============================================================================
/** Evaluates 1 test (given by 5 numbers: x1, y1, x2, y2, channel).
 * It gets the feature channel and then it accesses it at the 2 randomly selected
 * points and gets the difference between them.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::evaluateTest(std::vector<std::vector<Index> >& valSet,\
const long double* test,const std::vector<std::vector<const T*> >& TrainSet,\
const F *features){
	// [0] Loop over classes and patches
	for(unsigned int l=0;l<TrainSet.size();++l){
		valSet[l].resize(TrainSet[l].size());
		for(unsigned int i=0;i<TrainSet[l].size();++i){
			// [1] Get a pointer to the test channel
			CvMat* ptC = TrainSet[l][i]->feat(features,static_cast<unsigned>(test[4]));
			// [2] Get pixel values:  cvPtr2D(img, y, x, NULL);
			int p1 = static_cast<int>(*static_cast<uchar*>(cvPtr2D(ptC,static_cast\
				<unsigned>(test[1]),static_cast<unsigned>(test[0]))));
			int p2 = 0;
			switch(static_cast<unsigned>(test[6])){
				case 0: // I(x1,y1) > t
					valSet[l][i].val(p1);
					break;
				case 1: // I(x1,y1) - I (x2,y2) > t
					p2 = static_cast<int>(*static_cast<uchar*>(cvPtr2D(ptC,\
						static_cast<unsigned>(test[3]),\
						static_cast<unsigned>(test[2]))));
					valSet[l][i].val(p1 - p2);
					break;
				case 2: // I(x1,y1) + I (x2,y2) > t
					p2 = static_cast<int>(*static_cast<uchar*>(cvPtr2D(ptC,\
						static_cast<unsigned>(test[3]),\
						static_cast<unsigned>(test[2]))));
					valSet[l][i].val(p1 + p2);
					break;
				case 3: // |I(x1,y1) - I (x2,y2)| > t
					p2 = static_cast<int>(*static_cast<uchar*>(cvPtr2D(ptC,\
						static_cast<unsigned>(test[3]),\
						static_cast<unsigned>(test[2]))));
					valSet[l][i].val(std::abs(p1 - p2));
					break;
			}
			valSet[l][i].index(i);
			cvReleaseMat(&ptC);
		}
		std::sort(valSet[l].begin(),valSet[l].end());
	}
}
//==============================================================================
/** Writes the current tree into a given file.
 */
template <class M,class T,class F,class N,class U>
bool StructuredTree<M,T,F,N,U>::saveTree() const{
	if(this->binary_){
		this->saveTreeBin();
	}else{
		this->saveTreeTxt();
	}
}
//==============================================================================
/** Writes the current tree into a given file.
 */
template <class M,class T,class F,class N,class U>
bool StructuredTree<M,T,F,N,U>::saveTreeTxt() const{
	std::cout<<"["<<this->treeId_<<"] [StructuredTree<M,T,U>::saveTree] Save Tree "<<\
		this->path2models_<<" "<<this->treeId_<<std::endl;
	std::string dummy(this->path2models_);
	bool done      = false;
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%s%03d.txt",this->path2models_,this->treeId_);
	std::ofstream out(filename);
	if(out.is_open()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		// [0] Write general stuff (sizes, trees, etc.)
		out<<this->treeId_<<" "<<this->labSz_<<" "<<this->patchW_<<" "<<\
			this->patchH_<<" "<<this->patchCh_<<" "<<this->maxDepth_<<std::endl;
		// [1] Write the prior class frequencies
		out<<this->clsFreq_.size()<<" ";
		for(std::vector<float>::const_iterator ptF=this->clsFreq_.begin();\
		ptF!=this->clsFreq_.end();++ptF){
			out<<(*ptF)<<" ";
		}
		out<<std::endl;
		// [2] Write the co-frequencies from the data
		out<<this->coFreq_.size()<<" ";
		for(std::vector<std::vector<float> >::const_iterator ptC=this->coFreq_.begin();\
		ptC!=this->coFreq_.end();++ptC){
			for(std::vector<float>::const_iterator pc=ptC->begin();pc!=ptC->end();++pc){
				out<<(*pc)<<" ";
			}
		}
		out<<std::endl;
		this->preorderTxt(this->root_,out);
	}
	delete [] filename;
	out.close();
	done = true;
	return done;
}
//==============================================================================
/** Writes the current tree into a given file.
 */
template <class M,class T,class F,class N,class U>
bool StructuredTree<M,T,F,N,U>::saveTreeBin() const{
	std::cout<<"["<<this->treeId_<<"] [StructuredTree<M,T,U>::saveTree] Save Tree "<<\
		this->path2models_<<" "<<this->treeId_<<std::endl;
	std::ofstream out;
	std::string dummy(this->path2models_);
	bool done      = false;
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%s%03d.bin",this->path2models_,this->treeId_);
	out.open(filename,std::ios::out|std::ios::binary);
	// FIRST WRITE THE DIMENSIONS OF THE MATRIX
	if(out.is_open()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		// [0] Write general stuff (sizes, trees, etc.)
		unsigned treeId = this->treeId_;
		out.write(reinterpret_cast<const char*>(&treeId),sizeof(unsigned));
		unsigned labSz = this->labSz_;
		out.write(reinterpret_cast<const char*>(&labSz),sizeof(unsigned));
		unsigned patchW = this->patchW_;
		out.write(reinterpret_cast<const char*>(&patchW),sizeof(unsigned));
		unsigned patchH = this->patchH_;
		out.write(reinterpret_cast<const char*>(&patchH),sizeof(unsigned));
		unsigned patchCh = this->patchCh_;
		out.write(reinterpret_cast<const char*>(&patchCh_),sizeof(unsigned));
		unsigned maxDepth = this->maxDepth_;
		out.write(reinterpret_cast<const char*>(&maxDepth),sizeof(unsigned));
		unsigned nocls = this->clsFreq_.size();
		out.write(reinterpret_cast<const char*>(&nocls),sizeof(unsigned));
		for(std::vector<float>::const_iterator ptF=this->clsFreq_.begin();\
		ptF!=this->clsFreq_.end();++ptF){
			float afreq = (*ptF);
			out.write(reinterpret_cast<const char*>(&afreq),sizeof(float));
		}
		unsigned nococls = this->coFreq_.size();
		out.write(reinterpret_cast<const char*>(&nococls),sizeof(unsigned));
		for(std::vector<std::vector<float> >::const_iterator ptC=this->coFreq_.begin();\
		ptC!=this->coFreq_.end();++ptC){
			for(std::vector<float>::const_iterator pc=ptC->begin();pc!=ptC->end();++pc){
				float acoFreq = (*pc);
				out.write(reinterpret_cast<const char*>(&acoFreq),sizeof(float));
			}
		}
		this->preorderBin(this->root_,out);
	}
	delete [] filename;
	out.close();
	done = true;
	return done;
}
//==============================================================================
/** Applies a test to a patch --- hack to get features on 2x2 bins.
 */
template <class M,class T,class F,class N,class U>
bool StructuredTree<M,T,F,N,U>::applyTestHack(const long double *test,const T* testPatch,\
const F* features) const{
	// [0] Node: leaf-index [1] x1 [2] y1 [3] x2 [4] y2 [5] channel [6] \
		   threshold [7] testType
	int ch1   = static_cast<int>(*(test+5))%this->patchCh_;
	int pt1   = (*(test+5))/this->patchCh_;
	int ch2   = static_cast<int>(*(test+4))%this->patchCh_;
	int pt2   = (*(test+4))/this->patchCh_;
	assert((this->patchW_-2)%3==0 && (this->patchH_-2)%3==0);
	int stepW = (this->patchW_-2)/3;
	int stepH = (this->patchH_-2)/3;
	std::vector<cv::Point> points;
	points.push_back(cv::Point(stepW,stepH));
	points.push_back(cv::Point(stepW,this->patchH_-stepH));
	points.push_back(cv::Point(this->patchW_-stepW,stepH));
	points.push_back(cv::Point(this->patchW_-stepW,this->patchH_-stepH));
	cv::Point point1 = points[pt1];
	cv::Point point2 = points[pt2];
	CvMat* ptC1 = testPatch->feat(features,ch1);
	CvMat* ptC2 = testPatch->feat(features,ch2);
	cv::Mat channel1(ptC1,true); channel1.convertTo(channel1,CV_32FC1);
	cv::Mat channel2(ptC2,true); channel2.convertTo(channel2,CV_32FC1);
	long double p1 = static_cast<long double>(channel1.at<float>(point1.x,point1.y));
	long double p2 = static_cast<long double>(channel2.at<float>(point2.x,point2.y));
	bool result;
	switch(static_cast<unsigned>(*(test+7))){
		case 0: // I(x1,y1) > t
			result = (p1>=(*(test+6)));
			break;
		case 1: // I(x1,y1) - I(x2,y2) > t
			result = ((p1-p2)>=(*(test+6)));
			break;
		case 2: // I(x1,y1) + I(x2,y2) > t
			result = ((p1+p2)>=(*(test+6)));
			break;
		case 3: // |I(x1,y1) - I(x2,y2)| > t
			result = (std::abs(p1-p2)>=(*(test+6)));
			break;
	}
	channel1.release(); channel2.release();
	cvReleaseMat(&ptC1);cvReleaseMat(&ptC2);
	return result;
}
//==============================================================================
template <class M,class T,class F,class N,class U>
bool StructuredTree<M,T,F,N,U>::applyTest(const long double *test,const T* testPatch,\
const F* features) const{
	// [0] Node: leaf-index [1] x1 [2] y1 [3] x2 [4] y2 [5] channel [6] \
		   threshold [7] testType
	CvMat* ptC = testPatch->feat(features,(*(test+5)));
	int p1     = static_cast<int>(*(uchar*)cvPtr2D(ptC,(*(test+2)),(*(test+1))));
	int p2     = 0;
	bool result;
	// [1] Based on the node type evaluate:
	switch(static_cast<unsigned>(*(test+7))){
		case 0: // I(x1,y1) > t
			result = (p1>=(*(test+6)));
			break;
		case 1: // I(x1,y1) - I(x2,y2) > t
			p2     = static_cast<int>(*(uchar*)cvPtr2D(ptC,(*(test+4)),(*(test+3))));
			result = ((p1-p2)>=(*(test+6)));
			break;
		case 2: // I(x1,y1) + I(x2,y2) > t
			p2     = static_cast<int>(*(uchar*)cvPtr2D(ptC,(*(test+4)),(*(test+3))));
			result = ((p1+p2)>=(*(test+6)));
			break;
		case 3: // |I(x1,y1) - I(x2,y2)| > t
			p2     = static_cast<int>(*(uchar*)cvPtr2D(ptC,(*(test+4)),(*(test+3))));
			result = (std::abs(p1-p2)>=(*(test+6)));
			break;
	}
	cvReleaseMat(&ptC);
	return result;
}
//==============================================================================
/** Predicts on a one single test patch.
 * A node contains: [0] -- node type (0,1,-1),[1] -- x1,[2] -- y1,[3] -- x2,
 * 					[4] -- y2,[5] -- channel,[6] -- threshold, [7] -- test type,
 * 					[8] -- node ID
 */
template <class M,class T,class F,class N,class U>
const U* StructuredTree<M,T,F,N,U>::regression(const T* testPatch,const F* features,\
N* node,unsigned treeid){
	if(node->left()!=NULL || node->right()!=NULL){
		// [0] If test-node, apply test and go left/right
		typename Tree<N,U>::SIDE side = static_cast<typename Tree<N,U>::SIDE>\
			(StructuredTree<M,T,F,N,U>::applyTestHack(node->test(),testPatch,features));
		if(side == Tree<N,U>::LEFT){
			return this->regression(testPatch,features,node->left(),treeid);
		// [0.2] If the side to continue is right, go right
		}else if(side == Tree<N,U>::RIGHT){
			return this->regression(testPatch,features,node->right(),treeid);
		}
	// [1] If we reached a leaf, return it
	}else{
		if(this->treeId_!=treeid){
			std::cout<<"[StructuredTree<M,T,F,U>::regression] this->treeID: "<<\
				this->treeId_<<" != treeID: "<<treeid<<std::endl;
		}
		const U* leaf = new U(this->path2models_,node->nodeid(),treeid,\
			this->binary_);
		return leaf;
	}
}
//==============================================================================
/** Displays the leaves of the tree.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::showLeaves(unsigned labWidth,unsigned labHeight,\
const std::map<cv::Vec3b,unsigned,vec3bCompare> &classinfo){
	// TODO: fill this in
}
//==============================================================================
/** Splits the training samples into a left set and a right set.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::split(std::vector<std::vector<const T*> >& SetA,\
std::vector<std::vector<const T*> >& SetB,const std::vector<std::vector<const T*> >& \
TrainSet,const std::vector<std::vector<Index> >& valSet,long double t){
	SetA.resize(TrainSet.size());
	SetB.resize(TrainSet.size());
	// [0] Loop over all classes
	for(unsigned int l = 0; l<TrainSet.size(); ++l){
		// [1] Search for largest value such that val<t
		vector<Index>::const_iterator it = valSet[l].begin();
		while(it!=valSet[l].end() && (it->val())<t){
			++it;
		}
		// [2] Make the split at this point
		SetA[l].resize(it-valSet[l].begin());
		SetB[l].resize(TrainSet[l].size()-SetA[l].size());
		it = valSet[l].begin();
		for(unsigned int i=0; i<SetA[l].size(); ++i,++it){
			SetA[l][i] = TrainSet[l][it->index()];
		}
		it = valSet[l].begin()+SetA[l].size();
		for(unsigned int i=0; i<SetB[l].size(); ++i,++it){
			SetB[l][i] = TrainSet[l][it->index()];
		}
	}
}
//==============================================================================
/** Recursively read tree from file.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::readNodeTxt(N *parent,std::ifstream &in,\
typename Tree<N,U>::SIDE side){
	// [0] Read the node info
	unsigned nodetype;in>>nodetype;
	unsigned nodesize;in>>nodesize;
	long unsigned nodeid; in>>nodeid;
	long double* test = new long double[nodesize]();
	std::fill_n(test,nodesize,0);
	long double* pTest = test;
	for(long double n=0;n<nodesize;++n,++pTest){
		in>>(*pTest);
	}
	N *newnode;
	// [1] Add the node on the proper side
	if(!nodetype){
		U* leaf = new U(this->path2models_,nodeid,this->treeId_,false);
		newnode = this->addNode(parent,side,test,nodesize,nodeid,leaf);
		delete [] test;
		delete leaf;
		return;
	}else{
		newnode = this->addNode(parent,side,test,nodesize,nodeid,NULL);
		delete [] test;
		this->readNodeTxt(newnode,in,Tree<N,U>::LEFT);
		this->readNodeTxt(newnode,in,Tree<N,U>::RIGHT);
	}
}
//==============================================================================
/** Recursively read tree from binary file.
 */
template <class M,class T,class F,class N,class U>
void StructuredTree<M,T,F,N,U>::readNodeBin(N *parent,std::ifstream &in,\
typename Tree<N,U>::SIDE side){
	// [0] Read the node info
	unsigned nodetype; in.read(reinterpret_cast<char*>(&nodetype),sizeof(unsigned));
	unsigned nodesize; in.read(reinterpret_cast<char*>(&nodesize),sizeof(unsigned));
	long unsigned nodeid; in.read(reinterpret_cast<char*>(&nodeid),sizeof(long unsigned));
	long double* test = new long double[nodesize]();
	std::fill_n(test,nodesize,0);
	long double* pTest = test;
	for(long double n=0;n<nodesize;++n,++pTest){
		long double testval;
		in.read(reinterpret_cast<char*>(&testval),sizeof(long double));
		(*pTest) = testval;
	}
	N *newnode;
	// [1] Add the node on the proper side
	if(!nodetype){
		U* leaf = new U(this->path2models_,nodeid,this->treeId_,true);
		newnode = this->addNode(parent,side,test,nodesize,nodeid,leaf);
		delete [] test;
		delete leaf;
		return;
	}else{
		newnode = this->addNode(parent,side,test,nodesize,nodeid,NULL);
		delete [] test;
		this->readNodeBin(newnode,in,Tree<N,U>::LEFT);
		this->readNodeBin(newnode,in,Tree<N,U>::RIGHT);
	}
}
//==============================================================================
/** Adds a node to the tree given the parent node and the side.
 */
template <class M,class T,class F,class N,class U>
N* StructuredTree<M,T,F,N,U>::addNode(N *parent,typename Tree<N,U>::SIDE side,\
const long double *test,unsigned nodeSize,long unsigned nodeid,const U *leaf){
	// [0] If we don't have a root yet, then we add it
	if(!this->root_ && side==Tree<N,U>::ROOT){
		this->root_ = new N(nodeid,leaf,test,nodeSize);
		return this->root_;
	// [1] Else we put it on the indicated side
	}else{
		N* newNode = new N(nodeid,leaf,test,nodeSize);
		if(side == Tree<N,U>::LEFT){
			parent->left(newNode);
		}else if(side == Tree<N,U>::RIGHT){
			parent->right(newNode);
		}
		return newNode;
	}
	std::cerr<<"[Tree<evalFct,U>::addNode]: no correct side to add on."<<std::endl;
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif // STRUCTUREDTREED_CPP_





























