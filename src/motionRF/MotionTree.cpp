/* MotionTree.cpp
 * Author: Silvia-Laura Pintea
 */
#include "MotionTree.h"
#include "MotionTreeNode.h"
#include <stdlib.h>
#include <RunningStat.h>
//==============================================================================
template <class M,class T,class F,class N,class U>
MotionTree<M,T,F,N,U>::MotionTree(const char* filename,unsigned treeid,bool binary){
	// [0] Initialize to default the variables from CRTree (useless stuff).
	this->minSamples_     = 0;
	this->root_           = NULL;
	this->usepick_        = false;
	this->treeId_         = treeid;
	this->path2models_    = filename;
	this->binary_         = binary;
	this->hogOrSift_      = 0;
	this->leafavg_        = true;
	this->motionH_        = 0;
	this->motionW_        = 0;
	this->parentFreq_     = false;
	this->leafParentFreq_ = false;
	this->entropythresh_  = 0;
	this->log_.open(("log_predict"+Auxiliary<int,1>::number2string\
		(this->treeId_)+".txt").c_str());
	if(!this->log_.is_open()){
		std::cerr<<"[MotionTree::MotionTree]: could not open log file"<<std::endl;
	}
	std::cout<<"[MotionTree<evalFct,T,U>::Tree] Load Tree "<<filename<<" "<<\
		treeid<<std::endl;
	if(binary){
		this->readTreeBin();
	}else{
		this->readTree();
	}
}
//==============================================================================
/** Reads the tree from a regular text file.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::readTree(){
	std::string dummy(this->path2models_);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%s%03dtree.txt",this->path2models_,this->treeId_);
	std::ifstream in(filename);
	if(in.is_open()){
		// [1] Read general tree info.
		unsigned dummy;
		in>>dummy;
		in>>this->patchW_;
		in>>this->patchH_;
		in>>this->patchCh_;
		in>>this->maxDepth_;
		in>>this->entropythresh_;
		// [2] Read the histogram info
		unsigned infosize;in>>infosize;
		this->histinfo_ = std::vector<float>(infosize,0);
		for(std::vector<float>::iterator ptB=this->histinfo_.begin();\
		ptB!=this->histinfo_.end();++ptB){
			in>>*ptB;
		}
		// [4] Read the actual tree nodes
		this->readNode(NULL,in,Tree<N,U>::ROOT);
		// [5] Get the used node size
		this->nodeSize_ = this->root_->nodeSize();
	}else{
		std::cerr<<"Could not read tree: "<<filename<<std::endl;
	}
	delete [] filename;
	in.close();
};
//==============================================================================
/** Reads the tree from a binary file.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::readTreeBin(){
	// [0] Initialize to default the variables from CRTree (useless stuff).
	std::ifstream in;
	std::string dummy(this->path2models_);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%s%03dtree.bin",this->path2models_,this->treeId_);
	in.open(filename,std::ios::in | std::ios::binary);
	if(in.is_open()){
		// [1] Read general tree info.
		unsigned treeId,patchW,patchH,patchCh,maxDepth,entropyTh;
		in.read(reinterpret_cast<char*>(&treeId),sizeof(unsigned));
		in.read(reinterpret_cast<char*>(&patchW),sizeof(unsigned));
		this->patchW_ = patchW;
		in.read(reinterpret_cast<char*>(&patchH),sizeof(unsigned));
		this->patchH_ = patchH;
		in.read(reinterpret_cast<char*>(&patchCh),sizeof(unsigned));
		this->patchCh_ = patchCh;
		in.read(reinterpret_cast<char*>(&maxDepth),sizeof(unsigned));
		this->maxDepth_ = maxDepth;
		in.read(reinterpret_cast<char*>(&entropyTh),sizeof(float));
		this->entropythresh_ = entropyTh;
		// [2] Read the histogram info
		unsigned infosize; in.read(reinterpret_cast<char*>(&infosize),sizeof(unsigned));
		this->histinfo_ = std::vector<float>(infosize,0);
		for(std::vector<float>::iterator ptB=this->histinfo_.begin();\
		ptB!=this->histinfo_.end();++ptB){
			float info;
			in.read(reinterpret_cast<char*>(&info),sizeof(float));
			(*ptB) = info;
		}
		// [3] Read the actual tree nodes
		this->readNodeBin(NULL,in,Tree<N,U>::ROOT);
		// [4] Get the used node size
		this->nodeSize_ = this->root_->nodeSize();
	}else{
		std::cerr<<"Could not read tree: "<<filename<<std::endl;
	}
	delete [] filename;
	in.close();
};
//==============================================================================
/** Recursively read tree from file.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::readNode(N *parent,std::ifstream &in,typename \
Tree<N,U>::SIDE side){
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
		U* leaf     = new U(); // don;t load the leaves yet.
		N* leafnode = new N(nodeid,leaf,test,nodesize);
		leafnode    = this->addNode(leafnode,parent,side);
		delete [] test;
		delete leaf;
		return;
	}else{
		N* newnode = new N(nodeid,NULL,test,nodesize);
		newnode    = this->addNode(newnode,parent,side);
		delete [] test;
		this->readNode(newnode,in,Tree<N,U>::LEFT);
		this->readNode(newnode,in,Tree<N,U>::RIGHT);
	}
}
//==============================================================================
/** Recursively read tree from binary file.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::readNodeBin(N *parent,std::ifstream &in,typename \
Tree<N,U>::SIDE side){
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
		U* leaf     = new U(); // cheapper not to load the leaves not
		N* nodeleaf = new N(nodeid,leaf,test,nodesize);
		nodeleaf    = this->addNode(nodeleaf,parent,side);
		delete [] test;
		delete leaf;
		return;
	}else{
		N* newnode = new N(nodeid,NULL,test,nodesize);
		newnode    = this->addNode(newnode,parent,side);
		delete [] test;
		this->readNodeBin(newnode,in,Tree<N,U>::LEFT);
		this->readNodeBin(newnode,in,Tree<N,U>::RIGHT);
	}
}
//==============================================================================
/** Initializes the size of the labels, number of channels, etc.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::initDataSizes(const M& trData){
	if(!this->labSz_){
		this->labSz_ = trData.getLabelSize();
	}
	if(!this->patchH_){
		this->patchH_ = trData.featH();
	}
	if(!this->patchW_){
		this->patchW_ = trData.featW();
	}
	if(!this->motionH_){
		this->motionH_ = trData.motionH();
	}
	if(!this->motionW_){
		this->motionW_ = trData.motionW();
	}
	if(!this->patchCh_){
		this->patchCh_ = trData.getPatchChannels();
	}
}
//==============================================================================
/** Get "class inverse frequencies" --- inverse priors for reweighting.
 */
template <class M,class T,class F,class N,class U>
std::vector<cv::Mat> MotionTree<M,T,F,N,U>::setFreq(const F* features,\
const std::vector<std::vector<const T*> > &allTrainSet){
	// [0] Find the frequencies of the data
	this->histinfo_ = features->histinfo();
	std::vector<cv::Mat> freq;
	if(this->usepick_){
		freq.push_back(cv::Mat::zeros(cv::Size(this->histinfo_[0],1),CV_32FC1));
	}else{
		freq.resize(this->histinfo_[0],cv::Mat());
	}
	// [1] For each bin = prod_dim K(sample[dim]-bin[dim])
	for(vectConstIterT l=allTrainSet.begin();l!=allTrainSet.end();++l){
		for(constIterT p=l->begin();p!=l->end();++p){ // over patches
			if(this->usepick_){
				for(unsigned i=0;i<std::sqrt(this->motionH_*this->motionW_);++i){
					cv::Point pick = cv::Point(cvRandInt(this->cvRNG_)%(this->motionW_),\
						cvRandInt(this->cvRNG_)%(this->motionH_));
					// the x and y histograms gave the same lengths
					cv::Mat asmpl = (*p)->histo(features,pick);
					freq[0]      += asmpl;
					asmpl.release();
				}
			}else{
				std::vector<cv::Mat> histo       = (*p)->histo(features);
				std::vector<cv::Mat>::iterator f = freq.begin();
				for(std::vector<cv::Mat>::iterator h=histo.begin();h!=histo.end(),\
				f!=freq.end();++h,++f){
					if(f->empty()){
						h->copyTo(*f);
					}else{
						(*f) += (*h);
					}
					h->release();
				} // over bins
			}
		} // over patches
	} // over classes - not the case here
	return freq;
}
//==============================================================================
/** Given and input sample, find its corresponding inverse frequency.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::getProbMagni(const std::vector<cv::Mat> &probs,\
const std::vector<float> &bininfo,const std::vector<float> &values,\
const cv::Point &pos){
	float value = 0.0;
	assert(values.size()==2 || values.size()==4);
	if(values.size()==4){
		/*
			0 - histinfo.push_back(this->bins_);
			1 - histinfo.push_back(stepXX);
			2 - histinfo.push_back(stepXY);
			3 - histinfo.push_back(stepYX);
			4 - histinfo.push_back(stepYY);
			5 - histinfo.push_back(stdxx);
			6 - histinfo.push_back(stdxy);
			7 - histinfo.push_back(stdyx);
			8 - histinfo.push_back(stdyy);
		 */
		float bins = std::pow(static_cast<double>(bininfo[0]),1.0/4.0),
			 minXX = bininfo[1],minXY = bininfo[2],minYX = bininfo[3],
			 minYY = bininfo[4],maxXX = bininfo[5],maxXY = bininfo[6],
			 maxYX = bininfo[7],maxYY = bininfo[8];
		// [0] Recover the bin position
		float stepXX = (maxXX-minXX)/bins;
		float stepXY = (maxXY-minXY)/bins;
		float stepYX = (maxYX-minYX)/bins;
		float stepYY = (maxYY-minYY)/bins;
		float valxx  = std::max(std::min(values[0],maxXX),minXX);
		float valxy  = std::max(std::min(values[1],maxXY),minXY);
		float valyx  = std::max(std::min(values[2],maxYX),minYX);
		float valyy  = std::max(std::min(values[3],maxYY),minYY);
		int xxpos    = std::floor((valxx-minXX)/stepXX);
		int xypos    = std::floor((valxy-minXY)/stepXY);
		int yxpos    = std::floor((valyx-minYX)/stepYX);
		int yypos    = std::floor((valyy-minYY)/stepYY);
		if(xxpos >= bins){xxpos = bins-1;}
		if(xypos >= bins){xypos = bins-1;}
		if(yxpos >= bins){yxpos = bins-1;}
		if(yypos >= bins){yypos = bins-1;}
		// pos -- the position in the patch if !usepick, else 0
		value           = probs[yypos+bins*(yxpos+bins*(xypos+bins*xxpos))].at<float>(pos);
		float neighbors = 0.0;
		unsigned index  = 0;
		if(xxpos-1>=0){ // bin-xx - 1
			neighbors += probs[yypos+bins*(yxpos+bins*(xypos+bins*(xxpos-1)))].at<float>(pos);
			++index;
		}
		if(xxpos+1<bins){ // bin-xx + 1
			neighbors += probs[yypos+bins*(yxpos+bins*(xypos+bins*(xxpos+1)))].at<float>(pos);
			++index;
		}
		if(xypos-1>=0){ // bin-xy - 1
			neighbors += probs[yypos+bins*(yxpos+bins*((xypos-1)+bins*xxpos))].at<float>(pos);
			++index;
		}
		if(xypos+1<bins){ // bin-xy + 1
			neighbors += probs[yypos+bins*(yxpos+bins*((xypos+1)+bins*xxpos))].at<float>(pos);
			++index;
		}
		if(yxpos-1>=0){ // bin-yx - 1
			neighbors += probs[yypos+bins*((yxpos-1)+bins*(xypos+bins*xxpos))].at<float>(pos);
			++index;
		}
		if(yxpos+1<bins){ // bin-yx + 1
			neighbors = probs[yypos+bins*((yxpos+1)+bins*(xypos+bins*xxpos))].at<float>(pos);
			++index;
		}
		if(yypos-1>=0){ // bin-yy - 1
			neighbors = probs[(yypos-1)+bins*(yxpos+bins*(xypos+bins*xxpos))].at<float>(pos);
			++index;
		}
		if(yypos+1<bins){ // bin-yy + 1
			neighbors += probs[(yypos+1)+bins*(yxpos+bins*(xypos+bins*xxpos))].at<float>(pos);
			++index;
		}
		value = 0.5*value+0.5*neighbors/static_cast<float>(index);
	}else{
		/*
			0 - histinfo.push_back(this->bins_);
			1 - histinfo.push_back(stepX);
			2 - histinfo.push_back(stepY);
			3 - histinfo.push_back(stdx);
			4 - histinfo.push_back(stdy);
		 */
		float bins = std::pow(static_cast<double>(bininfo[0]),1.0/2.0),
			 minX = bininfo[1], minY = bininfo[2],
			 maxX = bininfo[3], maxY = bininfo[4];
		float valx  = std::max(std::min(values[0],maxX),minX);
		float valy  = std::max(std::min(values[1],maxY),minY);
		float stepX = (maxX-minX)/bins;
		float stepY = (maxY-minY)/bins;
		// [0] Recover the bin position on X
		int xpos    = std::floor((valx-minX)/stepX);
		int ypos    = std::floor((valy-minY)/stepY);
		if(xpos >= bins){xpos = bins-1;}
		if(ypos >= bins){ypos = bins-1;}
		// pos -- the position in the patch if !usepick, else 0
		value           = probs[ypos+bins*xpos].at<float>(pos);
		float neighbors = 0.0;
		unsigned index  = 0;
		if(xpos-1>=0){ // bin-X - 1
			neighbors += probs[ypos+bins*(xpos-1)].at<float>(pos);
			++index;
		}
		if(xpos+1<bins){ // bin-X + 1
			neighbors += probs[ypos+bins*(xpos+1)].at<float>(pos);
			++index;
		}
		if(ypos-1>=0){ // bin-Y - 1
			neighbors += probs[(ypos-1)+bins*xpos].at<float>(pos);
			++index;
		}
		if(ypos+1<bins){ // bin-Y + 1
			neighbors += probs[(ypos+1)+bins*xpos].at<float>(pos);
			++index;
		}
		value = 0.5*value+0.5*neighbors/static_cast<float>(index);
	}
	return value;
}
//==============================================================================
/** Given and input sample, find its corresponding inverse frequency.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::getProbAngle(const std::vector<cv::Mat> &probs,\
const std::vector<float> &bininfo,const std::vector<float> &values,const \
cv::Point &pos){
	float value = 0.0;
	assert(values.size()==2 || values.size()==4);
	if(values.size()==4){
		/*
		 * 0 - histinfo.push_back(bins);
		 * 1 - histinfo.push_back(step);
		 */
		float angleX = std::atan2(values[1],values[0])+M_PI;
		float angleY = std::atan2(values[3],values[2])+M_PI;
		float bins   = std::sqrt(bininfo[0]),step = bininfo[1];
		// [0] Recover the bin position on X
		int posX = std::floor(angleX/step);
		int posY = std::floor(angleY/step);
		// [1] Recover the bin position on Y
		if(posX >= bins){posX = bins-1;}
		if(posY >= bins){posY = bins-1;}
		value = probs[posY+bins*posX].at<float>(pos);
		float neighbors = 0.0;
		unsigned index  = 0;
		if(posX-1>=0){ // bin-X - 1
			neighbors += probs[posY+bins*(posX-1)].at<float>(pos);
			++index;
		}
		if(posX+1<bins){ // bin-X + 1
			neighbors += probs[posY+bins*(posX+1)].at<float>(pos);
			++index;
		}
		if(posY-1>=0){ // bin-Y - 1
			neighbors += probs[(posY-1)+bins*posX].at<float>(pos);
			++index;
		}
		if(posY+1<bins){ // bin-Y + 1
			neighbors += probs[(posY+1)+bins*posX].at<float>(pos);
			++index;
		}
		value = 0.5*value+0.5*neighbors/static_cast<float>(index);
	}else{
		std::vector<float> histinfo;
		/*
		 * 0 - histinfo.push_back(bins);
		 * 1 - histinfo.push_back(step);
		 */
		float angle = std::atan2(values[1],values[0])+M_PI;
		float bins  = bininfo[0], step = bininfo[1];
		// [0] Recover the bin position on X
		int anglepos = std::floor(angle/step);
		// [1] Recover the bin position on Y
		if(anglepos >= bins){anglepos = bins-1;}
		value = probs[anglepos].at<float>(pos);
		float neighbors = 0.0;
		unsigned index  = 0;
		if(anglepos-1>=0){ // bin - 1
			neighbors += probs[anglepos-1].at<float>(pos);
			++index;
		}
		if(anglepos+1<bins){ // bin + 1
			neighbors += probs[anglepos+1].at<float>(pos);
			++index;
		}
		value = 0.5*value+0.5*neighbors/static_cast<float>(index);
	}
	return value;
}
//==============================================================================
/** Implementing the <<growTee>> with multiple labels.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::growTree(const M& trData,unsigned nodeiters,long unsigned maxleaves){
	// [0] Get ratio positive patches/negative patches
	this->initDataSizes(trData);
	int pos = 0;
	// [1] Copy trData into trainSet because we will change it
	std::vector<std::vector<const T*> > trainSet = trData.patches();
	// [2] No class frequencies and probability
	std::vector<cv::Mat> freq;
	if(this->entropy_!=MotionTree<M,T,F,N,U>::MEAN_DIFF){
		freq = this->setFreq(trData.features(),trainSet);
	}
	// [3] Grow the tree recursi5ely from the root
	long unsigned nodeid = 0.0;
	this->grow(trainSet,trData.features(),nodeid,0,nodeiters,NULL,Tree<N,U>::ROOT,\
		freq,freq,maxleaves);
	for(std::vector<cv::Mat>::iterator f=freq.begin();f!=freq.end();++f){
		f->release();
	}
	freq.clear();
}
//==============================================================================
/** Creates the actual tree from the samples.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::grow(const std::vector<std::vector<const T*> > &trainSet,\
const F *features,long unsigned &nodeid,unsigned int depth,unsigned nodeiters,N* \
parent,typename Tree<N,U>::SIDE side,std::vector<cv::Mat> &prevfreq,\
std::vector<cv::Mat> &prevprevfreq,long unsigned maxleaves,bool showSplits){
	switch(this->growthtype_){
		case(Tree<N,U>::DEPTH_FIRST):
			this->growDepth(trainSet,features,nodeid,0,nodeiters,NULL,\
				Tree<N,U>::ROOT,prevfreq,prevprevfreq,showSplits);
			break;
		case(Tree<N,U>::WORST_FIRST):
			this->growLimit(trainSet,features,prevfreq,maxleaves,\
				nodeiters,showSplits);
			break;
		case(Tree<N,U>::BREADTH_FIRST):
			this->growLimit(trainSet,features,prevfreq,maxleaves,\
				nodeiters,showSplits);
			break;
		default:
			this->growLimit(trainSet,features,prevfreq,maxleaves,\
				nodeiters,showSplits);
			break;
	}
}
//==============================================================================
/** Creates the actual tree from the samples.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::growDepth(const std::vector<std::vector<const T*> > &trainSet,\
const F *features,long unsigned &nodeid,unsigned int depth,unsigned nodeiters,N* \
parent,typename Tree<N,U>::SIDE side,std::vector<cv::Mat> &prevfreq,\
std::vector<cv::Mat> &prevprevfreq,bool showSplits){
	unsigned noPatches = this->getNoPatches(trainSet);
	// [-1] You can not make a leaf on 1 patch!
	if(noPatches<=this->minSamples_){
		if(static_cast<int>(nodeid)%1000==0){
			std::cout<<"["<<this->treeId_<<"] [MotionTree::grow] make a leaf "<<\
				"[less than 5] "<<noPatches<<std::endl;
		}
		this->makeLeaf(features,trainSet,nodeid,parent,side,noPatches,prevprevfreq,0);
	}else{
		std::vector<std::vector<const T*> > SetA;
		std::vector<std::vector<const T*> > SetB;
		std::vector<cv::Mat> freqA, freqB;
		for(std::vector<cv::Mat>::iterator pf=prevfreq.begin();pf!=prevfreq.end();++pf){
			freqA.push_back(pf->clone()); freqB.push_back(pf->clone());
		}
		SetA.resize(trainSet.size());
		SetB.resize(trainSet.size());
		// [1] Pick a random position in the motion patch.
		this->mpick_ = cv::Point(cvRandInt(this->cvRNG_)%(this->motionW_),\
			cvRandInt(this->cvRNG_)%(this->motionH_));
		if(static_cast<int>(nodeid)%1000==0){
			std::cout<<"["<<this->treeId_<<"] [MotionTree<M,T,U>::grow]: "<<\
				"optimize node: "<<nodeid<<" depth: "<<depth<<std::endl;
			this->log_<<"["<<this->treeId_<<"] [MotionTree<M,T,U>::grow]: "<<\
				"optimize node: "<<nodeid<<" depth: "<<depth<<std::endl;
		}
		// [2] Node: leaf-index x1 y1 x2 y2 channel threshold testType
		long double *test = new long double[this->nodeSize_]();
		std::fill_n(test,this->nodeSize_,0);
		test[0] = 1; // not leaf
		test[6] = 0; // initial threshold
		// [3] Find the optimal test for this node
		unsigned there  = 0;
		float bestSplit = std::numeric_limits<float>::max();
		float entropyA,entropyB;
		bool istest     = this->optimizeTest(SetA,SetB,trainSet,features,(test+1),\
			nodeiters,0,freqA,freqB,bestSplit,entropyA,entropyB);
		while(!istest && there<10){ // try again 10 times
			delete [] test;
			test = new long double[this->nodeSize_]();
			std::fill_n(test,this->nodeSize_,0);
			test[0] = 1;
			test[6] = 0;
			std::vector<cv::Mat>::iterator fA = freqA.begin();
			std::vector<cv::Mat>::iterator fB = freqB.begin();
			for(std::vector<cv::Mat>::iterator pf=prevfreq.begin();pf!=prevfreq.end(),\
			fA!=freqA.end(),fB!=freqB.end();++pf,++fA,++fB){
				fA->release(); pf->copyTo(*fA);
				fB->release(); pf->copyTo(*fB);
			}
			istest = this->optimizeTest(SetA,SetB,trainSet,features,(test+1),\
				nodeiters,0,freqA,freqB,bestSplit,entropyA,entropyB);
			++there;
		}
		if(istest){
			// [4] Add a new test node to the tree
			N* newnode = new N(nodeid,NULL,test,this->nodeSize_);
			newnode    = this->addNode(newnode,parent,side);
			delete [] test;
			// [5] Split the data according to the new test-node
			unsigned countA = 0;unsigned countB = 0;
			vectIterT a=SetA.begin();
			for(vectIterT b=SetB.begin();b!=SetB.end(),a!=SetA.end();++a,++b){
				countA += a->size();
				countB += b->size();
			}
			std::cout<<"["<<this->treeId_<<"] #A:"<<countA<<" #B:"<<countB<<\
				" entropyA:"<<entropyA<<" entropyB:"<<entropyB<<std::endl;
			if(static_cast<int>(nodeid)%1000==0){
				this->log_<<"["<<this->treeId_<<"] #A:"<<countA<<" #B:"<<countB<<\
					" entropyA:"<<entropyA<<" entropyB:"<<entropyB<<std::endl;
			}
			if(showSplits && (countA>100 || countB>100) && this->treeId_==0){
				this->showPickedSplit(SetA,SetB,features,nodeid);
			}
			// [5.1] Go left. If enough patches are left continue growing else stop
  			bool stopA = false,stopB = false;
  		
			if(this->entropythresh_){
				stopA = (entropyA<=this->entropythresh_);
				stopB = (entropyB<=this->entropythresh_);
  			}else{
				switch(this->entropy_){
					case(MotionTree<M,T,F,N,U>::APPROX_ANGLE_KERNEL):
						stopA = this->stopCosSimilarity(SetA,features);
						stopB = this->stopCosSimilarity(SetB,features);
						break;
					case(MotionTree<M,T,F,N,U>::APPROX_MAGNI_KERNEL):
						stopA = this->stopEuclDist(SetA,features);
						stopB = this->stopEuclDist(SetB,features);
						break;
					default:
						stopA = this->stopCosSimilarity(SetA,features);
						stopB = this->stopCosSimilarity(SetB,features);
						break;
				}
			}
			if(!stopA && noPatches>this->minSamples_ && depth<this->maxDepth_){
				this->growDepth(SetA,features,++nodeid,depth+1,nodeiters,\
					newnode,Tree<N,U>::LEFT,freqA,prevfreq);
				for(std::vector<cv::Mat>::iterator fA=freqA.begin();fA!=freqA.end();++fA){
					fA->release();
				}
				freqA.clear();
			}else{
				std::cout<<"["<<this->treeId_<<"] [MotionTree::grow] make a leaf "<<\
					"[left] "<<countA<<std::endl;
				this->makeLeaf(features,SetA,++nodeid,newnode,Tree<N,U>::LEFT,countA,prevfreq,entropyA);
			}
			// [5.2] Go right. If enough patches are left continue growing else stop
			if(!stopB && noPatches>this->minSamples_ && depth<this->maxDepth_){
				this->growDepth(SetB,features,++nodeid,depth+1,nodeiters,\
					newnode,Tree<N,U>::RIGHT,freqB,prevfreq);
				for(std::vector<cv::Mat>::iterator fA=freqA.begin();fA!=freqA.end();++fA){
					fA->release();
				}
				freqA.clear();
			}else{
				std::cout<<"["<<this->treeId_<<"] [MotionTree::grow] make a leaf "<<\
					"[right] "<<countB<<std::endl;
				this->makeLeaf(features,SetB,++nodeid,newnode,Tree<N,U>::RIGHT,countB,prevfreq,entropyB);
			}
		}else{
			// [6] Could not find split (only invalid one leave split)
			std::cout<<"["<<this->treeId_<<"] [MotionTree::grow] make a leaf "<<\
				"[no valid test] "<<noPatches<<std::endl;
			this->makeLeaf(features,trainSet,nodeid,parent,side,noPatches,prevprevfreq,0);
			delete [] test;
		}
		std::vector<cv::Mat>::iterator fA=freqA.begin();
		for(std::vector<cv::Mat>::iterator fB=freqB.begin();fB!=freqB.end(),\
		fA!=freqA.end();++fA,++fB){
			fA->release(); fB->release();
		}
		freqA.clear(); freqB.clear();
	}
}
//==============================================================================
/** Get the node info. Does all administrative bits to get the info be saved in the node.
 */
template <class M,class T,class F,class N,class U>
bool MotionTree<M,T,F,N,U>::addNodeInfo(N*current,N *parent,const typename Tree<N,U>::SIDE side,\
const F *features,unsigned &countA,unsigned &countB,float &entropyA,float &entropyB,\
unsigned nodeiters,bool showsplits){
	std::vector<std::vector<const T*> > SetA;
	std::vector<std::vector<const T*> > SetB;
	std::vector<cv::Mat> freqA, freqB;
	for(std::vector<cv::Mat>::const_iterator pf=current->nodefreq().begin();pf!=\
	current->nodefreq().end();++pf){
		freqA.push_back(pf->clone()); freqB.push_back(pf->clone());
	}
	std::vector<std::vector<const T*> > trainSet = current->setA();
	SetA.resize(trainSet.size());
	SetB.resize(trainSet.size());
	// [1] Pick a random position in the motion patch.
	this->mpick_ = cv::Point(cvRandInt(this->cvRNG_)%(this->motionW_),\
		cvRandInt(this->cvRNG_)%(this->motionH_));
	// [2] Node: leaf-index x1 y1 x2 y2 channel threshold testType
	long double *test = new long double[this->nodeSize_]();
	std::fill_n(test,this->nodeSize_,0);
	test[0] = 1; // not leaf
	test[6] = 0; // initial threshold
	// [3] Find the optimal test for this node
	entropyA=0; entropyB=0;
	unsigned there  = 0;
	float bestSplit = std::numeric_limits<float>::max();
	bool istest     = this->optimizeTest(SetA,SetB,trainSet,features,(test+1),\
		nodeiters,0,freqA,freqB,bestSplit,entropyA,entropyB);
	while(!istest && there<10){ // try again 10 times
		delete [] test;
		test = new long double[this->nodeSize_]();
		std::fill_n(test,this->nodeSize_,0);
		test[0] = 1;
		test[6] = 0;
		std::vector<cv::Mat>::iterator fA = freqA.begin();
		std::vector<cv::Mat>::iterator fB = freqB.begin();
		for(std::vector<cv::Mat>::const_iterator pf=current->nodefreq().begin();\
		pf!=current->nodefreq().end(),fA!=freqA.end(),fB!=freqB.end();++pf,++fA,++fB){
			fA->release(); pf->copyTo(*fA);
			fB->release(); pf->copyTo(*fB);
		}
		istest = this->optimizeTest(SetA,SetB,trainSet,features,(test+1),\
			1,0,freqA,freqB,bestSplit,entropyA,entropyB);
		++there;
	}
	countA=0, countB=0;
	vectIterT      a = SetA.begin();
	vectConstIterT t = trainSet.begin();
	for(vectIterT b=SetB.begin();b!=SetB.end(),a!=SetA.end(),t!=trainSet.end();\
	++t,++a,++b){
		countA += a->size(); countB += b->size();
	}
	std::cout<<"[Check]: valid:"<<istest<<" && #left:"<<countA<<">0 && #right:"<<\
		countB<<">0"<<std::endl;
	bool added = istest && countB>0 && countA>0;
	if(added){
		std::cout<<"["<<this->treeId_<<"] #A:"<<countA<<" #B:"<<countB<<\
			" entropyA:"<<entropyA<<" entropyB:"<<entropyB<<std::endl;
		this->log_<<"["<<this->treeId_<<"] #A:"<<countA<<" #B:"<<countB<<\
			" entropyA:"<<entropyA<<" entropyB:"<<entropyB<<std::endl;
		if(showsplits && (countA>100 || countB>100) && this->treeId_==0){
			this->showPickedSplit(SetA,SetB,features,current->nodeid());
		}
		// [4] Add a new test node to the tree
		current->test(test);
		current->freqA(freqA);
		current->freqB(freqB);
		current->setA(SetA);
		current->setB(SetB);
		current = this->addNode(current,parent,side);
	}
	for(unsigned b=0;b<freqA.size();++b){
		freqA[b].release();
	}
	freqA.clear();
	for(unsigned b=0;b<freqB.size();++b){
		freqB[b].release();
	}
	freqB.clear();
	delete [] test;
	return added;
}
//==============================================================================
/** Adds a node to the tree given the parent node and the side.
 */
template <class M,class T,class F,class N,class U>
N* MotionTree<M,T,F,N,U>::addNode(N *current,N *parent,typename Tree<N,U>::SIDE side){
	// [0] If we don't have a root yet, then we add it
	if(!this->root_ && side==Tree<N,U>::ROOT){
		this->root_ = current;
		return this->root_;
	// [1] Else we put it on the indicated side
	}else{
		if(side == Tree<N,U>::LEFT){
			parent->left(current);
		}else if(side == Tree<N,U>::RIGHT){
			parent->right(current);
		}
		return current;
	}
	std::cerr<<"[Tree<evalFct,U>::addNode]: no correct side to add on."<<std::endl;
}
//==============================================================================
/** Grows the tree either breath-first or worst-first until a leaf is reached.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::growLimit(const std::vector<std::vector<const T*> > \
&trainSet,const F *features,const std::vector<cv::Mat> &prevfreq,long unsigned maxleaves,\
unsigned nodeiters,bool showSplits){
	long unsigned nodeid = 0;
	unsigned noPatches   = this->getNoPatches(trainSet);
	// [0] Too few patches in the set
	if(noPatches==1){
		std::cout<<"["<<this->treeId_<<"] [MotionTree::grow] make a leaf "<<\
			"[1 patch] "<<noPatches<<std::endl;
		N *dummy = NULL;
		this->makeLeaf(features,trainSet,nodeid,dummy,Tree<N,U>::ROOT,\
			noPatches,prevfreq,0);
	}else{
		this->noleaves_ = 0;
		// [1] Gets the node info and adds the first root node
		unsigned countA=0, countB=0, countAll=0;
		float entropyA=0, entropyB=0;
		float rootentropy = std::numeric_limits<float>::max();
		N* rootnode = new N(0,NULL,NULL,this->nodeSize_,prevfreq,prevfreq,prevfreq,trainSet,trainSet);
		this->queue_.push_back(std::pair<float,N*>(rootentropy,rootnode));
		this->queueinfo_.push_back(std::pair<unsigned,N*>(static_cast<unsigned>(Tree<N,U>::ROOT),NULL));
		while(!this->queue_.empty()){
			// Sort the list and take out the wanted node
			std::pair<float,N*> parent;
			std::pair<unsigned,N*> parentinfo;
			if(static_cast<typename Tree<N,U>::GROWTH_TYPE>(this->growthtype_)==\
			Tree<N,U>::WORST_FIRST){
				typename std::vector<std::pair<float,N*> >::iterator pos  = this->queue_.begin();
				typename std::vector<std::pair<unsigned,N*> >::iterator posi = this->queueinfo_.begin();
				float worst = -std::numeric_limits<float>::min();
				for(typename std::vector<std::pair<float,N*> >::iterator q=\
				this->queue_.begin();q!=this->queue_.end();++q){
					if(worst<q->first){
						worst = q->first;
						pos   = q;
						posi  = this->queueinfo_.begin()+(q-this->queue_.begin());
					}
				}
				parentinfo = (*posi);
				parent     = (*pos);
				this->queue_.erase(pos);
				this->queueinfo_.erase(posi);
			}else if(static_cast<typename Tree<N,U>::GROWTH_TYPE>(this->growthtype_)\
			==Tree<N,U>::BREADTH_FIRST){
				parent = this->queue_[0];
				this->queue_.erase(this->queue_.begin());
				parentinfo = this->queueinfo_[0];
				this->queueinfo_.erase(this->queueinfo_.begin());
			}
			// Now split it and add it in the tree
			unsigned countA=0, countB=0; 
			float entropyA=0, entropyB=0;
			N *currentnode    = parent.second;
			N *parentnode     = parentinfo.second;
			float nodeentropy = parent.first;
			typename Tree<N,U>::SIDE side = static_cast<typename Tree<N,U>::SIDE>\
				((parentinfo.first)-(currentnode->nodeid()));
			// If added Updates the parent node info with the split info
			
std::cout<<"node-entropy:"<<nodeentropy<<" #queue:"<<this->queue_.size()<<std::endl;


			bool tobeadded = nodeentropy>this->entropythresh_ && (this->noleaves_+\
				this->queue_.size()+1)<maxleaves;
			if(tobeadded){
				tobeadded = this->addNodeInfo(currentnode,parentnode,side,features,\
					countA,countB,entropyA,entropyB,nodeiters,showSplits);
			}
			// Now in the parent node we have child info
			if(!tobeadded){ // make a leaf and be done
				std::cout<<"["<<this->treeId_<<"] [MotionTree::grow] make a leaf "<<\
					"[right] "<<countA+countB<<std::endl;
				this->makeLeaf(features,currentnode->setA(),currentnode->nodeid(),\
					parentnode,side,countA+countB,currentnode->nodefreq(),nodeentropy);
				++this->noleaves_;
				delete currentnode; currentnode = NULL;
			}else{	
				// Add child node on left side
				++nodeid;
				N* nodeleft = new N(nodeid,NULL,NULL,this->nodeSize_,currentnode->freqA(),\
					currentnode->freqA(),currentnode->freqA(),currentnode->setA(),currentnode->setA());
				this->queue_.push_back(std::pair<float,N*>(entropyA,nodeleft));
				this->queueinfo_.push_back(std::pair<unsigned,N*>(static_cast<unsigned>(Tree<N,U>::LEFT)+\
					nodeid,currentnode));
				// Add child node on left side
				++nodeid;
				N* noderight = new N(nodeid,NULL,NULL,this->nodeSize_,currentnode->freqB(),\
					currentnode->freqB(),currentnode->freqB(),currentnode->setB(),currentnode->setB());
				this->queue_.push_back(std::pair<float,N*>(entropyB,noderight));
				this->queueinfo_.push_back(std::pair<unsigned,N*>(static_cast<unsigned>(Tree<N,U>::RIGHT)+\
					nodeid,currentnode));
			}

		} // end queue looping
	}
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
bool MotionTree<M,T,F,N,U>::optimizeTest(std::vector<std::vector<const T*> >& SetA,\
std::vector<std::vector<const T*> >& SetB,const std::vector<std::vector<const T*> >& \
TrainSet,const F* features,long double* test,unsigned int iter,unsigned pick,\
std::vector<cv::Mat> &freqA,std::vector<cv::Mat> &freqB,float &best,float &entropyA,\
float &entropyB){
	std::vector<cv::Mat> finfreqA, finfreqB;
	std::vector<cv::Mat>::iterator fA = freqA.begin();
	for(std::vector<cv::Mat>::iterator fB=freqB.begin();fB!=freqB.end(),\
	fA!=freqA.end();++fB,++fA){
		finfreqA.push_back(fA->clone());
		finfreqB.push_back(fB->clone());
	}
	// [0] Temporary data for finding best test
	std::vector<std::vector<Index> > valSet(TrainSet.size());
	best               = std::numeric_limits<float>::max();
	bool found         = false;
	unsigned bestsizeA = 0;
	unsigned bestsizeB = 0;
	long double tmpTest[this->nodeSize_-1];
	std::fill_n(tmpTest,this->nodeSize_-1,0);
	// [1] Find best test of ITER iterations
	for(unsigned int i =0;i<iter;++i){
		std::cout<<"[MotionTree<M,T,F,N,U>::optimizeTest] iter "<<i<<" .."<<std::endl;
		// [2] Generate binary test without threshold
		if(this->hogOrSift_){
			this->generateTestHack(&tmpTest[0],this->patchW_,this->patchH_,this->patchCh_);
			// [3] Compute value for each patch
			this->evaluateTestHack(valSet,&tmpTest[0],TrainSet,features);
		}else{
			this->siftgenerateTest(&tmpTest[0],this->patchW_,this->patchH_,this->patchCh_);
			// [3] Compute value for each patch
			this->siftevaluateTest(valSet,&tmpTest[0],TrainSet,features);
		}
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
		std::cout<<maxDist<<" "<<vmax<<" "<<vmin<<std::endl;
		if(maxDist>1){
			// [5] Generate all 10 thresholds
			std::vector<std::vector<const T*> > tmpA;
			std::vector<std::vector<const T*> > tmpB;
			for(unsigned int j=0;j<10;++j){
				long double thresh = static_cast<double>(cvRandInt(this->cvRNG_) %\
					static_cast<int>(maxDist))+vmin;
				thresh /= 1000.0;
				std::cout<<"threshold: "<<thresh<<std::endl;
				// [6] Evaluate how well this threshold splits the data
				unsigned sizeA=0, sizeB=0;
				std::vector<cv::Mat> tmpfreqA, tmpfreqB;
				std::vector<cv::Mat>::iterator fA = freqA.begin();
				for(std::vector<cv::Mat>::iterator fB=freqB.begin();fB!=freqB.end(),\
				fA!=freqA.end();++fB,++fA){
					tmpfreqA.push_back(fA->clone());
					tmpfreqB.push_back(fB->clone());
				}
				float tmpentropyA, tmpentropyB;
				float gain = this->performSplit(tmpA,tmpB,TrainSet,features,\
					valSet,pick,thresh,sizeA,sizeB,tmpfreqA,tmpfreqB,\
					tmpentropyA,tmpentropyB);
				// [7] If this is best, update the best test with the current test
				if(gain<best && sizeA>0 && sizeB>0){
					for(int t=0;t<this->nodeSize_-1;++t){
						test[t] = tmpTest[t];
					}
					test[5]     = static_cast<long double>(thresh);
					best        = gain;
					SetA        = tmpA;        SetB        = tmpB;
					entropyA    = tmpentropyA; entropyB    = tmpentropyB;
					bestsizeA   = sizeA;       bestsizeB   = sizeB;
					std::vector<cv::Mat>::iterator ffA = finfreqA.begin();
					std::vector<cv::Mat>::iterator tfA = tmpfreqA.begin();
					std::vector<cv::Mat>::iterator ffB = finfreqB.begin();
					for(std::vector<cv::Mat>::iterator tfB=tmpfreqB.begin();tfB!=\
					tmpfreqB.end(),ffB!=finfreqB.end(),tfA!=tmpfreqA.end(),ffA!=\
					finfreqA.end();++ffB,++ffA,++tfA,++tfB){
						ffA->release(); tfA->copyTo(*ffA); tfA->release();
						ffB->release(); tfB->copyTo(*ffB); tfB->release();
					}
					found = true;
				}
			}
		}
	}
	// [8] If we found a best split then get the frequencies of the sides
	if(found && this->parentFreq_){
		std::vector<cv::Mat>::iterator ffA = finfreqA.begin();
		std::vector<cv::Mat>::iterator fA  = freqA.begin();
		std::vector<cv::Mat>::iterator ffB = finfreqB.begin();
		for(std::vector<cv::Mat>::iterator fB=freqB.begin();fB!=\
		freqB.end(),ffB!=finfreqB.end(),fA!=freqA.end(),ffA!=\
		finfreqA.end();++ffB,++ffA,++fA,++fB){
			fA->release(); ffA->copyTo(*fA); ffA->release();
			fB->release(); ffB->copyTo(*fB); ffB->release();
		}
	}
	return found;
}
//==============================================================================
/** Just splits the data into subsets and makes sure the subsets are not empty
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::performSplit(std::vector<std::vector<const T*> >& tmpA,\
std::vector<std::vector<const T*> >& tmpB,const std::vector<std::vector<const T*> >& \
TrainSet,const F* features,const std::vector<std::vector<Index> > &valSet,\
unsigned pick,long double threshold,unsigned &sizeA,unsigned &sizeB,std::vector\
<cv::Mat> &parentfreqA,std::vector<cv::Mat> &parentfreqB,float &entropyA,float &entropyB){
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
		gain = this->measureSet(tmpA,tmpB,features,pick,parentfreqA,\
			parentfreqB,entropyA,entropyB);
	}
	return gain;
}
//==============================================================================
/** Predicts on a one single test patch.
 * A node contains: [0] -- node type (0,1,-1),[1] -- x1,[2] -- y1,[3] -- x2,
 * 					[4] -- y2,[5] -- channel,[6] -- threshold, [7] -- test type,
 * 					[8] -- node ID
 */
template <class M,class T,class F,class N,class U>
const U* MotionTree<M,T,F,N,U>::siftregression(const T* testPatch,const F* features,\
N* node,unsigned treeid){
	if(node->left()!=NULL || node->right()!=NULL){
		// [0] If test-node, apply test and go left/right
		typename Tree<N,U>::SIDE side = static_cast<typename Tree<N,U>::SIDE>\
			(MotionTree<M,T,F,N,U>::siftapplyTest(node->test(),testPatch,features));
		// [0.1] If the side to continue is left, go left
		if(side == Tree<N,U>::LEFT){
			return this->siftregression(testPatch,features,node->left(),treeid);
		// [0.2] If the side to continue is right, go right
		}else if(side == Tree<N,U>::RIGHT){
			return this->siftregression(testPatch,features,node->right(),treeid);
		}
	// [1] If we reached a leaf, return it
	}else{
		if(this->treeId_!=treeid){
			std::cout<<"[MotionTree<M,T,F,N,U>::siftregression] this->treeID: "<<\
				this->treeId_<<" != treeID: "<<treeid<<std::endl;
		}
		const U* leaf = new U(this->path2models_,node->nodeid(),treeid,this->binary_);
		return leaf;
	}
}
//==============================================================================
/** Applied the test on a feature patch. The center is fixed and we look at the
 * sift dimensions/channels.
 */
template <class M,class T,class F,class N,class U>
bool MotionTree<M,T,F,N,U>::siftapplyTest(const long double *test,const T* testPatch,\
const F* features) const{
	// [0] Node: leaf-index [1] x1 [2] y1 [3] x2 [4] y2 [5] channel [6] \
		   threshold [7] testType
	CvMat* ptC1 = testPatch->feat(features,static_cast<unsigned>(*(test+5)));
	CvMat* ptC2 = testPatch->feat(features,static_cast<unsigned>(*(test+4)));
	cv::Mat channel1(ptC1,true); channel1.convertTo(channel1,CV_32FC1);
	cv::Mat channel2(ptC2,true); channel2.convertTo(channel2,CV_32FC1);
	// [2] Get pixel values:  cvPtr2D(img, y, x, NULL);
	long double p1 = static_cast<long double>(channel1.at<float>(test[2],test[1]));
	long double p2 = static_cast<long double>(channel2.at<float>(test[2],test[1]));
	bool result;
	// [1] Based on the node type evaluate:
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
	cvReleaseMat(&ptC1); cvReleaseMat(&ptC2);
	return result;
}
//==============================================================================
/** Evaluates 1 test (given by 5 numbers: x1, y1, x2, y2, channel).
 * It gets the feature channel and then it accesses it at the 2 randomly selected
 * points and gets the difference between them.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::siftevaluateTest(std::vector<std::vector<Index> >& valSet,\
const long double* test,const std::vector<std::vector<const T*> >& TrainSet,\
const F *features){
	// [0] Loop over classes and patches
	for(unsigned int l=0;l<TrainSet.size();++l){
		valSet[l].resize(TrainSet[l].size());
		for(unsigned int i=0;i<TrainSet[l].size();++i){
			// [1] Get a pointer to the test channel
			CvMat* ptC1 = TrainSet[l][i]->feat(features,static_cast<unsigned>(test[4]));
			CvMat* ptC2 = TrainSet[l][i]->feat(features,static_cast<unsigned>(test[3]));
			cv::Mat channel1(ptC1,true); channel1.convertTo(channel1,CV_32FC1);
			cv::Mat channel2(ptC2,true); channel2.convertTo(channel2,CV_32FC1);
			// [2] Get pixel values:  cvPtr2D(img, y, x, NULL);
			long double p1 = static_cast<long double>(channel1.at<float>(test[1],test[0]));
			long double p2 = static_cast<long double>(channel2.at<float>(test[1],test[0]));
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
/** Generates a random test of a random type.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::siftgenerateTest(long double* test,unsigned int max_w,\
unsigned int max_h,unsigned int max_c){
	// test[0] - leaf-index; test[1] - x1;      test[2] - y1;        test[3] - x2;
	// test[4] - y2;         test[5] - channel; test[6] - threshold; test[7] - testType
	// test[8] - nodeID
	assert(this->nodeSize_>6);
	// Here we start with position 1 in real test set:
	// test[7] - generate a random test-type:
	// 			 [0] I(x1,y1) > t
	// 			 [1] I(x1,y1) - I(x2,y2) > t
	// 			 [2] I(x1,y1) + I(x2,y2) > t
	// 			 [3] |I(x1,y1) - I(x2,y2)| > t
	//
	test[6] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % 4);
	// Here we start with position 1 in real test set:
	// test[1] - x1, test[2] - y1
	// test[3] - x2, test[4] - y2
	//
	// Hard-code the position to always be the center
	test[0] = static_cast<long double>(max_w/2); // x1
	test[1] = static_cast<long double>(max_h/2); // y1
	test[2] = 0;
	test[3] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_c); // x2
	// Here we start with position 1 in real test set:
	// test[5] - channel
	//
	test[4] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_c);
	while(test[3]==test[4]){
		test[4] = static_cast<long double>(cvRandInt( this->cvRNG_ ) % max_c);
	}
	test[5] = 0;
}
//==============================================================================
/** show the mean to the samples for the picked best test.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::showPickedSplitDerivatives(const std::vector\
<std::vector<const T*> > &SetA,const std::vector<std::vector<const T*> > &SetB,\
const F *features,long unsigned nodeid){
	cv::Mat scatter = cv::Mat::zeros(cv::Size(500,500),CV_8UC3);
	unsigned scale  = 50;
	// [0] Get the points and the mean for setA
	float sizeA     = 0.0,meanXXA = 0.0, meanXYA = 0.0, meanYXA = 0.0, meanYYA = 0.0;
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *motionXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionXX,motionXY,motionYX,motionYY);
			// [2] Get the x,y position
			float flowXX = motionXX->at<float>(this->mpick_);
			float flowXY = motionXY->at<float>(this->mpick_);
			float flowYX = motionYX->at<float>(this->mpick_);
			float flowYY = motionYY->at<float>(this->mpick_);
			cv::circle(scatter,cv::Point(250+flowXX*scale,250+flowXY*scale),0.5,\
				CV_RGB(255,150,150),1,1,0);
			cv::circle(scatter,cv::Point(250+flowYX*scale,250+flowYY*scale),0.5,\
				CV_RGB(255,250,150),1,1,0);
			// [3] Get the mean of x,y-coordinates
			meanXXA += motionXX->at<float>(this->mpick_);
			meanXYA += motionXY->at<float>(this->mpick_);
			meanYXA += motionYX->at<float>(this->mpick_);
			meanYYA += motionYY->at<float>(this->mpick_);
			// [4] Increase counts for normalization
			++sizeA;
			motionXX->release(); delete motionXX;
			motionXY->release(); delete motionXY;
			motionYX->release(); delete motionYX;
			motionYY->release(); delete motionYY;
		}
	}
	meanXXA /= sizeA; meanXYA /= sizeA; meanYXA /= sizeA; meanYYA /= sizeA;
	cv::circle(scatter,cv::Point(250+meanXXA*scale,250+meanXYA*scale),4.0,CV_RGB\
		(255,0,0),-1,1,0);
	cv::circle(scatter,cv::Point(250+meanYXA*scale,250+meanYYA*scale),4.0,CV_RGB\
		(255,100,0),-1,1,0);
	Auxiliary<uchar,1>::file_exists("images",true);
	cv::imwrite((std::string("images")+std::string(PATH_SEP)+Auxiliary<long unsigned,1>::\
		number2string(nodeid)+"_setA.png"),scatter);
	// [5] We do the same thing for setB
	float sizeB = 0.0,meanXXB = 0.0, meanXYB = 0.0,meanYXB = 0.0, meanYYB = 0.0;
	for(vectConstIterT l=SetB.begin();l!=SetB.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *motionXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionXX,motionXY,motionYX,motionYY);
			// [2] Get the x,y position
			float flowXX = motionXX->at<float>(this->mpick_);
			float flowXY = motionXY->at<float>(this->mpick_);
			float flowYX = motionYX->at<float>(this->mpick_);
			float flowYY = motionYY->at<float>(this->mpick_);
			cv::circle(scatter,cv::Point(250+flowXX*scale,250+flowXY*scale),0.5,\
				CV_RGB(150,255,150),1,1,0);
			cv::circle(scatter,cv::Point(250+flowYX*scale,250+flowYY*scale),0.5,\
				CV_RGB(150,255,250),1,1,0);
			// [3] Get the mean of x,y-coordinates
			meanXXB += motionXX->at<float>(this->mpick_);
			meanXYB += motionXY->at<float>(this->mpick_);
			meanYXB += motionYX->at<float>(this->mpick_);
			meanYYB += motionYY->at<float>(this->mpick_);
			// [4] Increase counts for normalization
			++sizeB;
			motionXX->release(); delete motionXX;
			motionXY->release(); delete motionXY;
			motionYX->release(); delete motionYX;
			motionYY->release(); delete motionYY;
		}
	}
	meanXXB /= sizeB; meanXYB /= sizeB; meanYXB /= sizeB;meanYYB /= sizeB;
	cv::circle(scatter,cv::Point(250+meanXXB*scale,250+meanXYB*scale),4.0,CV_RGB\
		(0,255,0),-1,1,0);
	cv::circle(scatter,cv::Point(250+meanYXB*scale,250+meanYYB*scale),4.0,CV_RGB\
		(0,255,100),-1,1,0);
	cv::imwrite((std::string("images")+std::string(PATH_SEP)+Auxiliary<long unsigned,1>::\
		number2string(nodeid)+"_setB.png"),scatter);
	scatter.release();
}
//==============================================================================
/** show the mean to the samples for the picked best test.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::showPickedSplit(const std::vector\
<std::vector<const T*> > &SetA,const std::vector<std::vector<const T*> > &SetB,\
const F *features,long unsigned nodeid){
	if(features->usederivatives()){
		return this->showPickedSplitDerivatives(SetA,SetB,features,nodeid);
	}else{
		return this->showPickedSplitFlow(SetA,SetB,features,nodeid);
	}
}
//==============================================================================
/** show the mean to the samples for the picked best test.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::showPickedSplitFlow(const std::vector\
<std::vector<const T*> > &SetA,const std::vector<std::vector<const T*> > &SetB,\
const F *features,long unsigned nodeid){
	cv::Mat scatter = cv::Mat::zeros(cv::Size(500,500),CV_8UC3);
	unsigned scale  = 50;
	// [0] Get the points and the mean for setA
	float sizeA     = 0.0,meanXA = 0.0, meanYA = 0.0;
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *motionX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionX,motionY);
			// [2] Get the x,y position
			float flowX = motionX->at<float>(this->mpick_);
			float flowY = motionY->at<float>(this->mpick_);
			cv::circle(scatter,cv::Point(250+flowX*scale,250+flowY*scale),0.5,\
				CV_RGB(255,150,150),1,1,0);
			// [3] Get the mean of x,y-coordinates
			meanXA += motionX->at<float>(this->mpick_);
			meanYA += motionY->at<float>(this->mpick_);
			// [4] Increase counts for normalization
			++sizeA;
			motionX->release(); delete motionX;
			motionY->release(); delete motionY;
		}
	}
	meanXA /= sizeA; meanYA /= sizeA;
	cv::circle(scatter,cv::Point(250+meanXA*scale,250+meanYA*scale),4.0,CV_RGB\
		(255,0,0),-1,1,0);
	Auxiliary<uchar,1>::file_exists("images",true);
	cv::imwrite((std::string("images")+std::string(PATH_SEP)+Auxiliary<long unsigned,1>::\
		number2string(nodeid)+"_setA.png"),scatter);
	// [5] We do the same thing for setB
	float sizeB = 0.0,meanXB = 0.0, meanYB = 0.0;
	for(vectConstIterT l=SetB.begin();l!=SetB.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *motionX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionX,motionY);
			// [2] Get the x,y position
			float flowX = motionX->at<float>(this->mpick_);
			float flowY = motionY->at<float>(this->mpick_);
			cv::circle(scatter,cv::Point(250+flowX*scale,250+flowY*scale),0.5,\
				CV_RGB(150,255,150),1,1,0);
			// [3] Get the mean of x,y-coordinates
			meanXB += motionX->at<float>(this->mpick_);
			meanYB += motionY->at<float>(this->mpick_);
			// [4] Increase counts for normalization
			++sizeB;
			motionX->release(); delete motionX;
			motionY->release(); delete motionY;
		}
	}
	meanXB /= sizeB; meanXB /= sizeB; meanYB /= sizeB;
	cv::circle(scatter,cv::Point(250+meanXB*scale,250+meanYB*scale),4.0,CV_RGB\
		(0,255,0),-1,1,0);
	cv::imwrite((std::string("images")+std::string(PATH_SEP)+Auxiliary<long unsigned,1>::\
		number2string(nodeid)+"_setB.png"),scatter);
	scatter.release();
}
//==============================================================================
/** Overloading the function to carry around the labels matrices.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::measureSet(const std::vector<std::vector<const T*> > \
&SetA,const std::vector<std::vector<const T*> > &SetB,const F *features,unsigned pick,\
std::vector<cv::Mat> &parentfreqA,std::vector<cv::Mat> &parentfreqB,\
float &motionA,float &motionB){
	float sizeA = 0.0, sizeB   = 0.0;
	motionA     = 0.0; motionB = 0.0;
	// [0] Make/update the motion model for each class
	switch(this->entropy_){
		case(MotionTree<M,T,F,N,U>::APPROX_ANGLE_KERNEL):
			if(this->usepick_){
				motionA = this->splitApproxKernel(SetA,features,sizeA,parentfreqA);
				motionB = this->splitApproxKernel(SetB,features,sizeB,parentfreqB);
			}else{
				motionA = this->splitApproxKernelPatch(SetA,features,sizeA,parentfreqA);
				motionB = this->splitApproxKernelPatch(SetB,features,sizeB,parentfreqB);
			}
			break;
		case(MotionTree<M,T,F,N,U>::APPROX_MAGNI_KERNEL):
			if(this->usepick_){
				motionA = this->splitApproxKernel(SetA,features,sizeA,parentfreqA);
				motionB = this->splitApproxKernel(SetB,features,sizeB,parentfreqB);
			}else{
				motionA = this->splitApproxKernelPatch(SetA,features,sizeA,parentfreqA);
				motionB = this->splitApproxKernelPatch(SetB,features,sizeB,parentfreqB);
			}
			break;
		case(MotionTree<M,T,F,N,U>::MEAN_DIFF):
			motionA = this->splitDistance2mean(SetA,features,sizeA);
			motionB = this->splitDistance2mean(SetB,features,sizeB);
			break;
		default:
			motionA = this->splitDistance2mean(SetA,features,sizeA);
			motionB = this->splitDistance2mean(SetB,features,sizeB);
			break;
	}
	// [1] We will minimize this over candidate tests
	return (sizeA*motionA+sizeB*motionB)/(sizeA+sizeB);
}
//==============================================================================
/** Check if all patches have converged to a single pattern by looking as MSE.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::stopCosSimilarity(const std::vector<std::vector\
<const T*> > &trainSet,const F *features){
	if(features->usederivatives()){
		return this->stopCosSimilarityDerivatives(trainSet,features);
	}else{
		return this->stopCosSimilarityFlows(trainSet,features);
	}
}
//==============================================================================
/** Check if all patches have converged to a single pattern by looking as MSE.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::stopCosSimilarityDerivatives(const std::vector\
<std::vector<const T*> > &trainSet,const F *features){
	std::vector<cv::Mat> vectXX;
	std::vector<cv::Mat> vectXY;
	std::vector<cv::Mat> vectYX;
	std::vector<cv::Mat> vectYY;
	bool converged = true;
	float epsilon  = 0.7;
	// [0] First put all the samples together and loop for the first one.
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *sampleXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *sampleXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *sampleYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *sampleYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,sampleXX,sampleXY,sampleYX,sampleYY);
			// [2] Put the samples in their vectors
			vectXX.push_back(sampleXX->clone());
			vectXY.push_back(sampleXY->clone());
			vectYX.push_back(sampleYX->clone());
			vectYY.push_back(sampleYY->clone());
			// [3] Now for each sample loop over all its previous samples.
			std::vector<cv::Mat>::iterator xy=vectXY.begin();
			std::vector<cv::Mat>::iterator yx=vectYX.begin();
			std::vector<cv::Mat>::iterator yy=vectYY.begin();
			for(std::vector<cv::Mat>::iterator xx=vectXX.begin();xx!=vectXX.end()-1,\
			xy!=vectXY.end()-1,yx!=vectYX.end()-1,yy!=vectYY.end()-1;++xx,++xy,\
			++yx,++yy){
				cv::Mat splxx = (*xx);
				cv::Mat splxy = (*xy);
				cv::Mat splyx = (*yx);
				cv::Mat splyy = (*yy);
				cv::Mat dot   = cv::Mat(splxx.size(),CV_32FC1);
				// [3.1] Get the cosine similarity at each position
				cv::Mat_<float>::iterator d    = dot.begin<float>();
				cv::Mat_<float>::iterator saxx = sampleXX->begin<float>();
				cv::Mat_<float>::iterator saxy = sampleXY->begin<float>();
				cv::Mat_<float>::iterator sayx = sampleYX->begin<float>();
				cv::Mat_<float>::iterator sayy = sampleYY->begin<float>();
				cv::Mat_<float>::iterator sxy  = splxy.begin<float>();
				cv::Mat_<float>::iterator syx  = splyx.begin<float>();
				cv::Mat_<float>::iterator syy  = splyy.begin<float>();
				float small                    = 1.0e-10;
				for(cv::Mat_<float>::iterator sxx=splxx.begin<float>();sxx!=\
				splxx.end<float>(),sxy!=splxy.end<float>(),syx!=splyx.end<float>(),\
				syy!=splyy.end<float>(),saxx!=sampleXX->end<float>(),saxy!=sampleXY->end\
				<float>(),sayx!=sampleYX->end<float>(),sayy!=sampleYY->end<float>(),\
				d!=dot.end<float>();++sxx,++sxy,++syx,++syy,++saxx,++saxy,++sayx,\
				++sayy,++d){
					(*d) = ((*sxx)*(*saxx)+(*sxy)*(*saxy)+(*syx)*(*sayx)+\
						(*syy)*(*sayy)+small)/\
						(std::sqrt((*sxx)*(*sxx)+(*sxy)*(*sxy)+\
						(*syx)*(*syx)+(*syy)*(*syy)+small)*
						std::sqrt((*saxx)*(*saxx)+(*saxy)*(*saxy)+\
						(*sayx)*(*sayx)+(*sayy)*(*sayy)+small)*\
						static_cast<float>(this->motionW_*this->motionH_));
				}
				float sim = (cv::sum(dot).val[0]);
				dot.release();
				if(sim<=epsilon){
					converged = false;
					break;
				}
			}
			if(!converged){break;}
		}
		if(!converged){break;}
	}
	std::vector<cv::Mat>::iterator xy=vectXY.begin();
	std::vector<cv::Mat>::iterator yx=vectYX.begin();
	std::vector<cv::Mat>::iterator yy=vectYY.begin();
	for(std::vector<cv::Mat>::iterator xx=vectXX.begin();xx!=vectXX.end(),\
	xy!=vectXY.end(),yx!=vectYX.end(),yy!=vectYY.end();++xx,++xy,++yx,++yy){
		xx->release(); xy->release(); yx->release(); yy->release();
	}
	return converged;
}
//==============================================================================
/** Check if all patches have converged to a single pattern by looking as MSE.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::stopCosSimilarityFlows(const std::vector\
<std::vector<const T*> > &trainSet,const F *features){
	std::vector<cv::Mat> vectX;
	std::vector<cv::Mat> vectY;
	bool converged = true;
	float epsilon  = 0.7;
	// [0] First put all the samples together and loop for the first one.
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *sampleX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *sampleY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,sampleX,sampleY);
			// [2] Put the samples in their vectors
			vectX.push_back(sampleX->clone());
			vectY.push_back(sampleY->clone());
			// [3] Now for each sample loop over all its previous samples.
			std::vector<cv::Mat>::iterator y=vectY.begin();
			for(std::vector<cv::Mat>::iterator x=vectX.begin();x!=vectX.end()-1,\
			y!=vectY.end()-1;++x,++y){
				cv::Mat splx = (*x);
				cv::Mat sply = (*y);
				cv::Mat dot  = cv::Mat(splx.size(),CV_32FC1);
				// [3.1] Get the cosine similarity at each position
				cv::Mat_<float>::iterator d   = dot.begin<float>();
				cv::Mat_<float>::iterator sax = sampleX->begin<float>();
				cv::Mat_<float>::iterator say = sampleY->begin<float>();
				cv::Mat_<float>::iterator sy  = sply.begin<float>();
				float small                   = 1.0e-10;
				for(cv::Mat_<float>::iterator sx=splx.begin<float>();sx!=\
				splx.end<float>(),sy!=sply.end<float>(),sax!=sampleX->end<float>(),\
				say!=sampleY->end<float>(),d!=dot.end<float>();++sx,++sy,++sax,\
				++say,++d){
					(*d) = ((*sx)*(*sax)+(*sy)*(*say)+small)/\
						(std::sqrt((*sx)*(*sx)+(*sy)*(*sy)+small)*
						std::sqrt((*sax)*(*sax)+(*say)*(*say)+small)*\
						static_cast<float>(this->motionW_*this->motionH_));
				}
				float sim = (cv::sum(dot).val[0]);
				dot.release();
				if(sim<=epsilon){
					converged = false;
					break;
				}
			}
			if(!converged){break;}
		}
		if(!converged){break;}
	}
	std::vector<cv::Mat>::iterator y=vectY.begin();
	for(std::vector<cv::Mat>::iterator x=vectX.begin();x!=vectX.end(),\
	y!=vectY.end();++x,++y){
		x->release(); y->release();
	}
	return converged;
}
//==============================================================================
/** Check if all patches have converged to a single pattern by looking as MSE.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::stopEuclDist(const std::vector<std::vector\
<const T*> > &trainSet,const F *features){
	if(features->usederivatives()){
		return this->stopEuclDistDerivaties(trainSet,features);
	}else{
		return this->stopEuclDistFlows(trainSet,features);
	}
}
//==============================================================================
/** Check if all patches have converged to a single pattern by looking as MSE.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::stopEuclDistDerivaties(const std::vector<std::vector\
<const T*> > &trainSet,const F *features){
	std::vector<cv::Mat> vectXX;
	std::vector<cv::Mat> vectXY;
	std::vector<cv::Mat> vectYX;
	std::vector<cv::Mat> vectYY;
	bool converged = true;
	float epsilon  = 5;
	// [0] First put all the samples together and loop for the first one.
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *sampleXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *sampleXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *sampleYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *sampleYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,sampleXX,sampleXY,sampleYX,sampleYY);
			// [2] Put the samples in their vectors
			vectXX.push_back(sampleXX->clone());
			vectXY.push_back(sampleXY->clone());
			vectYX.push_back(sampleYX->clone());
			vectYY.push_back(sampleYY->clone());
			// [3] Now for each sample loop over all its previous samples.
			std::vector<cv::Mat>::iterator xy = vectXY.begin();
			std::vector<cv::Mat>::iterator yx = vectYX.begin();
			std::vector<cv::Mat>::iterator yy = vectYY.begin();
			for(std::vector<cv::Mat>::iterator xx=vectXX.begin();xx!=vectXX.end()-1,\
			xy!=vectXY.end()-1,yx!=vectYX.end()-1,yy!=vectYY.end()-1;++xx,++xy,\
			++yx,++yy){
				cv::Mat splxx = (*xx); cv::Mat splyx = (*yx);
				cv::Mat splxy = (*xy); cv::Mat splyy = (*yy);
				cv::Mat dot   = cv::Mat(splxx.size(),CV_32FC1);
				// [3.1] Get the cosine similarity at each position
				cv::Mat_<float>::iterator d    = dot.begin<float>();
				cv::Mat_<float>::iterator saxx = sampleXX->begin<float>();
				cv::Mat_<float>::iterator saxy = sampleXY->begin<float>();
				cv::Mat_<float>::iterator sayx = sampleYX->begin<float>();
				cv::Mat_<float>::iterator sayy = sampleYY->begin<float>();
				cv::Mat_<float>::iterator sxy  = splxy.begin<float>();
				cv::Mat_<float>::iterator syx  = splyx.begin<float>();
				cv::Mat_<float>::iterator syy  = splyy.begin<float>();
				for(cv::Mat_<float>::iterator sxx=splxx.begin<float>();\
				sxx!=splxx.end<float>(),sxy!=splxy.end<float>(),syx!=\
				splyx.end<float>(),syy!=splyy.end<float>(),\
				saxx!=sampleXX->end<float>(),saxy!=sampleXY->end<float>(),\
				sayx!=sampleYX->end<float>(),sayy!=sampleYY->end<float>(),\
				d!=dot.end<float>();++sxx,++sxy,++syx,++syy,++saxx,++saxy,++sayx,\
				++sayy,++d){
					(*d) = std::sqrt(((*sxx)-(*saxx))*((*sxx)-(*saxx)) +\
						((*sxy)-(*saxy))*((*sxy)-(*saxy))+((*syx)-(*sayx))*\
						((*syx)-(*sayx))+((*syy)-(*sayy))*((*syy)-(*sayy)))/\
						static_cast<float>(this->motionW_*this->motionH_);
				}
				float sim = (cv::sum(dot).val[0]);
				dot.release();
				if(sim<=epsilon){
					converged = false;
					break;
				}
			}
			if(!converged){break;}
		}
		if(!converged){break;}
	}
	std::vector<cv::Mat>::iterator xy = vectXY.begin();
	std::vector<cv::Mat>::iterator yx = vectYX.begin();
	std::vector<cv::Mat>::iterator yy = vectYY.begin();
	for(std::vector<cv::Mat>::iterator xx=vectXX.begin();xx!=vectXX.end(),\
	xy!=vectXY.end(),yx!=vectYX.end(),yy!=vectYY.end();++xx,++xy,++yx,++yy){
		xx->release(); xy->release(); yx->release(); yy->release();
	}
	return converged;
}
//==============================================================================
/** Check if all patches have converged to a single pattern by looking as MSE.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::stopEuclDistFlows(const std::vector<std::vector\
<const T*> > &trainSet,const F *features){
	std::vector<cv::Mat> vectX;
	std::vector<cv::Mat> vectY;
	bool converged = true;
	float epsilon  = 5;
	// [0] First put all the samples together and loop for the first one.
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *sampleX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *sampleY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,sampleX,sampleY);
			// [2] Put the samples in their vectors
			vectX.push_back(sampleX->clone());
			vectY.push_back(sampleY->clone());
			// [3] Now for each sample loop over all its previous samples.
			std::vector<cv::Mat>::iterator y = vectY.begin();
			for(std::vector<cv::Mat>::iterator x=vectX.begin();x!=vectX.end()-1,\
			y!=vectY.end()-1;++x,++y){
				cv::Mat splx = (*x); cv::Mat sply = (*y);
				cv::Mat dot  = cv::Mat(splx.size(),CV_32FC1);
				// [3.1] Get the cosine similarity at each position
				cv::Mat_<float>::iterator d   = dot.begin<float>();
				cv::Mat_<float>::iterator sax = sampleX->begin<float>();
				cv::Mat_<float>::iterator say = sampleY->begin<float>();
				cv::Mat_<float>::iterator sy  = sply.begin<float>();
				for(cv::Mat_<float>::iterator sx=splx.begin<float>();sx!=\
				splx.end<float>(),sy!=sply.end<float>(),sax!=\
				sampleX->end<float>(),say!=sampleY->end<float>(),\
				d!=dot.end<float>();++sx,++sy,++sax,++say,++d){
					(*d) = std::sqrt(((*sx)-(*sax))*((*sx)-(*sax)) +\
						((*sy)-(*say))*((*sy)-(*say)))/\
						static_cast<float>(this->motionW_*this->motionH_);
				}
				float sim = (cv::sum(dot).val[0]);
				dot.release();
				if(sim<=epsilon){
					converged = false;
					break;
				}
			}
			if(!converged){break;}
		}
		if(!converged){break;}
	}
	std::vector<cv::Mat>::iterator y = vectY.begin();
	for(std::vector<cv::Mat>::iterator x=vectX.begin();x!=vectX.end(),\
	y!=vectY.end();++x,++y){
		x->release(); y->release();
	}
	return converged;
}
//==============================================================================
/** Sum-Squared-Distance to the mean of the samples at the picked position.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::splitDistance2mean(const std::vector<std::vector\
<const T*> > &SetA,const F *features,float &sizeA){
	if(features->usederivatives()){
		return this->splitDistance2meanDerivatives(SetA,features,sizeA);
	}else{
		return this->splitDistance2meanFlows(SetA,features,sizeA);
	}
}
//==============================================================================
/** Sum-Squared-Distance to the mean of the samples at the picked position.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::splitDistance2meanDerivatives(const std::vector\
<std::vector<const T*> > &SetA,const F *features,float &sizeA){
	// [0] Loop over classes (no classes here)
	cv::Mat meanXX, meanXY, meanYX, meanYY;
	sizeA = 0.0;
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		// [1] Loop over patches in current class
		sizeA += l->size();
	}
	if(sizeA==1){return 0;}
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *motionXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionXX,motionXY,motionYX,motionYY);
			// [2] Get the mean of x,y-coordinates
			if(meanXX.empty()){motionXX->copyTo(meanXX);
			}else{meanXX += (*motionXX);}
			if(meanXY.empty()){motionXY->copyTo(meanXY);
			}else{meanXY += (*motionXY);}
			if(meanYX.empty()){motionYX->copyTo(meanYX);
			}else{meanYX += (*motionYX);}
			if(meanYY.empty()){motionYY->copyTo(meanYY);
			}else{meanYY += (*motionYY);}
			// [3] Increase counts for normalization
			motionXX->release(); delete motionXX;
			motionXY->release(); delete motionXY;
			motionYX->release(); delete motionYX;
			motionYY->release(); delete motionYY;
		}
	} // over patches
	// [4] Finally, get the SSD to mean
	meanXX /= sizeA; meanXY /= sizeA; meanYX /= sizeA; meanYY /= sizeA;
	assert(meanXX.cols==meanXY.cols && meanXX.cols==meanYX.cols && meanXX.cols\
		==meanYY.cols && meanXX.rows==meanXY.rows && meanXX.rows==meanYX.rows && \
		meanXX.rows==meanYY.rows);
	cv::Mat ssd = cv::Mat::zeros(meanXX.size(),meanXX.type());
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		// [5] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *motionXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionXX,motionXY,motionYX,motionYY);
			cv::Mat diffXX = meanXX-(*motionXX);
			cv::multiply(diffXX,diffXX,diffXX);
			cv::Mat diffXY = meanXY-(*motionXY);
			cv::multiply(diffXY,diffXY,diffXY);
			cv::Mat diffYX = meanYX-(*motionYX);
			cv::multiply(diffYX,diffYX,diffYX);
			cv::Mat diffYY = meanYY-(*motionYY);
			cv::multiply(diffYY,diffYY,diffYY);
			ssd += (diffXX+diffXY+diffYX+diffYY); // /(sizeA-1.0);
			motionXX->release(); delete motionXX;
			motionXY->release(); delete motionXY;
			motionYX->release(); delete motionYX;
			motionYY->release(); delete motionYY;
			diffXX.release(); diffXY.release();
			diffYX.release(); diffYY.release();
		}// over patches
	}
	meanXX.release(); meanXY.release(); meanYX.release(); meanYY.release();
	cv::Scalar sum   = cv::sum(ssd);
	unsigned ssdsize = ssd.rows*ssd.cols;	
	ssd.release();
	return sum.val[0]/static_cast<float>(ssdsize);
}
//==============================================================================
/** Sum-Squared-Distance to the mean of the samples at the picked position.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::splitDistance2meanFlows(const std::vector<std::vector\
<const T*> > &SetA,const F *features,float &sizeA){
	// [0] Loop over classes (no classes here)
	sizeA = 0.0;
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		// [1] Loop over patches in current class
		sizeA += l->size();
	}
	if(sizeA==1){return 0;}
	cv::Mat meanX, meanY;
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		// [1] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *motionX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionY = new cv::Mat(cv::Size((*p)->motionW(),\
			(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionX,motionY);
			// [2] Get the mean of x,y-coordinates
			if(meanX.empty()){ motionX->copyTo(meanX);
			}else{ meanX += (*motionX); }
			if(meanY.empty()){ motionY->copyTo(meanY);
			}else{ meanY += (*motionY); }
			// [3] Increase counts for normalization
			motionX->release(); delete motionX;
			motionY->release(); delete motionY;
		}
	}
	// [4] Finally, get the SSD to mean
	meanX /= sizeA; meanY /= sizeA;
	assert(meanX.cols==meanY.cols && meanX.rows==meanY.rows);
	cv::Mat ssd = cv::Mat::zeros(meanX.size(),meanX.type());
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		// [5] Loop over patches in current class
		for(constIterT p=l->begin();p!=l->end();++p){
			cv::Mat *motionX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionX,motionY);
			cv::Mat diffX, diffY;
			diffX = meanX - (*motionX);
			diffY = meanY - (*motionY);
			cv::multiply(diffX,diffX,diffX);
			cv::multiply(diffY,diffY,diffY);
			ssd += (diffX+diffY); // /(sizeA-1.0);
			diffX.release(); diffY.release();
			motionX->release(); delete motionX;
			motionY->release(); delete motionY;
		} // over patches
	}
	meanX.release(); meanY.release();
	cv::Scalar sum   = cv::sum(ssd);
	unsigned ssdsize = ssd.rows*ssd.cols;
	ssd.release();
	return sum.val[0]/static_cast<float>(ssdsize);
}
//==============================================================================
/** Approximating continuous entropy with sum over sample probability, in turn
 * approximated the density kernel estimation with pixel-wise kernels over
 * complete patch.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::splitApproxKernelPatch(const std::vector<std::vector\
<const T*> > &SetA,const F *features,float &sizeA,std::vector<cv::Mat> &prevfreq){
	// [1] Get the total probability mass in this set
	sizeA         = 0.0;
	cv::Mat normA = cv::Mat::zeros(cv::Size(this->motionW_,this->motionH_),CV_32FC1);
	std::vector<cv::Mat> probs(this->histinfo_[0],cv::Mat());
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		for(constIterT p=l->begin();p!=l->end();++p){ // over patches
			// [1] Get the frequency of the current sample
			std::vector<cv::Mat> histos = (*p)->histo(features);
			assert(histos.size()==probs.size());
			std::vector<cv::Mat>::iterator pr = probs.begin();
			for(std::vector<cv::Mat>::iterator h=histos.begin();h!=histos.end(),\
			pr!=probs.end();++h,++pr){
				if(pr->empty()){ h->copyTo(*pr);
				}else{ (*pr) += (*h); }
				normA += (*h); h->release();
			} // over bins
		} // over patches
	} // over classes --- not the case
	// [2] Before weighting we save it for the children.
	std::vector<cv::Mat> unweightedprobs;
	for(std::vector<cv::Mat>::iterator p=probs.begin();p!=probs.end();++p){
		unweightedprobs.push_back(p->clone());
	}
	// [3] If parent weighting then divide at the end.
	cv::Mat weightedsize;
	if(this->parentFreq_){
		assert(prevfreq.size()==probs.size());
		weightedsize = cv::Mat::zeros(normA.size(),normA.type());
		std::vector<cv::Mat>::iterator pr=probs.begin();
		for(std::vector<cv::Mat>::iterator pf=prevfreq.begin();\
		pr!=probs.end(),pf!=prevfreq.end();++pf,++pr){
			cv::Mat mask;
			cv::inRange((*pf),0,SMALL,mask);
			pf->setTo(1.0,mask); mask.release();
			cv::divide((*pr),(*pf),(*pr)); // inverse parent frequency
			weightedsize += (*pr);
		} // over bins
	}else{normA.copyTo(weightedsize);}
	cv::Scalar sum = cv::sum(weightedsize);
	sizeA          = sum.val[0];
	normA.release();
	// [4] Mask the weights to not get nans
	cv::Mat mask;
	cv::inRange(weightedsize,0,SMALL,mask);
	weightedsize.setTo(1,mask); mask.release();
	// [5] Now normalize and distribution entropy: sum_bins p(bin) log p(bin)
	cv::Mat entropy = cv::Mat::zeros(cv::Size(this->motionW_,this->motionH_),CV_32FC1);
	for(std::vector<cv::Mat>::iterator pr=probs.begin();pr!=probs.end();++pr){
		// [5.1] Now normalize the probabilities to be probabilities
		cv::divide((*pr),weightedsize,(*pr));
		// [5.2] The actual entropy: sum_bins p(bin) log p(bin)
		cv::Mat_<float>::iterator e = entropy.begin<float>();
		for(cv::Mat_<float>::iterator d=pr->begin<float>();d!=pr->end<float>(),\
		e!=entropy.end<float>();++d,++e){
			assert((*d)>=0 && (*d)<=1);
			if((*d)>SMALL){
				(*e) += -(*d)*log2(*d);
			}
		} // over patch dimensions
	} // over bins
	weightedsize.release();
	// [6] Unnormalized frequencies and bininfo update.
	std::vector<cv::Mat>::iterator uw = unweightedprobs.begin();
	std::vector<cv::Mat>::iterator pr = probs.begin();
	for(std::vector<cv::Mat>::iterator pf=prevfreq.begin();pf!=prevfreq.end(),\
	uw!=unweightedprobs.end(),pr!=probs.end();++pf,++uw,++pr){
		pf->release(); uw->copyTo(*pf); uw->release(); pr->release();
	}
	// [7] Return the entropy as normalized sum over dimensions
	cv::Scalar sentropy = cv::sum(entropy); entropy.release();
	float n_entropy     = sentropy.val[0]/static_cast<float>(this->motionW_*this->motionH_);
	assert(!isinf(n_entropy) && !isnan(n_entropy));
	return n_entropy;
}
//==============================================================================
/** Approximating continuous entropy with sum over sample probability, in turn
 * approximated the density kernel estimation with pixel-wise kernels.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::splitApproxKernel(const std::vector<std::vector\
<const T*> > &SetA,const F *features,float &sizeA,std::vector<cv::Mat> &prevfreq){
	// [1] Get the total probability mass of the set
	sizeA = 0.0;
	std::vector<cv::Mat> probs;
	probs.push_back(cv::Mat::zeros(cv::Size(this->histinfo_[0],1),CV_32FC1));
	for(vectConstIterT l=SetA.begin();l!=SetA.end();++l){
		for(constIterT p=l->begin();p!=l->end();++p){ // over patches
			// [1.1] Get the frequency of the current sample
			cv::Mat asmpl = (*p)->histo(features,this->mpick_);
			probs[0] += asmpl; asmpl.release();
		} // over patch
	}
	cv::Scalar psum = cv::sum(probs[0]);
	sizeA           = psum.val[0];
	// [2] Before weighting we save it for the children.
	std::vector<cv::Mat> unweightedprobs;
	unweightedprobs.push_back(probs[0].clone());
	// [3] If parent weighting then divide at the end.
	float weightedsize = 0;
	if(this->parentFreq_){
		assert(prevfreq.size()==1); // we pick only 1 random position
		cv::Mat mask;
		cv::inRange(prevfreq[0],0,SMALL,mask);
		prevfreq[0].setTo(1,mask); mask.release();
		cv::divide(probs[0],prevfreq[0],probs[0]); // inverse parent frequency
		cv::Scalar wsum = cv::sum(probs[0]);
		weightedsize    = wsum.val[0];
	}else{weightedsize = sizeA;}
	// [4] Normalize the probabilities to be probabilities
	if(weightedsize>SMALL){
		probs[0] /= weightedsize;
	}
	// [5] Get the distribution entropy as: sum_bins p(bin) log p(bin)
	float n_entropy = 0.0;
	for(cv::Mat_<float>::iterator pr=probs[0].begin<float>();pr!=\
	probs[0].end<float>();++pr){
		assert((*pr)>=0);
		if((*pr)>SMALL){
			n_entropy += (*pr)*log2(*pr);
		}
	}
	prevfreq[0].release(); unweightedprobs[0].copyTo(prevfreq[0]);
	unweightedprobs[0].release(); probs[0].release();
	assert(!isinf(n_entropy) && !isnan(n_entropy));
	return -n_entropy;
}
//==============================================================================
/** Create leaf node from all patches.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::makeLeaf(const F* features,const std::vector<std::vector<const T*> >\
&trainSet,long unsigned nodeid,N* parent,typename Tree<N,U>::SIDE side,unsigned nopatches,\
const std::vector<cv::Mat> &prevfreq,float entropy,bool showLeaves){
	// [0] Find the first non-empty set of patches
	int first        = -1;
	unsigned totsize = 0;
	for(vectConstIterT it=trainSet.begin();it!=trainSet.end();++it){ // over classes
		totsize += it->size();
		if(!it->empty() && first<0){
			first = it-trainSet.begin();
		}
	}
	float bestAppProb = 0.0, bestMotionProb = 0.0;
	cv::Mat *bestMotion            = trainSet[first][0]->motion(features);
	cv::Mat *bestApp               = trainSet[first][0]->image(features);
	std::vector<cv::Mat> bestHisto;
	if(this->entropy_!=MotionTree<M,T,F,N,U>::MEAN_DIFF){
		std::vector<cv::Mat> bestHisto = trainSet[first][0]->histo(features);
	}
	// [0] Patch choosing in the leaf has to agree with the splitting method
	if(totsize>1){
		switch(this->entropy_){
			case(MotionTree<M,T,F,N,U>::APPROX_ANGLE_KERNEL):
				this->leafApprox(features,trainSet,first,totsize,\
					bestAppProb,bestMotionProb,bestApp,bestMotion,bestHisto,\
					prevfreq,nodeid);
				break;
			case(MotionTree<M,T,F,N,U>::APPROX_MAGNI_KERNEL):
				this->leafApprox(features,trainSet,first,totsize,\
					bestAppProb,bestMotionProb,bestApp,bestMotion,bestHisto,\
					prevfreq,nodeid);
				break;
			case(MotionTree<M,T,F,N,U>::MEAN_DIFF):
				this->leafMean(features,trainSet,first,totsize,\
					bestAppProb,bestMotionProb,bestMotion,bestApp);
				break;
			default:
				this->leafMean(features,trainSet,first,totsize,\
					bestAppProb,bestMotionProb,bestMotion,bestApp);
				break;
		}
	}
	// [1] We do not use any labels so set them to 0
	float bestLabelProb             = 0.0;
	std::vector<unsigned> bestLabel = std::vector<unsigned>(this->labSz_,0.0);
	// [2] Add leaf: Now add the label patch and its probability in the tree.
	long double* test = new long double[this->nodeSize_]();
	std::fill_n(test,this->nodeSize_,0);
	U* aleaf = new U();
	// [3] Now fill the leaf with the patches and their probabilities
	aleaf->labelProb(bestLabelProb);
	aleaf->motionProb(bestMotionProb);
	aleaf->appearanceProb(bestAppProb);
	aleaf->vAppearance(bestApp);
	aleaf->vMotion(bestMotion);
	aleaf->vLabels(bestLabel);
	aleaf->vHistos(bestHisto);
	// [4] Add NULL in the tree and save the leaf separately
	U* dummy    = new U();
	N* leafnode = new N(nodeid,dummy,test,this->nodeSize_);
	leafnode    = this->addNode(leafnode,parent,side);
	delete dummy; dummy = NULL;
	// [5] Now write the leaf and remove it
	if(this->binary_){
		aleaf->showLeafBin(this->path2models_,nodeid,this->treeId_);
	}else{
		aleaf->showLeafTxt(this->path2models_,nodeid,this->treeId_);
	}
	delete [] test;
	delete aleaf;
	for(unsigned b=0;b<bestHisto.size();++b){bestHisto[b].release();}
	// [5] If we want to see how the leaves look like
	if(showLeaves && this->treeId_==0){
		this->showSamples(trainSet,features,nodeid,entropy,bestMotion,bestApp);
	}
	bestApp->release(); delete bestApp;
	bestMotion->release();delete bestMotion;
}
//==============================================================================
/** Take the mean of all patches arriving to the leaf.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::leafMean(const F* features,const std::vector\
<std::vector<const T*> > &trainSet,int first,unsigned totsize,float &bestAppProb,\
float &bestMotionProb,cv::Mat *bestMotion,cv::Mat *bestApp){
	// [1] Now loop over patches and take the mean of the patches.
	bestAppProb    = 0.0; // no appearance here
	bestMotionProb = 0.0;
	unsigned flowdim ;
	if(features->usederivatives()){flowdim = 4;
	}else{flowdim = 2;}
	cv::Mat *tmpMotion = new cv::Mat(cv::Mat::zeros(cv::Size(flowdim*\
		this->motionH_*this->motionW_,1),CV_32FC1));
	cv::Mat *tmpApp = new cv::Mat(cv::Mat::zeros(cv::Size(this->motionH_,\
		this->motionW_),CV_32FC3));
	unsigned motionClsId=first, motionPatchId=0, labelClsId=first;
	for(typename std::vector<std::vector<const T*> >::const_iterator t=\
	trainSet.begin();t!=trainSet.end();++t){
		for(typename std::vector<const T*>::const_iterator p=t->begin();p!=t->end();++p){
			cv::Mat *motion = (*p)->motion(features);
			cv::Mat *app    = (*p)->image(features);
			app->convertTo(*app,CV_32FC3);
			assert(tmpMotion->size()==motion->size());
			(*tmpMotion) += (*motion);
			(*tmpApp)    += (*app);
			app->release(); delete app;
			motion->release(); delete motion;
		}
	}
	// [3] Normalize and be done with it.
	(*tmpMotion) /= static_cast<float>(totsize);
	(*tmpApp) /= static_cast<float>(totsize);
	tmpApp->convertTo(*tmpApp,CV_8UC3);
	bestMotion->release();
	tmpMotion->copyTo(*bestMotion);
	bestApp->release();
	tmpApp->copyTo(*bestApp);
	tmpApp->release(); delete tmpApp;
	tmpMotion->release();delete tmpMotion;
}
//==============================================================================
/** Keeps the most likely patch in the leaf given the approximation of kernel
 * density estimation for the patch probability.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::leafApprox(const F* features,const std::vector\
<std::vector<const T*> > &trainSet,int first,unsigned totsize,float &bestAppProb,\
float &bestMotionProb,cv::Mat *bestApp,cv::Mat *bestMotion,std::vector<cv::Mat> \
&bestHisto,const std::vector<cv::Mat> &prevfreq,long unsigned nodeid,bool writeprobs){
	// [0] Find patch probabilities using kernel density estimation
	std::vector<std::vector<float> > mProb = this->patchApprox(features,trainSet,\
		totsize,prevfreq);
	// [2] Find patch probabilities (wrt motion and appearance)
	bestMotionProb = mProb[first][0];
	bestAppProb    = 0;
	// [2] Now loop over patches to find the best
	unsigned bestmotionClsId=first,bestmotionPatchId=0;
	// [3] Loop over allpatches and take a weighted average:
	cv::Mat avgMotion;
	unsigned numPatches = 0;
	for(std::vector<std::vector<float> >::iterator m=mProb.begin();m!=mProb.end();++m){
		numPatches += m->size();
		for(std::vector<float>::iterator pm=m->begin();pm!=m->end();++pm){
			// [2.1] I want to minimize distance to mean
			unsigned motionClsId   = m-mProb.begin();
			unsigned motionPatchId = pm-(m->begin());
			cv::Mat *motion        = trainSet[motionClsId][motionPatchId]->motion\
				(features);
			if(this->leafavg_){
				if(avgMotion.empty()){
					motion->copyTo(avgMotion);
				}else{
					avgMotion += (*motion);
				}
			}
			// Get it anyway for the best appearance and best histogram
			if((*pm)>bestMotionProb){
				bestMotionProb    = (*pm);
				bestmotionClsId   = m-mProb.begin();
				bestmotionPatchId = pm-(m->begin());
			}
			motion->release(); delete motion;
		}
	}
	// [3] Save the best patch and its appearance probability
	cv::Mat* tmpApp = trainSet[bestmotionClsId][bestmotionPatchId]->image(features);
	bestApp->release();
	tmpApp->copyTo(*bestApp);
	tmpApp->release(); delete tmpApp;
	// [4] Save the nest motion patch
	if(this->leafavg_){
		avgMotion /= static_cast<float>(numPatches);
		bestMotion->release();
		avgMotion.copyTo(*bestMotion);
	}else{
		cv::Mat* tmpMotion = trainSet[bestmotionClsId]\
			[bestmotionPatchId]->motion(features);
		bestMotion->release();
		tmpMotion->copyTo(*bestMotion);
		tmpMotion->release(); delete tmpMotion;
	}
	avgMotion.release();
	// [5] Save the histogram patch
	std::vector<cv::Mat> tmpHisto = trainSet[bestmotionClsId]\
		[bestmotionPatchId]->histo(features);
	for(unsigned b=0;b<tmpHisto.size();++b){
		bestHisto[b].release();
		tmpHisto[b].copyTo(bestHisto[b]);
		tmpHisto[b].release();
	}
	if(writeprobs && this->treeId_==0){
		this->writeprobs(mProb,nodeid,bestmotionPatchId);
	}
}
//==============================================================================
/** Writes down the probability for each leaf. As a check.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::writeprobs(const std::vector<std::vector<float> > &mProb,\
long unsigned nodeid,unsigned bestmotionPatchId){
	std::string folder;
	switch(this->entropy_){
		case(MotionTree<M,T,F,N,U>::APPROX_ANGLE_KERNEL):
			folder = "images_angle"+this->runname_;
			break;
		case(MotionTree<M,T,F,N,U>::APPROX_MAGNI_KERNEL):
			folder = "images_magni"+this->runname_;
			break;
		case(MotionTree<M,T,F,N,U>::MEAN_DIFF):
			folder = "images_mean"+this->runname_;
			break;
	}
	Auxiliary<uchar,1>::file_exists(folder.c_str(),true);
	std::string outputDir = folder+std::string(PATH_SEP)+\
		std::string("leaves")+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	outputDir += Auxiliary<long unsigned,1>::number2string(nodeid)+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	std::string probsfilename = outputDir+"probs.txt";
	std::ofstream probsfile;
	try{
		probsfile.open(probsfilename.c_str(),std::ios::out);
	}catch(std::exception &e){
		std::cerr<<"Cannot open file: %s"<<e.what()<<std::endl;
		exit(1);
	}
	probsfile.precision(std::numeric_limits<double>::digits10);
	probsfile.precision(std::numeric_limits<float>::digits10);
	for(std::vector<std::vector<float> >::const_iterator p=mProb.begin();p!=\
	mProb.end();++p){
		for(std::vector<float>::const_iterator pa=p->begin();pa!=p->end();++pa){
			probsfile<<"leaf_"<<Auxiliary<int,1>::number2string(pa-(p->begin()))<<\
				" >>> "<<(*pa);
			if(pa-(p->begin())==bestmotionPatchId){
				probsfile<<" >>> best patch"<<std::endl;
			}else{
				probsfile<<std::endl;
			}
		} // over patches
	} // over classes - not here
	probsfile.close();
}
//==============================================================================
/** Gets the closest patch to the mean-motion in the leaf (euclidian distance).
 */
template <class M,class T,class F,class N,class U>
std::vector<std::vector<float> > MotionTree<M,T,F,N,U>::patchDist2Mean\
(const std::vector<std::vector<const T*> > &trainSet,const F* features,\
unsigned totsize,const std::vector<float> &prevfreq){
	if(features->usederivatives()){
		return this->patchDist2MeanDerivatives(trainSet,features,totsize,\
			prevfreq);
	}else{
		return this->patchDist2MeanFlows(trainSet,features,totsize,prevfreq);
	}
}
//==============================================================================
/** Gets the closest patch to the mean-motion in the leaf (euclidian distance).
 */
template <class M,class T,class F,class N,class U>
std::vector<std::vector<float> > MotionTree<M,T,F,N,U>::patchDist2MeanDerivatives\
(const std::vector<std::vector<const T*> > &trainSet,const F* features,\
unsigned totsize,const std::vector<float> &prevfreq){
	std::vector<std::vector<float> > mProb(trainSet.size(),std::vector<float>());
	if(totsize<=1){
		for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
			mProb[l-trainSet.begin()] = std::vector<float>(l->size(),0.0);
		}
		return mProb;
	}
	// [0] Get the mean angle patch
	cv::Mat *meanXX       = new cv::Mat(cv::Mat::zeros(cv::Size(this->motionW_,\
		this->motionH_),CV_32FC1));
	cv::Mat *meanXY       = new cv::Mat(cv::Mat::zeros(cv::Size(this->motionW_,\
		this->motionH_),CV_32FC1));
	cv::Mat *meanYX       = new cv::Mat(cv::Mat::zeros(cv::Size(this->motionW_,\
		this->motionH_),CV_32FC1));
	cv::Mat *meanYY       = new cv::Mat(cv::Mat::zeros(cv::Size(this->motionW_,\
		this->motionH_),CV_32FC1));
	unsigned totalPatches = 0;
	// [1] Loop over classes (none) and patches
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		for(constIterT p=l->begin();p!=l->end();++p){ // over leaf patches
			cv::Mat *motionXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionXX,motionXY,motionYX,motionYY);
			// [2] Now find the means on x,y - direction
			(*meanXX) += (*motionXX);
			(*meanXY) += (*motionXY);
			(*meanYX) += (*motionYX);
			(*meanYY) += (*motionYY);
			++totalPatches;
			motionXX->release(); delete motionXX;
			motionXY->release(); delete motionXY;
			motionYX->release(); delete motionYX;
			motionYY->release(); delete motionYY;
		}
	}
	// [2] Normalize the mean to be a mean
	(*meanXX) *= 1.0/static_cast<float>(totalPatches);
	(*meanXY) *= 1.0/static_cast<float>(totalPatches);
	(*meanYX) *= 1.0/static_cast<float>(totalPatches);
	(*meanYY) *= 1.0/static_cast<float>(totalPatches);
	// [3] Now get the distances to mean
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		mProb[l-trainSet.begin()] = std::vector<float>(l->size(),0.0);
		for(constIterT p=l->begin();p!=l->end();++p){ // over patches
			cv::Mat *motionXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionXX,motionXY,motionYX,motionYY);
			// [4] Get again the motions and get the RSD to mean
			cv::Mat tmpXX = (*meanXX)-(*motionXX);
			cv::Mat tmpXY = (*meanXY)-(*motionXY);
			cv::Mat tmpYX = (*meanYX)-(*motionYX);
			cv::Mat tmpYY = (*meanYY)-(*motionYY);
			double dotXX  = tmpXX.dot(tmpXX);
			double dotXY  = tmpXY.dot(tmpXY);
			double dotYX  = tmpYX.dot(tmpYX);
			double dotYY  = tmpYY.dot(tmpYY);
			tmpXX.release(); tmpXY.release(); tmpYX.release(); tmpYY.release();
			mProb[l-trainSet.begin()][p-(l->begin())] = std::sqrt(dotXX+dotXY+\
				dotYX+dotYY);
			motionXX->release(); delete motionXX;
			motionXY->release(); delete motionXY;
			motionYX->release(); delete motionYX;
			motionYY->release(); delete motionYY;
		}
	}
	meanXX->release(); delete meanXX;
	meanXY->release(); delete meanXY;
	meanYX->release(); delete meanYX;
	meanYY->release(); delete meanYY;
	return mProb;
}
//==============================================================================
/** Gets the closest patch to the mean-motion in the leaf (euclidian distance).
 */
template <class M,class T,class F,class N,class U>
std::vector<std::vector<float> > MotionTree<M,T,F,N,U>::patchDist2MeanFlows\
(const std::vector<std::vector<const T*> > &trainSet,const F* features,\
unsigned totsize,const std::vector<float> &prevfreq){
	std::vector<std::vector<float> > mProb(trainSet.size(),std::vector<float>());
	if(totsize<=1){
		for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
			mProb[l-trainSet.begin()] = std::vector<float>(l->size(),0.0);
		}
		return mProb;
	}
	// [0] Get the mean angle patch
	cv::Mat *meanX  = new cv::Mat(cv::Mat::zeros(cv::Size(this->motionW_,\
		this->motionH_),CV_32FC1));
	cv::Mat *meanY = new cv::Mat(cv::Mat::zeros(cv::Size(this->motionW_,\
		this->motionH_),CV_32FC1));
	unsigned totalPatches = 0;
	// [1] Loop over classes (none) and patches
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		for(constIterT p=l->begin();p!=l->end();++p){ // over leaf patches
			cv::Mat *motionX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionX,motionY);
			// [2] Now find the means on x,y - direction
			(*meanX) += (*motionX);
			(*meanY) += (*motionY);
			++totalPatches;
			motionX->release(); delete motionX;
			motionY->release(); delete motionY;
		}
	}
	// [2] Normalize the mean to be a mean
	(*meanX) *= 1.0/static_cast<float>(totalPatches);
	(*meanY) *= 1.0/static_cast<float>(totalPatches);
	// [3] Now get the distances to mean
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		mProb[l-trainSet.begin()] = std::vector<float>(l->size(),0.0);
		for(constIterT p=l->begin();p!=l->end();++p){ // over patches
			cv::Mat *motionX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionX,motionY);
			// [4] Get again the motions and get the RSD to mean
			cv::Mat tmpX = (*meanX)-(*motionX);
			cv::Mat tmpY = (*meanY)-(*motionY);
			double dotX  = tmpX.dot(tmpX);
			double dotY  = tmpY.dot(tmpY);
			tmpX.release(); tmpY.release();
			mProb[l-trainSet.begin()][p-(l->begin())] = std::sqrt(dotX+dotY);
			motionX->release(); delete motionX;
			motionY->release(); delete motionY;
		}
	}
	meanX->release(); delete meanX;
	meanY->release(); delete meanY;
	return mProb;
}
//==============================================================================
/** For each patch finds it probability as 1/#bins sum_bins k(sample-bin).
 */
template <class M,class T,class F,class N,class U>
std::vector<std::vector<float> > MotionTree<M,T,F,N,U>::patchApprox(const F* \
features,const std::vector<std::vector<const T*> > &trainSet,unsigned totPatches,\
const std::vector<cv::Mat> &prevfreq){
	std::vector<std::vector<float> > mProb(trainSet.size(),std::vector<float>());
	if(totPatches<=1){
		for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
			mProb[l-trainSet.begin()] = std::vector<float>(l->size(),0);
		}
		return mProb;
	}
	// [0] Loop over patches and get the probabilities per dimension per bin
	std::vector<cv::Mat> probs; // bins => patch size
	cv::Mat norm = cv::Mat::zeros(cv::Size(this->motionW_,motionH_),CV_32FC1);
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
		for(constIterT p=l->begin();p!=l->end();++p){ // over patches
			std::vector<cv::Mat> histo = (*p)->histo(features);
			if(probs.empty()){ probs.resize(histo.size()); }
			std::vector<cv::Mat>::iterator pr = probs.begin();
			for(std::vector<cv::Mat>::iterator hi=histo.begin();hi!=histo.end(),\
			pr!=probs.end();++hi,++pr){
				if(pr->empty()){ hi->copyTo(*pr);
				}else{ (*pr) += (*hi); }
				norm += (*hi);
				hi->release();
			} // over bins
		} // over patches
	} // over classes -- not here
	cv::Mat mask;
	cv::inRange(norm,0,SMALL,mask);
	norm.setTo(1,mask); mask.release();
	for(std::vector<cv::Mat>::iterator pr=probs.begin();pr!=probs.end();++pr){
		cv::divide((*pr),norm,(*pr));
	}
	norm.release();
	// [1] Do the parent weighting if wanted/needed
	if(this->leafParentFreq_){
		if(this->usepick_){
			cv::Mat_<float>::const_iterator fr = prevfreq[0].begin<float>();
			for(std::vector<cv::Mat>::iterator pr=probs.begin();pr!=probs.end(),\
			fr!=prevfreq[0].end<float>();++pr,++fr){
				if((*fr)>SMALL){
					(*pr) /= (*fr);
				}
			} // over bins
		}else{
			std::vector<cv::Mat>::const_iterator fr = prevfreq.begin();
			for(std::vector<cv::Mat>::iterator pr=probs.begin();pr!=probs.end(),\
			fr!=prevfreq.end();++pr,++fr){
				cv::Mat mask;
				cv::Mat frclone = fr->clone();
				cv::inRange(frclone,0,SMALL,mask);
				frclone.setTo(1,mask); mask.release();
				cv::divide((*pr),frclone,(*pr));
				frclone.release();
			} // over bins
		}
	}
	// [2] Loop again over patches and get the patch probability
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
		if(mProb[l-trainSet.begin()].empty()){
			mProb[l-trainSet.begin()].resize(l->size(),0.0);
		}
		for(constIterT p=l->begin();p!=l->end();++p){ // over patches
			// [3] Get the patch probability and store it
			mProb[l-trainSet.begin()][p-(l->begin())] = this->patchProb\
				(probs,(*p),features);
		} // over patches
	} // over classes - not here
	for(std::vector<cv::Mat>::iterator pr=probs.begin();pr!=probs.end();++pr){
		pr->release();
	}
	return mProb;
}
//==============================================================================
/** Gets the patch probabilities as sum_px log p(px)
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::patchProb(const std::vector<cv::Mat> &probs,const T* \
patch,const F* features){
	float patchprob = 0.0;
	if(features->usederivatives()){
		cv::Mat *motionXX = new cv::Mat(cv::Size(patch->motionW(),\
			patch->motionH()),CV_32FC1);
		cv::Mat *motionXY = new cv::Mat(cv::Size(patch->motionW(),\
			patch->motionH()),CV_32FC1);
		cv::Mat *motionYX = new cv::Mat(cv::Size(patch->motionW(),\
			patch->motionH()),CV_32FC1);
		cv::Mat *motionYY = new cv::Mat(cv::Size(patch->motionW(),\
			patch->motionH()),CV_32FC1);
		patch->motion(features,motionXX,motionXY,motionYX,motionYY);
		cv::Mat_<float>::iterator yy = motionYY->begin<float>();
		cv::Mat_<float>::iterator yx = motionYX->begin<float>();
		cv::Mat_<float>::iterator xy = motionXY->begin<float>();
		for(cv::Mat_<float>::iterator xx=motionXX->begin<float>();xx!=motionXX->\
		end<float>(),xy!=motionXY->end<float>(),yx!=motionYX->end<float>(),\
		yy!=motionYY->end<float>();++xx,++xy,++yx,++yy){
			float pxprob      = 0.0;
			unsigned patchpos = (xx-motionXX->begin<float>());
			cv::Point ptpos   = cv::Point((static_cast<unsigned>(patchpos)%\
				this->motionW_),std::floor(static_cast<float>(patchpos)/\
				static_cast<float>(this->motionW_)));
			std::vector<float> values;
			values.push_back(*xx); values.push_back(*xy);
			values.push_back(*yx); values.push_back(*yy);
			switch(this->entropy_){
				case(MotionTree<M,T,F,N,U>::APPROX_ANGLE_KERNEL):
					pxprob = MotionTree<M,T,F,N,U>::getProbAngle(probs,\
						this->histinfo_,values,ptpos);
					break;
				case(MotionTree<M,T,F,N,U>::APPROX_MAGNI_KERNEL):
					pxprob = MotionTree<M,T,F,N,U>::getProbMagni(probs,\
						this->histinfo_,values,ptpos);
					break;
				default:
					pxprob = MotionTree<M,T,F,N,U>::getProbMagni(probs,\
						this->histinfo_,values,ptpos);
					break;
			}
			patchprob += (std::isinf(std::log(pxprob))?std::log(SMALL):\
				std::log(pxprob));
		} // over dimensions
		motionXX->release(); delete motionXX;
		motionXY->release(); delete motionXY;
		motionYX->release(); delete motionYX;
		motionYY->release(); delete motionYY;
	}else{
		cv::Mat *motionX = new cv::Mat(cv::Size(patch->motionW(),\
			patch->motionH()),CV_32FC1);
		cv::Mat *motionY = new cv::Mat(cv::Size(patch->motionW(),\
			patch->motionH()),CV_32FC1);
		patch->motion(features,motionX,motionY);
		cv::Mat_<float>::iterator x = motionX->begin<float>();
		for(cv::Mat_<float>::iterator y=motionY->begin<float>();y!=motionY->\
		end<float>(),x!=motionX->end<float>();++x,++y){
			float pxprob      = 0.0;
			unsigned patchpos = (x-motionX->begin<float>());
			cv::Point ptpos   = cv::Point((static_cast<unsigned>(patchpos)%\
				this->motionW_),std::floor(static_cast<float>(patchpos)/\
				static_cast<float>(this->motionW_)));
			std::vector<float> values;
			values.push_back(*x); values.push_back(*y);
			switch(this->entropy_){
				case(MotionTree<M,T,F,N,U>::APPROX_ANGLE_KERNEL):
					pxprob = MotionTree<M,T,F,N,U>::getProbAngle(probs,\
						this->histinfo_,values,ptpos);
					break;
				case(MotionTree<M,T,F,N,U>::APPROX_MAGNI_KERNEL):
					pxprob = MotionTree<M,T,F,N,U>::getProbMagni(probs,\
						this->histinfo_,values,ptpos);
					break;
				default:
					pxprob = MotionTree<M,T,F,N,U>::getProbMagni(probs,\
						this->histinfo_,values,ptpos);
					break;
			}
			patchprob += (std::isinf(std::log(pxprob))?std::log(SMALL):\
				std::log(pxprob));
		} // over dimensions
		motionX->release(); delete motionX;
		motionY->release(); delete motionY;
	}
	return patchprob;
}
//==============================================================================

/** Just dot product between vectors.
 */
template <class M,class T,class F,class N,class U>
float MotionTree<M,T,F,N,U>::dotProd(const std::vector<float> &asmpl,const \
std::vector<float> &dimprobs){
	float freq = 0.0;
	assert(dimprobs.size()==asmpl.size());
	std::vector<float>::const_iterator a = asmpl.begin();
	for(std::vector<float>::const_iterator d=dimprobs.begin();d!=dimprobs.end(),\
	a!=asmpl.end();++d,++a){
		freq += (*a)*(*d);
	}
	return freq;
}
//==============================================================================
/** Gets the appearance probabilities in the leaf based on similarity.
 */
template <class M,class T,class F,class N,class U>
std::vector<std::vector<float> > MotionTree<M,T,F,N,U>::patchAppearanceSim\
(const std::vector<std::vector<const T*> > &trainSet,const F* features,unsigned \
totPatches){
	std::vector<std::vector<float> > appProb(trainSet.size(),std::vector<float>());
	if(totPatches<=1){
		for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){
			appProb[l-trainSet.begin()] = std::vector<float>(l->size(),0);
		}
		return appProb;
	}
	// [0] Make a table of appearance similarities between patches: p(app,app')
	std::vector<std::vector<float> > similarity(totPatches,std::vector<float>\
		(totPatches,0));
	unsigned indexApp = 0;
	float toNormalize = 0.0;
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		for(constIterT p=l->begin();p!=l->end();++p){ // over leaf patches
			cv::Mat* appearance = (*p)->image(features);
			unsigned indexOth  = 0;
			// [1] Get the mean similarity with all the others
			for(vectConstIterT ll=trainSet.begin();ll!=trainSet.end();++ll){ // over classes
				for(constIterT pp=ll->begin();pp!=ll->end();++pp){ // over leaf patches
					if(indexOth!=indexApp){
						cv::Mat* other                 = (*pp)->image(features);
						similarity[indexApp][indexOth] = appearance->dot(*other)/\
							(std::sqrt(appearance->dot(*appearance))*std::sqrt\
							(other->dot(*other)));
						other->release(); delete other;
						toNormalize += similarity[indexApp][indexOth];
					}
					++indexOth;
				}
			}
			appearance->release(); delete appearance;
			++indexApp;
		}
	}
	// [2] Normalize all the similarities and pick the final ones
	indexApp = 0;
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		appProb[l-trainSet.begin()] = std::vector<float>(l->size(),0);
		for(constIterT p=l->begin();p!=l->end();++p){ // over leaf patches
			// [3] Get p(app) = sum_app' p(app,app')
			for(unsigned row=0;row<similarity[indexApp].size();++row){
				appProb[l-trainSet.begin()][p-(l->begin())] += similarity\
					[indexApp][row]/toNormalize;
			}
			++indexApp;
		}
	}
	return appProb;
}
//==============================================================================
/** Displays the samples among which we need to choose to make a leaf
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::showSamples(const std::vector<std::vector<const T*> > \
&trainSet,const F* features,long unsigned nodeid,float entropy,const cv::Mat* bestMotion,\
const cv::Mat *bestApp,bool justdisplay){
	if(features->usederivatives()){
		this->showSamplesDerivatives(trainSet,features,nodeid,entropy,\
			bestMotion,bestApp,justdisplay);
	}else{
		this->showSamplesFlows(trainSet,features,nodeid,entropy,\
			bestMotion,bestApp,justdisplay);
	}
}
//==============================================================================
/** Displays the samples among which we need to choose to make a leaf
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::showSamplesDerivatives(const std::vector<std::vector<const T*> > \
&trainSet,const F* features,long unsigned nodeid,float entropy,const cv::Mat* bestMotion,\
const cv::Mat *bestApp,bool justdisplay){
	std::string folder;
	switch(this->entropy_){
		case(MotionTree<M,T,F,N,U>::APPROX_ANGLE_KERNEL):
			folder = "images_angle"+this->runname_;
			break;
		case(MotionTree<M,T,F,N,U>::APPROX_MAGNI_KERNEL):
			folder = "images_magni"+this->runname_;
			break;
		case(MotionTree<M,T,F,N,U>::MEAN_DIFF):
			folder = "images_mean"+this->runname_;
			break;
	}
	Auxiliary<uchar,1>::file_exists(folder.c_str(),true);
	// [1] Loop over classes (none) and patches
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		for(constIterT p=l->begin();p!=l->end();++p){ // over leaf patches
			cv::Mat *motionXX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionXY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionYY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionXX,motionXY,motionYX,motionYY);
			cv::Mat *patch = (*p)->image(features);
			cv::copyMakeBorder(*patch,*patch,10,10,10,10,cv::BORDER_CONSTANT,0);
			// [2] Make some borders around so we can see them better
			cv::copyMakeBorder(*motionXX,*motionXX,10,10,10,10,cv::BORDER_CONSTANT,0);
			cv::copyMakeBorder(*motionXY,*motionXY,10,10,10,10,cv::BORDER_CONSTANT,0);
			cv::copyMakeBorder(*motionYX,*motionYX,10,10,10,10,cv::BORDER_CONSTANT,0);
			cv::copyMakeBorder(*motionYY,*motionYY,10,10,10,10,cv::BORDER_CONSTANT,0);
			// [3] Make the directory to store them into
			std::string outputDir = folder+std::string(PATH_SEP)+\
				std::string("leaves")+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
			outputDir += Auxiliary<long unsigned,1>::number2string(nodeid)+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
			// [4] Now display the OF vectors on the patch
			cv::Mat garbage = MotionPatch<T,F>::showOFderi(*motionXX,*motionXY,\
				*motionYX,*motionYY,*patch,5,false);
			// [5] Now write the patch out
			float scaleVal = static_cast<float>(500.0/std::min(garbage.cols,garbage.rows));
			cv::resize(garbage,garbage,cv::Size(garbage.cols*scaleVal,garbage.rows*scaleVal));
			if(justdisplay){
				cv::imshow("garbage",garbage);
				cv::waitKey(10);
			}else{
				std::string imgPath = outputDir+"leaf_"+Auxiliary<int,1>::\
					number2string(p-l->begin())+".jpg";
				cv::imwrite(imgPath,garbage);
			}
			garbage.release();
			motionXX->release(); delete motionXX;
			motionXY->release(); delete motionXY;
			motionYX->release(); delete motionYX;
			motionYY->release(); delete motionYY;
			patch->release(); delete patch;
		}
	}
	cv::Mat apatch      = cv::Mat::zeros(cv::Size(this->motionW_+20,\
		this->motionH_+20),CV_8UC3);
	cv::Mat amotionXX   = bestMotion->colRange(0,bestMotion->cols/4).clone();
	cv::Mat amotionXY   = bestMotion->colRange(bestMotion->cols/4,bestMotion->cols/2).clone();
	cv::Mat amotionYX   = bestMotion->colRange(bestMotion->cols/2,bestMotion->cols*3/4).clone();
	cv::Mat amotionYY   = bestMotion->colRange(bestMotion->cols*3/4,bestMotion->cols).clone();
	cv::Mat tmpmotionXX = amotionXX.reshape(0,motionH_);
	cv::Mat tmpmotionXY = amotionXY.reshape(0,motionH_);
	cv::Mat tmpmotionYX = amotionYX.reshape(0,motionH_);
	cv::Mat tmpmotionYY = amotionYY.reshape(0,motionH_);
	amotionXX.release();amotionXY.release();amotionYX.release();amotionYY.release();
	cv::copyMakeBorder(tmpmotionXX,tmpmotionXX,10,10,10,10,cv::BORDER_CONSTANT,0);
	cv::copyMakeBorder(tmpmotionXY,tmpmotionXY,10,10,10,10,cv::BORDER_CONSTANT,0);
	cv::copyMakeBorder(tmpmotionYX,tmpmotionYX,10,10,10,10,cv::BORDER_CONSTANT,0);
	cv::copyMakeBorder(tmpmotionYY,tmpmotionYY,10,10,10,10,cv::BORDER_CONSTANT,0);
	cv::Mat garbage       = MotionPatch<T,F>::showOFderi(tmpmotionXX,tmpmotionXY,\
		tmpmotionYX,tmpmotionYY,apatch,5,false);
	std::string outputDir = folder+std::string(PATH_SEP)+std::string("leaves")+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	outputDir            += Auxiliary<long unsigned,1>::number2string(nodeid)+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	std::string imgPath   = outputDir+"leaf_best.jpg";
	cv::imwrite(imgPath,garbage);
	std::string appPath   = outputDir+"leaf_bestApp.jpg";
	cv::imwrite(appPath,*bestApp);
	apatch.release(); garbage.release();
	tmpmotionXX.release(); tmpmotionXY.release();
	tmpmotionYX.release(); tmpmotionYY.release();
}
//==============================================================================
/** Displays the samples among which we need to choose to make a leaf
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::showSamplesFlows(const std::vector<std::vector<const T*> > \
&trainSet,const F* features,long unsigned nodeid,float entropy,const cv::Mat* bestMotion,\
const cv::Mat *bestApp,bool justdisplay){
	std::string folder;
	switch(this->entropy_){
		case(MotionTree<M,T,F,N,U>::APPROX_ANGLE_KERNEL):
			folder = "images_angle"+this->runname_;
			break;
		case(MotionTree<M,T,F,N,U>::APPROX_MAGNI_KERNEL):
			folder = "images_magni"+this->runname_;
			break;
		case(MotionTree<M,T,F,N,U>::MEAN_DIFF):
			folder = "images_mean"+this->runname_;
			break;
	}
	Auxiliary<uchar,1>::file_exists(folder.c_str(),true);
	// [1] Loop over classes (none) and patches
	for(vectConstIterT l=trainSet.begin();l!=trainSet.end();++l){ // over classes
		for(constIterT p=l->begin();p!=l->end();++p){ // over leaf patches
			cv::Mat *motionX = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			cv::Mat *motionY = new cv::Mat(cv::Size((*p)->motionW(),\
				(*p)->motionH()),CV_32FC1);
			(*p)->motion(features,motionX,motionY);
			cv::Mat *patch = (*p)->image(features);
			cv::copyMakeBorder(*patch,*patch,10,10,10,10,cv::BORDER_CONSTANT,0);
			// [2] Make some borders around so we can see them better
			cv::copyMakeBorder(*motionX,*motionX,10,10,10,10,cv::BORDER_CONSTANT,0);
			cv::copyMakeBorder(*motionY,*motionY,10,10,10,10,cv::BORDER_CONSTANT,0);
			// [3] Make the directory to store them into
			std::string outputDir = folder+std::string(PATH_SEP)+\
				std::string("leaves")+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
			outputDir += Auxiliary<long unsigned,1>::number2string(nodeid)+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
			// [4] Now display the OF vectors on the patch
			cv::Mat garbage = MotionPatch<T,F>::showOF(*motionX,*motionY,\
				*patch,5,false);
			// [5] Now write the patch out
			float scaleVal = static_cast<float>(500.0/std::min(garbage.cols,garbage.rows));
			cv::resize(garbage,garbage,cv::Size(garbage.cols*scaleVal,garbage.rows*scaleVal));
			if(justdisplay){
				cv::imshow("garbage",garbage);
				cv::waitKey(10);
			}else{
				std::string imgPath = outputDir+"leaf_"+Auxiliary<int,1>::\
					number2string(p-l->begin())+".jpg";
				cv::imwrite(imgPath,garbage);
			}
			garbage.release();
			motionX->release(); delete motionX;
			motionY->release(); delete motionY;
			patch->release(); delete patch;
		}
	}
	cv::Mat apatch   = cv::Mat::zeros(cv::Size(this->motionW_+20,\
		this->motionH_+20),CV_8UC3);
	cv::Mat amotionX = bestMotion->colRange(0,bestMotion->cols/2);
	cv::Mat amotionY = bestMotion->colRange(bestMotion->cols/2,bestMotion->cols);
	cv::Mat tmpmotionX, tmpmotionY;
	amotionX.copyTo(tmpmotionX);
	amotionY.copyTo(tmpmotionY);
	tmpmotionX = tmpmotionX.reshape(0,motionH_);
	tmpmotionY = tmpmotionY.reshape(0,motionH_);
	cv::copyMakeBorder(tmpmotionX,tmpmotionX,10,10,10,10,cv::BORDER_CONSTANT,0);
	cv::copyMakeBorder(tmpmotionY,tmpmotionY,10,10,10,10,cv::BORDER_CONSTANT,0);
	cv::Mat garbage       = MotionPatch<T,F>::showOF(tmpmotionX,tmpmotionY,\
		apatch,5,false);
	std::string outputDir = folder+std::string(PATH_SEP)+std::string("leaves")+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	outputDir            += Auxiliary<long unsigned,1>::number2string(nodeid)+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	std::string imgPath   = outputDir+"leaf_best.jpg";
	cv::imwrite(imgPath,garbage);
	std::string appPath   = outputDir+"leaf_bestapp.jpg";
	cv::imwrite(appPath,*bestApp);
	apatch.release(); garbage.release();
	tmpmotionX.release(); tmpmotionY.release();
}
//==============================================================================
/** Displays the set of predicted leaves.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::showSamplesDerivatives(const std::vector<const U*> &leaves,\
unsigned sampleW,unsigned sampleH,const cv::Point &point){
	cv::Mat patch         = cv::Mat::zeros(cv::Size(sampleW+50,sampleH+50),CV_8UC3);
	Auxiliary<uchar,1>::file_exists("images",true);
	std::string outputDir = std::string("images")+std::string(PATH_SEP)+\
		std::string("trees")+std::string(PATH_SEP);
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	outputDir += "trees_["+Auxiliary<int,1>::number2string\
		(point.x)+"]["+Auxiliary<int,1>::number2string(point.y)+"]"+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	// [1] Loop over trees
	for(typename std::vector<const U*>::const_iterator l=leaves.begin();l!=leaves.end();++l){
		cv::Mat *motion  = (*l)->vMotion();
		cv::Mat motionXX = cv::Mat::zeros(cv::Size(sampleW*sampleH,1),CV_32FC1);
		cv::Mat motionXY = cv::Mat::zeros(cv::Size(sampleW*sampleH,1),CV_32FC1);
		cv::Mat motionYX = cv::Mat::zeros(cv::Size(sampleW*sampleH,1),CV_32FC1);
		cv::Mat motionYY = cv::Mat::zeros(cv::Size(sampleW*sampleH,1),CV_32FC1);
		cv::MatIterator_<float> xx = motionXX.begin<float>();
		cv::MatIterator_<float> xy = motionXY.begin<float>();
		cv::MatIterator_<float> yx = motionYX.begin<float>();
		cv::MatIterator_<float> yy = motionYY.begin<float>();
		for(cv::MatConstIterator_<float> m=motion->begin<float>();m!=motion->end<float>()-\
		(motion->cols/4),xx!=motionXX.end<float>(),xy!=motionXY.end<float>(),\
		yx!=motionYX.end<float>(),yy!=motionYY.end<float>();++m,++xx,++xy,++yx,++yy){
			(*xx) = (*m);
			(*xy) = *(m+(motion->cols)/4);
			(*yx) = *(m+(motion->cols)/2);
			(*yy) = *(m+(motion->cols)*3/4);
		}
		motionXX = motionXX.reshape(0,sampleH);
		motionXY = motionXY.reshape(0,sampleH);
		motionYX = motionYX.reshape(0,sampleH);
		motionYY = motionYY.reshape(0,sampleH);
		// [2] Make some borders around so we can see them better
		cv::copyMakeBorder(motionXX,motionXX,25,25,25,25,cv::BORDER_CONSTANT,0);
		cv::copyMakeBorder(motionXY,motionXY,25,25,25,25,cv::BORDER_CONSTANT,0);
		cv::copyMakeBorder(motionYX,motionYX,25,25,25,25,cv::BORDER_CONSTANT,0);
		cv::copyMakeBorder(motionYY,motionYY,25,25,25,25,cv::BORDER_CONSTANT,0);
		// [3] Now display the OF vectors on the patch
		cv::Mat garbage = MotionPatch<MotionPatchFeature<FeaturesMotion>,FeaturesMotion>::\
			showOFderi(motionXX,motionXY,motionYX,motionYY,patch,5,true);
		// [4] Now write the patch out
		float scaleVal = static_cast<float>(500.0/std::min(garbage.cols,garbage.rows));
		cv::resize(garbage,garbage,cv::Size(garbage.cols*scaleVal,garbage.rows*scaleVal));
		std::string imgPath = outputDir+"leaf_"+Auxiliary<int,1>::number2string\
			(l-leaves.begin())+".jpg";
		cv::imwrite(imgPath,garbage);
		garbage.release();
		motionXX.release(); motionXY.release(); motionYX.release(); motionYY.release();
	}
	patch.release();
}
//==============================================================================
/** Displays the set of predicted leaves.
 */
template <class M,class T,class F,class N,class U>
void MotionTree<M,T,F,N,U>::showSamplesFlows(const std::vector<const U*> &leaves,\
unsigned sampleW,unsigned sampleH,const cv::Point &point){
	cv::Mat patch         = cv::Mat::zeros(cv::Size(sampleW+50,sampleH+50),CV_8UC3);
	Auxiliary<uchar,1>::file_exists("images",true);
	std::string outputDir = std::string("images")+std::string(PATH_SEP)+\
		std::string("trees")+std::string(PATH_SEP);
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	outputDir += "trees_["+Auxiliary<int,1>::number2string\
		(point.x)+"]["+Auxiliary<int,1>::number2string(point.y)+"]"+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(outputDir.c_str(),true);
	// [1] Loop over trees
	for(typename std::vector<const U*>::const_iterator l=leaves.begin();l!=leaves.end();++l){
		cv::Mat *motion = (*l)->vMotion();
		cv::Mat motionX = cv::Mat::zeros(cv::Size(sampleW*sampleH,1),CV_32FC1);
		cv::Mat motionY = cv::Mat::zeros(cv::Size(sampleW*sampleH,1),CV_32FC1);
		cv::MatIterator_<float> x = motionX.begin<float>();
		cv::MatIterator_<float> y = motionY.begin<float>();
		for(cv::MatConstIterator_<float> m=motion->begin<float>();m!=motion->end<float>()-\
		(motion->cols/2),x!=motionX.end<float>(),y!=motionY.end<float>();++m,++x,++y){
			(*x) = (*m);
			(*y) = *(m+(motion->cols)/2);
		}
		motionX = motionX.reshape(0,sampleH);
		motionY = motionY.reshape(0,sampleH);
		// [2] Make some borders around so we can see them better
		cv::copyMakeBorder(motionX,motionX,25,25,25,25,cv::BORDER_CONSTANT,0);
		cv::copyMakeBorder(motionY,motionY,25,25,25,25,cv::BORDER_CONSTANT,0);
		// [3] Now display the OF vectors on the patch
		cv::Mat garbage = MotionPatch<MotionPatchFeature<FeaturesMotion>,FeaturesMotion>::\
			showOF(motionX,motionY,patch,5,true);
		// [4] Now write the patch out
		float scaleVal = static_cast<float>(500.0/std::min(garbage.cols,garbage.rows));
		cv::resize(garbage,garbage,cv::Size(garbage.cols*scaleVal,garbage.rows*scaleVal));
		std::string imgPath = outputDir+"leaf_"+Auxiliary<int,1>::number2string\
			(l-leaves.begin())+".jpg";
		cv::imwrite(imgPath,garbage);
		garbage.release();
		motionX.release(); motionY.release();
	}
	patch.release();
}
//==============================================================================
/** Writes the current tree into a given file.
 */
template <class M,class T,class F,class N,class U>
bool MotionTree<M,T,F,N,U>::saveTree(){
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
bool MotionTree<M,T,F,N,U>::saveTreeTxt(){
	std::cout<<"["<<this->treeId_<<"] [MotionTree<M,T,U>::saveTree] Save Tree "<<\
		this->path2models_<<" "<<this->treeId_<<std::endl;
	std::string dummy(this->path2models_);
	bool done      = false;
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%s%03dtree.txt",this->path2models_,this->treeId_);
	std::ofstream out(filename);
	if(out.good()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		// [0] Write general stuff (sizes, trees, etc.)
		out<<this->treeId_<<" "<<" "<<this->patchW_<<" "<<\
			this->patchH_<<" "<<this->patchCh_<<" "<<this->maxDepth_<<" "<<\
			this->entropythresh_<<std::endl;
		// [1] Write the prior class frequencies
		out<<this->histinfo_.size()<<" ";
		for(std::vector<float>::const_iterator ptB=this->histinfo_.begin();\
		ptB!=this->histinfo_.end();++ptB){
			out<<(*ptB)<<" ";
		}
		out<<std::endl;
		this->preorderTxt(this->root_,out);
		out.close();
	}
	delete [] filename;
	// [3] Save the times into the log file
	clock_t clockend = clock();
	std::cout<<"[MotionTree::saveTreeBin]: Training tree ["<<this->treeId_<<\
		" time elapsed: "<<double(Auxiliary<uchar,1>::diffclock(clockend,\
		this->clockbegin_))<<" sec"<<std::endl;
	this->log_<<"[MotionTree::saveTreeBin]: Training tree ["<<this->treeId_<<\
		" time elapsed: "<<double(Auxiliary<uchar,1>::diffclock(clockend,\
		this->clockbegin_))<<" sec"<<std::endl;
	done = true;
	return done;
}
//==============================================================================
/** Writes the current tree into a given binary file.
 */
template <class M,class T,class F,class N,class U>
bool MotionTree<M,T,F,N,U>::saveTreeBin(){
	std::cout<<"["<<this->treeId_<<"] [MotionTree<M,T,F,N,U>::saveTreeBin] Save Tree "<<\
		this->path2models_<<" "<<this->treeId_<<std::endl;
	std::string dummy(this->path2models_);
	bool done      = false;
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%s%03dtree.bin",this->path2models_,this->treeId_);
	std::ofstream out;
	out.open(filename,std::ios::out|std::ios::binary);
	// FIRST WRITE THE DIMENSIONS OF THE MATRIX
	if(out.good()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		// [0] Write general stuff (sizes, trees, etc.)
		unsigned treeId = this->treeId_;
		out.write(reinterpret_cast<const char*>(&treeId),sizeof(unsigned));
		unsigned patchW = this->patchW_;
		out.write(reinterpret_cast<const char*>(&patchW),sizeof(unsigned));
		unsigned patchH = this->patchH_;
		out.write(reinterpret_cast<const char*>(&patchH),sizeof(unsigned));
		unsigned patchCh = this->patchCh_;
		out.write(reinterpret_cast<const char*>(&patchCh),sizeof(unsigned));
		unsigned maxDepth = this->maxDepth_;
		out.write(reinterpret_cast<const char*>(&maxDepth),sizeof(unsigned));
		unsigned entropythresh = this->entropythresh_;
		out.write(reinterpret_cast<const char*>(&entropythresh),sizeof(float));
		// [1] Write the prior class frequencies
		unsigned infosize = this->histinfo_.size();
		out.write(reinterpret_cast<const char*>(&infosize),sizeof(unsigned));
		for(std::vector<float>::const_iterator ptB=this->histinfo_.begin();\
		ptB!=this->histinfo_.end();++ptB){
			float info = (*ptB);
			out.write(reinterpret_cast<const char*>(&info),sizeof(float));
		}
		this->preorderBin(this->root_,out);
		out.close();
	}
	delete [] filename;
	done = true;
	// [3] Save the times into the log file
	clock_t clockend = clock();
	std::cout<<"[MotionTree::saveTreeBin]: Training tree ["<<this->treeId_<<\
		"] time elapsed: "<<double(Auxiliary<uchar,1>::diffclock(clockend,\
		this->clockbegin_))<<" sec"<<std::endl;
	this->log_<<"[MotionTree::saveTreeBin]: Training tree ["<<this->treeId_<<\
		"] time elapsed: "<<double(Auxiliary<uchar,1>::diffclock(clockend,\
		this->clockbegin_))<<" sec"<<std::endl;
	return done;
}
//==============================================================================
//==============================================================================
//==============================================================================
template class MotionTree<MotionPatch<MotionPatchFeature<FeaturesMotion>,\
FeaturesMotion>,MotionPatchFeature<FeaturesMotion>,FeaturesMotion,MotionTreeNode\
<MotionLeafNode,MotionPatchFeature<FeaturesMotion> >,MotionLeafNode>;









































