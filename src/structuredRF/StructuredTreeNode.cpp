/* StructuredTreeNode.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef STRUCTUREDTREENODE_CPP_
#define STRUCTUREDTREENODE_CPP_
#include "StructuredTreeNode.h"
//==============================================================================
inline LabelLeafNode::LabelLeafNode(const char* path2models,long unsigned leafid,\
unsigned treeid,bool binary){
	this->labelProb_ = 0.0;
	this->vLabels_   = std::vector<unsigned>();
	this->isempty_   = true;
	if(binary){
		this->readLeafBin(path2models,leafid,treeid);
	}else{
		this->readLeafTxt(path2models,leafid,treeid);
	}
	if(!this->vLabels_.empty()){this->isempty_ = false;}
}
//==============================================================================
/** Reads the leaf from a regular file.
 */
inline void LabelLeafNode::readLeafTxt(const char *path2models,long unsigned leafid,\
unsigned treeid){
	std::string dummy(path2models);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%sleaf%lu_%d.txt",path2models,leafid,treeid);
	std::ifstream in(filename);
	if(in.is_open()){
		unsigned lSize;in>>lSize;
		in>>this->labelProb_;
		this->vLabels_.resize(lSize,0);
		for(std::vector<unsigned>::iterator it=this->vLabels_.begin();\
		it!=this->vLabels_.end();++it){
			in>>(*it);
		}
	}
	delete [] filename;
	in.close();
}
//==============================================================================
/** Reads the leaf from a regular file.
 */
inline void LabelLeafNode::readLeafBin(const char *path2models,long unsigned leafid,\
unsigned treeid){
	std::string dummy(path2models);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%sleaf%lu_%d.bin",path2models,leafid,treeid);
	std::ifstream in;
	in.open(filename,std::ios::in|std::ios::binary);
	if(in.is_open()){
		unsigned lSize;
		in.read(reinterpret_cast<char*>(&lSize),sizeof(unsigned));
		float labProb;
		in.read(reinterpret_cast<char*>(&labProb),sizeof(float));
		this->labelProb_ = labProb;
		this->vLabels_.resize(lSize);
		for(std::vector<unsigned>::iterator it=this->vLabels_.begin();\
		it!=this->vLabels_.end();++it){
			unsigned alab;
			in.read(reinterpret_cast<char*>(&alab),sizeof(unsigned));
			(*it) = alab;
		}
	}
	delete [] filename;
	in.close();
}
//==============================================================================
/** Writes the leaf info into an opened file.
 */
inline void LabelLeafNode::showLeafTxt(const char *path2models,long unsigned leafid,\
unsigned treeid){
	std::string dummy(path2models);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%sleaf%lu_%d.txt",path2models,leafid,treeid);
	std::ofstream out;
	out.open(filename,std::ios::out);
	if(out.is_open()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		out<<this->vLabels_.size()<<" "<<this->labelProb_<<" ";
		for(std::vector<unsigned>::iterator it=this->vLabels_.begin();\
		it!=this->vLabels_.end();++it){
			out<<(*it)<<" ";
		}
	}
	delete [] filename;
	out<<std::endl;
	out.close();
}
//==============================================================================
/** Writes the leaf info into an opened file.
 */
inline void LabelLeafNode::showLeafBin(const char *path2models,long unsigned leafid,\
unsigned treeid){
	std::string dummy(path2models);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%sleaf%lu_%d.bin",path2models,leafid,treeid);
	std::ofstream out;
	out.open(filename,std::ios::out|std::ios::binary);
	if(out.is_open()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		unsigned labsize = this->vLabels_.size();
		out.write(reinterpret_cast<const char*>(&labsize),sizeof(unsigned));
		float labprob    = this->labelProb_;
		out.write(reinterpret_cast<const char*>(&labprob),sizeof(float));
		for(std::vector<unsigned>::iterator it=this->vLabels_.begin();\
		it!=this->vLabels_.end();++it){
			unsigned alab = (*it);
			out.write(reinterpret_cast<const char*>(&alab),sizeof(unsigned));
		}
	}
	delete [] filename;
	out.close();
}
//==============================================================================
/** Shows a leaf nicely with colors.
 */
inline void LabelLeafNode::display(unsigned labW,unsigned labH,const std::map<cv::Vec3b,\
unsigned,vec3bCompare> &classinfo,const std::string &path2model) const{
	cv::Mat tmp(cv::Size(labW,labH),CV_8UC3);
	cv::MatIterator_<cv::Vec3b> iTmp = tmp.begin<cv::Vec3b>();
	for(std::vector<unsigned>::const_iterator it=this->vLabels_.begin();\
	it!=this->vLabels_.end(),iTmp!=tmp.end<cv::Vec3b>();++it,++iTmp){
		// find color for label
		cv::Vec3b b(0,0,0);
		for(std::map<cv::Vec3b,unsigned,vec3bCompare>::const_iterator m=\
		classinfo.begin();m!=classinfo.end();++m){
			if(m->second==(*it)){
				b=m->first;
				break;
			}
		}
		(*iTmp) = b;
	}
	std::vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(100);
	cv::imwrite(path2model,tmp,params);
	tmp.release();
}
//==============================================================================
//==============================================================================
//==============================================================================
template <class U>
StructuredTreeNode<U>::StructuredTreeNode(long unsigned nodeid,const U* leaf,\
const long double* test,unsigned nodeSize){
	this->nodeid_   = nodeid;
	this->nodeSize_ = nodeSize;
	if(leaf){
		this->leaf_ = new U(*leaf);
	}else{
		this->leaf_ = NULL;
	}
	this->test_ = new long double[nodeSize]();
	std::fill_n(this->test_,nodeSize,0);
	long double *ptT1       = this->test_;
	const long double *ptT2 = test;
	for(unsigned i=0;i<this->nodeSize_;++i,++ptT1,++ptT2){
		(*ptT1) = (*ptT2);
	}
	this->right_ = NULL;
	this->left_  = NULL;
};
//==============================================================================
/** Write the node to a text output stream.
 */
template <class U>
void StructuredTreeNode<U>::showNodeTxt(std::ofstream &out,const char* path2model,\
unsigned treeid){
	out.precision(std::numeric_limits<double>::digits10);
	out.precision(std::numeric_limits<float>::digits10);
	if(this->leaf_){
		out<<0<<" "; // leaf-node
	}else{
		out<<1<<" "; // test-node
	}
	out<<this->nodeSize_<<" ";
	out<<this->nodeid_<<" ";
	long double *ptT = this->test_;
	for(unsigned i=0;i<this->nodeSize_;++i,++ptT){
		out<<(*ptT)<<" ";
	}
	out<<std::endl;
	if(this->leaf_ && !this->leaf_->isempty()){
		this->leaf_->showLeafTxt(path2model,this->nodeid_,treeid);
	}
}
//==============================================================================
/** Write the node to a binary output stream.
 */
template <class U>
void StructuredTreeNode<U>::showNodeBin(std::ofstream &out,const char *path2model,\
unsigned treeid){
	out.precision(std::numeric_limits<double>::digits10);
	out.precision(std::numeric_limits<float>::digits10);
	if(this->leaf_){
		unsigned nodetype = 0;
		out.write(reinterpret_cast<const char*>(&nodetype),sizeof(unsigned));
	}else{
		unsigned nodetype = 1;
		out.write(reinterpret_cast<const char*>(&nodetype),sizeof(unsigned));
	}
	unsigned nodesize = this->nodeSize_;
	out.write(reinterpret_cast<const char*>(&nodesize),sizeof(unsigned));
	long unsigned nodeid = this->nodeid_;
	out.write(reinterpret_cast<const char*>(&nodeid),sizeof(long unsigned));
	long double *ptT = this->test_;
	for(unsigned i=0;i<this->nodeSize_;++i,++ptT){
		long double val = (*ptT);
		out.write(reinterpret_cast<const char*>(&val),sizeof(long double));
	}
	if(this->leaf_ && !this->leaf_->isempty()){
		this->leaf_->showLeafBin(path2model,this->nodeid_,treeid);
	}
}
//==============================================================================
/** clone to make a deep copy. For the shallow ones use default copy constructors.
 */
template <class U>
StructuredTreeNode<U>& StructuredTreeNode<U>::clone(StructuredTreeNode<U> const &rhs){
	this->nodeid(rhs.nodeid());
	this->leaf(rhs.leaf());
	this->test(rhs.test());
	this->nodeSize(rhs.nodeSize());
	this->left(rhs.left());
	this->right(rhs.right());
	return *this;
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif /* STRUCTUREDTREENODE_CPP_ */














