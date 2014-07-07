/* MotionTreeNode.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONTREENODE_CPP_
#define MOTIONTREENODE_CPP_
#include "MotionTreeNode.h"
//==============================================================================
inline MotionLeafNode::MotionLeafNode(){
	this->labelProb_      = 0.0;
	this->vLabels_        = std::vector<unsigned>();
	this->vHistos_        = std::vector<cv::Mat>();
	this->vMotion_        = NULL;
	this->vAppearance_    = NULL;
	this->motionProb_     = 0.0;
	this->appearanceProb_ = 0.0;
	this->isempty_        = true;
}
//==============================================================================
inline MotionLeafNode::MotionLeafNode(const char *path2models,long unsigned leafid,\
unsigned treeid,bool binary){
	this->labelProb_      = 0.0;
	this->vLabels_        = std::vector<unsigned>();
	this->vHistos_        = std::vector<cv::Mat>();
	this->vMotion_        = NULL;
	this->vAppearance_    = NULL;
	this->motionProb_     = 0.0;
	this->appearanceProb_ = 0.0;
	this->isempty_        = true;
	if(binary){
		this->readLeafBin(path2models,leafid,treeid);
	}else{
		this->readLeafTxt(path2models,leafid,treeid);
	}
	if(this->vMotion_ || this->vAppearance_){
		this->isempty_ = false;
	}
}
//==============================================================================
/** Reads the leaf from a binary file.
 */
inline void MotionLeafNode::readLeafBin(const char *path2models,long unsigned leafid,\
unsigned treeid){
	std::string dummy(path2models);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%sleaf%lu_%d.bin",path2models,leafid,treeid);
	std::ifstream in;
	in.open(filename,std::ios::in|std::ios::binary);
	if(in.good()){
		// [1] Read the motion-patch
		unsigned mSize;
		in.read(reinterpret_cast<char*>(&mSize),sizeof(unsigned));
		float motionProb;
		in.read(reinterpret_cast<char*>(&motionProb),sizeof(float));
		this->motionProb_ = motionProb;
		this->vMotion_    = new cv::Mat(1,mSize,CV_32FC1);
		for(unsigned i=0;i<mSize;++i){
			float val;
			in.read(reinterpret_cast<char*>(&val),sizeof(float));
			this->vMotion_->at<float>(0,i) = val;
		}
		unsigned aSizeX,aSizeY;
		in.read(reinterpret_cast<char*>(&aSizeX),sizeof(unsigned));
		in.read(reinterpret_cast<char*>(&aSizeY),sizeof(unsigned));
		float appearanceProb;
		in.read(reinterpret_cast<char*>(&appearanceProb),sizeof(float));
		this->appearanceProb_ = appearanceProb;
		this->vAppearance_    = new cv::Mat(cv::Size(aSizeX,aSizeY),CV_8UC3);
		for(unsigned y=0;y<aSizeY;++y){
			for(unsigned x=0;x<aSizeX;++x){
				unsigned val0,val1,val2;
				in.read(reinterpret_cast<char*>(&val0),sizeof(unsigned));
				in.read(reinterpret_cast<char*>(&val1),sizeof(unsigned));
				in.read(reinterpret_cast<char*>(&val2),sizeof(unsigned));
				cv::Vec3b vals = cv::Vec3b(val0,val1,val2);
				this->vAppearance_->at<cv::Vec3b>(y,x) = vals;
			}
		}
		// [2] Write the number of channels and the size of the patch
		unsigned channels=0;
		in.read(reinterpret_cast<char*>(&channels),sizeof(unsigned));
		for(unsigned c=0;c<channels;++c){
			cv::Mat tmp = cv::Mat::zeros(aSizeY,aSizeX,CV_32FC1);
			for(unsigned y=0;y<aSizeY;++y){
				for(unsigned x=0;x<aSizeX;++x){
					float val;
					in.read(reinterpret_cast<char*>(&val),sizeof(float));
					tmp.at<float>(y,x) = val;
				}
			}
			this->vHistos_.push_back(tmp.clone());
			tmp.release();
		}
		in.close();
	}
	delete [] filename;
}
//==============================================================================
/** Reads the leaf from a regular file.
 */
inline void MotionLeafNode::readLeafTxt(const char *path2models,long unsigned leafid,\
unsigned treeid){
	std::string dummy(path2models);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%sleaf%lu_%d.txt",path2models,leafid,treeid);
	std::ifstream in(filename);
	if(in.good()){
		// [1] Read the motion-patch
		unsigned mSize;in>>mSize;
		in>>this->motionProb_;
		this->vMotion_ = new cv::Mat(1,mSize,CV_32FC1);
		for(unsigned i=0;i<mSize;++i){
			float val;in>>val;
			this->vMotion_->at<float>(0,i) = val;
		}
		unsigned aSizeX,aSizeY;
		in>>aSizeX;in>>aSizeY;in>>this->appearanceProb_;
		this->vAppearance_ = new cv::Mat(cv::Size(aSizeX,aSizeY),CV_8UC3);
		for(unsigned y=0;y<aSizeY;++y){
			for(unsigned x=0;x<aSizeX;++x){
				unsigned val0,val1,val2;
				in>>val0;in>>val1;in>>val2;
				cv::Vec3b vals = cv::Vec3b(val0,val1,val2);
				this->vAppearance_->at<cv::Vec3b>(y,x) = vals;
			}
		}
		// [2] Write the number of channels and the size of the patch
		unsigned channels=0; in>>channels;
		for(unsigned c=0;c<channels;++c){
			cv::Mat tmp = cv::Mat::zeros(aSizeY,aSizeX,CV_32FC1);
			for(unsigned y=0;y<aSizeY;++y){
				for(unsigned x=0;x<aSizeX;++x){
					float val; in>>val;
					tmp.at<float>(y,x) = val;
				}
			}
			this->vHistos_.push_back(tmp.clone());
			tmp.release();
		}
		in.close();
	}
	delete [] filename;
}
//==============================================================================
/** Writes the leaf info into an opened file.
 */
inline void MotionLeafNode::showLeafTxt(const char *path2models,long unsigned leafid,\
unsigned treeid){
	std::string dummy(path2models);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%sleaf%lu_%d.txt",path2models,leafid,treeid);
	std::ofstream out;
	out.open(filename,std::ios::out);
	if(out.good()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		// outputs the motion-patch
		out<<this->vMotion_->cols<<" "<<(std::isinf(this->motionProb_)|| \
			std::isnan(this->motionProb_)?-std::numeric_limits<float>::max():\
			this->motionProb_)<<" ";
		for(unsigned i=0;i<this->vMotion_->cols;++i){
			out<<this->vMotion_->at<float>(0,i)<<" ";
		}
		out<<std::endl;
		// outputs the appearance-patch
		out<<this->vAppearance_->cols<<" "<<this->vAppearance_->rows<<" "<<\
			(std::isinf(this->appearanceProb_)|| \
			std::isnan(this->appearanceProb_)?-std::numeric_limits<float>::max():\
			this->appearanceProb_)<<" ";
		for(unsigned y=0;y<this->vAppearance_->cols;++y){
			for(unsigned x=0;x<this->vAppearance_->rows;++x){
				cv::Vec3b vals = this->vAppearance_->at<cv::Vec3b>(y,x);
				out<<static_cast<unsigned>(vals.val[0])<<" "<<\
					static_cast<unsigned>(vals.val[1])<<" "<<\
					static_cast<unsigned>(vals.val[2])<<" ";
			}
		}
		out<<std::endl;
		// Write the number of channels and the size of the patch
		if(this->vHistos_.size()){
			out<<this->vHistos_.size()<<" ";
			for(unsigned c=0;c<this->vHistos_.size();++c){
				for(unsigned y=0;y<this->vHistos_[c].rows;++y){
					for(unsigned x=0;x<this->vHistos_[c].cols;++x){
						float value = this->vHistos_[c].at<float>(y,x);
						out<<value<<" ";
					}
				}
			}
			out<<std::endl;
		}
		out.close();
	}
	delete [] filename;
}
//==============================================================================
/** Writes the leaf info into an opened binary file.
 */
inline void MotionLeafNode::showLeafBin(const char *path2models,long unsigned leafid,\
unsigned treeid){
	std::string dummy(path2models);
	char *filename = new char[dummy.size()+100]();
	sprintf(filename,"%sleaf%lu_%d.bin",path2models,leafid,treeid);
	std::ofstream out;
	out.open(filename,std::ios::out|std::ios::binary);
	if(out.good()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		// outputs the motion-patch
		unsigned motioncols = this->vMotion_->cols;
		out.write(reinterpret_cast<const char*>(&motioncols),sizeof(unsigned));
		float motionprob    = (std::isinf(this->motionProb_)|| \
			std::isnan(this->motionProb_)?-std::numeric_limits<float>::max():\
			this->motionProb_);
		out.write(reinterpret_cast<const char*>(&motionprob),sizeof(float));
		for(unsigned i=0;i<this->vMotion_->cols;++i){
			float val = this->vMotion_->at<float>(0,i);
			out.write(reinterpret_cast<const char*>(&val),sizeof(float));
		}
		// outputs the appearance-patch
		unsigned appearcol = this->vAppearance_->cols;
		unsigned appearrow = this->vAppearance_->rows;
		out.write(reinterpret_cast<const char*>(&appearcol),sizeof(unsigned));
		out.write(reinterpret_cast<const char*>(&appearrow),sizeof(unsigned));
		float appearprob  = (std::isinf(this->appearanceProb_)|| \
			std::isnan(this->appearanceProb_)?-std::numeric_limits<float>::max():\
			this->appearanceProb_);
		out.write(reinterpret_cast<const char*>(&appearprob),sizeof(float));
		for(unsigned y=0;y<this->vAppearance_->cols;++y){
			for(unsigned x=0;x<this->vAppearance_->rows;++x){
				cv::Vec3b vals = this->vAppearance_->at<cv::Vec3b>(y,x);
				unsigned val0 = vals.val[0];
				unsigned val1 = vals.val[1];
				unsigned val2 = vals.val[2];
				out.write(reinterpret_cast<const char*>(&val0),sizeof(unsigned));
				out.write(reinterpret_cast<const char*>(&val1),sizeof(unsigned));
				out.write(reinterpret_cast<const char*>(&val2),sizeof(unsigned));
			}
		}
		// Write the number of channels and the size of the patch
		if(this->vHistos_.size()){
			unsigned channels = this->vHistos_.size();
			out.write(reinterpret_cast<const char*>(&channels),sizeof(unsigned));
			for(unsigned c=0;c<this->vHistos_.size();++c){
				for(unsigned y=0;y<this->vHistos_[c].rows;++y){
					for(unsigned x=0;x<this->vHistos_[c].cols;++x){
						float value = this->vHistos_[c].at<float>(y,x);
						out.write(reinterpret_cast<const char*>(&value),\
							sizeof(float));
					}
				}
			}
		}
		out.close();
	}
	delete [] filename;
}
//==============================================================================
//==============================================================================
//==============================================================================
template <class U,class T>
MotionTreeNode<U,T>::MotionTreeNode(long unsigned nodeid,const U* leaf,\
const long double* test,unsigned nodeSize,const std::vector<cv::Mat> &nodefreq,\
const std::vector<cv::Mat> &freqA,const std::vector<cv::Mat> &freqB,const typename \
std::vector<std::vector<const T*> > &setA,const typename std::vector<std::vector\
<const T*> > &setB){
	this->nodeid_   = nodeid;
	this->nodeSize_ = nodeSize;
	if(leaf){
		this->leaf_ = new U(*leaf);
	}else{
		this->leaf_ = NULL;
	}
	if(test){
		this->test_ = new long double[nodeSize]();
		std::fill_n(this->test_,nodeSize,0);
		long double *ptT1       = this->test_;
		const long double *ptT2 = test;
		for(unsigned i=0;i<this->nodeSize_;++i,++ptT1,++ptT2){
			(*ptT1) = (*ptT2);
		}
	}
	this->nodefreq(nodefreq);
	this->freqA(freqA);
	this->freqB(freqB);
	this->setA(setA);
	this->setB(setB);
	this->right_ = NULL;
	this->left_  = NULL;
}
//==============================================================================
/** Use default Copy and assignment constructors.
 */
template <class U,class T>
MotionTreeNode<U,T>& MotionTreeNode<U,T>::clone(MotionTreeNode<U,T> \
const &rhs){
	this->nodeid(rhs.nodeid());
	this->leaf(rhs.leaf());
	this->test(rhs.test());
	this->nodeSize(rhs.nodeSize());
	this->left(rhs.left());
	this->right(rhs.right());
	this->freqA(rhs.freqB());
	this->freqA(rhs.freqB());
	this->nodefreq(rhs.nodefreq());
	this->setA(rhs.setA());
	this->setB(rhs.setB());
	return *this;
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif /* MOTIONTREENODE_CPP_ */























