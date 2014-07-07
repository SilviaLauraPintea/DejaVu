/* RunRF.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef RUNRF_CPP_
#define RUNRF_CPP_
#include <stdexcept>
#include <time.h>
#include "RunRF.h"
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
RunRF<L,M,T,F,N,U>::RunRF(const char* config){
	std::ifstream in(config);
	unsigned charsize = 1000;
	char *buffer      = new char[charsize]();
	if(in.is_open()){
		// [0] Run name: std::string runName_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->runName_ = std::string(buffer);
		this->runName_ = Auxiliary<uchar,1>::trim(this->runName_);
		// [1] Path to results: std::string path2results_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2results_ = std::string(buffer);
		this->path2results_ = Auxiliary<uchar,1>::trim(this->path2results_);
		Auxiliary<uchar,1>::fixPath(this->path2results_);
		Auxiliary<uchar,1>::file_exists(this->path2results_.c_str(),true);
		// [2] Path to labels: std::string path2labs_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2labs_ = std::string(buffer);
		this->path2labs_ = Auxiliary<uchar,1>::trim(this->path2labs_);
		Auxiliary<uchar,1>::fixPath(this->path2labs_);
		// [3] Path to train: std::string path2train_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2train_ = std::string(buffer);
		this->path2train_ = Auxiliary<uchar,1>::trim(this->path2train_);
		Auxiliary<uchar,1>::fixPath(this->path2train_);
		// [4] Path 2 test: std::string path2test_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2test_ = std::string(buffer);
		this->path2test_ = Auxiliary<uchar,1>::trim(this->path2test_);
		Auxiliary<uchar,1>::fixPath(this->path2test_);
		// [5] Path 2 baseline models: std::string path2model_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2model_ = std::string(buffer);
		this->path2model_ = Auxiliary<uchar,1>::trim(this->path2model_);
		Auxiliary<uchar,1>::fixPath(this->path2model_);
		Auxiliary<uchar,1>::file_exists(this->path2model_.c_str(),true);
		// [7] Path to baseline features: std::string path2feat_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2feat_ = std::string(buffer);
		this->path2feat_ = Auxiliary<uchar,1>::trim(this->path2feat_);
		Auxiliary<uchar,1>::fixPath(this->path2feat_);
		Auxiliary<uchar,1>::file_exists(this->path2feat_.c_str(),true);
		// [9] Extension for the images: std::string ext_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->ext_ = std::string(buffer);
		this->ext_ = Auxiliary<uchar,1>::trim(this->ext_);
		// [10] Label extension (at the end of the image naming if any)
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->labTerm_ = std::string(buffer);
		this->labTerm_ = Auxiliary<uchar,1>::trim(this->labTerm_);
		// [11] Feature patch size: unsigned patchWidth_;unsigned patchHeigth_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		in>>this->patchWidth_;in>>this->patchHeight_;
		// [12] Label patch size: unsigned labWidth_;unsigned labHeigth_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		in>>this->labWidth_;in>>this->labHeight_;
		// unsigned noTrees_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->noTrees_;
		// [13] Considered classes: unsigned consideredCls_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->consideredCls_;
		// [14] Pyramid scales for prediction: std::vector<unsigned> pyrScales_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);unsigned ssize;in>>ssize;
		for(unsigned s=0;s<ssize;++s){
			float val;in>>val;
			if(s==0 && val!=1){
				std::cerr<<"[RunRF::RunRF] first level of the pyramid "<<\
					"should be 1"<<std::endl;
				throw std::exception();
			}
			this->pyrScales_.push_back(val);
		}
		// [15] Mapping from colors to labels: std::map<cv::Scalar,unsigned> classInfo_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>ssize;
		for(unsigned s=0;s<ssize;++s){
			unsigned val1,val2,val3,val;
			in>>val1;in>>val2;in>>val3;in>>val;
			cv::Vec3b tmp(val3,val2,val1); // we use BGR not RGB
			this->classInfo_[tmp] = val;
		}
		// [16] If we want to balance the samples per class: bool balance_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->balance_;
		// [17] Number of training images to use: unsigned trainSize_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->trainSize_;
		// [18] Total number of patches to use: unsigned noPatches_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->noPatches_;
		// [19] Number of iterations per node: unsigned iterPerNode_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->iterPerNode_;
		// [20] Entropy type: typename StructuredTree<M,T,F,U>::ENTROPY entropy_;
		unsigned dummy;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>dummy;
		this->entropy_ = static_cast<typename StructuredTree<M,T,F,N,U>::\
			ENTROPY>(dummy);
		// [22] Prediction method: typename StructuredRFdetector::METHOD predMethod_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>dummy;
		this->predMethod_ = static_cast<Puzzle<PuzzlePatch>::METHOD>(dummy);
		// [24] Step for sampling patches on grid: unsigned step_
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->step_;
		// [24] Step for sampling patches on grid: unsigned step_
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->binary_;
		delete [] buffer;
		// [27] The feature patch should be the largest at all times
		assert(this->labWidth_<=this->patchWidth_);
		assert(this->labHeight_<=this->patchHeight_);
	}else{
		delete [] buffer;
		std::cerr<<"File not found "<<config<<std::endl;
		std::exit(-1);
	}
	in.close();
}
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
RunRF<L,M,T,F,N,U>::~RunRF() {
	this->ext_.clear();
	this->path2train_.clear();
	this->path2labs_.clear();
	this->path2test_.clear();
	this->path2results_.clear();
	this->path2model_.clear();
	this->pyrScales_.clear();
	this->path2feat_.clear();
	this->classInfo_.clear();
}
//==============================================================================
/** Performs the RF detection on test images.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void RunRF<L,M,T,F,N,U>::detect(StructuredRFdetector<L,M,T,F,N,U> &crDetect){
	// [0] Load image names
	std::vector<std::string> vFilenames = Auxiliary<uchar,1>::listDir\
		(this->path2test_,this->ext_);
	// [1] Loop over all images
	for(unsigned int i=0;i<vFilenames.size();++i){
		// [2] Allocate space for output over pyramid
		std::vector<cv::Mat> vImgDetect(this->pyrScales_.size());
		std::string justname = vFilenames[i].substr(0,vFilenames[i].size()-4);
		// [3] Perform detection for all scales
		clock_t begin = clock();
		crDetect.detectPyramid(justname,this->path2test_,this->path2feat_,\
			this->ext_,this->pyrScales_,vImgDetect);
		clock_t end = clock();
		std::cout<<"Prediction 1 img time elapsed: "<<double(Auxiliary\
			<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
		// [4] Loop over all scales and store result
		for(unsigned int k=0;k<vImgDetect.size();++k){ // over scales
			// [5] Create the directories to store outputs in
			Auxiliary<uchar,1>::file_exists((this->path2results_+"labels/").c_str(),true);
			Auxiliary<uchar,1>::file_exists((this->path2results_+"color/").c_str(),true);
			// [6] Make the color image and save it.
			cv::Mat color = RunRF<L,M,T,F,N,U>::getColorLabels(vImgDetect[k],\
				this->classInfo_);
			// [7] Get the names of the outputs to save
			std::string tmpOut   = (this->path2results_+"labels/"+Auxiliary\
				<int,1>::number2string(k)+"_"+justname+".bin").c_str();
			std::string colorOut = (this->path2results_+"color/"+Auxiliary\
				<int,1>::number2string(k)+"_"+vFilenames[i]).c_str();
			// [8] Define parameters for the predicted label-images
			std::vector<int> params;
			if(strcmp(this->ext_.c_str(),".jpg")){
				params.push_back(CV_IMWRITE_JPEG_QUALITY);
				params.push_back(100);
				cv::imwrite(colorOut,color,params);
			}else if(strcmp(this->ext_.c_str(),".png")){
				params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				params.push_back(4);
				cv::imwrite(colorOut,color,params);
			}else{
				cv::imwrite(colorOut,color);
			}
			color.release();
			// [9] For the actual labels we use binary files:
			Auxiliary<uchar,1>::mat2bin(vImgDetect[k],tmpOut.c_str(),false);
			vImgDetect[k].release();
		}
		vImgDetect.clear();
	}
}
//==============================================================================
/** Gets the color labels for the image.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
cv::Mat RunRF<L,M,T,F,N,U>::getColorLabels(const cv::Mat &output,const \
std::map<cv::Vec3b,unsigned,vec3bCompare> &classinfo){
	// [0] Convert label Id values to color RGB values
	cv::Mat color = cv::Mat::zeros(output.size(),CV_8UC3);
	cv::MatIterator_<cv::Vec3b> it2 = color.begin<cv::Vec3b>();
	for(cv::MatConstIterator_<uchar> it1=output.begin<uchar>();\
	it1!=output.end<uchar>(),it2!=color.end<cv::Vec3b>();++it1,++it2){
		// [1] Find color for current label
		cv::Vec3b b(0,0,0);
		for(std::map<cv::Vec3b,unsigned,vec3bCompare>::const_iterator m=\
		classinfo.begin();m!=classinfo.end();++m){
			if(m->second==(*it1)){
				b=m->first;
				break;
			}
		}
		// [2] Save the color at this position
		(*it2) = b;
	}
	return color;
}
//==============================================================================
/** Initialize and start detector on test set.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void RunRF<L,M,T,F,N,U>::runDetect(){
	// [0] Initialize forest with number of trees
	StructuredRF<L,M,T,F,N,U> forest(this->noTrees_);
	// [1] Load forest
	forest.loadForest(this->path2model_,this->binary_);
	// [2] Initialize detector
	StructuredRFdetector<L,M,T,F,N,U> detect(&forest,this->patchWidth_,\
		this->patchHeight_,this->classInfo_.size(),this->labWidth_,\
		this->labHeight_,this->predMethod_,this->step_);
	// [3] Create directory for output
	Auxiliary<uchar,1>::file_exists(this->path2results_.c_str(),true);
	// [4] Run detector over all images
	this->detect(detect);
}
//==============================================================================
/** Initialize and start training.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void RunRF<L,M,T,F,N,U>::runTrain() {
	// [0] Initialize forest with number of trees
	StructuredRF<L,M,T,F,N,U> forest(this->noTrees_);
	// [1] Create directory for storing the trees
	Auxiliary<uchar,1>::file_exists(this->path2model_.c_str(),true);
	std::vector<std::string> files = Auxiliary<uchar,1>::listDir(this->path2train_,\
		this->ext_);
	// [2] Train each tree on a different subset if needed
	#if DO_PRAGMA
		#pragma omp parallel for schedule(dynamic,1)
	#endif
	// [3] Loop over all trees
	for(unsigned t=0;t<this->noTrees_;++t){
		// [4] Initialize random number generator
		time_t times = time(NULL);
		int seed     = (int)times+t;
		CvRNG cvRNG(seed);
		// [5] Initialize training data
		M train(&cvRNG,this->patchWidth_,this->patchHeight_,this->classInfo_.\
			size(),this->labWidth_,this->labHeight_,this->trainSize_,\
			this->noPatches_,this->consideredCls_,this->balance_,this->step_);
		// [6] Extract training patches and features
		std::cout<<"[RunRF::run_train]: Extracting patches for tree "<<t<<".."<<std::endl;
		train.extractPatches(this->path2train_,this->path2labs_,this->path2feat_,\
			files,this->classInfo_,this->labTerm_,this->ext_);
		// [7] Train forest tree on images
		std::cout<<"[RunRF::run_train] Forest training ..."<<std::endl;
		clock_t begin = clock();
		forest.trainForestTree(5,100,&cvRNG,train,this->iterPerNode_,t,\
			this->path2model_.c_str(),this->runName_,this->entropy_,\
			this->consideredCls_,this->binary_);
		clock_t end   = clock();
		std::cout<<"Train forest time elapsed: "<<double\
			(Auxiliary<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
		// [8] Save forest trees in separate files
		std::cout<<"[RunRF::run_train] Saving forest ..."<<std::endl;
		forest.saveTree(this->path2model_.c_str(),t);
	}
}
//==============================================================================
/** Initialize and start training.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void RunRF<L,M,T,F,N,U>::runExtract() {
	Auxiliary<uchar,1>::file_exists(this->path2model_.c_str(),true);
	std::vector<std::string> trainFiles = Auxiliary<uchar,1>::listDir\
		(this->path2train_,this->ext_);
	std::vector<std::string> testFiles = Auxiliary<uchar,1>::listDir\
		(this->path2test_,this->ext_);
	// [0] Initialize random number generator
	time_t times = time(NULL);
	int seed     = (int)times;
	CvRNG cvRNG(seed);
	// [1] Initialize training data
	M train(&cvRNG,this->patchWidth_,this->patchHeight_,this->classInfo_.\
		size(),this->labWidth_,this->labHeight_,1e+5,0,this->consideredCls_,0,\
		this->step_);
	// [2] Extract training/test patches and features
	std::cout<<"[RunRF::extract]: Extracting patches for training"<<std::endl;
	train.extractPatches(this->path2train_,this->path2labs_,this->path2feat_,\
		trainFiles,this->classInfo_,this->labTerm_,this->ext_);
	std::cout<<"[RunRF::extract]: Extracting patches for test"<<std::endl;
	train.trainingSize(1e+5);
	train.extractPatches(this->path2train_,this->path2labs_,this->path2feat_,\
		testFiles,this->classInfo_,this->labTerm_,this->ext_);
}
//==============================================================================
/** Initialize and start training.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
void RunRF<L,M,T,F,N,U>::run(RunRF::MODE mode){
	switch(mode){
		case RunRF<L,M,T,F,N,U>::TRAIN_RF:
			// train forest
			this->runTrain();
		break;
		case RunRF<L,M,T,F,N,U>::TEST_RF:
			// test forest
			this->runDetect();
			break;
		case RunRF<L,M,T,F,N,U>::TRAIN_TEST_RF:
			// train forest
			this->runTrain();
			this->runDetect();
			break;
		case RunRF<L,M,T,F,N,U>::EXTRACT:
			// test forest
			this->runExtract();
			break;
		default:
			std::cerr<<"[RunRF::run] option not implemented"<<std::endl;
			break;
	}
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif // RUNRF_CPP_







































