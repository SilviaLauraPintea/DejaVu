/* Evaluation.cpp
 * Author: Silvia-Laura Pintea
 */
#include "LabelEval.h"
//==============================================================================
LabelEval::LabelEval(const std::string &config){
	std::ifstream in(config.c_str());
	char *buffer         = new char[500];
	this->consideredCls_ = 0;
	if(in.is_open()){
		// std::string runName_;
		in.getline(buffer,500);in.getline(buffer,500);
		// std::string path2results_;
		in.getline(buffer,500);in.getline(buffer,500);
		this->path2predi_ = std::string(buffer);
		this->path2predi_ = Auxiliary<uchar,1>::trim(this->path2predi_);
		Auxiliary<uchar,1>::fixPath(this->path2predi_);
		this->path2mpredi_ = this->path2predi_+"mlabels/";
		this->path2predi_ += "labels/";
		// std::string path2labs_;
		in.getline(buffer,500);in.getline(buffer,500);
		this->path2gt_ = std::string(buffer);
		this->path2gt_ = Auxiliary<uchar,1>::trim(this->path2gt_);
		Auxiliary<uchar,1>::fixPath(this->path2gt_);
		// std::string path2train_;
		in.getline(buffer,500);in.getline(buffer,500);
		// std::string path2test_;
		in.getline(buffer,500);in.getline(buffer,500);
		// std::string path2model_;
		in.getline(buffer,500);in.getline(buffer,500);
		// std::string path2model_ motion model;
		in.getline(buffer,500);in.getline(buffer,500);
		// std::string path2feat_;
		in.getline(buffer,500);in.getline(buffer,500);
		// std::string path2feat_ motion features;
		in.getline(buffer,500);in.getline(buffer,500);
		// std::string ext_;
		in.getline(buffer,500);in.getline(buffer,500);
		this->ext_ = std::string(buffer);
		this->ext_ = Auxiliary<uchar,1>::trim(this->ext_);
		// std::string labTerm_;
		in.getline(buffer,500);in.getline(buffer,500);
		this->labTerm_ = std::string(buffer);
		this->labTerm_ = Auxiliary<uchar,1>::trim(this->labTerm_);
		// unsigned patchWidth_;unsigned patchHeigth_;
		in.getline(buffer,500);in.getline(buffer,500);
		// what for we need to get line so may times?
		in>>this->featW_;in>>this->featH_;
		// unsigned labWidth_;unsigned labHeigth_;
		in.getline(buffer,500);in.getline(buffer,500);
		in>>this->labW_;in>>this->labH_;
		// ignore this: unsigned motionWidth_;unsigned motionHeigth_;
		in.getline(buffer,500);in.getline(buffer,500);
		// unsigned noTrees_;
		in.getline(buffer,500);in.getline(buffer,500);
		// unsigned consdieredCls_;
		in.getline(buffer,500);in.getline(buffer,500);in>>this->consideredCls_;
		// std::vector<unsigned> pyrScales_;
		in.getline(buffer,500);in.getline(buffer,500);
		// std::map<cv::Scalar,unsigned> classInfo_;
		in.getline(buffer,500);in.getline(buffer,500);unsigned ssize;in>>ssize;
		for(unsigned s=0;s<ssize;++s){
			unsigned val1,val2,val3,val;
			in>>val1;in>>val2;in>>val3;
			cv::Vec3b tmp(val3,val2,val1); // we use BGR not RGB
			in>>this->classinfo_[tmp];
		}
		delete [] buffer;
	}else{
		delete [] buffer;
		std::cerr<<"File not found "<<config<<std::endl;
		std::exit(-1);
	}
	in.close();
}
//==============================================================================
LabelEval::~LabelEval(){
	this->classinfo_.clear();
	this->path2gt_.clear();
	this->path2predi_.clear();
	this->ext_.clear();
}
//==============================================================================
/** Gets the global scores: percentage of pixels correctly classified.
 */
float LabelEval::global(std::string &predi){
	// Load image names
	std::vector<std::string> vPredictions = Auxiliary<uchar,1>::listDir(predi,".bin");
	float totAgreement = 0.0;
	for(std::vector<std::string>::iterator iP=vPredictions.begin();iP!=\
	vPredictions.end();++iP){ // for each image
		std::string gtName   = (*iP).substr(2,(*iP).size()-6)+this->labTerm_+this->ext_;
		cv::Mat groundtruth  = cv::imread((std::string(this->path2gt_+gtName)).c_str(),1);
		std::string justname = (*iP).substr(0,iP->size()-4);
		cv::Mat prediction   = Auxiliary<uchar,1>::bin2mat((std::string\
			(predi+justname+".bin")).c_str());
		unsigned limitW      = std::max(this->labW_,this->featW_);
		unsigned limitH      = std::max(this->labH_,this->featH_);
		cv::Rect roi(limitW/2,limitH/2,groundtruth.cols-limitW,groundtruth.rows-limitH);
		cv::Mat ground       = groundtruth(roi).clone();
		groundtruth.release();
		if(ground.size()!=prediction.size()){
			std::cerr<<"[LabelEval::global]: the ground truth and the prediction "<<\
				"should have the same size"<<std::endl;
		}
		cv::Mat truth = this->color2label(ground);
		ground.release();
		// just percentage of pixels correctly classified
		unsigned agreement = 0;
		unsigned norm      = 0;
		cv::MatConstIterator_<uchar> mP = prediction.begin<uchar>();
		for(cv::MatConstIterator_<uchar> mG=truth.begin<uchar>();mG!=\
		truth.end<uchar>(),mP!=prediction.end<uchar>();++mG,++mP){
			if((*mG)<this->consideredCls_){
				++norm;
				if((*mG)==(*mP)){
					++agreement;
				}
			}
		}
		truth.release();
		prediction.release();
		totAgreement += static_cast<float>(agreement)/static_cast<float>(norm);
	}
	totAgreement /= static_cast<float>(vPredictions.size());
	std::cout<<"[LabelEval::global] percentage of correct pixels: "<<\
		totAgreement<<std::endl;
	return totAgreement;
}
//==============================================================================
/** Converts the input matrix to label IDs.
 */
cv::Mat LabelEval::color2label(const cv::Mat &mat){
	assert(mat.channels()==3);
	cv::Mat out = cv::Mat::zeros(mat.size(),CV_8UC1);
	for(unsigned y=0;y<mat.rows;++y){
		for(unsigned x=0;x<mat.cols;++x){
			out.at<uchar>(y,x) = this->classinfo_[mat.at<cv::Vec3b>(y,x)];
		}
	}
	return out;
}
//==============================================================================
/** Average recall of all classes: avg_cls TP[cls]/(TP[cls]+FN[cls]).
 */
float LabelEval::avgClass(std::string &predi){
	std::vector<std::string> vGroundTruth = Auxiliary<uchar,1>::listDir\
		(this->path2gt_,this->ext_,"_L");
	std::vector<std::string> vPredictions = Auxiliary<uchar,1>::listDir(predi,".bin");
	std::vector<float> truePos(this->consideredCls_,0.0);
	std::vector<float> allPos(this->consideredCls_,0.0);
	for(std::vector<std::string>::iterator iP = vPredictions.begin();iP!=\
	vPredictions.end();++iP){ // for each image
		std::string gtName   = (*iP).substr(2,(*iP).size()-6)+this->labTerm_+this->ext_;
		cv::Mat ground       = cv::imread((std::string(this->path2gt_+gtName)).c_str(),1);
		std::string justname = (*iP).substr(0,iP->size()-4);
		cv::Mat prediction   = Auxiliary<uchar,1>::bin2mat((std::string\
			(predi+justname+".bin")).c_str());
		unsigned limitW      = std::max(this->labW_,this->featW_);
		unsigned limitH      = std::max(this->labH_,this->featH_);
		cv::Rect roi(limitW/2,limitH/2,ground.cols-limitW,ground.rows-limitH);
		cv::Mat groundtruth  = ground(roi).clone();
		ground.release();
		if(groundtruth.size()!=prediction.size()){
			std::cerr<<"[LabelEval::global]: the ground truth and the prediction "<<\
				"should have the same size"<<std::endl;
		}
		cv::Mat truth = this->color2label(groundtruth);
		groundtruth.release();
		// for each class I want the TP and FP
		for(unsigned c=0;c<this->consideredCls_;++c){ // for each class
			cv::Mat maskGT,maskPredi;
			cv::inRange(truth,c,c,maskGT);
			maskGT.convertTo(maskGT,CV_32FC1);
			maskGT      /= 255.0;
			cv::inRange(prediction,c,c,maskPredi);
			maskPredi.convertTo(maskPredi,CV_32FC1);
			maskPredi   /= 255.0;
			cv::Mat tmp  = maskGT+maskPredi;
			maskPredi.release();
			// true positive --- intersection(maskGT,maskPredi)
			cv::Mat maskCorrect;
			cv::inRange(tmp,2,2,maskCorrect);
			maskCorrect.convertTo(maskCorrect,CV_32FC1);
			maskCorrect /= 255.0;
			tmp.release();
			cv::Scalar hereTruePos = cv::sum(maskCorrect);
			// true positive + false negatives --- all positives = maskGT
			cv::Scalar hereAllPos  = cv::sum(maskGT);
			maskGT.release();
			maskCorrect.release();
			truePos[c] += hereTruePos.val[0];
			allPos[c]  += hereAllPos.val[0];
		}
		truth.release();
		prediction.release();
	}
	// now get the average over classes:
	float avgClass                  = 0.0;
	std::vector<float>::iterator tp = truePos.begin();
	for(std::vector<float>::iterator ap=allPos.begin();ap!=allPos.end(),tp!=\
	truePos.end();++ap,++tp){
		if(*ap){
			avgClass += (*tp)/(*ap);
		}
	}
	avgClass /= static_cast<float>(truePos.size());
	std::cout<<"[LabelEval::avgClass] average class recall: "<<avgClass<<std::endl;
	return avgClass;
}
//==============================================================================
/** Average intersection vs. union: avg_cla TP[cls]/(TP[cls]+FN[cls]+FP[cls]).
 */
float LabelEval::avgPascal(std::string &predi){
	std::vector<std::string> vPredictions = Auxiliary<uchar,1>::listDir(predi,".bin");
	std::vector<float> truePos(this->consideredCls_,0.0);
	std::vector<float> allClass(this->consideredCls_,0.0);
	for(std::vector<std::string>::iterator iP = vPredictions.begin();iP!=\
	vPredictions.end();++iP){ // for each image
		std::string gtName   = (*iP).substr(2,(*iP).size()-6)+this->labTerm_+this->ext_;
		cv::Mat ground       = cv::imread((std::string(this->path2gt_+gtName)).c_str(),1);
		std::string justname = (*iP).substr(0,iP->size()-4);
		cv::Mat prediction   = Auxiliary<uchar,1>::bin2mat((std::string\
			(predi+justname+".bin")).c_str());
		unsigned limitW      = std::max(this->labW_,this->featW_);
		unsigned limitH      = std::max(this->labH_,this->featH_);
		cv::Rect roi(limitW/2,limitH/2,ground.cols-limitW,ground.rows-limitH);
		cv::Mat groundtruth  = ground(roi).clone();
		ground.release();
		if(groundtruth.size()!=prediction.size()){
			std::cerr<<"[LabelEval::global]: the ground truth and the prediction "<<\
				"should have the same size"<<std::endl;
		}
		cv::Mat truth = this->color2label(groundtruth);
		groundtruth.release();
		// for each class I want the TP and FP
		for(unsigned c=0;c<this->consideredCls_;++c){ // for each class
			cv::Mat maskGT,maskPredi;
			cv::inRange(truth,c,c,maskGT);
			maskGT.convertTo(maskGT,CV_32FC1);
			maskGT     /= 255.0;
			cv::inRange(prediction,c,c,maskPredi);
			maskPredi.convertTo(maskPredi,CV_32FC1);
			maskPredi  /= 255.0;
			cv::Mat tmp = maskGT+maskPredi;
			maskPredi.release();
			maskGT.release();
			// true positives --- intersection(maskGT, maskPredi)
			cv::Mat maskCorrect,maskClass;
			cv::inRange(tmp,2,2,maskCorrect);
			maskCorrect.convertTo(maskCorrect,CV_32FC1);
			maskCorrect /= 255.0;
			// all class --- maskGT + maskPredi - intersection(maskGT, maskPredi)
			cv::inRange(tmp,1,2,maskClass);
			maskClass.convertTo(maskClass,CV_32FC1);
			maskClass  /= 255.0;
			tmp.release();
			cv::Scalar hereTruePos  = cv::sum(maskCorrect);
			cv::Scalar hereAllClass = cv::sum(maskClass);
			maskCorrect.release();
			maskClass.release();
			truePos[c]  += hereTruePos.val[0];
			allClass[c] += hereAllClass.val[0];
		}
		truth.release();
		prediction.release();
	}
	// now get the average over classes:
	float avgPascal                 = 0.0;
	std::vector<float>::iterator tp = truePos.begin();
	for(std::vector<float>::iterator ac=allClass.begin();ac!=allClass.end(),tp!=\
	truePos.end();++ac,++tp){
		if(*ac){
			avgPascal += (*tp)/(*ac);
		}
	}
	avgPascal /= static_cast<float>(truePos.size());
	std::cout<<"[LabelEval::avgPascal] average pascal score: "<<avgPascal<<std::endl;
	return avgPascal;
}
//==============================================================================
/** Evaluates one or all of the above methods.
 */
void LabelEval::run(LabelEval::METHOD method){
	switch(method){
		case(LabelEval::GLOBAL):
			std::cout<<"Baseline global:"<<std::endl;
			this->global(this->path2predi_);
			std::cout<<"Motion global:"<<std::endl;
			this->global(this->path2mpredi_);
			break;
		case(LabelEval::AVG_CLASS):
			std::cout<<"Baseline avg-class:"<<std::endl;
			this->avgClass(this->path2predi_);
			std::cout<<"Motion avg-class:"<<std::endl;
			this->avgClass(this->path2mpredi_);
			break;
		case(LabelEval::AVG_PASCAL):
			std::cout<<"Baseline avg-pascal:"<<std::endl;
			this->avgPascal(this->path2predi_);
			std::cout<<"Motion avg-pascal:"<<std::endl;
			this->avgPascal(this->path2mpredi_);
			break;
		case(LabelEval::ALL):
			std::cout<<"Baseline global:"<<std::endl;
			this->global(this->path2predi_);
			std::cout<<"Motion global:"<<std::endl;
			this->global(this->path2mpredi_);
			std::cout<<std::endl;
			std::cout<<"Baseline avg-class:"<<std::endl;
			this->avgClass(this->path2predi_);
			std::cout<<"Motion avg-class:"<<std::endl;
			this->avgClass(this->path2mpredi_);
			std::cout<<std::endl;
			std::cout<<"Baseline avg-pascal:"<<std::endl;
			this->avgPascal(this->path2predi_);
			std::cout<<"Motion avg-pascal:"<<std::endl;
			this->avgPascal(this->path2mpredi_);
			break;
		default:
			std::cerr<<"[LaeblEval::run] option not implemented"<<std::endl;
			break;
	}
}
//==============================================================================
