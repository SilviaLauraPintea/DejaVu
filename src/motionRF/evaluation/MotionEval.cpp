/* MotionEval.cpp
 * Author: Silvia-Laura Pintea
 */
#include "MotionEval.h"
#include <MotionPatch.h>
typedef MotionPatch<MotionPatchFeature<FeaturesMotion>,\
	FeaturesMotion> MotionPatchClass;
//==============================================================================
MotionEval::MotionEval(const std::string &config){
	std::ifstream in(config.c_str());
	unsigned buffersz = 1000;
	char *buffer      = new char[buffersz];
	if(in.is_open()){
		// [0] Core type to be ignored
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		// [1] std::string runName_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		// [2] std::string path2results_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		this->path2predi_ = std::string(buffer);
		this->path2predi_ = Auxiliary<uchar,1>::trim(this->path2predi_);
		Auxiliary<uchar,1>::fixPath(this->path2predi_);
		// [3] std::string path2train_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		// [4] std::string path2test_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		this->path2test_ = std::string(buffer);
		this->path2test_ = Auxiliary<uchar,1>::trim(this->path2test_);
		Auxiliary<uchar,1>::fixPath(this->path2predi_);
		// [5] std::string path2model_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		// [6] std::string path2feat_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		this->path2gt_ = std::string(buffer);
		this->path2gt_ = Auxiliary<uchar,1>::trim(this->path2gt_);
		Auxiliary<uchar,1>::fixPath(this->path2gt_);
		// [7] std::string ext_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		this->ext_ = "."+std::string(buffer);
		// [8] unsigned patchWidth_;unsigned patchHeigth_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		in>>this->featW_;in>>this->featH_;
		// [9] unsigned motionWidth_;unsigned motionHeigth_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		in>>this->motionW_;in>>this->motionH_;
		// [10] unsigned noTrees_;
		in.getline(buffer,buffersz);in.getline(buffer,buffersz);
		// that's all we needed from here
		delete [] buffer;
	}else{
		delete [] buffer;
		std::cerr<<"File not found "<<config<<std::endl;
		std::exit(-1);
	}
	in.close();
}
//==============================================================================
MotionEval::~MotionEval(){
	this->path2predi_.clear();
	this->path2gt_.clear();
}
//==============================================================================
/** Load only the motion ground truth from the features.
 */
cv::Mat MotionEval::loadGTfromFeat(const std::string &path2feat){
	cv::Mat of;
	if(!Auxiliary<uchar,1>::file_exists(path2feat.c_str())){
		std::cerr<<"[MotionEval::loadGTfromFeat]: Error opening the file: "<<\
			path2feat<<std::endl;
		std::exception e;
		throw(e);
	}
	std::ifstream pFile;
	pFile.open(path2feat.c_str(),std::ios::in | std::ios::binary);
	if(pFile.is_open()){
		pFile.seekg (0,std::ios::beg);
		// [0] Read the features matrices (one per channel)
		unsigned vsize;
		pFile.read(reinterpret_cast<char*>(&vsize),sizeof(unsigned));
		if(vsize>0){
			for(unsigned s=0;s<vsize;++s){
				int width,height,nChannels,depth;
				pFile.read(reinterpret_cast<char*>(&nChannels),sizeof(int));
				pFile.read(reinterpret_cast<char*>(&depth),sizeof(int));
				pFile.read(reinterpret_cast<char*>(&width),sizeof(int));
				pFile.read(reinterpret_cast<char*>(&height),sizeof(int));
				IplImage* im = cvCreateImage(cvSize(width,height),depth,nChannels);
				for(int y=0;y<height;++y){
					for(int x=0;x<width;++x){
						uchar val;
						pFile.read(reinterpret_cast<char*>(&val),sizeof(uchar));
					}
				}
			}
		}
		// [2] Read the OF matrices
		int cols, rows;
		pFile.read(reinterpret_cast<char*>(&rows),sizeof(int));
		pFile.read(reinterpret_cast<char*>(&cols),sizeof(int));
		cv::Mat tmpof = cv::Mat(cv::Size(cols,rows),CV_32FC2);
		// [3] Loop over the cols and rows
		for(int y=0;y<rows;++y){
			for(int x=0;x<cols;++x){
				float valX,valY;
				pFile.read(reinterpret_cast<char*>(&valX),sizeof(float));
				pFile.read(reinterpret_cast<char*>(&valY),sizeof(float));
				tmpof.at<cv::Vec2f>(y,x) = cv::Vec2f(valX,valY);
			}
		}
		cv::Rect roi(this->featW_/2,this->featH_/2,tmpof.cols-(this->featW_),\
			tmpof.rows-(this->featH_));
		of = tmpof(roi).clone();
		tmpof.release();
		pFile.close();
	}
	return of;
}
//==============================================================================
/** Gets the global motion error: mean-euclidian distance between endpoints.
 */
float MotionEval::meanEPE(bool fromfeat,bool display){
	// Load image names
	std::vector<std::string> dirs = Auxiliary<uchar,1>::listDir(this->path2predi_);
	float overall                 = 0.0;
	float overall0                = 0.0;
	float canny                   = 0.0;
	float canny0                  = 0.0;
	float coscanny                = 0.0;
	float coscanny0               = 0.0;
	float abscoscanny             = 0.0;
	float abscoscanny0            = 0.0;
	unsigned processed            = 0;
	for(std::vector<std::string>::iterator d=dirs.begin();d!=dirs.end();++d){
		std::string adir = this->path2predi_+(*d)+PATH_SEP+std::string("flow_motion");
		std::vector<std::string> vPredictions;
		if(Auxiliary<char,1>::file_exists(adir.c_str())){
			vPredictions = Auxiliary<uchar,1>::listDir(adir,".bin");
		}
		if(vPredictions.empty()){continue;}
		for(std::vector<std::string>::iterator iP=vPredictions.begin();iP!=\
		vPredictions.end()-1;++iP){ // for each image
			std::string gtName = (*iP).substr(2,(*iP).size()-2);
			std::string imName = this->path2test_+(*d)+PATH_SEP+gtName.substr\
				(0,gtName.size()-4)+this->ext_;
			cv::Mat groundtruth,prediction;
			try{
				if(fromfeat){
					groundtruth = this->loadGTfromFeat(this->path2gt_+(*d)+\
						PATH_SEP+gtName);
				}else{
					std::cout<<"Loading: "<<(this->path2gt_+(*d)+PATH_SEP+\
						"flow_motion"+PATH_SEP+gtName).c_str()<<std::endl;
					groundtruth = Auxiliary<float,2>::bin2mat((this->path2gt_+\
						(*d)+PATH_SEP+"flow_motion"+PATH_SEP+gtName).c_str());
				}
				std::cout<<"Prediction: "<<std::string(adir+(*iP)).c_str()<<std::endl;

				prediction  = Auxiliary<float,2>::bin2mat(std::string\
					(adir+(*iP)).c_str());
			}catch(std::exception &e){
				std::cerr<<"[MotionEval::meanEPE]: Image "<<this->path2gt_+gtName<<\
					" not processed"<<std::endl;
				continue;
			}
			cv::resize(groundtruth,groundtruth,prediction.size());
			cv::Mat image = cv::imread(imName,1);
			cv::resize(image,image,prediction.size());
			// [2] Get the cCanny edges:
			cv::Mat gray, edge;
			cv::cvtColor(image,gray,CV_BGR2GRAY);
			cv::blur(gray,gray,cv::Size(3,3));
			cv::Canny(gray,edge,50,150,3);
			gray.release();
			// [3] Now loop over prediction and GT and get the EPEs
			cv::Mat zeropred = cv::Mat::zeros(prediction.size(),CV_32FC2);
			// [4] just percentage of pixels correctly classified
			float tmpoverall           = 0.0;
			float tmpoverall0          = 0.0;
			float tmpcanny             = 0.0;
			float tmpcanny0            = 0.0;
			float tmpcoscanny          = 0.0;
			float tmpcoscanny0         = 0.0;
			float tmpabscoscanny       = 0.0;
			float tmpabscoscanny0      = 0.0;
			float small                = 1.0e-10;
			unsigned fgelem            = 0;
			unsigned bgelem            = 0;
			unsigned cannyelem         = 0;
			cv::MatConstIterator_<cv::Vec2f> zP = zeropred.begin<cv::Vec2f>();
			cv::MatConstIterator_<cv::Vec2f> mG = groundtruth.begin<cv::Vec2f>();
			cv::Mat_<uchar>::iterator ed        = edge.begin<uchar>();
			for(cv::MatConstIterator_<cv::Vec2f> mP=prediction.begin<cv::Vec2f>();\
			mP!=prediction.end<cv::Vec2f>(),mG!=groundtruth.end<cv::Vec2f>(),\
			zP!=zeropred.end<cv::Vec2f>(),ed!=edge.end<uchar>();++mP,++mG,++zP,++ed){
				cv::Vec2f truth = *mG, predi = *mP, zerop = *zP;
				tmpoverall += std::sqrt((truth.val[0]-predi.val[0])*\
					(truth.val[0]-predi.val[0])+\
					(truth.val[1]-predi.val[1])*(truth.val[1]-predi.val[1]));
				tmpoverall0 += std::sqrt((truth.val[0]-zerop.val[0])*\
					(truth.val[0]-zerop.val[0])+\
					(truth.val[1]-zerop.val[1])*(truth.val[1]-zerop.val[1]));
				float norm1 = std::sqrt(truth.val[0]*truth.val[0]+truth.val[1]*\
					truth.val[1]+small);
				float norm2 = std::sqrt(predi.val[0]*predi.val[0]+predi.val[1]*\
					predi.val[1]+small);
				float zeronorm2 = std::sqrt(zerop.val[0]*zerop.val[0]+zerop.val[1]*\
					zerop.val[1]+small);
				if(static_cast<int>(*ed)>0){
					tmpcanny += std::sqrt((truth.val[0]-predi.val[0])*\
						(truth.val[0]-predi.val[0])+\
						(truth.val[1]-predi.val[1])*(truth.val[1]-predi.val[1]));
					tmpcanny0 += std::sqrt((truth.val[0]-zerop.val[0])*\
						(truth.val[0]-zerop.val[0])+(truth.val[1]-zerop.val[1])*\
						(truth.val[1]-zerop.val[1]));
					tmpcoscanny += (truth.val[0]*predi.val[0]+truth.val[1]*\
						predi.val[1])/(norm1*norm2);
					tmpcoscanny0 += (truth.val[0]*zerop.val[0]+truth.val[1]*\
						zerop.val[1])/(norm1*zeronorm2);
					tmpabscoscanny += std::abs(truth.val[0]*predi.val[0]+truth.val[1]*\
						predi.val[1])/(norm1*norm2);
					tmpabscoscanny0 += std::abs(truth.val[0]*zerop.val[0]+truth.val[1]*\
						zerop.val[1])/(norm1*zeronorm2);
					++cannyelem;
				}
			}
			// MSE = 1/N sum_X (X-barX)^2 = 1/N sum_x (X-barX).(X-barX)
			overall           += tmpoverall/static_cast<float>(groundtruth.cols*groundtruth.rows);
			overall0          += tmpoverall0/static_cast<float>(groundtruth.cols*groundtruth.rows);
			if(cannyelem){
				canny        += tmpcanny/static_cast<float>(cannyelem);
				canny0       += tmpcanny0/static_cast<float>(cannyelem);
				coscanny     += tmpcoscanny/static_cast<float>(cannyelem);
				coscanny0    += tmpcoscanny0/static_cast<float>(cannyelem);
				abscoscanny  += tmpabscoscanny/static_cast<float>(cannyelem);
				abscoscanny0 += tmpabscoscanny0/static_cast<float>(cannyelem);
			}
			groundtruth.release(); prediction.release(); edge.release();
			++processed;
			image.release();
		}
	}
	overall           /= static_cast<float>(processed);
	overall0          /= static_cast<float>(processed);
	canny             /= static_cast<float>(processed);
	canny0            /= static_cast<float>(processed);
	coscanny          /= static_cast<float>(processed);
	coscanny0         /= static_cast<float>(processed);
	abscoscanny       /= static_cast<float>(processed);
	abscoscanny0      /= static_cast<float>(processed);
	std::cout<<"[MotionEval::meanEPE] frames processed: "<<processed<<std::endl;
	std::cout<<"[MotionEval::meanEPE] overall-epe: "<<overall<<\
		" overall-0-epe: "<<overall0<<std::endl;
	std::cout<<"[MotionEval::meanEPE] canny-epe: "<<canny<<\
		" canny-0-epe: "<<canny0<<std::endl;
	std::cout<<"[MotionEval::meanEPE] canny-direction: "<<coscanny<<\
		" canny-0-direction: "<<coscanny0<<std::endl;
	std::cout<<"[MotionEval::meanEPE] canny-orientation: "<<abscoscanny<<\
		" canny-0-orientation: "<<abscoscanny0<<std::endl;
	std::cout<<overall<<" "<<overall0<<" "<<canny<<" "<<canny0<<" "<<coscanny<<\
		" "<<coscanny0<<" "<<abscoscanny<<" "<<abscoscanny0<<std::endl;
	return overall;
}
//==============================================================================
/** Gets unexpected event.
 */
float MotionEval::unexpectedEPE(bool fromfeat,bool display){
	// Load image names
	std::vector<std::string> dirs = Auxiliary<uchar,1>::listDir(this->path2predi_);
	float overall                 = 0.0;
	unsigned processed            = 0;
	std::vector<float> epes;
	for(std::vector<std::string>::iterator d=dirs.begin();d!=dirs.end();++d){
		std::string adir = this->path2predi_+(*d)+PATH_SEP+std::string("flow_motion");
		std::vector<std::string> vPredictions;
		if(Auxiliary<char,1>::file_exists(adir.c_str())){
			vPredictions = Auxiliary<uchar,1>::listDir(adir,".bin");
		}
		if(vPredictions.empty()){continue;}
		for(std::vector<std::string>::iterator iP=vPredictions.begin();iP!=\
		vPredictions.end()-1;++iP){ // for each image
			std::string gtName = (*iP).substr(2,(*iP).size()-2);
			std::string imName = this->path2test_+(*d)+PATH_SEP+gtName.substr\
				(0,gtName.size()-4)+this->ext_;
			cv::Mat groundtruth,prediction;
			try{
				if(fromfeat){
					groundtruth = this->loadGTfromFeat(this->path2gt_+(*d)+\
						PATH_SEP+gtName);
				}else{
					std::cout<<"Loading: "<<(this->path2gt_+(*d)+PATH_SEP+\
						"flow_motion"+PATH_SEP+gtName).c_str()<<std::endl;
					groundtruth = Auxiliary<float,2>::bin2mat((this->path2gt_+\
						(*d)+PATH_SEP+"flow_motion"+PATH_SEP+gtName).c_str());
				}
				prediction  = Auxiliary<float,2>::bin2mat(std::string\
					(adir+(*iP)).c_str());
			}catch(std::exception &e){
				std::cerr<<"[MotionEval::unexpectedEPE]: Image "<<this->path2gt_+gtName<<\
					" not processed"<<std::endl;
				continue;
			}
			assert(groundtruth.cols==prediction.cols && groundtruth.rows==prediction.rows);
			// [4] just percentage of pixels correctly classified
			cv::Mat image = cv::imread(imName,1);
			cv::resize(image,image,prediction.size());
			prediction *= 10;
			cv::Mat diff = groundtruth-prediction;
			std::vector<cv::Mat> splitsDiff;
			std::vector<cv::Mat> splitsGt;
			std::vector<cv::Mat> splitsPred;
			cv::split(diff,splitsDiff);
			cv::split(groundtruth,splitsGt);
			cv::split(prediction,splitsPred);
			if(display){
				MotionPatchClass::showOF(splitsDiff[0],splitsDiff[1],image,5,true,\
					"Predicted Thresholded Diff");
				MotionPatchClass::showOF(splitsGt[0],splitsGt[1],image,5,true,\
					"Predicted Thresholded Gt");
				MotionPatchClass::showOF(splitsPred[0],splitsPred[1],image,5,true,\
					"Predicted Thresholded Predi");
			}
			splitsDiff[0].release(); splitsDiff[1].release();
			splitsGt[0].release(); splitsGt[1].release();
			splitsPred[0].release(); splitsPred[1].release();
			diff.release();
			// [2] Get the cCanny edges:
			cv::Mat gray, edge;
			cv::cvtColor(image,gray,CV_BGR2GRAY);
			cv::blur(gray,gray,cv::Size(3,3));
			cv::Canny(gray,edge,50,150,3);
			gray.release(); image.release();
			float tmpoverall = 0.0;
			unsigned pixels  = 0;
			edge.convertTo(edge,CV_8UC1);
			cv::MatConstIterator_<uchar> ed     = edge.begin<uchar>();
			cv::MatConstIterator_<cv::Vec2f> mG = groundtruth.begin<cv::Vec2f>();
			for(cv::MatConstIterator_<cv::Vec2f> mP=prediction.begin<cv::Vec2f>();\
			mP!=prediction.end<cv::Vec2f>(),mG!=groundtruth.end<cv::Vec2f>(),\
			ed!=edge.end<uchar>();++mP,++mG,++ed){
				cv::Vec2f truth = *mG, predi = *mP;
				if((int)(*ed)>0){
					tmpoverall += std::sqrt((truth.val[0]-predi.val[0])*\
						(truth.val[0]-predi.val[0])+\
						(truth.val[1]-predi.val[1])*
						(truth.val[1]-predi.val[1]));
					++pixels;
				}
			} // loop over pixel positions
			// MSE = 1/N sum_X (X-barX)^2 = 1/N sum_x (X-barX).(X-barX)
			epes.push_back(tmpoverall/static_cast<float>(pixels));
			overall += tmpoverall/static_cast<float>(pixels);
			++processed;
		} // over images/frames
	} // over directories
	overall  /= static_cast<float>(processed);
	std::cout<<"[MotionEval::unexpectedEPE] frames processed: "<<processed<<std::endl;
	std::cout<<"[MotionEval::unexpectedEPE] overall-epe: "<<overall<<std::endl;
	// Now find the frames whose EPE is over the mean
	float std = 0;
	for(std::vector<float>::iterator e=epes.begin();e!=epes.end();++e){
		std += ((*e)-overall)*((*e)-overall);
	}
	std            = std::sqrt(std/static_cast<float>(epes.size()));
	for(std::vector<std::string>::iterator d=dirs.begin();d!=dirs.end();++d){
		// Create the dir at which to write the per frame EPEs
		std::string writefile  = this->path2predi_+(*d)+PATH_SEP;
		Auxiliary<uchar,1>::file_exists(writefile.c_str(),true);
		writefile              = writefile+"unexpected.txt";
		// Now compute the epes
		std::string adir = this->path2predi_+(*d)+PATH_SEP+std::string("flow_arrows");
		std::vector<std::string> vPredictions;
		if(Auxiliary<char,1>::file_exists(adir.c_str())){
			vPredictions = Auxiliary<uchar,1>::listDir(adir,".png");
		}
		if(vPredictions.empty()){continue;}
		// Open the file to write to
		std::ofstream pFile;
		std::cout<<"EPEs saved to "<<writefile<<std::endl;
		try{
			pFile.open(writefile.c_str(),std::ios::out);
		}catch(std::exception &e){
			std::cerr<<"[MotionEval::unexpectedPrev]: Cannot open file: %s"<<\
				e.what()<<std::endl;
			std::exit(-1);
		}
		pFile.precision(std::numeric_limits<double>::digits10);
		pFile.precision(std::numeric_limits<float>::digits10);
		// Loop over frames
		unsigned index = 0;
		for(std::vector<std::string>::iterator iP=vPredictions.begin();iP!=\
		vPredictions.end()-1;++iP){ // for each image
			std::string gtName = (*iP).substr(2,(*iP).size()-2);
			std::string imName = this->path2test_+(*d)+PATH_SEP+gtName.substr\
					(0,gtName.size()-4)+this->ext_;
			pFile<<index<<" "<<epes[index]<<std::endl;
			if(epes[index]>overall+std){
				std::cout<<"Over? "<<index<<" "<<epes[index]<<">"<<overall<<" >>> "<<\
					(adir+PATH_SEP+(*iP))<<std::endl;
				cv::Mat which = cv::imread(adir+PATH_SEP+(*iP),1);
				cv::Mat image = cv::imread(imName,1);
				cv::resize(image,image,which.size());

				std::string justname = gtName.substr(0,gtName.size()-4);
				std::string flowname = "unexpected-flow"+Auxiliary<unsigned,1>::\
					number2string(index)+"_"+justname+".png";
				std::string predname = "unexpected-pred"+Auxiliary<unsigned,1>::\
					number2string(index)+"_"+justname+".png";
				cv::imwrite(flowname,which);
				cv::imwrite(predname,image);
				if(display){
					cv::imshow("Unexpected-flow?",which);
					cv::waitKey(0);
					cv::imshow("Unexpected-pred?",image);
					cv::waitKey(0);
				}
				which.release(); image.release();
			}
			++index;
		} // over images
		pFile.close();
	} // over dirs
	return overall;
}
//==============================================================================
/** Gets unexpected event looking at previous frame.
 */
float MotionEval::unexpectedEPEPrev(bool fromfeat,bool display){
	// Load image names
	std::vector<std::string> dirs = Auxiliary<uchar,1>::listDir(this->path2predi_);
	float overall                 = 0.0;
	unsigned processed            = 0;
	std::vector<float> epes;
	for(std::vector<std::string>::iterator d=dirs.begin();d!=dirs.end();++d){
		std::string adir = this->path2predi_+(*d)+PATH_SEP+std::string("flow_motion");
		std::vector<std::string> vPredictions;
		if(Auxiliary<char,1>::file_exists(adir.c_str())){
			vPredictions = Auxiliary<uchar,1>::listDir(adir,".bin");
		}
		if(vPredictions.empty()){continue;}
		for(std::vector<std::string>::iterator iP=vPredictions.begin();iP!=\
		vPredictions.end()-2;++iP){ // for each image
			std::string gtName1 = (*iP).substr(2,(*iP).size()-2);
			std::string gtName2 = (*(iP+1)).substr(2,(*(iP+1)).size()-2);
			std::string imName = this->path2test_+(*d)+PATH_SEP+gtName1.substr\
				(0,gtName1.size()-4)+this->ext_;
			cv::Mat groundtruth1,groundtruth2;
			try{
				if(fromfeat){
					groundtruth1 = this->loadGTfromFeat(this->path2gt_+(*d)+\
						PATH_SEP+gtName1);
					groundtruth2 = this->loadGTfromFeat(this->path2gt_+(*d)+\
						PATH_SEP+gtName2);

				}else{
					std::cout<<"Loading: "<<(this->path2gt_+(*d)+PATH_SEP+\
						"flow_motion"+PATH_SEP+gtName1).c_str()<<std::endl;
					groundtruth1 = Auxiliary<float,2>::bin2mat((this->path2gt_+\
						(*d)+PATH_SEP+"flow_motion"+PATH_SEP+gtName1).c_str());
					groundtruth2 = Auxiliary<float,2>::bin2mat((this->path2gt_+\
						(*d)+PATH_SEP+"flow_motion"+PATH_SEP+gtName2).c_str());
				}
			}catch(std::exception &e){
				std::cerr<<"[MotionEval::unexpectedEPE]: Image "<<this->path2gt_+\
					gtName1<<" not processed"<<std::endl;
				continue;
			}
			// [4] just percentage of pixels correctly classified
			cv::Mat image = cv::imread(imName,1);
			cv::resize(image,image,groundtruth1.size());

			cv::Mat diff = groundtruth1-groundtruth2;
			std::vector<cv::Mat> splitsDiff;
			std::vector<cv::Mat> splitsGt1;
			std::vector<cv::Mat> splitsGt2;
			cv::split(diff,splitsDiff);
			cv::split(groundtruth1,splitsGt1);
			cv::split(groundtruth2,splitsGt2);
			if(display){
				MotionPatchClass::showOF(splitsDiff[0],splitsDiff[1],image,5,true,\
					"Predicted Thresholded Diff");
				MotionPatchClass::showOF(splitsGt1[0],splitsGt1[1],image,5,true,\
					"Predicted Thresholded Gt1");
				MotionPatchClass::showOF(splitsGt2[0],splitsGt2[1],image,5,true,\
					"Predicted Thresholded Gt2");
			}
			splitsGt1[0].release(); splitsGt1[1].release();
			splitsDiff[0].release(); splitsDiff[1].release();
			splitsGt2[0].release(); splitsGt2[1].release();
			diff.release();
			// [2] Get the cCanny edges:
			cv::Mat gray, edge;
			cv::cvtColor(image,gray,CV_BGR2GRAY);
			cv::blur(gray,gray,cv::Size(3,3));
			cv::Canny(gray,edge,50,150,3);
			gray.release(); image.release();
			float tmpoverall = 0.0;
			unsigned pixels  = 0;
			edge.convertTo(edge,CV_8UC1);
			cv::MatConstIterator_<uchar> ed      = edge.begin<uchar>();
			cv::MatConstIterator_<cv::Vec2f> mG1 = groundtruth1.begin<cv::Vec2f>();
			for(cv::MatConstIterator_<cv::Vec2f> mG2=groundtruth2.begin<cv::Vec2f>();\
			mG2!=groundtruth2.end<cv::Vec2f>(),mG1!=groundtruth1.end<cv::Vec2f>(),\
			ed!=edge.end<uchar>();++mG1,++mG2,++ed){
				cv::Vec2f truth1 = *mG1, truth2 = *mG2;
				if((int)(*ed)>0){
					tmpoverall += std::sqrt((truth1.val[0]-truth2.val[0])*\
						(truth1.val[0]-truth2.val[0])+
						(truth1.val[1]-truth2.val[1])*\
						(truth1.val[1]-truth2.val[1]));
					++pixels;
				}
			} // loop over pixel positions
			// MSE = 1/N sum_X (X-barX)^2 = 1/N sum_x (X-barX).(X-barX)
			epes.push_back(tmpoverall/static_cast<float>(pixels));
			overall += tmpoverall/static_cast<float>(pixels);
			++processed;
		} // over images/frames
	} // over directories
	overall /= static_cast<float>(processed);
	std::cout<<"[MotionEval::unexpectedEPEPrev] frames processed: "<<processed<<std::endl;
	std::cout<<"[MotionEval::unexpectedEPEPrev] overall-epe: "<<overall<<std::endl;
	// Now find the frames whose EPE is over the mean
	float std = 0;
	for(std::vector<float>::iterator e=epes.begin();e!=epes.end();++e){
		std += ((*e)-overall)*((*e)-overall);
	}
	std            = std::sqrt(std/static_cast<float>(epes.size()));
	unsigned index = 0;
	for(std::vector<std::string>::iterator d=dirs.begin();d!=dirs.end();++d){
		// Create the dir at which to write the per frame EPEs
		std::string writefile  = this->path2predi_+(*d)+PATH_SEP;
		Auxiliary<uchar,1>::file_exists(writefile.c_str(),true);
		writefile              = writefile+"unexpected_prev.txt";
		// Now compute the epes
		std::string adir = this->path2predi_+(*d)+PATH_SEP+std::string("flow_arrows");
		std::vector<std::string> vPredictions;
		if(Auxiliary<char,1>::file_exists(adir.c_str())){
			vPredictions = Auxiliary<uchar,1>::listDir(adir,".png");
		}
		if(vPredictions.empty()){continue;}
		// Open the file to write to
		std::ofstream pFile;
		std::cout<<"EPEs saved to "<<writefile<<std::endl;
		try{
			pFile.open(writefile.c_str(),std::ios::out);
		}catch(std::exception &e){
			std::cerr<<"[MotionEval::unexpectedPrev]: Cannot open file: %s"<<\
				e.what()<<std::endl;
			std::exit(-1);
		}
		pFile.precision(std::numeric_limits<double>::digits10);
		pFile.precision(std::numeric_limits<float>::digits10);
		// Loop over frames
		for(std::vector<std::string>::iterator iP=vPredictions.begin();iP!=\
		vPredictions.end()-1;++iP){ // for each image
			std::string gtName = (*iP).substr(2,(*iP).size()-2);
			std::string imName = this->path2test_+(*d)+PATH_SEP+gtName.substr\
					(0,gtName.size()-4)+this->ext_;
			pFile<<index<<" "<<epes[index]<<std::endl;
			if(epes[index]>overall+std){
				std::cout<<"Over? "<<epes[index]<<">"<<overall<<" >>> "<<\
					(adir+PATH_SEP+(*iP))<<std::endl;
				cv::Mat which = cv::imread(adir+PATH_SEP+(*iP),1);
				cv::Mat image = cv::imread(imName,1);
				cv::resize(image,image,which.size());
				std::string justname = gtName.substr(0,gtName.size()-4);
				std::string flowname = "unexpected-flow"+justname+".png";
				std::string predname = "unexpected-pred"+justname+".png";
				cv::imwrite(flowname,which);
				cv::imwrite(predname,image);
				if(display){
					cv::imshow("Unexpected-flow?",which);
					cv::waitKey(0);
					cv::imshow("Unexpected-pred?",image);
					cv::waitKey(0);
				}
				which.release(); image.release();
			}
			++index;
		} // over images
		pFile.close();
	} // over dirs
	return overall;
}
//==============================================================================s
/** Raw files to be used with python.
 */
void MotionEval::raw4python(bool fromfeat){
	// [0] Load image names
	std::vector<std::string> dirs = Auxiliary<uchar,1>::listDir(this->path2predi_);
	for(std::vector<std::string>::iterator d=dirs.begin();d!=dirs.end();++d){
		std::string adir    = this->path2predi_+(*d)+PATH_SEP+std::string("flow_motion");
		std::string outdir  = this->path2predi_+(*d)+PATH_SEP+std::string("file")+PATH_SEP;
		Auxiliary<uchar,1>::file_exists(outdir.c_str(),true);
		std::vector<std::string> vPredictions = Auxiliary<uchar,1>::listDir(adir,".bin");
		for(std::vector<std::string>::iterator iP=vPredictions.begin();iP!=\
		vPredictions.end()-1;++iP){ // for each image
			std::string gtName = (*iP).substr(2,(*iP).size()-2);
			cv::Mat groundtruth,prediction;
			try{
				if(fromfeat){
					groundtruth = this->loadGTfromFeat(this->path2gt_+(*d)+\
						PATH_SEP+gtName);
				}else{
					groundtruth = Auxiliary<float,2>::bin2mat((this->path2gt_+\
						(*d)+PATH_SEP+"flow_motion"+PATH_SEP+gtName).c_str());
				}
				prediction = Auxiliary<float,2>::bin2mat(std::string\
					(adir+(*iP)).c_str());
			}catch(std::exception &e){
				groundtruth.release();
				prediction.release();
				std::cerr<<"[MotionEval::globalSSD]: Image "<<this->path2gt_+gtName<<\
					" not processed"<<std::endl;
				continue;
			}
			if(groundtruth.cols!=prediction.cols || groundtruth.rows!=prediction.rows){
				std::cerr<<"[MotionEval::globalSSD]: the ground truth and the prediction "<<\
					"should have the same size: "<<groundtruth.size()<<" "<<\
					prediction.size()<<std::endl;
			}
			// [1] the file in which to store the current data
			std::string outfile = outdir+(*iP).substr(0,(*iP).size()-4)+".txt";
			std::ofstream pFile;
			std::cout<<"Coordinates saved to "<<outfile<<std::endl;
			try{
				pFile.open(outfile.c_str(),std::ios::out);
			}catch(std::exception &e){
				std::cerr<<"[MotionEval::raw4python]: Cannot open file: %s"<<\
					e.what()<<std::endl;
				std::exit(-1);
			}
			pFile.precision(std::numeric_limits<double>::digits10);
			pFile.precision(std::numeric_limits<float>::digits10);
			// [0] Write down the data (each pixel with its own line)
			for(unsigned r=0;r<prediction.rows;++r){
				for(unsigned c=0;c<prediction.cols;++c){
					cv::Vec2f predi = prediction.at<cv::Vec2f>(r,c);
					cv::Vec2f truth = groundtruth.at<cv::Vec2f>(r,c);
					pFile<<c<<" "<<r<<" "<<predi.val[0]<<" "<<predi.val[1]<<" "<<\
						truth.val[0]<<" "<<truth.val[1]<<std::endl;
				}
			}
			pFile.close();
			groundtruth.release();
			prediction.release();
		}
	}
}
//==============================================================================
/** Show a flow image.
 */
void MotionEval::flow(bool fromfeat){
	// [0] Load image names
	std::vector<std::string> dirs = Auxiliary<uchar,1>::listDir(this->path2predi_);
	unsigned scale                = 10;
	for(std::vector<std::string>::iterator d=dirs.begin();d!=dirs.end();++d){
		std::string adir   = this->path2predi_+(*d)+PATH_SEP+std::string("flow_motion");
		std::string outdir = this->path2predi_+(*d)+PATH_SEP+std::string("flow")+PATH_SEP;
		Auxiliary<uchar,1>::file_exists(outdir.c_str(),true);
		std::vector<std::string> vPredictions = Auxiliary<uchar,1>::listDir(adir,".bin");
		for(std::vector<std::string>::iterator iP=vPredictions.begin();iP!=\
		vPredictions.end()-1;++iP){ // for each image
			cv::Mat prediction,groundtruth;
			try{
				std::string gtName = (*iP).substr(2,(*iP).size()-2);
				if(fromfeat){
					groundtruth = this->loadGTfromFeat(this->path2gt_+(*d)+\
						PATH_SEP+gtName);
				}else{
					groundtruth = Auxiliary<float,2>::bin2mat((this->path2gt_+\
						(*d)+PATH_SEP+"flow_motion"+PATH_SEP+gtName).c_str());
				}
				prediction  = Auxiliary<float,2>::bin2mat(std::string\
					(adir+(*iP)).c_str());
			}catch(std::exception &e){
				prediction.release();
				std::cerr<<"[MotionEval::globalSSD]: Image "<<std::string(adir+(*iP))<<\
					" not processed"<<std::endl;
				continue;
			}
			cv::Mat flow = MotionEval::flow2im(prediction);
			std::string plotout = outdir+(*iP).substr(0,(*iP).size()-4)+".png";
			cv::imwrite(plotout,flow);
			prediction.release(); flow.release();
			cv::Mat flowGt = MotionEval::flow2im(groundtruth);
			std::string plotoutGt = outdir+(*iP).substr(0,(*iP).size()-4)+"_gt.png";
			cv::imwrite(plotoutGt,flowGt);
			groundtruth.release(); flowGt.release();
		}
	}
}
//==============================================================================
/** Displays the flow as it is usually displayed.
 */
cv::Mat MotionEval::flow2im(const cv::Mat &flow,bool resize){
	//[1] Extract x and y channels
	cv::Mat xy[2];
	cv::split(flow,xy);
	//[2] Calculate angle and magnitude
	cv::Mat magnitude,angle;
	cv::cartToPolar(xy[0],xy[1],magnitude,angle,true);
	//[3] Translate magnitude to range [0;1]
	double magMax,magMin;
	cv::minMaxLoc(magnitude,&magMin,&magMax);
	magnitude.convertTo(magnitude,-1,1.0/magMax);
	//[4] Build hsv image
	cv::Mat hsvIm[3],hsv;
	hsvIm[0] = angle;
	hsvIm[1] = cv::Mat::ones(angle.size(),CV_32FC1);
	hsvIm[2] = magnitude;
	cv::merge(hsvIm,3,hsv);
	magnitude.release(); angle.release();
	//[5] Convert to BGR and show
	cv::Mat bgr;
	cv::cvtColor(hsv,bgr,cv::COLOR_HSV2BGR);
	hsv.release();
	//[6] Convert it to uchar to save it
	double minVal, maxVal;
	cv::minMaxLoc(bgr,&minVal,&maxVal);
	cv::Mat draw;
	bgr.convertTo(draw,CV_8UC3,255.0/(maxVal-minVal),-minVal*255.0/(maxVal-minVal));
	bgr.release();
	if(resize){
		float step = 500.0/static_cast<float>(std::max(draw.cols,draw.rows));
		cv::Size size(draw.cols*step,draw.rows*step);
		cv::resize(draw,draw,size);
	}
	return draw;
}
//==============================================================================
/** Evaluates one or all of the above methods.
 */
void MotionEval::run(MotionEval::METHOD method){
	switch(method){
		case(MotionEval::FLOW):
			this->flow();
			break;
		case(MotionEval::RAW4PYTHON):
			this->raw4python();
			break;
		case(MotionEval::MEAN_EPE):
			this->meanEPE();
			break;
		case(MotionEval::ALL):
			this->meanEPE();
			this->raw4python();
			this->flow();
			break;
		case(MotionEval::UNEXPECTED):
			this->unexpectedEPE();
			break;
		case(MotionEval::UNEXPECTED_PREV):
			this->unexpectedEPEPrev();
			break;
		default:
			std::cerr<<"[MotionEval::run] option not implemented"<<std::endl;
			break;
	}
}
//==============================================================================

