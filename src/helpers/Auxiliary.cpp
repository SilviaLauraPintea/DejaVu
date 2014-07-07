/* Auxiliary.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef AUXILIARY_CPP_
#define AUXILIARY_CPP_
#include "Auxiliary.h"
#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
/** Note: Matrixes are stored as: mat.cols*r+c (the same other for reshape also).
 */
//==============================================================================
//==============================================================================
//==============================================================================
/** Splits a string on a specific character and does not include that character
 * in the result.
 */
template <typename T,int U>
std::deque<std::string> Auxiliary<T,U>::splitLine(char *str,char charct){
	std::deque<std::string> res;
	unsigned strSize = std::string(str).size()+1;
	char *prev = str;// they both point to the beginning of the string
	while(*str){
		if(*str == charct && str!=prev){
			char *tmp      = new char[strSize];
			unsigned int k = 0;
			while(prev<str){
				tmp[k] = (*prev);//the value at which prev points
				++prev;
				++k;
			}
			while((*str) == charct){
				++str;
			}
			tmp[k] = '\0';
			res.push_back(std::string(tmp));
			delete [] tmp;
			prev = str;
		}else{
			++str;
		}
	}
	// THE LAST PART OF THE STRING IF ANY
	if(str!=prev){
		char *tmp      = new char[strSize];
		unsigned int k = 0;
		while(prev<str){
			tmp[k] = (*prev);//the value at which prev points
			++prev;
			++k;
		}
		tmp[k] = '\0';
		res.push_back(std::string(tmp));
		delete [] tmp;
	}
	return res;
}
//==============================================================================
/** Check if a file exists.
 */
template <typename T,int U>
int Auxiliary<T,U>::file_exists(const char* fileName,bool create){
	 struct stat buf;
	 int i = stat(fileName,&buf);
	 // IF THE FILE/FOLDER WAS FOUND
	 if(i == 0){return 1;}
	 // FILE/FOLDER NOT FOUND AND WE WANT TO CREATE IT:
	 if(create){
		int stat = mkdir(fileName,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	 }
	 return 0;
}
//==============================================================================
/** Converts a pointer to an IplImage to an OpenCV Mat. No hard copy.
 */
template <typename T,int U>
cv::Mat Auxiliary<T,U>::ipl2mat(IplImage* ipl_image){
	cv::Mat mat_image = cv::Mat(ipl_image);
	return mat_image;
}
//==============================================================================
/** Converts an OpenCV Mat to a pointer to an IplImage. No hard copy (just creates
 * a header and points to the cv::Mat).
 */
template <typename T,int U>
IplImage* Auxiliary<T,U>::mat2ipl(const cv::Mat_<cv::Vec<T,U> > &image){
	IplImage* ipl_image = new IplImage(image);
	return ipl_image;
}
//==============================================================================
/** Write a 2D-matrix to a text file (first row is the dimension of the matrix).
 */
template <typename T,int U>
void Auxiliary<T,U>::mat2txt(cv::Mat_<cv::Vec<T,U> > &matrix,char* fileName,\
bool append){
	std::ofstream dictOut;
	try{
		if(append){
			dictOut.open(fileName,std::ios::out | std::ios::app);
			dictOut.seekp(0,std::ios::end);
		}else{
			dictOut.open(fileName,std::ios::out);
		}
	}catch(std::exception &e){
		std::cerr<<"Cannot open file: %s"<<e.what()<<std::endl;
		exit(1);
	}
	dictOut.precision(std::numeric_limits<double>::digits10);
	dictOut.precision(std::numeric_limits<float>::digits10);
	dictOut<<matrix.cols<<" "<<matrix.rows<<std::endl;
	for(int ch=0;ch<matrix.channels();++ch){
		for(int y=0;y<matrix.rows;++y){
			for(int x=0;x<matrix.cols;++x){
				dictOut<<(matrix.template at<cv::Vec<T,U> >(y,x)).val[ch]<<" ";
			}
			dictOut<<std::endl;
		}
		dictOut<<std::endl;
	}
	dictOut.close();
}
//==============================================================================
/** Reads a 2D-matrix from a text file (first row is the dimension of the matrix).
 */
template <typename T,int U>
void Auxiliary<T,U>::txt2mat(cv::Mat_<cv::Vec<T,U> > &matrix,char* fileName){
	std::ifstream dictFile(fileName);
	if(dictFile.is_open()){
		// FIRST LINE IS THE SIZE OF THE MATRIX
		std::string fline;
		std::getline(dictFile,fline);
		std::deque<std::string> flineVect = Auxiliary::splitLine(const_cast<char*>\
			(fline.c_str()),' ');
		if(flineVect.size() == 2){
			char *pRows,*pCols;
			int cols = strtol(flineVect[0].c_str(),&pCols,10);
			int rows = strtol(flineVect[1].c_str(),&pRows,10);
			matrix   = cv::Mat::zeros(cv::Size(cols,rows),\
				cv::DataType<cv::Vec<T,U> >::type);
		}else return;
		fline.clear();
		flineVect.clear();

		// THE REST OF THE LINES ARE READ ONE BY ONE
		int ch=0,y=0;
		while(dictFile.good()){
			std::string line;
			std::getline(dictFile,line);
			std::deque<std::string> lineVect = Auxiliary::splitLine(const_cast<char*>\
				(line.c_str()),' ');
			if(lineVect.size()>=1){
				// read the matrix by: channels x [rows x cols]
				for(std::size_t x=0;x<lineVect.size();++x){
					char *pValue;
					(matrix.template at<cv::Vec<T,U> >(y,static_cast<int>(x))).\
						val[ch] = static_cast<T>(strtod(lineVect[x].c_str(),&pValue));
				}
				++y;
				if(y==matrix.cols){ // 1 channel full
					y=0;
					++ch;
				}
			}
			line.clear();
			lineVect.clear();
		}
		dictFile.close();
	}
}
//==============================================================================
/** Write a 2D-matrix to a binary file (first the dimension of the matrix).
 */
template <typename T,int U>
void Auxiliary<T,U>::mat2bin(const cv::Mat_<cv::Vec<T,U> > &matrix,\
const char* fileName,bool append){
	std::ofstream mxFile;
	try{
		if(append){
			mxFile.open(fileName,std::ios::out|std::ios::app|std::ios::binary);
			mxFile.seekp(0,std::ios::end);
		}else{
			mxFile.open(fileName,std::ios::out|std::ios::binary);
		}
	}catch(std::exception &e){
		std::cerr<<"Cannot open file: %s"<<e.what()<<std::endl;
		exit(1);
	}
	// FIRST WRITE THE DIMENSIONS OF THE MATRIX
	mxFile.precision(std::numeric_limits<double>::digits10);
	mxFile.precision(std::numeric_limits<float>::digits10);
	int channels = matrix.channels();
	mxFile.write(reinterpret_cast<const char*>(&channels),sizeof(int));
	mxFile.write(reinterpret_cast<const char*>(&matrix.cols),sizeof(int));
	mxFile.write(reinterpret_cast<const char*>(&matrix.rows),sizeof(int));
	// WRITE THE MATRIX TO THE FILE
	for(int ch=0;ch<matrix.channels();++ch){
		for(int y=0;y<matrix.rows;++y){
			for(int x=0;x<matrix.cols;++x){
				mxFile.write(reinterpret_cast<const char*>(&(matrix.template at\
					<cv::Vec<T,U> >(y,x)).val[ch]),sizeof(T));
			}
		}
	}
	mxFile.close();
}
//==============================================================================
/** Write a 2D-matrix to a binary file (first the dimension of the matrix).
 */
template <typename T,int U>
void Auxiliary<T,U>::matvec2bin(const std::vector<cv::Mat_<cv::Vec<T,U> > > &matrix,\
const char* fileName,bool append){
	std::ofstream mxFile;
	try{
		if(append){
			mxFile.open(fileName,std::ios::out|std::ios::app|std::ios::binary);
			mxFile.seekp(0,std::ios::end);
		}else{
			mxFile.open(fileName,std::ios::out|std::ios::binary);
		}
	}catch(std::exception &e){
		std::cerr<<"Cannot open file: %s"<<e.what()<<std::endl;
		exit(1);
	}
	mxFile.precision(std::numeric_limits<double>::digits10);
	mxFile.precision(std::numeric_limits<float>::digits10);
	// FIRST WRITE THE DIMENSIONS OF THE VECTOR
	std::size_t msize = matrix.size();
	mxFile.write(reinterpret_cast<char*>(&msize),sizeof(int));
	for(std::size_t s=0;s<matrix.size();++s){
		// FIRST WRITE THE DIMENSIONS OF THE MATRIX
		int channels = matrix[s].channels();
		mxFile.write(reinterpret_cast<char*>(&channels),sizeof(int));
		mxFile.write(reinterpret_cast<char*>(&matrix[s].cols),sizeof(int));
		mxFile.write(reinterpret_cast<char*>(&matrix[s].rows),sizeof(int));
		// WRITE THE MATRIX TO THE FILE
		for(int ch=0;ch<matrix[s].channels();++ch){
			for(int y=0;y<matrix[s].rows;++y){
				for(int x=0;x<matrix[s].cols;++x){
					mxFile.write(reinterpret_cast<char*>(&(matrix[s].template at\
						<cv::Vec<T,U> >(y,x)).val[ch]),sizeof(T));
				}
			}
		}
	}
	mxFile.close();
}
//==============================================================================
/** Reads a 2D-matrix from a binary file (first the dimension of the matrix).
 */
template <typename T,int U>
void Auxiliary<T,U>::bin2mat(cv::Mat_<cv::Vec<T,U> > &matrix,const char* \
fileName){
	if(!Auxiliary::file_exists(fileName)){
		std::cerr<<"Error opening the file: "<<fileName<<std::endl;
		exit(1);
	}
	std::ifstream mxFile;
	mxFile.open(fileName,std::ios::in | std::ios::binary);
	if(mxFile.is_open()){
		// FIRST READ THE MATRIX SIZE AND ALLOCATE IT
		int cols,rows,channels;
		mxFile.read(reinterpret_cast<char*>(&channels),sizeof(int));
		mxFile.read(reinterpret_cast<char*>(&cols),sizeof(int));
		mxFile.read(reinterpret_cast<char*>(&rows),sizeof(int));
		matrix = cv::Mat::zeros(cv::Size(cols,rows),cv::DataType<cv::Vec<T,U> >::type);
		// READ THE CONTENT OF THE MATRIX
		for(int ch=0;ch<channels;++ch){
			for(int y=0;y<matrix.rows;++y){
				for(int x=0;x<matrix.cols;++x){
					mxFile.read(reinterpret_cast<char*>(&(matrix.template at\
						<cv::Vec<T,U> >(y,x)).val[ch]),sizeof(T));
				}
			}
		}
		mxFile.close();
	}
}
//==============================================================================
/** Reads a 2D-matrix from a binary file (first the dimension of the matrix).
 */
template <typename T,int U>
cv::Mat_<cv::Vec<T,U> > Auxiliary<T,U>::bin2mat(const char* fileName){
	if(!Auxiliary::file_exists(fileName)){
		std::cerr<<"Error opening the file: "<<fileName<<std::endl;
		exit(1);
	}
	std::ifstream mxFile;
	mxFile.open(fileName,std::ios::in | std::ios::binary);
	cv::Mat_<cv::Vec<T,U> > matrix;
	if(mxFile.is_open()){
		// FIRST READ THE MATRIX SIZE AND ALLOCATE IT
		int cols,rows,channels;
		mxFile.read(reinterpret_cast<char*>(&channels),sizeof(int));
		mxFile.read(reinterpret_cast<char*>(&cols),sizeof(int));
		mxFile.read(reinterpret_cast<char*>(&rows),sizeof(int));
		matrix = cv::Mat::zeros(cv::Size(cols,rows),cv::DataType<cv::Vec<T,U> >::type);
		// READ THE CONTENT OF THE MATRIX
		for(int ch=0;ch<channels;++ch){
			for(int y=0;y<matrix.rows;++y){
				for(int x=0;x<matrix.cols;++x){
					mxFile.read(reinterpret_cast<char*>(&(matrix.template at\
						<cv::Vec<T,U> >(y,x)).val[ch]),sizeof(T));
				}
			}
		}
		mxFile.close();
	}
	return matrix;
}
//==============================================================================
/** Reads a 2D-matrix from a binary file (first the dimension of the matrix).
 */
template <typename T,int U>
void Auxiliary<T,U>::bin2matvec(std::vector<cv::Mat_<cv::Vec<T,U> > > &matrix,\
const char* fileName){
	if(!Auxiliary::file_exists(fileName)){
		std::cerr<<"Error opening the file: "<<fileName<<std::endl;
		exit(1);
	}
	std::ifstream mxFile(fileName,std::ios::in | std::ios::binary);
	if(mxFile.is_open()){
		// FIRST READ THE DIMENSIONS OF THE VECTOR
		int vecSize;
		mxFile.read(reinterpret_cast<char*>(&vecSize),sizeof(int));
		for(unsigned s=0;s<vecSize;++s){
			// FIRST READ THE MATRIX SIZE AND ALLOCATE IT
			int cols,rows,channels;
			mxFile.read(reinterpret_cast<char*>(&channels),sizeof(int));
			mxFile.read(reinterpret_cast<char*>(&cols),sizeof(int));
			mxFile.read(reinterpret_cast<char*>(&rows),sizeof(int));
			cv::Mat_<cv::Vec<T,U> > amatrix = cv::Mat::zeros(cv::Size(cols,rows),\
				cv::DataType<cv::Vec<T,U> >::type);
			// READ THE CONTENT OF THE MATRIX
			for(int ch=0;ch<channels;++ch){
				for(int y=0;y<amatrix.rows;++y){
					for(int x=0;x<amatrix.cols;++x){
						mxFile.read(reinterpret_cast<char*>(&(amatrix.template at\
							<cv::Vec<T,U> >(y,x)).val[ch]),sizeof(T));
					}
				}
			}
			matrix.push_back(amatrix.clone());
			amatrix.release();
		}
		mxFile.close();
	}
}
//==============================================================================
/** Convert int to string.
 */
template <typename T,int U>
std::string Auxiliary<T,U>::number2string(T i){
	std::stringstream out;
	out << i;
	return out.str();
}
//==============================================================================
/** A function that transforms the data such that it has zero mean and unit
 * variance: img = (img-mean(img(:)))/std(img(:)).
 */
template <typename T,int U>
void Auxiliary<T,U>::mean0Variance1(cv::Mat_<cv::Vec<T,U> > &mat,\
cv::Mat_<cv::Vec<T,U> > &mean, cv::Mat_<cv::Vec<T,U> > &var){
	// GE THE MEAN AND COVARIANCE
	if(mean.empty() || var.empty()){
		mean = cv::Mat::zeros(cv::Size(mat.cols,1),cv::DataType<cv::Vec<T,U> >::type);
		var  = cv::Mat::zeros(cv::Size(mat.cols,1),cv::DataType<cv::Vec<T,U> >::type);
		for(unsigned c=0;c<mat.cols;++c){
			cv::Mat_<cv::Vec<T,U> > aCol = mat.col(c);
			cv::Scalar smean,stddev;
			cv::meanStdDev(aCol,smean,stddev);
			for(unsigned ch=0;ch<mat.channels();++ch){
				(mean.template at<cv::Vec<T,U> >(0,c)).val[ch] = static_cast<T>\
					(smean.val[ch]);
				(var.template at<cv::Vec<T,U> >(0,c)).val[ch]  = static_cast<T>\
					(stddev.val[ch]);
			}
		}
	}
	for(unsigned r=0;r<mat.rows;++r){
		cv::Mat_<cv::Vec<T,U> > aRow = mat.row(r);
		aRow                         = (aRow - mean);
		cv::Mat_<cv::Vec<T,U> > tmp;
		cv::divide(aRow,var,tmp,1);
		tmp.copyTo(aRow);
		tmp.release();
	}
}
//==============================================================================
/** A function that transforms the data such that it has zero mean and unit
 * variance: img = (img-mean(img(:)))/std(img(:)).
 */
template <typename T,int U>
cv::Mat Auxiliary<T,U>::mean0Variance1(const cv::Mat_<cv::Vec<T,U> > &inmat){
	cv::Mat mat = inmat.clone();
	unsigned rows = mat.rows;
	if(rows != 1){
		mat = mat.reshape(1);
	}
	cv::Scalar mean,stddev;
	cv::meanStdDev(mat,mean,stddev);
	// the matrix is on 1 row now
	cv::Mat_<cv::Vec<T,U> > aRow = mat.row(0);
	for(unsigned c=0;c<mat.cols;++c){
		for(unsigned ch=0;ch<mat.channels();++ch){
			if(stddev.val[ch]){
				(aRow.template at<cv::Vec<T,U> >(0,c)).val[ch] = \
					((aRow.template at<cv::Vec<T,U> >(0,c)).val[ch]-mean.val[ch])/\
					stddev.val[0];
			}
		}
	}
	if(rows != 1){
		mat = mat.reshape(rows);
	}
	return mat;
}
//==============================================================================
/** Normalizes the input image to have all values between 0 and 1.
 */
template <typename T,int U>
cv::Mat Auxiliary<T,U>::normalize01(const cv::Mat_<cv::Vec<T,U> > &inmat){
	cv::Mat mat = inmat.clone();
	double mini=0,maxi=0;
	cv::minMaxLoc(mat,&mini,&maxi);
	if(mini!=0){
		mat = mat-mini;
	}
	cv::minMaxLoc(mat,&mini,&maxi);
	if(maxi!=0){
		mat = mat/maxi;
	}
	return mat;
}
//==============================================================================
/** Fix directory path to not have "/" at the end.
 */
template <typename T,int U>
void Auxiliary<T,U>::fixPath(std::string &path){
	path.erase(path.find_last_not_of(" \n\r\t")+1);
	if(path.substr(path.size()-1,1).compare(PATH_SEP)){
		path.push_back('/');
	}
}
//==============================================================================
/** Lists the files in a directory with a certain extension.
 */
template <typename T,int U>
std::vector<std::string> Auxiliary<T,U>::listDir(std::string &dir,\
const std::string &ext,const std::string &contains){
	DIR *dp;
	struct dirent *dirp;
	Auxiliary::fixPath(dir);
	if((dp=opendir(dir.c_str())) == NULL){
		std::cerr<<"Error("<<errno<<") opening "<<dir<<std::endl;
		std::exit(-1);
	}
	std::vector<std::string> files;
	while((dirp = readdir(dp)) != NULL){
		std::string aname   = std::string(dirp->d_name);
		std::string dirpath = std::string(dir+aname);
		if(ext.empty()){
			if(boost::filesystem::is_directory(boost::filesystem::path(dirpath))\
			&& aname.find(".")!=0){
				std::size_t pos2 = 0;
				if(!contains.empty()){pos2 = aname.find(contains);}
				if(static_cast<int>(pos2)!=-1){
					files.push_back(aname);
				}
			}else if(aname.find(".")!=0){
				std::size_t pos2 = 0;
				if(!contains.empty()){pos2 = aname.find(contains);}
				if(static_cast<int>(pos2)!=-1){
					files.push_back(aname);
				}
			}
		}else{
			std::size_t pos1 = aname.find(ext);
			std::size_t pos2 = 0;
			if(!contains.empty()){pos2 = aname.find(contains);}
			if(static_cast<int>(pos1)!=-1 && static_cast<int>(pos2)!=-1){
				files.push_back(aname);
			}
		}
	}
	std::sort(files.begin(),files.end());
	closedir(dp);
	return files;
}
//==============================================================================
/** Applies a function to every element of a matrix.
 */
template <typename T,int U>
void Auxiliary<T,U>::applyFct2Mat(cv::Mat_<cv::Vec<T,U> > &mat,void *fct\
(cv::Vec<T,U>)){
	for(cv::MatConstIterator_<cv::Vec<T,U> > it=mat.begin();it!=mat.end();++it){
		fct(*it);
	}
}
//==============================================================================
/** Computes the difference between 2 time stamps in seconds.
 */
template <typename T,int U>
double Auxiliary<T,U>::diffclock(const clock_t &clock1,const clock_t &clock2){
	double diffticks = clock1-clock2;
	double diffms    = diffticks/static_cast<double>(CLOCKS_PER_SEC);
	return diffms;
}
//==============================================================================
/** Computes the covariance the the mean of a data matrix.
 */
template <typename T,int U>
void Auxiliary<T,U>::getGaussian(const cv::Mat_<cv::Vec<T,1> > &data,\
cv::Mat &mean,cv::Mat &cov,cv::Mat &invcov,float &determinant){
	// get the mean matrix
	cv::Mat dataClone = data.clone();
	dataClone.convertTo(dataClone,CV_32FC1);
	mean = cv::Mat::zeros(cv::Size(data.cols,1),CV_32FC1);
	for(unsigned r=0;r<data.rows;++r){ // over samples
		cv::add(mean,data.row(r),mean);
	}
	mean = mean*1.0/static_cast<float>(data.rows);
	// compute: data-mean
	for(unsigned r=0;r<dataClone.rows;++r){ // over samples
		dataClone.row(r) = dataClone.row(r)-mean;
	}
	// and the cov matrix
	cov = dataClone.t()*dataClone*1.0/static_cast<float>(data.rows-1);
	for(unsigned c=0;c<cov.cols;++c){
		cov.at<float>(c,c) += 1e-10;
	}
	dataClone.release();
	// determinant of cov:
	cv::Mat eye = cv::Mat::eye(cov.size(),cov.type());
	determinant = cov.dot(eye);
	eye.release();
	// the inverse of the covariance
	invcov = cv::Mat(cov.size(),cov.type());
	cv::invert(cov,invcov);
}
//==============================================================================
/** Evaluate given Gaussian at a sample x.
 */
template <typename T,int U>
float Auxiliary<T,U>::evalGaussian(const cv::Mat_<cv::Vec<T,1> > &sample,\
const cv::Mat &mean,const cv::Mat &cov,const cv::Mat &invcov,float determinant){
	cv::Mat dataClone = sample.clone();
	dataClone.convertTo(dataClone,CV_32FC1);
	dataClone         = dataClone-mean;
	cv::Mat tmp       = dataClone*invcov*dataClone.t();
	dataClone.release();
	float density = 1.0/std::sqrt(std::pow(2.0*M_PI,static_cast<float>(cov.cols))*\
		determinant)*std::exp(-0.5*tmp.at<float>(0,0));
	return density;
}
//==============================================================================
/** Trim spaces from the string start.
 */
template <typename T,int U>
std::string &Auxiliary<T,U>::ltrim(std::string &s){
	s.erase(s.begin(),std::find_if(s.begin(),s.end(),std::not1(std::ptr_fun\
		<int, int>(std::isspace))));
	return s;
}
//==============================================================================
/** Trim spaces from the string end.
 */
template <typename T,int U>
std::string &Auxiliary<T,U>::rtrim(std::string &s){
	s.erase(std::find_if(s.rbegin(),s.rend(),std::not1(std::ptr_fun<int, int>\
		(std::isspace))).base(),s.end());
	return s;
}
//==============================================================================
/** Trim spaces from the both ends of the string.
 */
template <typename T,int U>
std::string &Auxiliary<T,U>::trim(std::string &s){
	return Auxiliary<T,U>::ltrim(Auxiliary<T,U>::rtrim(s));
}
//==============================================================================
/** Print the min and max values in the matrix.
 */
template <typename T,int U>
void Auxiliary<T,U>::printMinMax(const cv::Mat_<cv::Vec<T,U> > &mat){
	double mini=0.0,maxi=0.0;
	cv::minMaxLoc(mat,&mini,&maxi);
	std::cout<<"[Auxiliary::printMinMax] min="<<mini<<" max="<<maxi<<std::endl;
}
//==============================================================================
/** For searching in a vector of matrix values.
 */
template <typename T,int U>
bool Auxiliary<T,U>::isEqual(const cv::Vec<T,U> &val1,const cv::Vec<T,U> &val2){
	bool equal = true;
	for(unsigned i=0;i<U;++i){
		if(val1.val[i]!=val2.val[i]){
			equal = false;
			break;
		}
	}
	return equal;
}
//==============================================================================
/** Simple if-else to get the quadrant of the angle.
 */
template <typename T,int U>
unsigned Auxiliary<T,U>::angle2quadrant(float angle){
	assert(angle<=2.0*M_PI);
	assert(angle>=0.0);
	unsigned quadrant;
	if(angle<=M_PI/2.0){
		quadrant = 0;
	}else if(angle<=M_PI){
		quadrant = 1;
	}else if(angle<=3.0*M_PI/2.0){
		quadrant = 2;
	}else if(angle<=2.0*M_PI){
		quadrant = 3;
	}
	return quadrant;
}
//==============================================================================
/** Sum over matrix elements.
 */
template <typename T,int U>
typename cv::Vec<T,U> Auxiliary<T,U>::sum(const cv::Mat_<cv::Vec<T,U> > &mat){
	cv::Vec<T,U> sum;
	typename cv::Mat_<cv::Vec<T,U> >::const_iterator it1 = mat.begin();
	typename cv::Mat_<cv::Vec<T,U> >::const_iterator it2 = mat.end();
	for(typename cv::Mat_<cv::Vec<T,U> >::const_iterator it=it1;it!=it2;++it){
		sum += (*it);
	}
	return sum;
}
//==============================================================================
/** Displays a matrix.
 */
template <typename T,int U>
void Auxiliary<T,U>::display(const cv::Mat_<cv::Vec<T,U> > &mat){
	// [0] First re-normalize the channels between 0-255
	double minVal, maxVal;
	cv::minMaxLoc(mat,&minVal,&maxVal);
	cv::Mat draw;
	mat.convertTo(draw,CV_8UC3,255.0/(maxVal-minVal),-minVal*255.0/(maxVal-minVal));
	// [3] Resize it to a visible size
	float step = 500.0/static_cast<float>(std::max(draw.cols,draw.rows));
	cv::Size size(draw.cols*step,draw.rows*step);
	cv::resize(draw,draw,size);
	// [4] Now display the larger image
	cv::imshow("display",draw);
	cv::waitKey(10);
	draw.release();
}
//==============================================================================
/** Convolves an image with a filter -- if more channels, each is processed separately.
 */
template <typename T,int U>
cv::Mat_<cv::Vec<T,U> > Auxiliary<T,U>::conv2(const cv::Mat_<cv::Vec<T,U> > &image,\
const cv::Mat_<cv::Vec<T,U> > &kernel){
	// [0] First make border half of filter size
	std::vector<cv::Mat> split;
	cv::split(image,split);
	for(unsigned c=0;c<split.size();++c){
		cv::copyMakeBorder(split[c],split[c],image.rows/2,image.rows/2,image.cols/2,\
			image.cols/2,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
		// [2] Then convolve the image
		cv::filter2D(split[c],split[c],CV_32FC1,kernel);
	}
	// [3] Merge it back into 1 image
	cv::Mat_<cv::Vec<T,U> > result;
	cv::merge(split,result);
	return result;
}
//==============================================================================
/** Computes log-of-sum in term of logs:
 * if(c>a):
 * 			log(a+c) = log(a) + log( 1 + b^(log(c)-log(a)) )
 * else:
 * 			log(a+c) = log(c) + log( 1 + b^(log(a)-log(c)) ).
 */
template <typename T,int U>
float Auxiliary<T,U>::logOfSum(float loga,float logc){
	if(!loga){return logc;}
	if(!logc){return loga;}
	float realLogA,realLogC;
	if(loga>logc){
		// log(a+c) = log(a) + log( 1 + b^(log(c)-log(a)) )
		realLogA = loga;
		realLogC = logc;
	}else{
		// log(a+c) = log(c) + log( 1 + b^(log(a)-log(c)) )
		realLogA = logc;
		realLogC = loga;
	}
	return realLogA+std::log(1.0+std::pow(M_E,(realLogC-realLogA)));
}
//==============================================================================
/** Estimates the Gaussian kernel between two input samples given the sigma.
 */
template <typename T,int U>
float Auxiliary<T,U>::gaussianKernelDensity(const cv::Mat *sampleX,const \
cv::Mat *sampleXn,float sigma){
	double difference = ((*sampleX)-(*sampleXn)).dot((*sampleX)-(*sampleXn));
	return 1.0/(std::sqrt(2.0*M_PI)*sigma)*std::exp(-difference/(2.0*sigma*sigma));
}
//==============================================================================
/** Throws a standard exception.
 */
template <typename T,int U>
void Auxiliary<T,U>::except(const char *message){
	throw helperException(message);
}
//==============================================================================
/** Checks the memory usage.
 */
template <typename T,int U>
std::string Auxiliary<T,U>::memusage(){
	long pageSize = sysconf(_SC_PAGE_SIZE);
	char buf[100];
	sprintf(buf,"/proc/%u/statm",(unsigned)getpid());
	FILE* fp = fopen(buf,"r");
	int usage;
	if(fp){
		unsigned size;
		fscanf(fp,"%u",&size);
		usage = (int)size*(int)pageSize;
		fclose(fp);
	}
	const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB"};
	std::string sign;
	if(usage<0){
		sign  = "-";
		usage = -usage;
	}
	std::vector<int> vec;
	for(unsigned i=0 ;i<7;++i){
		vec.push_back(usage % 1024);
		usage /= 1024;
	}
	if(usage!=0){return "Out of range";}
	int u = 6;
	while((vec[u] == 0) && (u > 0)) u--;
	std::string val = Auxiliary<int,1>::number2string(vec[u]);
	if(u>0){
		val += "." + Auxiliary<int,1>::number2string(vec[u-1]);
	}
	return sign+val+std::string(units[u]);
}
//==============================================================================
/** Splits a string on a character and returns the wanted bit.
 */
template <typename T,int U>
std::string Auxiliary<T,U>::getStringSplit(const std::string &thestring,\
const char *needle,int pos){
	std::deque<std::string> splits = Auxiliary<T,U>::splitLine\
		(const_cast<char*>(thestring.c_str()),needle[0]);
	int ssize = splits.size();
	assert(pos<ssize);
	if(pos<0){
		pos = ssize+pos;
		assert(pos>=0);
	}
	return splits[pos];
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif // AUXILIARY_CPP_















































