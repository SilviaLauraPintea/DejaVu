/* Auxiliary.h
 * Author: Silvia-Laura Pintea
 */
#ifndef AUXILIARY_H_
#define AUXILIARY_H_
//==============================================================================
#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <omp.h>
#include <set>
#include <time.h>
#include <assert.h>
#include <tr1/memory>
#include <opencv2/opencv.hpp>
#include <limits>
#include <cmath>
#include <exception>
#include <cctype>
//==============================================================================
// A macro to disallow the copy constructor and operator= functions.
// This should be used in the private: declarations for a class.
#define DO_PRAGMA 1
#define SMALL 1e-20
#define PATH_SEP "/"
#undef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName)\
	TypeName(const TypeName&);\
	TypeName& operator=(const TypeName&)
//==============================================================================
struct helperException:public std::exception{
	public:
		helperException(){this->message_ = NULL;}
		helperException(const char* message){
			this->message_ = message;
		}
		virtual const char* what() const throw() {return this->message_;}
		//----------------------------------------------------------------------
		//---GETTERS & SETTERS--------------------------------------------------
		//----------------------------------------------------------------------
		void message(const char* message) throw() {
			this->message_ = message;
		}
		const char* message() const throw() {return this->message_;}
	private:
		const char* message_;
};
//==============================================================================
/** Class containing all auxiliary, helpful functions needed for outputs and other
 * administrative bits.
 */
template <typename T,int U>
class Auxiliary{
	public:
		/** Splits a string on a character and returns the wanted bit.
		 */
		static std::string getStringSplit(const std::string &thestring,\
			const char *needle,int pos);
		/** Splits a string on a specific character and does not include that
		 * character in the result.
		 */
		static std::deque<std::string> splitLine(char *str,char charct);
		/** Check if a file exists.
		 */
		static int file_exists(const char *fileName,bool create = false);
		/** Converts a pointer to an IplImage to an OpenCV Mat.
		 */
		static cv::Mat ipl2mat(IplImage *ipl_image);
		/** Converts an OpenCV Mat to a pointer to an IplImage.
		 */
		static IplImage* mat2ipl(const cv::Mat_<cv::Vec<T,U> > &image);
		/** Write a 2D-matrix to a text file (first row is the dimension of the matrix).
		 */
		static void mat2txt(cv::Mat_<cv::Vec<T,U> > &matrix,\
			char *fileName,bool append = false);
		/** Reads a 2D-matrix from a text file (first row is the dimension of the matrix).
		 */
		static void txt2mat(cv::Mat_<cv::Vec<T,U> > &matrix,char *fileName);
		/** Write a 2D-matrix to a binary file (first the dimension of the matrix).
		 */
		static void mat2bin(const cv::Mat_<cv::Vec<T,U> > &matrix,\
			const char *fileName,bool append = false);
		/** Reads a 2D-matrix from a binary file (first the dimension of the matrix).
		 */
		static void bin2mat(cv::Mat_<cv::Vec<T,U> > &matrix,const char *fileName);
		/** Reads a 2D-matrix from a binary file (first the dimension of the matrix).
		 */
		static cv::Mat_<cv::Vec<T,U> > bin2mat(const char* fileName);
		/** Convert int to string.
		 */
		static std::string number2string(T i);
		/** A function that transforms the data such that it has zero mean and unit
		 * variance: img = (img-mean(img(:)))/std(img(:)).
		 */
		static cv::Mat mean0Variance1(const cv::Mat_<cv::Vec<T,U> > &inmat);
		/** Mean and stddev for matrices.
		 */
		static void mean0Variance1(cv::Mat_<cv::Vec<T,U> > &mat,\
			cv::Mat_<cv::Vec<T,U> > &mean, cv::Mat_<cv::Vec<T,U> > &var);
		/** Normalizes the input image to have all values between 0 and 1.
		 */
		static cv::Mat normalize01(const cv::Mat_<cv::Vec<T,U> > &inmat);
		/** Write a vector of 2D-matrices to a binary file (first the dimension
		 * of the matrix).
		 */
		static void matvec2bin(const std::vector<cv::Mat_<cv::Vec<T,U> > > &matrix,\
			const char *fileName,bool append = false);
		/** Reads a vector of 2D-matrices from a binary file (first the dimension
		 * of the matrix).
		 */
		static void bin2matvec(std::vector<cv::Mat_<cv::Vec<T,U> > > &matrix,\
			const char *fileName);
		/** Lists the files in a directory with a certain extension.
		 */
		static std::vector<std::string> listDir(std::string &dir,\
			const std::string &ext="",const std::string &contains="");
		/** Fix directory path to not have "/" at the end.
		 */
		static void fixPath(std::string &path);
		/** Applies a function to every element of a matrix.
		 */
		static void applyFct2Mat(cv::Mat_<cv::Vec<T,U> > &mat,void *fct\
			(cv::Vec<T,U>));
		/** Computes the difference between 2 time stamps in seconds.
		 */
		static double diffclock(const clock_t &clock1,const clock_t &clock2);
		/** Computes the covariance the the mean of a data matrix.
		 */
		static void getGaussian(const cv::Mat_<cv::Vec<T,1> > &data,\
			cv::Mat &mean,cv::Mat &cov,cv::Mat &invcov,float &determinant);
		/** Evaluate given Gaussian at a sample x.
		 */
		static float evalGaussian(const cv::Mat_<cv::Vec<T,1> > &sample,\
			const cv::Mat &mean,const cv::Mat &cov,const cv::Mat &invcov,\
			float determinant);
		/** Trim spaces from the string start.
		 */
		static std::string &ltrim(std::string &s);
		/** Trim spaces from the string end.
		 */
		static std::string &rtrim(std::string &s);
		/** Trim spaces from the both ends of the string.
		 */
		static std::string &trim(std::string &s);
		/** Print the min and max values in the matrix.
		 */
		static void printMinMax(const cv::Mat_<cv::Vec<T,U> > &mat);
		/** For searching in a vector of matrix values.
		 */
		static bool isEqual(const cv::Vec<T,U> &val1,const cv::Vec<T,U> &val2);
		/** Simple if-else to get the quadrant of the angle.
		 */
		static unsigned angle2quadrant(float angle);
		/** Sum over matrix elements.
		 */
		static typename cv::Vec<T,U> sum(const cv::Mat_<cv::Vec<T,U> > &mat);
		/** Displays a matrix.
		 */
		static void display(const cv::Mat_<cv::Vec<T,U> > &mat);
		/** Convolves an image with a filter.
		 */
		static cv::Mat_<cv::Vec<T,U> > conv2(const cv::Mat_<cv::Vec<T,U> > &image,\
			const cv::Mat_<cv::Vec<T,U> > &kernel);
		/** Computes log-of-sum in term of logs:
		 * if(c>a):
		 * 			log(a+c) = log(a) + log( 1 + b^(log(c)-log(a)) )
		 * else:
		 * 			log(a+c) = log(c) + log( 1 + b^(log(a)-log(c)) )
		 */
		static float logOfSum(float loga,float logc);
		/** Estimates the Gaussian kernel between two input samples given the sigma.
		 */
		static float gaussianKernelDensity(const cv::Mat *sampleX,const \
			cv::Mat *sampleXn,float sigma);
		/** Throws an exception with a given message.
		 */
		static void except(const char *message);
		/** Get the mem usage on linux.
		 */
		static std::string memusage();
	private:
		DISALLOW_COPY_AND_ASSIGN(Auxiliary);
};
//==============================================================================
//==============================================================================
//==============================================================================
/** Equal operator for vectors.
 */
template <typename T,int U>
bool operator==(const cv::Vec<T,U> &l,const cv::Vec<T,U> &r){
	for(unsigned i=0;i<U;++i){
		if(l.val[i]!=r.val[i]){
			return false;
		}
	}
	return true;
}
//==============================================================================
/** Writing a cv rectangle to the output to the output stream.
 */
inline std::ostream& operator<<(std::ostream &output,const cv::Rect &r){
	output<<"[("<<r.x<<","<<r.y<<"),"<<r.width<<","<<r.height<<"]";
	return output;
}
//==============================================================================
/** Writing a cv::Size to the output to the output stream.
 */
inline std::ostream& operator<<(std::ostream &output,const cv::Size &s){
	return output<<"["<<s.width<<","<<s.height<<"]";
}
//==============================================================================
/** Writing a 2D point to the output stream.
 */
inline std::ostream& operator<<(std::ostream &output,const cv::Point &p){
	output<<"("<<p.x<<","<<p.y<<")";
	return output;
}
//==============================================================================
/** Writing a 2D point to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream &output,const cv::Point_<T> &p){
	output<<"("<<p.x<<","<<p.y<<")";
	return output;
}
//==============================================================================
/** Writing a 3D point to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream &output,const cv::Point3_<T> &p){
	output<<"("<<p.x<<","<<p.y<<","<<p.z<<")";
	return output;
}
//==============================================================================
/** Writing a matrix to the output stream.
 */
template <typename T,int U>
std::ostream& operator<<(std::ostream &output,cv::Mat_<cv::Vec<T,U> > &m){
	for(unsigned ch=0;ch<m.channels();++ch){
		output<<"m(:,:,"<<ch<<")="<<std::endl;
		for(unsigned r=0;r<m.rows;++r){
			for(unsigned c=0;c<m.cols;++c){
				output<<static_cast<T>((m.template at<cv::Vec<T,U> >\
					(r,c)).val[ch])<<" ";
			}
			output<<std::endl;
		}
		output<<std::endl;
	}
	return output;
}
//==============================================================================
/** Structure for sorting pixel values.
 */
struct vec3bCompare{
	bool operator()(const cv::Vec3b &lhs,const cv::Vec3b &rhs) const{
		if(lhs.val[0]!=rhs.val[0]){
			return lhs.val[0]<rhs.val[0];
		}else if(lhs.val[1]!=rhs.val[1]){
			return lhs.val[1]<rhs.val[1];
		}else {
			return lhs.val[2]<rhs.val[2];
		}
	}
};
//==============================================================================
/** Writing a vector of vector to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream &output,const std::vector<std::vector<T> > \
&vect){
	for(typename std::vector<std::vector<T> >::const_iterator it1=vect.begin();\
	it1<vect.end();++it1){
		for(typename std::vector<T>::const_iterator it2=it1->begin();it2<\
		it1->end();++it2){
			output<<(*it2)<<" ";
		}
		output<<std::endl;
	}
	return output;
}
//==============================================================================
/** Writing a vector of vector to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream &output,const std::vector<T> &vect){
	for(typename std::vector<T>::const_iterator it=vect.begin();it<\
	vect.end();++it){
		output<<(*it)<<" ";
	}
	output<<std::endl;
	return output;
}
//==============================================================================
/** Writing a vector of vector to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream &output,const std::deque<std::deque<T> > \
&queue){
	for(typename std::deque<std::deque<T> >::const_iterator it1=queue.begin();\
	it1<queue.end();++it1){
		for(typename std::deque<T>::const_iterator it2=it1->begin();it2<\
		it1->end();++it2){
			output<<(*it2)<<" ";
		}
		output<<std::endl;
	}
	return output;
}
//==============================================================================
/** Writing a vector of vector to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream &output,const std::deque<T> &queue){
	for(typename std::deque<T>::const_iterator it=queue.begin();it<\
	queue.end();++it){
		output<<(*it)<<" ";
	}
	output<<std::endl;
	return output;
}
//==============================================================================
/** Writing an OpenCV vector to the output.
 */
template <typename T,int U>
std::ostream& operator<<(std::ostream &output,const cv::Vec<T,U> &vect){
	for(unsigned i=0;i<U;++i){
		output<<vect.val[i]<<" ";
	}
	output<<std::endl;
	return output;
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif /* AUXILIARY_H_ */
#include "Auxiliary.cpp"

