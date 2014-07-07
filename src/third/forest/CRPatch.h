/* Author: Juergen Gall,BIWI,ETH Zurich
* Email: gall@vision.ee.ethz.ch
*/
#ifndef CRPATCH_H_
#define CRPATCH_H_
#define _copysign copysign
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "HoG.h"
#include <Auxiliary.h>
//==============================================================================
/** Auxiliary structure.
 */
struct Index{
	public:
		bool operator<(const Index& a) const{return val_<a.val_;}
		long double val() const {return this->val_;}
		unsigned index() const {return this->index_;}
		void val(long double val){this->val_ = val;}
		void index(unsigned index){this->index_ = index;}
	protected:
		long double val_;
		unsigned int index_;
};
//==============================================================================
/** structure for image patch.
 */
struct PatchFeature{
	public:
		PatchFeature(){}
		~PatchFeature(){
			for(std::size_t i=0; i<this->vPatch_.size();++i){
				// created with: cvCloneMat
				if(this->vPatch_[i]){
					cvReleaseMat(&this->vPatch_[i]);
					this->vPatch_[i] = NULL;
				}
			}
			this->vPatch_.clear();
			this->center_.clear();
		}
		void print() const{
			std::cout << roi_.x << " " << roi_.y << " " << roi_.width << " " <<\
				roi_.height;
			for(unsigned int i=0; i<center_.size(); ++i){
				std::cout << " " << center_[i].x << " " << center_[i].y;
				std::cout << std::endl;
			}
		}
		void show(int delay) const;
		//----------------------------------------------------------------------
		/** Getters for members.
		 */
		CvRect roi() const {return this->roi_;};
		std::vector<CvPoint> center() const {return this->center_;};
		CvPoint center(unsigned pos) const {return this->center_[pos];};
		std::vector<CvMat*> vPatch() const {return this->vPatch_;};
		CvMat* vPatch(unsigned pos) const {return this->vPatch_[pos];};
		/** Setters for members.
		 */
		void centerResize(unsigned size){this->center_.resize(size);};
		void roi(const CvRect &roi){this->roi_ = roi;};
		void center(const std::vector<CvPoint> &center){this->center_ = center;};
		void center(unsigned pos,const CvPoint &pt){this->center_[pos] = pt;};
		void vPatchResize(unsigned size){this->vPatch_.resize(size);}
		void vPatch(const std::vector<CvMat*> &vPatch){
			for(std::size_t i=0; i<this->vPatch_.size();++i){
				// created with: cvCloneMat
				if(this->vPatch_[i]){
					cvReleaseMat(&this->vPatch_[i]);
					this->vPatch_[i] = NULL;
				}
			}
			this->vPatch_.clear();
			for(std::vector<CvMat*>::const_iterator i=vPatch.begin();i!=vPatch.end();++i){
				this->vPatch_.push_back(cvCloneMat(*i));
			}
		};
		void vPatch(unsigned pos,const CvMat* mat){
			if(this->vPatch_[pos]){
				// created with: cvCloneMat
				cvReleaseMat(&this->vPatch_[pos]);
				this->vPatch_[pos] = NULL;
			}
			std::vector<CvMat*>::iterator it = (this->vPatch_.begin())+pos;
			this->vPatch_.insert(it,cvCloneMat(mat));
		};
		//----------------------------------------------------------------------
	protected:
		CvRect roi_;
		std::vector<CvPoint> center_;
		std::vector<CvMat*> vPatch_;
	private:
		DISALLOW_COPY_AND_ASSIGN(PatchFeature);
};
//==============================================================================
//==============================================================================
//==============================================================================
template <class T>
class CRPatch{
	public:
		CRPatch(CvRNG* pRNG,int w,int h,int num_l):cvRNG_(pRNG),width_(w),\
		height_(h){
			vLPatches_.resize(num_l);
		}
		virtual ~CRPatch(){
			// created with: new HoG
			for(std::size_t s=0;s<this->vLPatches_.size();++s){
				// created with: push_back(new T)
				for(std::size_t t=0;t<this->vLPatches_[s].size();++t){
					delete this->vLPatches_[s][t];
					this->vLPatches_[s][t] = NULL;
				}
				this->vLPatches_[s].clear();
			}
			this->vLPatches_.clear();
		}
		/** Extract patches from image.
		 */
		void extractPatches(IplImage *img,unsigned int n,int label,CvRect* box = 0,\
			std::vector<CvPoint>* vCenter = 0);
		/** Extract features from image.
		 */
		static void extractFeatureChannels32(IplImage *img,std::vector<IplImage*> \
			&vImg,unsigned patchsize=5);
		static void extractFeatureChannels9(IplImage *img,std::vector<IplImage*> \
			&vImg,unsigned patchsize);
		static void maxfilt(uchar *data,uchar *maxvalues,unsigned int step,\
			unsigned int size,unsigned int width);
		static void maxfilt(uchar *data,unsigned int step,unsigned int size,\
			unsigned int width);
		static void minfilt(uchar *data,uchar *minvalues,unsigned int step,\
			unsigned int size,unsigned int width);
		static void minfilt(uchar *data,unsigned int step,unsigned int size,\
			unsigned int width);
		static void maxminfilt(uchar *data,uchar *maxvalues,uchar *minvalues,\
			unsigned int step,unsigned int size,unsigned int width);
		static void maxfilt(IplImage *src,unsigned int width);
		static void maxfilt(IplImage *src,IplImage *dst,unsigned int width);
		static void minfilt(IplImage *src,unsigned int width);
		static void minfilt(IplImage *src,IplImage *dst,unsigned int width);
		virtual unsigned getPatchChannels() const;
		//----------------------------------------------------------------------
		/** Getters for the members.
		 */
		int width() const {return this->width_;}
		int height() const {return this->height_;}
		std::vector<std::vector<T*> > vLPatches() const {return this->vLPatches_;};
		std::vector<T*> vLPatches(unsigned l) const {return this->vLPatches_[l];};
		T* vLPatches(unsigned l,unsigned p) const {
			return this->vLPatches_[l][p];
		};
		CvRNG *cvRNG() const {return this->cvRNG_;};
		/** Setters for the members.
		 */
		void width(int width){this->width_ = width;}
		void height(int height) const{this->height_ = height;}
		void vLPatches(const std::vector<std::vector<T*> > &vLPatches){
			for(std::size_t s=0;s<this->vLPatches_.size();++s){
				// created with: push_back(new T)
				for(std::size_t t=0;t<this->vLPatches_[s].size();++t){
					delete this->vLPatches_[s][t];
					this->vLPatches_[s][t] = NULL;
				}
				this->vLPatches_[s].clear();
			}
			this->vLPatches_.clear();
			this->vLPatches_.resize(vLPatches.size());
			for(unsigned l=0;l<vLPatches.size();++l){
				this->vLPatches_[l].resize(vLPatches[l].size());
				for(unsigned p=0;p<vLPatches[l].size();++p){
					this->vLPatches_[l][p] = new T(*vLPatches[l][p]);
				}
			}
		};
		void vLPatches(unsigned l,unsigned p,const T* tmp){
			if(this->vLPatches_[l][p]){
				// created with: new T
				delete this->vLPatches_[l][p];
				this->vLPatches_[l][p] = NULL;
			}
			this->vLPatches_[l][p] = new T(*tmp);
		};
		void cvRNG(CvRNG *cvRNG){this->cvRNG_ = cvRNG;};
		//----------------------------------------------------------------------
	protected:
		std::vector<std::vector<T*> > vLPatches_;
		CvRNG *cvRNG_;
		int width_;
		int height_;
	private:
		DISALLOW_COPY_AND_ASSIGN(CRPatch);
};
//==============================================================================
#endif // CRPATCH_H_
#include "CRPatch.cpp"

