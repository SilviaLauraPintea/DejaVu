/* StructuredRFdetector.h
 * Author: Silvia-Laura Pintea
 */
#ifndef STRUCTUREDRFDETECTOR_H_
#define STRUCTUREDRFDETECTOR_H_
#pragma once
#include "StructuredRF.h"
#include "Puzzle.h"
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,\
class T,class F,class N,class U>
class StructuredRFdetector{
	public:
		StructuredRFdetector(StructuredRF<L,M,T,F,N,U> *pRF,int w,int h,\
			unsigned cls,unsigned labW,unsigned labH,Puzzle<PuzzlePatch>::METHOD \
			method,unsigned step);
		virtual ~StructuredRFdetector(){
			// no forest delete because it's not created with new
		};
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Gets an input image and returns a detection image (RF regression).
		 * Given a set of predicted leafs for current pixel, get the final label:
		 * Simple: [1] Just get the most voted pixel label per position.
		 * Puzzle: [2] Optimized the patch selection label \cite{kontschider}.
		 */
		virtual void detectColor(const F* features,cv::Mat &imgDetect,const \
			std::vector<const T*> &patches,const cv::Size &imsize) const;
		/** Scales the image at a number of sizes and it labels each scale [?].
		 */
		virtual void detectPyramid(const std::string &imname,const std::string \
			&path2img,const std::string &path2feat,const std::string &ext,\
			const std::vector<float> &pyramid,std::vector<cv::Mat> &vImgDetect) const;
		/** Extracts or loads the test features for the current test image.
		 */
		virtual std::vector<std::vector<IplImage*> > getFeatures(const std::string \
			&path2img,const std::vector<std::string> &path2feat,const std::vector\
			<float> &pyr,std::vector<std::vector<const T*> > &patches,\
			cv::Size &origsize,bool showWhere=false) const;
		/** Loads the test features from file for the current test image.
		 */
		virtual std::vector<std::vector<IplImage*> > loadFeatures(const std::vector\
			<std::string> &path2feat,std::vector<std::vector<const T*> > &patches,\
			bool showWhere=false) const;
		/** Loads the test features from file for the current test image.
		 */
		virtual void saveFeatures(const std::vector<std::string> &path2feat,\
			const std::vector<std::vector<IplImage*> > &vImg,const std::vector\
			<std::vector<const T*> > &patches) const;
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned step() const {return this->step_;}
		const StructuredRF<L,M,T,F,N,U>* forest() const {return this->forest_;};
		int width() const {return this->width_;}
		int height() const {return this->height_;}
		unsigned labW() const {return this->labW_;}
		unsigned labH() const {return this->labH_;}
		unsigned noCls() const {return this->noCls_;}
		Puzzle<PuzzlePatch>::METHOD method() const {return this->method_;}
		unsigned maxsize() const {return this->maxsize_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void step(unsigned step){this->step_ = step;}
		void width(int width){this->width_ = width;}
		void height(int height){this->height_ = height;}
		void labH(unsigned labH){this->labH_ = labH;}
		void labW(unsigned labW){this->labW_ = labW;}
		void noCls(unsigned noCls){this->noCls_ = noCls;}
		void forest(StructuredRF<L,M,T,F,N,U> *forest){
			// shallow copy (no new)
			this->forest_ = forest;
		};
		void method(Puzzle<PuzzlePatch>::METHOD method){this->method_ = method;}
		void maxsize(unsigned maxsize){this->maxsize_ = maxsize;}
		//----------------------------------------------------------------------
	private:
		void detectColor(IplImage *img, std::vector<IplImage*> &imgDetect,\
			std::vector<float> &ratios);
	protected:
		/** @var maxsize_
		 * The maximum image size.
		 */
		unsigned maxsize_;
		/** @var step_
		 * The step of the grid for sampling.
		 */
		unsigned step_;
		/** @var forest_
		 * Pointer to the trained forest.
		 */
		StructuredRF<L,M,T,F,N,U>* forest_;
		/** @var width_
		 * The width of the feature patch.
		 */
		int width_;
		/** @var height_
		 * The height of the feature patch.
		 */
		int height_;
		/** @var labW_
		 * The width of the label patch.
		 */
		unsigned labW_;
		/** @var labH_
		 * The height of the feature patch.
		 */
		unsigned labH_;
		/** @var noCls_
		 * The number of classes.
		 */
		unsigned noCls_;
		/** @var method_
		 * The label selection method.
		 */
		Puzzle<PuzzlePatch>::METHOD method_;
		/** @var storefeat_
		 * If we store the features or not.
		 */
		bool storefeat_;
	private:
		DISALLOW_COPY_AND_ASSIGN(StructuredRFdetector);
};
//==============================================================================
#endif /* STRUCTUREDRFDETECTOR_H_ */
#include "StructuredRFdetector.cpp"
