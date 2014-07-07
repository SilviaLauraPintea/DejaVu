/* SIFTlike.h
 * Author: Silvia-Laura Pintea
 */
#ifndef SIFTLIKE_H_
#define SIFTLIKE_H_
#include <Auxiliary.h>
//==============================================================================
/** Extracts a certain SIFT type. For now only CSIFT or Gray-SIFT.
 */
class SIFTlike {
	public:
		SIFTlike(unsigned step,unsigned patchsize,const std::vector<unsigned> \
		&bounds=std::vector<unsigned>()):patchsize_(patchsize),step_(step),\
		bounds_(bounds){};
		virtual ~SIFTlike(){};
		//----------------------------------------------------------------------
		/** Extracts a certain SIFT descriptor.
		 */
		void getGray(const IplImage *image,std::vector<IplImage*> &feat,\
			std::vector<cv::Point2f> &points);
		/** Extracts a certain SIFT descriptor.
		 */
		void getOpponent(const IplImage *image,std::vector<IplImage*> &feat,\
			std::vector<cv::Point2f> &points);
		/** Just computes SIFT over a 1 channel image.
		 */
		cv::Mat oneChannelSIFT(const cv::Mat &input,std::vector<cv::Point2f> &points,\
			bool verbose=true);
		/** Puts descriptors into a vector of images.
		 */
		void desc2iplvect(std::vector<IplImage*> &feat,const cv::Mat &descriptors,\
			const std::vector<cv::Point2f> &points,const cv::Size &imsize);
		//----------------------------------------------------------------------
		//---SETTERS----------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned patchsize() const {return this->patchsize_;}
		unsigned step() const {return this->step_;}
		std::vector<unsigned> bounds() const {return this->bounds_;}
		//----------------------------------------------------------------------
		//---GETTERS----------------------------------------------------------
		//----------------------------------------------------------------------
		void patchsize(unsigned patchsize){this->patchsize_ = patchsize;}
		void step(unsigned step){this->step_ = step;}
		void bounds(const std::vector<unsigned> &bounds){this->bounds_ = bounds;}
		//----------------------------------------------------------------------
	protected:
		/** @var patchsize_
		 * The patch size for SIFT
		 */
		unsigned patchsize_;
		/** @var step_
		 * The step for extracting SIFT
		 */
		unsigned step_;
		/** @var bounds_
		 * The image boundaries.
		 */
		std::vector<unsigned> bounds_;
		//----------------------------------------------------------------------
	private:
		DISALLOW_COPY_AND_ASSIGN(SIFTlike);
};
//==============================================================================
#endif /* SIFTLIKE_H_ */
