/* Author: Juergen Gall, BIWI, ETH Zurich
* Email: gall@vision.ee.ethz.ch
*/
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
//==============================================================================
class HoG {
	public:
		HoG(unsigned patchsize=5);
		~HoG(){
			if(Gauss_){
				// created with: cvCreateMat
				cvReleaseMat(&Gauss_);Gauss_ = NULL;
			}
			if(ptGauss_){
				// created with: new float []
				delete [] ptGauss_;ptGauss_ = NULL;
			}
		}
		void extractOBin(IplImage *Iorient, IplImage *Imagn,\
			std::vector<IplImage*>& out, int off);
	private:
		void calcHoGBin(uchar* ptOrient, uchar* ptMagn, int step, float* desc);
		void binning(float v, float w, float* desc, int maxb);
	private:
		int bins_;
		float binsize_;
		int g_w_;
		CvMat* Gauss_;
		// Gauss as vector
		float* ptGauss_;
};
//==============================================================================
inline void HoG::calcHoGBin(uchar* ptOrient, uchar* ptMagn, int step, float* desc) {
	for(int i=0; i<bins_;i++)
		desc[i]=0;

	uchar* ptO = &ptOrient[0];
	uchar* ptM = &ptMagn[0];
	int i=0;
	for(int y=0;y<g_w_; ++y, ptO+=step, ptM+=step) {
		for(int x=0;x<g_w_; ++x, ++i) {
			binning((float)ptO[x]/binsize_, (float)ptM[x] * ptGauss_[i], desc, bins_);
		}
	}
}
//==============================================================================
inline void HoG::binning(float v, float w, float* desc, int maxb) {
	int bin1 = int(v);
	int bin2;
	float delta = v-bin1-0.5f;
	if(delta<0) {
		bin2 = bin1 < 1 ? maxb-1 : bin1-1; 
		delta = -delta;
	} else
		bin2 = bin1 < maxb-1 ? bin1+1 : 0; 
	desc[bin1] += (1-delta)*w;
	desc[bin2] += delta*w;
}
//==============================================================================

