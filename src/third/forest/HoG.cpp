/*Author: Juergen Gall, BIWI, ETH Zurich
* Email: gall@vision.ee.ethz.ch
*/
#include <vector>
#include <iostream>
#include "HoG.h"
using namespace std;
//==============================================================================
HoG::HoG(unsigned patchsize) {
	bins_        = 9;
	binsize_     = (3.14159265f*80.0f)/float(bins_);

	g_w_         = patchsize;
	Gauss_       = cvCreateMat( g_w_, g_w_, CV_32FC1 );
	float a      = -(g_w_-1)/2.0;
	float sigma2 = 2*(0.5*g_w_)*(0.5*g_w_);
	float count  = 0;
	for(int x    = 0; x<g_w_; ++x) {
		for(int y = 0; y<g_w_; ++y) {
			float tmp = exp(-( (a+x)*(a+x)+(a+y)*(a+y) )/sigma2);
			count += tmp;
			cvSet2D( Gauss_, x, y, cvScalar(tmp) );
		}
	}
	cvConvertScale( Gauss_, Gauss_, 1.0/count);

	ptGauss_ = new float[g_w_*g_w_];
	int i = 0;
	for(int y = 0; y<g_w_; ++y)
		for(int x = 0; x<g_w_; ++x)
			ptGauss_[i++] = (float)cvmGet( Gauss_, x, y );

}
//==============================================================================
void HoG::extractOBin(IplImage *Iorient, IplImage *Imagn, std::vector<IplImage*> \
&out, int off) {
	float* desc = new float[bins_];
	// reset output image (border=0) and get pointers
	uchar** ptOut     = new uchar*[bins_];
	uchar** ptOut_row = new uchar*[bins_];
	for(int k=off; k<bins_+off; ++k) {
		cvSetZero( out[k] );
		cvGetRawData( out[k], (uchar**)&(ptOut[k-off]));
	}
	// get pointers to orientation, magnitude
	int step;
	uchar* ptOrient;
	uchar* ptOrient_row;
	cvGetRawData( Iorient, (uchar**)&(ptOrient), &step);
	step /= sizeof(ptOrient[0]);
	uchar* ptMagn;
	uchar* ptMagn_row;
	cvGetRawData( Imagn, (uchar**)&(ptMagn));
	int off_w = int(g_w_/2.0);
	for(int l=0; l<bins_; ++l)
		ptOut[l] += off_w*step;
	for(int y=0;y<Iorient->height-g_w_; y++, ptMagn+=step, ptOrient+=step) {
		// Get row pointers
		ptOrient_row = &ptOrient[0];
		ptMagn_row = &ptMagn[0];
		for(int l=0; l<bins_; ++l)
			ptOut_row[l] = &ptOut[l][0]+off_w;
		for(int x=0; x<Iorient->width-g_w_; ++x, ++ptOrient_row, ++ptMagn_row) {
			calcHoGBin( ptOrient_row, ptMagn_row, step, desc );
			for(int l=0; l<bins_; ++l) {
				*ptOut_row[l] = (uchar)desc[l];
				++ptOut_row[l];
			}
		}
		// update pointer
		for(int l=0; l<bins_; ++l)
			ptOut[l] += step;
	}
	delete[] desc;
	delete[] ptOut;
	delete[] ptOut_row;
}
//==============================================================================



