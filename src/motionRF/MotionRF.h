/* MotionRF.h
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONRF_H_
#define MOTIONRF_H_
#include <Auxiliary.h>
#include <StructuredRF.h>
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
class MotionRF:public StructuredRF<L,M,T,F,N,U>{
	public:
		virtual ~MotionRF(){};
		MotionRF(int trees = 0,bool hogOrSift=true):hogOrSift_(hogOrSift),\
			StructuredRF<L,M,T,F,N,U>(trees){}
		/** Gets the histogram information for one tree.
		 */
		std::vector<float> histinfo(unsigned treeId);
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Trains a specified tree in the forest on the given patches.
		 */
		virtual void trainForestTree(unsigned min_s,unsigned max_d,CvRNG* pRNG,\
			const M& TrData,unsigned samples,unsigned treeId,const char \
			*path2models,const std::string &runName,typename StructuredTree\
			<M,T,F,N,U>::ENTROPY entropy,unsigned consideredCls,bool binary,\
			bool leafavg,bool parentfreq,bool leafparentfreq,const std::string \
			&runname,float entropysigma,bool usepick,bool hogOrSift,\
			unsigned growthtype,long unsigned maxleaves);
		/** Predicts on 1 single test patch.
		 */
		virtual void regression(std::vector<const U*> &result,\
			const T *testPatch,const F *features) const;
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		bool hogOrSift() const {return this->hogOrSift_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void hogOrSift(bool hogOrSift) {this->hogOrSift_ = hogOrSift;}
		//----------------------------------------------------------------------
	protected:
		/** @var hogOrSift_
		 * hog - 1, sift - 0
		 */
		bool hogOrSift_;
		//----------------------------------------------------------------------
	private:
		DISALLOW_COPY_AND_ASSIGN(MotionRF);
};
//==============================================================================
#endif /* MOTIONRF_H_ */
