/* RunningStat.h
 * Author: Silvia Pintea
 * Adapted from: John D. Cook
 */
#ifndef RUNNINGSTAT_H_
#define RUNNINGSTAT_H_
#include <Auxiliary.h>
//==============================================================================
/** For computing fast mean and variance over samples.
 */
class RunningStat {
	public:
		RunningStat(unsigned dims):samples_(0),dims_(dims){
			this->oldMean_     = std::vector<float>(dims,0.0);
			this->newMean_     = std::vector<float>(dims,0.0);
			this->oldVariance_ = std::vector<float>(dims,0.0);
			this->newVariance_ = std::vector<float>(dims,0.0);
		}
		virtual ~RunningStat(){};
		//----------------------------------------------------------------------
		void Clear(){ this->samples_ = 0; }
		void push(const std::vector<float> &vals){
			++this->samples_;
			// See Knuth TAOCP vol 2, 3rd edition, page 232
			if(this->samples_==1){
				std::vector<float>::const_iterator v = vals.begin();
				std::vector<float>::iterator om      = this->oldMean_.begin();
				for(std::vector<float>::iterator nm=this->newMean_.begin();\
				nm!=this->newMean_.end(),om!=this->oldMean_.end(),v!=vals.end();\
				++om,++nm,++v){
					(*om) = (*nm) = (*v);
				} // over dims
			}else{
				this->newMean_                       = this->oldMean_;
				this->newVariance_                   = this->oldVariance_;
				std::vector<float>::const_iterator v = vals.begin();
				std::vector<float>::iterator om      = this->oldMean_.begin();
				std::vector<float>::iterator ov      = this->oldVariance_.begin();
				std::vector<float>::iterator nv      = this->newVariance_.begin();
				for(std::vector<float>::iterator nm=this->newMean_.begin();\
				nm!=this->newMean_.end(),om!=this->oldMean_.end(),v!=vals.end(),\
				ov!=this->oldVariance_.end(),nv!=this->newVariance_.end();\
				++om,++nm,++v,++ov,++nv){
					(*nm) += ((*v)-(*om))/static_cast<float>(this->samples_);
					(*nv) += ((*v)-(*om))*((*v)-(*nm));
				} // over dims
				// set up for next iteration
				this->oldMean_     = this->newMean_;
				this->oldVariance_ = this->newVariance_;
			}
		}
		std::vector<float> mean() const {
			return (this->samples_>0)?this->newMean_:std::vector<float>\
				(this->dims_,0.0);
		}
		float variance() const {
			float variance = 0.0;
			if(this->samples_>1){
				for(std::vector<float>::const_iterator nv=this->newVariance_.begin();\
				nv!=this->newVariance_.end();++nv){
					variance += (*nv);
				} // over dims
				variance /= static_cast<float>(this->samples_-1);
			}
			return variance;
		}
		float StandardDeviation() const {
			return std::sqrt(this->variance());
		}
		//----------------------------------------------------------------------
		//---Getters------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned samples() const {return this->samples_;}
		std::vector<float> oldMean() const {return this->oldMean_;}
		std::vector<float> oldVariance() const {return this->oldVariance_;}
		std::vector<float> newMean() const {return this->newMean_;}
		std::vector<float> newVariance() const {return this->newVariance_;}
		unsigned dims() const {return this->dims_;}
		//----------------------------------------------------------------------
		//---Setters------------------------------------------------------------
		//----------------------------------------------------------------------
		void samples(unsigned samples){this->samples_ = samples;}
		void oldMean(const std::vector<float> &oldMean){this->oldMean_ = oldMean;}
		void oldVariance(const std::vector<float> &oldVariance){
			this->oldVariance_ = oldVariance;
		}
		void newMean(const std::vector<float> &newMean){this->newMean_ = newMean;}
		void newVariance(const std::vector<float> &newVariance){
			this->newVariance_ = newVariance;
		}
		void dims(unsigned dims){this->dims_ = dims;}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors for trees (to put them in the forest).
		 */
		RunningStat(RunningStat const &rhs){
			this->dims_        = rhs.dims();
			this->samples_     = rhs.samples();
			this->oldMean_     = rhs.oldMean();
			this->newMean_     = rhs.newMean();
			this->oldVariance_ = rhs.oldVariance();
			this->newVariance_ = rhs.newVariance();
		}
		RunningStat& operator=(RunningStat const &rhs){
			if(this == &rhs) return *this;
			if(this){delete this;}
			this->dims_        = rhs.dims();
			this->samples_     = rhs.samples();
			this->oldMean_     = rhs.oldMean();
			this->newMean_     = rhs.newMean();
			this->oldVariance_ = rhs.oldVariance();
			this->newVariance_ = rhs.newVariance();
			return *this;
		}
		//----------------------------------------------------------------------
	private:
		/** @var samples_
		 * The number of samples already seen.
		 */
		unsigned samples_; // m_n
		/** @var oldMean_
		 * The old mean.
		 */
		std::vector<float> oldMean_; // m_oldM
		/** @var oldVariance_
		 * The old variance.
		 */
		std::vector<float> oldVariance_; // m_oldS
		/** @var newMean_
		 * The new mean.
		 */
		std::vector<float> newMean_; // m_newM
		/** @var newVariance_
		 * The new variance.
		 */
		std::vector<float> newVariance_; // m_newS
		/** @var dims_
		 * The dimensions of the data.
		 */
		unsigned dims_;
};
//==============================================================================
#endif /* RUNNINGSTAT_H_ */
