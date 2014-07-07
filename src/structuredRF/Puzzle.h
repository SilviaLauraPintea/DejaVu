/** Puzzle.h
 * Author: Silvia-Laura Pintea
 */
#ifndef PUZZLE_H_
#define PUZZLE_H_
#pragma once
#include <Auxiliary.h>
//==============================================================================
/** For storing the possible label-ings.
 */
struct PuzzlePatch{
	public:
		PuzzlePatch(){
			this->center_.x   = 0;
			this->center_.y   = 0;
			this->piece_      = std::vector<unsigned>();
			this->logProb_    = 0.0;
		}
		PuzzlePatch(const cv::Point &center,const std::vector<unsigned> &piece,\
		float logProb){
			this->center_.x   = center.x;
			this->center_.y   = center.y;
			this->piece_      = piece;
			this->logProb_    = logProb;
		}
		virtual ~PuzzlePatch(){}
		//----------------------------------------------------------------------
		/** Compares this piece with another piece.
		 */
		unsigned agreement(const std::vector<unsigned> &other) const {
			unsigned agreement = 0;
			if(other.size()!=this->piece_.size()){
				std::cerr<<"[PuzzlePatch::agreement]: patches should have "<<\
					"equal size"<<std::endl;
			}
			std::vector<unsigned>::const_iterator it2=this->piece_.begin();
			for(std::vector<unsigned>::const_iterator it1=other.begin();\
			it1!=other.end(),it2!=this->piece_.end();++it1,++it2){
				if((*it1)==(*it2)){
					++agreement;
				}
			}
			return agreement;
		}
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Finds the corresponding position in the original image.
		 */
		virtual cv::Point pos2pt(unsigned pos,unsigned labW,unsigned labH) const{
			// mat.cols*r+c (r=y,c=x)
			cv::Point inLab(0,0);
			if(this->piece_.size()>1){
				inLab = cv::Point(static_cast<unsigned>(pos % labW),\
					std::floor(static_cast<float>(pos)/static_cast<float>(labW)));
			}
			return cv::Point(inLab.x+(this->center_.x-(labW/2)),\
				inLab.y+(this->center_.y-(labH/2)));
		}
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		float logProb() const {return this->logProb_;}
		cv::Point center() const {return this->center_;}
		std::vector<unsigned> piece() const {return this->piece_;}
		unsigned piece(unsigned pos) const {return this->piece_[pos];}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void logProb(float logProb){this->logProb_ = logProb;}
		void center(const cv::Point &center){
			this->center_.x = center.x;
			this->center_.y = center.y;
		}
		void piece(const std::vector<unsigned> &piece){this->piece_ = piece;}
		void piece(unsigned pos,unsigned piece){this->piece_[pos] = piece;}
		//----------------------------------------------------------------------
		//---COPY & ASSIGNMENT--------------------------------------------------
		//----------------------------------------------------------------------
		/** Copy constructors the default ones are not good with IplImages
		 */
		PuzzlePatch(PuzzlePatch const &rhs){
			this->center_     = rhs.center();
			this->logProb_    = rhs.logProb();
			this->piece(rhs.piece());
		}
		PuzzlePatch& operator=(PuzzlePatch const &rhs){
			if(this == &rhs) return *this;
			if(this){delete this;}
			this->center_     = rhs.center();
			this->logProb_    = rhs.logProb();
			this->piece(rhs.piece());
			return *this;
		}
		//----------------------------------------------------------------------
	protected:
		/** @var center_
		 * The point on which the patch is centered.
		 */
		cv::Point center_;
		/** @var piece_
		 * Puzzle piece --- the label vector at this point.
		 */
		std::vector<unsigned> piece_;
		/** @var logProb_
		 * The log of the patch probability from the RF.
		 */
		float logProb_;
};
//==============================================================================
template <class P>
class Puzzle {
	public:
		/** Label selection method.
		 */
		enum METHOD {SIMPLE, PUZZLE};
		Puzzle(){};
		virtual ~Puzzle(){};
		//----------------------------------------------------------------------
		/** It solves the recursive label optimization.
		 */
		static cv::Mat solve(const std::vector<std::vector<P> > \
			&patchLabels,const cv::Size &imsize,unsigned labW,unsigned labH,\
			unsigned noCls,std::vector<std::vector<float> > &clsFreq,\
			Puzzle<P>::METHOD method,unsigned maxIter=75);
		/** Picks the initial candidates as ones with maximum foreground probability.
		 */
		static std::vector<P> initialPick(const std::vector\
			<std::vector<P> > &patchLabels,unsigned labW,unsigned labH,\
			unsigned noCls,std::vector<std::vector<float> > &clsFreq);
		/** Selects candidate best patches from the given patches
		 */
		static std::vector<P> selectPatches(const cv::Mat &labeling,\
			const std::vector<std::vector<P> > &patchLabels,\
			unsigned labW,unsigned labH);
		/** Generates candidate labelings put of the proposed best patches.
		 */
		static cv::Mat proposeLabeling(const std::vector<P> &candidates,\
			const cv::Size &imsize,unsigned labW,unsigned labH,unsigned noCls);
		/** Checks to see how much the labeling has changed between iterations.
		 */
		static bool checkConvergence(const cv::Mat &labeling,const cv::Mat \
			&prevLabeling);
	private:
		DISALLOW_COPY_AND_ASSIGN(Puzzle);
};
//==============================================================================
#endif /* PUZZLE_H_ */
#include "Puzzle.cpp"
