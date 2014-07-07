/** Puzzle.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef PUZZLE_CPP_
#define PUZZLE_CPP_
#include "Puzzle.h"
//==============================================================================
/** It solves the recursive label optimization.
*/
template <class P>
cv::Mat Puzzle<P>::solve(const std::vector<std::vector<P> > &patchLabels,\
const cv::Size &featsize,unsigned labW,unsigned labH,unsigned noCls,\
std::vector<std::vector<float> > &clsFreq,Puzzle<P>::METHOD method,unsigned maxIter){
	cv::Mat labeling,prevLabeling;
	clock_t begin,end;
	bool converged            = false;
	unsigned iter             = 0;
	begin                     = clock();
	// [0] Get initial candidate patches over trees
	std::vector<P> candidates = Puzzle<P>::initialPick(patchLabels,\
		labW,labH,noCls,clsFreq);
	end                       = clock();
	std::cout<<"Initializing patch candidates time elapsed: "<<double\
		(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
	// [1] While not converged, iterate
	while(!converged){
		labeling.release();
		begin    = clock();
		// [2] Proposed labeling given the candidates
		labeling = Puzzle<P>::proposeLabeling(candidates,featsize,labW,labH,noCls);
		end      = clock();
		std::cout<<"Proposing labeling time elapsed: "<<double\
			(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		if(method == Puzzle<P>::SIMPLE){break;}
		candidates.clear();
		begin      = clock();
		// [3] Select candidates that agree the most with the labeling
		candidates = Puzzle<P>::selectPatches(labeling,patchLabels,labW,labH);
		end        = clock();
		std::cout<<"Selecting patches time elapsed: "<<double\
			(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		// [4] Check for convergence again
		if(iter>=maxIter){
			break;
		}else if(iter>0){
			begin     = clock();
			converged = Puzzle<P>::checkConvergence(labeling,prevLabeling);
			end       = clock();
			std::cout<<"Checking convergence time elapsed: "<<double\
				(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		}
		++iter;
		prevLabeling.release();
		labeling.copyTo(prevLabeling);
	}
	prevLabeling.release();
	return labeling;
}
//==============================================================================
/** Picks the initial candidates as ones with maximum foreground probability.
 */
template <class P>
std::vector<P> Puzzle<P>::initialPick(const std::vector<std::vector\
<P> > &patchLabels,unsigned labW,unsigned labH,unsigned noCls,\
std::vector<std::vector<float> > &clsFreq){
	std::vector<P> candidates;
	// [0] Loop over image positions and trees
	for(typename std::vector<std::vector<P> >::const_iterator p=\
	patchLabels.begin();p!=patchLabels.end();++p){
		std::vector<float> toNormalize(labH*labW,0.0);
		std::vector<std::vector<float> > classMarginals(noCls,std::vector<float>\
			(labH*labW,0.0));
		for(typename std::vector<P>::const_iterator t=p->begin();t!=p->end();++t){
			// [1] Get label patch at this position and in this tree
			std::vector<unsigned> tmp = t->piece();
			// [2] Loop over the pixels in the pathc
			for(std::vector<unsigned>::const_iterator v=tmp.begin();\
			v!=tmp.end();++v){
				// [3] Count class occurrences (*v) at the this position in label patch
				++classMarginals[*v][v-tmp.begin()];
				++toNormalize[v-tmp.begin()];
			}
		}
		unsigned bestPatchId = 0;
		float maxLogProb     = -std::numeric_limits<float>::max();
		// [4] Loop again over all trees
		for(typename std::vector<P>::const_iterator t=p->begin();t!=p->end();++t){
			float currentProb         = 0.0;
			std::vector<unsigned> tmp = t->piece();
			// [5] Loop over each pixel in this patch
			for(std::vector<unsigned>::const_iterator v=tmp.begin();\
			v!=tmp.end();++v){
				// [5.1] If patch, then use log probabilities
				if(tmp.size()>1 && classMarginals[*v][v-tmp.begin()]){
					currentProb += std::log(classMarginals[*v][v-tmp.begin()]/\
						toNormalize[v-tmp.begin()]);
				// [5.2] If 1 label then don't use log probabilities
				}else if(tmp.size()==1 && toNormalize[v-tmp.begin()]){
					currentProb = classMarginals[*v][v-tmp.begin()]/\
						toNormalize[v-tmp.begin()];
				}
			}
			// [6] Remember the id of the best patch for this position
			if(currentProb>maxLogProb){
				maxLogProb  = currentProb;
				bestPatchId = t-(p->begin());
			}
		}
		// [7] Add the candidate at this position
		candidates.push_back(p->at(bestPatchId));
	}
	return candidates;
}
//==============================================================================
/** Selects candidate best patches from the given patches
 */
template <class P>
std::vector<P> Puzzle<P>::selectPatches(const cv::Mat &labeling,\
const std::vector<std::vector<P> > &patchLabels,unsigned labW,\
unsigned labH){
	std::cout<<"[Puzzle::selectPatches] selecting candidate label-patches"<<std::endl;
	// [0] Find the candidate label patches that agree with this labeling
	std::vector<P> candidate;
	// [1] Loop over each image position
	for(typename std::vector<std::vector<P> >::const_iterator p=patchLabels.begin();\
	p!=patchLabels.end();++p){
		// [2] Get the point in the image corresponding to this position
		P patch           = p->at(0);
		cv::Point center  = patch.center();
		// [3] Cut out the roi at this position from the labeling
		cv::Rect roi(center.x-labW/2,center.y-labH/2,labW,labH);
		cv::Mat labelRoi            = (labeling(roi)).clone();
		labelRoi                    = labelRoi.reshape(1,1); // 1 channel, 1 row
		uchar* rowpt                = labelRoi.ptr(0);
		std::vector<unsigned> label = std::vector<unsigned>(rowpt,rowpt+labelRoi.cols);
		labelRoi.release();
		// [4] Compute agreement of label-roi with all patches at this position
		unsigned maxAgreement = 0;
		unsigned bestIdx      = 0;
		// [6] Loop over all trees
		for(typename std::vector<P>::const_iterator t=p->begin();t!=p->end();\
		++t){
			// [7] Find agreement between this patch and the label-roi
			unsigned agreement = t->agreement(label);
			// [8] Keep in ming the ID of the best patch
			if(agreement>maxAgreement){
				bestIdx      = t-(p->begin());
				maxAgreement = agreement;
			}
		}
		// [8] Store patch with largest agreement
		candidate.push_back(p->at(bestIdx));
	}
	return candidate;
}
//==============================================================================
/** Generates candidate labelings put of the proposed best patches.
 */
template <class P>
cv::Mat Puzzle<P>::proposeLabeling(const std::vector<P> &candidates,\
const cv::Size &featsize,unsigned labW,unsigned labH,unsigned noCls){
	std::cout<<"[Puzzle::proposeLabeling] proposing final labeling"<<std::endl;
	cv::Mat labeling           = cv::Mat::zeros(featsize,CV_8UC1);
	std::vector<cv::Mat> votes;
	for(unsigned c=0;c<noCls;++c){
		votes.push_back(cv::Mat::zeros(featsize,CV_32FC1));
	}
	cv::Mat allVotes           = cv::Mat::ones(featsize,CV_32FC1);
	cv::Mat maxi               = cv::Mat::ones(featsize,CV_32FC1);
	maxi                      *= -std::numeric_limits<float>::max();
	// [0] Loop over each image position
	for(typename std::vector<P>::const_iterator p=candidates.begin();\
	p!=candidates.end();++p){
		std::vector<unsigned> tmp    = p->piece();
		// [1] Loop over each pixel in the patch at this position
		for(std::vector<unsigned>::const_iterator l=tmp.begin();\
		l!=tmp.end();++l){
			// [2] Accumulate votes of current class at this position
			cv::Point pt             = p->pos2pt(l-(tmp.begin()),labW,labH);
			votes[*l].at<float>(pt) += 1.0;
			allVotes.at<float>(pt)  += 1.0;
		}
	}
	// [3] Loop over all classes
	for(unsigned s=0;s<votes.size();++s){
		cv::MatIterator_<float> iM     = maxi.begin<float>();
		cv::MatIterator_<uchar> iL     = labeling.begin<uchar>();
		cv::MatIterator_<float> iA     = allVotes.begin<float>();
		// [4] Loop over all image positions and pick the maximum label
		for(cv::MatIterator_<float> iV = votes[s].begin<float>();iV!=\
		votes[s].end<float>(),iM!=maxi.end<float>(),iL!=labeling.end<uchar>(),\
		iA!=allVotes.end<float>();++iV,++iM,++iL,++iA){
			if(*iA){(*iV) /= (*iA);}
			if((*iV)>(*iM)){
				*iM = *iV;
				*iL = s;
			}
		}
		votes[s].release();
	}
	allVotes.release();
	votes.clear();
	maxi.release();
	return labeling;
}
//==============================================================================
/** Checks to see how much the labeling has changed between iterations.
 */
template <class P>
bool Puzzle<P>::checkConvergence(const cv::Mat &labeling,const cv::Mat &prevLabeling){
	std::cout<<"[Puzzle::checkConvergence] checking for convergence"<<std::endl;
	if(labeling.cols!=prevLabeling.cols || labeling.rows!=prevLabeling.rows){
		std::cerr<<"[Puzzle::checkConvergence] the matrices dimensions do "<<\
			"not match"<<std::endl;
	}
	bool match                      = true;
	cv::MatConstIterator_<uchar> i1 = prevLabeling.begin<uchar>();
	for(cv::MatConstIterator_<uchar> i2=labeling.begin<uchar>();\
	i2!=labeling.end<uchar>(),i1!=prevLabeling.end<uchar>();++i2,++i1){
		if(*i1 != *i2){
			match = false;
			break;
		}
	}
	return match;
}
//==============================================================================
#endif // PUZZLE_CPP_
















