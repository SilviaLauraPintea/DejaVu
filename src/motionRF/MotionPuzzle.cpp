/* MotionPuzzle.cpp
 * Author: Silvia-Laura Pintea
 */
#include "MotionPuzzle.h"
//==============================================================================
//==============================================================================
//==============================================================================
/** Solves choosing the final motion prediction problem:
 * - choosing among the trees
 * - choosing in the neighborhood.
 */
template <class P>
void MotionPuzzle<P>::solve(const std::vector<std::vector<P*> > \
&patches,const cv::Size &featsize,unsigned motionW,unsigned motionH,\
cv::Mat &motionX,cv::Mat &motionY,cv::Mat &appear,MotionTreeClass::ENTROPY entropy,\
Puzzle<PuzzlePatch>::METHOD method,unsigned step,const std::string &imname,\
const std::string &path2results,unsigned maxIter,bool display,bool pertree){
	cv::Mat motion,prevMotion;
	clock_t begin,end;
	bool converged = false;
	unsigned iter  = 0;
	begin          = clock();
	// [0] Get initial candidate patches over trees
	std::vector<P*> candidates = MotionPuzzle<P>::initialPick(patches,motionW,\
		motionH,entropy,display,false);
	end = clock();
	std::cout<<"[MotionPuzzle::solve] Initializing candidates time elapsed: "<<\
		double(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
	// [1] While not converged, iterate
	while(!converged){
		motion.release(); appear.release();
		begin = clock();
		// [2] Proposed labeling given the candidates
		motion = MotionPuzzle<P>::proposePredictionSum(candidates,featsize,\
			motionW,motionH,appear,false);
		assert(motion.channels()==2);
		end = clock();
		std::cout<<"[MotionPuzzle::solve] Proposing labeling time elapsed: "<<\
			double(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		for(typename std::vector<P*>::iterator candi=candidates.begin();candi!=\
		candidates.end();++candi){
			if(*candi){
				delete (*candi); (*candi) = NULL;
			}
		}
		candidates.clear();
		if(method == Puzzle<PuzzlePatch>::SIMPLE){break;}
		begin      = clock();
		// [3] Select candidates that agree the most with the labeling
		candidates = MotionPuzzle<P>::selectPatches(motion,patches,motionW,motionH,false);
		end        = clock();
		std::cout<<"[MotionPuzzle::solve] Selecting patches time elapsed: "<<\
			double(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		// [4] Check for convergence again
		if(iter>=maxIter){
			break;
		}else if(iter){
			begin     = clock();
			converged = MotionPuzzle<P>::checkConvergence(motion,prevMotion,false);
			end       = clock();
			std::cout<<"[MotionPuzzle::solve] Checking convergence time elapsed: "<<\
				double(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		}
		++iter;
		prevMotion.release();
		motion.copyTo(prevMotion);
	}
	prevMotion.release();
	std::vector<cv::Mat> split;
	cv::split(motion,split);
	assert(split.size()==2);
	split[0].copyTo(motionX); split[0].release();
	split[1].copyTo(motionY); split[1].release();
	motion.release();
	// [5] No point in saving the per-tree-prediction at while running exp
	if(pertree){
		MotionPuzzle<P>::perTreePredictions(patches,motionW,motionH,featsize,\
			step,entropy,imname,path2results,display,false);
	}
}
//==============================================================================
/** Solves choosing the final motion prediction problem:
 * - choosing among the trees
 * - choosing in the neighborhood.
 */
template <class P>
void MotionPuzzle<P>::solve(const std::vector<std::vector<P*> > \
&patches,const cv::Size &featsize,unsigned motionW,unsigned motionH,\
cv::Mat &motionXX,cv::Mat &motionXY,cv::Mat &motionYX,cv::Mat &motionYY,\
cv::Mat &appear,MotionTreeClass::ENTROPY entropy,Puzzle<PuzzlePatch>::METHOD method,\
unsigned step,const std::string &imname,const std::string &path2results,\
unsigned maxIter,bool display,bool pertree){
	cv::Mat motion,prevMotion;
	clock_t begin,end;
	bool converged = false;
	unsigned iter  = 0;
	begin          = clock();
	// [0] Get initial candidate patches over trees
	std::vector<P*> candidates = MotionPuzzle<P>::initialPick(patches,motionW,\
		motionH,entropy,display,true);
	end = clock();
	std::cout<<"[MotionPuzzle::solve] Initializing candidates time elapsed: "<<\
		double(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
	// [1] While not converged, iterate
	while(!converged){
		motion.release(); appear.release();
		begin      = clock();
		// [2] Proposed labeling given the candidates
		motion     = MotionPuzzle<P>::proposePredictionSum(candidates,featsize,\
			motionW,motionH,appear,true);
		assert(motion.channels()==4);
		end        = clock();
		std::cout<<"[MotionPuzzle::solve] Proposing labeling time elapsed: "<<\
			double(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		for(typename std::vector<P*>::iterator candi=candidates.begin();candi!=\
		candidates.end();++candi){
			if(*candi){
				delete (*candi); (*candi) = NULL;
			}
		}
		candidates.clear();
		if(method == Puzzle<PuzzlePatch>::SIMPLE){break;}
		begin      = clock();
		// [3] Select candidates that agree the most with the labeling
		candidates = MotionPuzzle<P>::selectPatches(motion,patches,motionW,motionH,true);
		end        = clock();
		std::cout<<"[MotionPuzzle::solve] Selecting patches time elapsed: "<<\
			double(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		// [4] Check for convergence again
		if(iter>=maxIter){
			break;
		}else if(iter){
			begin     = clock();
			converged = MotionPuzzle<P>::checkConvergence(motion,prevMotion,true);
			end       = clock();
			std::cout<<"[MotionPuzzle::solve] Checking convergence time elapsed: "<<\
				double(Auxiliary<uchar,1>::diffclock(end,begin))<<" ms"<<std::endl;
		}
		++iter;
		prevMotion.release();
		motion.copyTo(prevMotion);
	}
	prevMotion.release();
	std::vector<cv::Mat> split;
	cv::split(motion,split);
	assert(split.size()==4);
	split[0].copyTo(motionXX); split[0].release();
	split[1].copyTo(motionXY); split[1].release();
	split[2].copyTo(motionYX); split[2].release();
	split[3].copyTo(motionYY); split[3].release();
	motion.release();
	if(pertree){
		MotionPuzzle<P>::perTreePredictions(patches,motionW,motionH,featsize,\
			step,entropy,imname,path2results,display,true);
	}
}
//==============================================================================
/** Generate a set of initial candidates (picks the most likely patches among
 * the trees).
 */
template <class P>
std::vector<P*> MotionPuzzle<P>::initialPick(const std::vector<std::vector<P*> > \
&candidates,unsigned motionW,unsigned motionH,MotionTreeClass::ENTROPY entropy,\
bool display,bool usederivatives){
	std::vector<P*> final;
	switch(entropy){
		case(MotionTreeClass::APPROX_MAGNI_KERNEL):
			final = MotionPuzzle<P>::pickApproxKernel(candidates,motionW,motionH,\
				entropy,display,usederivatives);
			break;
		case(MotionTreeClass::APPROX_ANGLE_KERNEL):
			final = MotionPuzzle<P>::pickApproxKernel(candidates,motionW,motionH,\
				entropy,display,usederivatives);
			break;
		case(MotionTreeClass::MEAN_DIFF):
			final = MotionPuzzle<P>::pickMean(candidates,motionW,motionH,display,\
				usederivatives);
			break;
		default:
			final = MotionPuzzle<P>::pickMean(candidates,motionW,motionH,display,\
				usederivatives);
			break;
	}
	return final;
}
//==============================================================================
/** Generates the best patch per position among trees as the mean of all patches.
 */
template <class P>
std::vector<P*> MotionPuzzle<P>::pickMean(const std::vector<std::vector<P*> > \
&candidates,unsigned motionW,unsigned motionH,bool display,bool usederivatives){
	std::cout<<"[MotionPuzzle<P>::pickMean] best prediction/position"<<std::endl;
	std::vector<P*> final;
	unsigned nocols = 2;
	if(usederivatives){nocols = 4;}
	// [0] Loop over all image positions
	for(typename std::vector<std::vector<P*> >::const_iterator p=candidates.begin();\
	p!=candidates.end();++p){
		// [1] Find the mean patch over the 10 trees at this position
		cv::Mat meanMotion = cv::Mat::zeros(cv::Size(nocols*motionW*motionH,1),CV_32FC1);
		cv::Mat meanAppear = cv::Mat::zeros(cv::Size(motionW,motionH),CV_32FC3);
		std::vector<cv::Mat> meanHisto;
		float motionProb     = 0.0;
		float appearanceProb = 0.0;
		for(typename std::vector<P*>::const_iterator t=p->begin();t!=p->end();++t){
			if(display){
				MotionPuzzle<P>::showSamples((**t),motionW,motionH,usederivatives);
			}
			// [1.1] For each tree I get a motion prediction at this position
			cv::Mat motion  = (*t)->motion();
			cv::Mat appear  = (*t)->appearance();
			appear.convertTo(appear,CV_32FC3);
			std::vector<cv::Mat> histo = (*t)->histo();
			motionProb                += (*t)->motionProb();
			appearanceProb            += (*t)->appearanceProb();
			// [1.2] Loop over the pixels inside the patch
			assert(meanMotion.size()==motion.size());
			meanMotion += motion;
			meanAppear += appear;
			if(meanHisto.empty()){meanHisto.resize(histo.size());}
			for(unsigned b=0;b<histo.size();++b){
				if(meanHisto[b].empty()){ histo[b].copyTo(meanHisto[b]);
				}else{ meanHisto[b] += histo[b]; }
			}
		} // over trees
		meanMotion       /= static_cast<float>(p->size());
		meanAppear       /= static_cast<float>(p->size());
		for(unsigned b=0;b<meanHisto.size();++b){
			meanHisto[b] /= static_cast<float>(p->size());
		}
		motionProb       /= static_cast<float>(p->size());
		appearanceProb   /= static_cast<float>(p->size());
		// [2] Copy it to this position
		typename std::vector<P*>::const_iterator t = p->begin();
		final.push_back(new MotionPuzzlePatch((*t)->center(),(*t)->piece(),meanMotion,\
			meanAppear,meanHisto,(*t)->logProb(),motionProb,appearanceProb,\
			(*t)->histinfo()));
		meanMotion.release(); meanAppear.release();
		for(unsigned b=0;b<meanHisto.size();++b){meanHisto[b].release();}
	} // over img positions
	return final;
}
//==============================================================================
/** Given a vector of candidate patches compute their probabilities.
 */
template <class P>
std::vector<float> MotionPuzzle<P>::approxKernel(const std::vector<P*> \
&candidates,unsigned motionW,unsigned motionH,unsigned &bestId,\
MotionTreeClass::ENTROPY entropy,bool usederivatives){
	// [2] For each dimension we have a histogram
	std::vector<cv::Mat> probs; // #bins => patch size
	cv::Mat norm = cv::Mat::zeros(cv::Size(motionW,motionH),CV_32FC1); // patch size
	for(typename std::vector<P*>::const_iterator p=candidates.begin();p!=\
	candidates.end();++p){ // over patches
		std::vector<cv::Mat> histo = (*p)->histo();
		if(probs.empty()){probs.resize(histo.size());}
		std::vector<cv::Mat>::iterator pr=probs.begin();
		for(std::vector<cv::Mat>::iterator hi=histo.begin();hi!=histo.end(),pr!=\
		probs.end();++hi,++pr){
			if(pr->empty()){ hi->copyTo(*pr);
			}else{ (*pr) += (*hi); }
			norm += (*hi);
		}// over bins
	}// over candidate patches at current position
	cv::Mat mask;
	cv::inRange(norm,0,SMALL,mask);
	norm.setTo(1,mask); mask.release();
	for(std::vector<cv::Mat>::iterator pr=probs.begin();pr!=probs.end();++pr){
		cv::divide((*pr),norm,(*pr));
	}
	norm.release();
	// [7] Now loop again and get the probabilities of each patch
	std::vector<float> patchprobs(candidates.size(),0.0);
	bestId            = 0;
	float bestLogprob = -std::numeric_limits<float>::max();
	for(typename std::vector<P*>::const_iterator p=candidates.begin();p!=\
	candidates.end();++p){ // over patches
		patchprobs[p-candidates.begin()] = MotionPuzzle<P>::patchProb(probs,(**p),\
			motionW,motionH,entropy,usederivatives);
		if(patchprobs[p-candidates.begin()]>bestLogprob){
			bestLogprob = patchprobs[p-candidates.begin()];
			bestId      = (p-candidates.begin());
		}
	}// over patches
	for(std::vector<cv::Mat>::iterator pr=probs.begin();pr!=probs.end();++pr){
		pr->release();
	}
	return patchprobs;
}
//==============================================================================
/** Gets the patch probabilities as sum_px log p(px).
 */
template <class P>
float MotionPuzzle<P>::patchProb(const std::vector<cv::Mat> &probs,const P \
&patch,unsigned motionW,unsigned motionH,MotionTreeClass::ENTROPY entropy,\
bool usederivatives){
	float patchprob = 0.0;
	if(usederivatives){
		cv::Mat motionXX = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		cv::Mat motionXY = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		cv::Mat motionYX = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		cv::Mat motionYY = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		patch.motion(motionXX,motionXY,motionYX,motionYY,motionW,motionW);
		std::vector<float> bininfo   = patch.histinfo();
		cv::Mat_<float>::iterator yy = motionYY.begin<float>();
		cv::Mat_<float>::iterator yx = motionYX.begin<float>();
		cv::Mat_<float>::iterator xy = motionXY.begin<float>();
		for(cv::Mat_<float>::iterator xx=motionXX.begin<float>();xx!=motionXX.\
		end<float>(),xy!=motionXY.end<float>(),yx!=motionYX.end<float>(),\
		yy!=motionYY.end<float>();++xx,++xy,++yx,++yy){
			float pxprob      = 0.0;
			unsigned patchpos = (xx-motionXX.begin<float>());
			cv::Point ptpos   = cv::Point((static_cast<unsigned>(patchpos)%\
				motionW),std::floor(static_cast<float>(patchpos)/\
				static_cast<float>(motionW)));
			std::vector<float> values;
			values.push_back(*xx); values.push_back(*xy);
			values.push_back(*yx); values.push_back(*yy);
			switch(entropy){
				case(MotionTreeClass::APPROX_ANGLE_KERNEL):
					pxprob = MotionTreeClass::getProbAngle(probs,bininfo,values,ptpos);
					break;
				case(MotionTreeClass::APPROX_MAGNI_KERNEL):
					pxprob = MotionTreeClass::getProbMagni(probs,bininfo,values,ptpos);
					break;
				default:
					pxprob = MotionTreeClass::getProbMagni(probs,bininfo,values,ptpos);
					break;
			}
			patchprob += (std::isinf(std::log(pxprob))?std::log(SMALL):\
				std::log(pxprob));
		} // over dimensions
		motionXX.release(); motionXY.release();
		motionYX.release(); motionYY.release();
	}else{
		cv::Mat motionX = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		cv::Mat motionY = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		patch.motion(motionX,motionY,motionW,motionH);
		std::vector<float> bininfo  = patch.histinfo();
		cv::Mat_<float>::iterator x = motionX.begin<float>();
		for(cv::Mat_<float>::iterator y=motionY.begin<float>();y!=motionY.\
		end<float>(),x!=motionX.end<float>();++x,++y){
			float pxprob      = 0.0;
			unsigned patchpos = (x-motionX.begin<float>());
			cv::Point ptpos   = cv::Point((static_cast<unsigned>(patchpos)%\
				motionW),std::floor(static_cast<float>(patchpos)/\
				static_cast<float>(motionW)));
			std::vector<float> values;
			values.push_back(*x); values.push_back(*y);
			switch(entropy){
				case(MotionTreeClass::APPROX_ANGLE_KERNEL):
					pxprob = MotionTreeClass::getProbAngle(probs,bininfo,values,ptpos);
					break;
				case(MotionTreeClass::APPROX_MAGNI_KERNEL):
					pxprob = MotionTreeClass::getProbMagni(probs,bininfo,values,ptpos);
					break;
				default:
					pxprob = MotionTreeClass::getProbMagni(probs,bininfo,values,ptpos);
					break;
			}
			patchprob += (std::isinf(std::log(pxprob))?std::log(SMALL):\
				std::log(pxprob));
		} // over dimensions
		motionX.release(); motionY.release();
	}
	return patchprob;
}
//==============================================================================
/** Picks the best patch per position among trees based on probability
 * approximated using kernel density estimation.
 */
template <class P>
std::vector<P*> MotionPuzzle<P>::pickApproxKernel(const std::vector<std::vector<P*> > \
&candidates,unsigned motionW,unsigned motionH,MotionTreeClass::ENTROPY entropy,\
bool display,bool usederivatives){
	std::cout<<"[MotionPuzzle<P>::pickApproxKernel] best prediction/position"<<std::endl;
	// [0] Loop over all image positions
	std::vector<std::vector<float> > probabilities;
	std::vector<P*> final;
	for(typename std::vector<std::vector<P*> >::const_iterator p=candidates.begin();\
	p!=candidates.end();++p){
		// [1] Get at each position: p(motion | tree), tree in {1,..10}
		unsigned bestId;
		std::vector<float> treeProb = MotionPuzzle<P>::approxKernel(*p,motionW,\
			motionH,bestId,entropy,usederivatives);
		probabilities.push_back(treeProb);
		P* patch = p->at(bestId);
		if(display){
			MotionPuzzle<P>::showSamples(*patch,motionW,motionH,usederivatives);
		}
		final.push_back(new MotionPuzzlePatch(*patch));
	}
	return final;
}
//==============================================================================
/** Displays the set of predicted leaves with flow derivatives.
 */
template <class P>
void MotionPuzzle<P>::showSamplesDerivatives(const P &leaf,unsigned sampleW,\
unsigned sampleH){
	cv::Mat patch = cv::Mat::zeros(cv::Size(sampleW+50,sampleH+50),CV_8UC3);
	cv::Mat motionXX,motionXY,motionYX,motionYY;
	leaf.motion(motionXX,motionXY,motionYX,motionYY,sampleW,sampleH);
	motionXX = motionXX.reshape(1,sampleW);
	motionXY = motionXY.reshape(1,sampleW);
	motionYX = motionYX.reshape(1,sampleW);
	motionYY = motionYY.reshape(1,sampleW);
	// [2] Make some borders around so we can see them better
	cv::copyMakeBorder(motionXX,motionXX,25,25,25,25,cv::BORDER_CONSTANT,0);
	cv::copyMakeBorder(motionXY,motionXY,25,25,25,25,cv::BORDER_CONSTANT,0);
	cv::copyMakeBorder(motionYX,motionYX,25,25,25,25,cv::BORDER_CONSTANT,0);
	cv::copyMakeBorder(motionYY,motionYY,25,25,25,25,cv::BORDER_CONSTANT,0);
	// [3] Now display the OF vectors on the patch
	cv::Mat garbage = MotionPatch<MotionPatchFeature<FeaturesMotion>,FeaturesMotion>::\
		showOFderi(motionXX,motionXY,motionYX,motionYY,patch,5,true);
	garbage.release();
	motionXX.release();motionXY.release();motionYX.release();motionYY.release();
	patch.release();
}
//==============================================================================
/** Displays the set of predicted leaves with flow.
 */
template <class P>
void MotionPuzzle<P>::showSamplesFlow(const P &leaf,unsigned sampleW,unsigned sampleH){
	cv::Mat patch = cv::Mat::zeros(cv::Size(sampleW+50,sampleH+50),CV_8UC3);
	cv::Mat motionX, motionY;
	leaf.motion(motionX,motionY,sampleW,sampleH);
	motionX = motionX.reshape(1,sampleW);
	motionY = motionY.reshape(1,sampleW);
	// [2] Make some borders around so we can see them better
	cv::copyMakeBorder(motionX,motionX,25,25,25,25,cv::BORDER_CONSTANT,0);
	cv::copyMakeBorder(motionY,motionY,25,25,25,25,cv::BORDER_CONSTANT,0);
	// [3] Now display the OF vectors on the patch
	cv::Mat garbage = MotionPatch<MotionPatchFeature<FeaturesMotion>,FeaturesMotion>::\
		showOF(motionX,motionY,patch,5,true);
	garbage.release();motionX.release();motionY.release();patch.release();
}
//==============================================================================
/** Displays the set of predicted leaves.
 */
template <class P>
void MotionPuzzle<P>::showSamples(const P &leaf,unsigned sampleW,unsigned sampleH,\
bool usederivatives){
	if(usederivatives){
		MotionPuzzle<P>::showSamplesDerivatives(leaf,sampleW,sampleH);
	}else{
		MotionPuzzle<P>::showSamplesFlow(leaf,sampleW,sampleH);
	}
}
//==============================================================================
/** Returns proposals for motion derivatives on X and Y.
 */
template <class P>
cv::Mat MotionPuzzle<P>::proposePredictionSumDerivatives(const std::vector<P*> \
&candidates,const cv::Size &featsize,unsigned motionW,unsigned motionH,cv::Mat \
&appearance){
	std::cout<<"[MotionPuzzle<P>::proposePredictionSumDerivatives] motion prediction"<<std::endl;
	appearance       = cv::Mat::zeros(featsize,CV_32FC3);
	cv::Mat motionXX = cv::Mat::zeros(featsize,CV_32FC1);
	cv::Mat motionXY = cv::Mat::zeros(featsize,CV_32FC1);
	cv::Mat motionYX = cv::Mat::zeros(featsize,CV_32FC1);
	cv::Mat motionYY = cv::Mat::zeros(featsize,CV_32FC1);
	cv::Mat norm     = cv::Mat::zeros(featsize,CV_32FC1);
	// [0] Loop over the image positions
	for(typename std::vector<P*>::const_iterator p=candidates.begin();\
	p!=candidates.end();++p){
		cv::Mat amotionXX,amotionXY,amotionYX,amotionYY;
		(*p)->motion(amotionXX,amotionXY,amotionYX,amotionYY,motionW,motionH);
		cv::Mat appear = (*p)->appearance();
		appear.convertTo(appear,CV_32FC3);
		// [5] Loop over 1 motion patch
		cv::Mat_<cv::Vec3f>::const_iterator a=appear.begin<cv::Vec3f>();
		cv::Mat_<float>::const_iterator mxx=amotionXX.begin<float>();
		cv::Mat_<float>::const_iterator mxy=amotionXY.begin<float>();
		cv::Mat_<float>::const_iterator myx=amotionYX.begin<float>();
		for(cv::Mat_<float>::const_iterator myy=amotionYY.begin<float>();myy!=\
		amotionYY.end<float>(),myx!=amotionYX.end<float>(),mxy!=amotionXY.end<float>(),\
		mxx!=amotionXX.end<float>(),a!=appear.end<cv::Vec3f>();++a,++myy,++myx,\
		++mxy,++mxx){
			// [6] For each point get the distance to its mean
			unsigned pos                  = mxx-(amotionXX.begin<float>());
			cv::Point pt                  = (*p)->pos2pt(pos,motionW,motionH);
			motionXX.at<float>(pt)       += (*mxx);
			motionXY.at<float>(pt)       += (*mxy);
			motionYX.at<float>(pt)       += (*myx);
			motionYY.at<float>(pt)       += (*myy);
			cv::Vec3f appear              = appearance.at<cv::Vec3f>(pt);
			appear.val[0]                += (*a).val[0];
			appear.val[1]                += (*a).val[1];
			appear.val[2]                += (*a).val[2];
			appearance.at<cv::Vec3f>(pt)  = appear;
			++norm.at<float>(pt);
		}
		amotionXX.release(); amotionXY.release(); amotionYX.release(); amotionYY.release();
	}
	std::vector<cv::Mat> splits;
	cv::split(appearance,splits);
	assert(splits.size()==3);
	cv::divide(splits[0],norm,splits[0]);
	cv::divide(splits[1],norm,splits[1]);
	cv::divide(splits[2],norm,splits[2]);
	cv::Mat inapp[] = {splits[0],splits[1],splits[2]};
	appearance.release();
	cv::merge(inapp,3,appearance);
	appearance.convertTo(appearance,CV_8UC3);
	splits[0].release(); splits[1].release(); splits[2].release();
	cv::divide(motionXX,norm,motionXX);
	cv::divide(motionXY,norm,motionXY);
	cv::divide(motionYX,norm,motionYX);
	cv::divide(motionYY,norm,motionYY);
	norm.release();
	cv::Mat in[] = {motionXX,motionXY,motionYX,motionYY};
	cv::Mat motion;
	cv::merge(in,4,motion);
	motionXX.release(); motionXY.release(); motionYX.release(); motionYY.release();
	return motion;
}
//==============================================================================
/** Returns proposals for motion flows on X and Y.
 */
template <class P>
cv::Mat MotionPuzzle<P>::proposePredictionSumFlows(const std::vector<P*> &candidates,\
const cv::Size &featsize,unsigned motionW,unsigned motionH,cv::Mat &appearance){
	std::cout<<"[MotionPuzzle<P>::proposePredictionSumFlows] motion prediction"<<std::endl;
	appearance      = cv::Mat::zeros(featsize,CV_32FC3);
	cv::Mat motionX = cv::Mat::zeros(featsize,CV_32FC1);
	cv::Mat motionY = cv::Mat::zeros(featsize,CV_32FC1);
	cv::Mat norm    = cv::Mat::zeros(featsize,CV_32FC1);
	// [0] Loop over the image positions
	for(typename std::vector<P*>::const_iterator p=candidates.begin();\
	p!=candidates.end();++p){
		cv::Mat amotionX,amotionY;
		(*p)->motion(amotionX,amotionY,motionW,motionW);
		cv::Mat appear = (*p)->appearance();
		appear.convertTo(appear,CV_32FC3);
		// [5] Loop over 1 motion patch
		cv::Mat_<cv::Vec3f>::const_iterator a = appear.begin<cv::Vec3f>();
		cv::Mat_<float>::const_iterator my    = amotionY.begin<float>();
		for(cv::Mat_<float>::const_iterator mx=amotionX.begin<float>();\
		mx!=amotionX.end<float>(),my!=amotionY.end<float>(),a!=appear.end\
		<cv::Vec3f>();++mx,++my,++a){
			// [6] For each point get the distance to its mean
			unsigned pos                 = mx-(amotionX.begin<float>());
			cv::Point pt                 = (*p)->pos2pt(pos,motionW,motionH);
			motionX.at<float>(pt)       += (*mx);
			motionY.at<float>(pt)       += (*my);
			cv::Vec3f appear             = appearance.at<cv::Vec3f>(pt);
			appear.val[0]               += (*a).val[0];
			appear.val[1]               += (*a).val[1];
			appear.val[2]               += (*a).val[2];
			appearance.at<cv::Vec3f>(pt) = appear;
			++norm.at<float>(pt);
		}
		amotionX.release(); amotionY.release();
	}
	std::vector<cv::Mat> splits;
	cv::split(appearance,splits);
	cv::divide(splits[0],norm,splits[0]);
	cv::divide(splits[1],norm,splits[1]);
	cv::divide(splits[2],norm,splits[2]);
	cv::Mat inapp[] = {splits[0],splits[1],splits[2]};
	appearance.release();
	cv::merge(inapp,3,appearance);
	appearance.convertTo(appearance,CV_8UC3);
	splits[0].release(); splits[1].release(); splits[2].release();
	cv::divide(motionX,norm,motionX);
	cv::divide(motionY,norm,motionY);
	norm.release();
	cv::Mat in[] = {motionX,motionY};
	cv::Mat motion;
	cv::merge(in,2,motion);
	motionX.release(); motionY.release();
	return motion;
}
//==============================================================================
/** Returns proposals for motions on X and Y.
 */
template <class P>
cv::Mat MotionPuzzle<P>::proposePredictionSum(const std::vector<P*> &candidates,\
const cv::Size &featsize,unsigned motionW,unsigned motionH,cv::Mat &appearance,\
bool usederivatives){
	if(usederivatives){
		return MotionPuzzle<P>::proposePredictionSumDerivatives(candidates,featsize,\
			motionW,motionH,appearance);
	}else{
		return MotionPuzzle<P>::proposePredictionSumFlows(candidates,featsize,\
			motionW,motionH,appearance);
	}
}
//==============================================================================
/** Selects the patches that agree the most with the previous prediction.
 */
template <class P>
std::vector<P*> MotionPuzzle<P>::selectPatches(const cv::Mat &motion,\
const std::vector<std::vector<P*> > &candidates,unsigned motionW,unsigned motionH,\
bool usederivatives){
	// [0] Loop over all image positions
	std::vector<P*> final;
	for(typename std::vector<std::vector<P*> >::const_iterator p=candidates.begin();\
	p!=candidates.end();++p){
		// [1] Get the motion at the current position
		cv::Point center  = p->at(0)->center();
		cv::Rect roi(center.x-motionW/2,center.y-motionH/2,motionW,motionH);
		cv::Mat motionRoi = (motion(roi)).clone();
		motionRoi         = motionRoi.reshape(1,1);
		float minDist     = std::numeric_limits<float>::max();
		unsigned bestIdx  = 0;
		// [2] Get the similarity between this patch and the labeling
		for(typename std::vector<P*>::const_iterator t=p->begin();t!=p->end();++t){
			float dist = (*t)->motionAgreement(motionRoi,usederivatives);
			if(dist<minDist){
				bestIdx = t-(p->begin());
				minDist = dist;
			}
		}
		motionRoi.release();
		final.push_back(new MotionPuzzlePatch(*(p->at(bestIdx))));
	}
	return final;
}
//==============================================================================
/** Checks to see how much the prediction has changed between iterations.
 */
template <class P>
bool MotionPuzzle<P>::checkConvergence(const cv::Mat &motion,const cv::Mat \
&prevMotion,bool usederivatives){
	std::cout<<"[MotionPuzzle::checkConvergence] checking for convergence"<<std::endl;
	if(motion.cols!=prevMotion.cols || motion.rows!=prevMotion.rows){
		std::cerr<<"[MotionPuzzle::checkConvergence] the matrices dimensions do "<<\
			"not match"<<std::endl;
	}
	bool match                          = true;
	if(usederivatives){
		cv::MatConstIterator_<cv::Vec4f> i1 = prevMotion.begin<cv::Vec4f>();
		for(cv::MatConstIterator_<cv::Vec4f> i2=motion.begin<cv::Vec4f>();\
		i2!=motion.end<cv::Vec4f>(),i1!=prevMotion.end<cv::Vec4f>();++i2,++i1){
			cv::Vec4f value1 = (*i1);
			cv::Vec4f value2 = (*i2);
			if(value1.val[0] != value2.val[0] || value1.val[1] != value2.val[1] ||\
			value1.val[2] != value2.val[2] || value1.val[3] != value2.val[3]){
				match = false;
				break;
			}
		}
	}else{
		cv::MatConstIterator_<cv::Vec2f> i1 = prevMotion.begin<cv::Vec2f>();
		for(cv::MatConstIterator_<cv::Vec2f> i2=motion.begin<cv::Vec2f>();\
		i2!=motion.end<cv::Vec2f>(),i1!=prevMotion.end<cv::Vec2f>();++i2,++i1){
			cv::Vec2f value1 = (*i1);
			cv::Vec2f value2 = (*i2);
			if(value1.val[0] != value2.val[0] || value1.val[1] != value2.val[1]){
				match = false;
				break;
			}
		}
	}
	return match;
}
//==============================================================================
/** Show per tree predictions and appearance.
 */
template <class P>
void MotionPuzzle<P>::perTreePredictions(const std::vector<std::vector<P*> > &candidates,\
unsigned motionW,unsigned motionH,const cv::Size &featsize,unsigned step,\
MotionTreeClass::ENTROPY entropy,const std::string &imname,const std::string \
&path2results,bool display,bool usederivatives){
	std::vector<std::vector<P*> > perTree;
	perTree.resize(candidates[0].size());
	// [0] Loop over all image positions
	for(typename std::vector<std::vector<P*> >::const_iterator p=candidates.begin();\
	p!=candidates.end();++p){
		for(typename std::vector<P*>::const_iterator t=p->begin();t!=p->end();++t){
			perTree[t-(p->begin())].push_back(*t);
		} // trees
	} // image positions
	// [1] For each tree get a labeling
	for(typename std::vector<std::vector<P*> >::const_iterator t=perTree.begin();\
	t!=perTree.end();++t){
		cv::Mat appearance,appearanceover;
		cv::Mat prediction     = MotionPuzzle<P>::proposePredictionSum((*t),featsize,\
			motionW,motionH,appearance,usederivatives);
		cv::Mat predictionover = MotionPuzzle<P>::proposePredictionOverlap((*t),\
			featsize,motionW,motionH,step,entropy,appearanceover,usederivatives);
		// [2] Draw the arrows on the images
		std::vector<cv::Mat> split;
		cv::split(prediction,split);
		cv::Mat arrows;
		if(usederivatives){
			assert(split.size()==4);
			arrows = MotionPatchClass::showOFderi(split[0],split[1],split[2],\
				split[3],appearance,5,false);
			split[0].release(); split[1].release();
			split[2].release(); split[3].release();
		}else{
			assert(split.size()==2);
			arrows = MotionPatchClass::showOF(split[0],split[1],appearance,5,false);
			split[0].release(); split[1].release();
		}
		prediction.release();
		// [3] Draw the arrows with overlap on the images
		std::vector<cv::Mat> splitover;
		cv::split(predictionover,splitover);
		cv::Mat arrowsover;
		if(usederivatives){
			assert(splitover.size()==4);
			arrowsover = MotionPatchClass::showOFderi(splitover[0],splitover[1],\
				splitover[2],splitover[3],appearanceover,5,false);
			splitover[0].release(); splitover[1].release();
			splitover[2].release(); splitover[3].release();
		}else{
			assert(splitover.size()==2);
			arrowsover = MotionPatchClass::showOF(splitover[0],splitover[1],\
				appearanceover,5,false);
			splitover[0].release(); splitover[1].release();
		}
		predictionover.release();
		std::string treenr = Auxiliary<int,1>::number2string(t-perTree.begin());
		// [4] Write the images somewhere locally
		boost::filesystem::path full_path = boost::filesystem::current_path();
		std::string parentpath = path2results+PATH_SEP+"results_parent"+PATH_SEP;
		Auxiliary<char,1>::file_exists(parentpath.c_str(),true);
		std::vector<int> params;
		params.push_back(CV_IMWRITE_JPEG_QUALITY);
		params.push_back(100);
		if(display){
			cv::imshow(("prediction "+treenr),arrows);
			cv::waitKey(10);
			cv::imshow(("prediction overlap "+treenr),arrowsover);
			cv::waitKey(10);
			cv::imshow(("appearance overlap "+treenr),appearanceover);
			cv::waitKey(10);
			cv::imshow(("appearance "+treenr),appearance);
			cv::waitKey(10);
		}
		std::string outpredi      = parentpath+"prediction"+treenr+"_im"+imname+".jpg";
		std::string outprediover  = parentpath+"prediction_overlap"+treenr+"_im"+imname+".jpg";
		std::string outappear     = parentpath+"appearance"+treenr+"_im"+imname+".jpg";
		std::string outappearover = parentpath+"appearance_overlap"+treenr+"_im"+imname+".jpg";
		cv::imwrite(outpredi,arrows,params);
		cv::imwrite(outprediover,arrowsover,params);
		cv::imwrite(outappearover,appearanceover,params);
		cv::imwrite(outappear,appearance,params);
		appearance.release(); appearanceover.release();
		arrowsover.release(); arrows.release();
	}
}
//==============================================================================
/** Propose the final motion derivative prediction over overlapping neighborhoods.
 */
template <class P>
cv::Mat MotionPuzzle<P>::proposePredictionOverlapDerivatives(const std::vector<P*> \
&candidates,const cv::Size &featsize,unsigned motionW,unsigned motionH,unsigned step,\
MotionTreeClass::ENTROPY entropy,cv::Mat &appearance){
	std::cout<<"[MotionPuzzle<P>::proposePredictionOverlapDerivatives] motion prediction"<<std::endl;
	assert(motionW>step && motionH>step);
	unsigned scaleX  = motionW-step;
	unsigned scaleY  = motionH-step;
	cv::Mat motionXX = cv::Mat::zeros(cv::Size(featsize.width*scaleX,\
		featsize.height*scaleY),CV_32FC1);
	cv::Mat motionXY = cv::Mat::zeros(cv::Size(featsize.width*scaleX,\
		featsize.height*scaleY),CV_32FC1);
	cv::Mat motionYX = cv::Mat::zeros(cv::Size(featsize.width*scaleX,\
		featsize.height*scaleY),CV_32FC1);
	cv::Mat motionYY = cv::Mat::zeros(cv::Size(featsize.width*scaleX,\
		featsize.height*scaleY),CV_32FC1);
	appearance       = cv::Mat::zeros(cv::Size(featsize.width*scaleX,\
		featsize.height*scaleY),CV_8UC3);
	// [0] Loop over candidate patches for current tree
	for(typename std::vector<P*>::const_iterator c=candidates.begin();c!=\
	candidates.end();++c){
		cv::Mat amotionXX = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		cv::Mat amotionXY = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		cv::Mat amotionYX = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		cv::Mat amotionYY = cv::Mat(cv::Size(motionW,motionH),CV_32FC1);
		(*c)->motion(amotionXX,amotionXY,amotionYX,amotionYY,motionW,motionH);
		cv::Mat appear      = (*c)->appearance();
		int patchpos        = c-candidates.begin();
		cv::Point initpoint = (*c)->center();
		cv::Point point;
		point.x = initpoint.x*(motionW-step);
		point.y = initpoint.y*(motionH-step);
		// [1.2] Loop over the pixels inside the patch
		cv::MatConstIterator_<cv::Vec3b> a=appear.begin<cv::Vec3b>();
		cv::MatConstIterator_<float> mXY=amotionXY.begin<float>();
		cv::MatConstIterator_<float> mYX=amotionYX.begin<float>();
		cv::MatConstIterator_<float> mYY=amotionYY.begin<float>();
		for(cv::MatConstIterator_<float> mXX=amotionXX.begin<float>();mXX!=\
		amotionXX.end<float>(),mXY!=amotionXY.end<float>(),mYX!=\
		amotionYX.end<float>(),mYY!=amotionYY.end<float>(),a!=appear.end<cv::Vec3b>();\
		++mXX,++a,++mXY,++mYY,++mYX){
			// [6] For each point get the distance to its mean
			unsigned pos                  = mXX-(amotionXX.begin<float>());
			cv::Point pt                  = (*c)->pos2pt(pos,motionW,motionH);
			pt.x                          = pt.x+point.x-initpoint.x;
			pt.y                          = pt.y+point.y-initpoint.y;
			float valXX                   = (*mXX);
			float valXY                   = (*mXY);
			float valYX                   = (*mYX);
			float valYY                   = (*mYY);
			motionXX.at<float>(pt)       += valXX;
			motionXY.at<float>(pt)       += valXY;
			motionYX.at<float>(pt)       += valYX;
			motionYY.at<float>(pt)       += valYY;
			appearance.at<cv::Vec3b>(pt) += (*a);
		}
		amotionXX.release();amotionXY.release();amotionYX.release();amotionYY.release();
	}
	cv::Mat in[] = {motionXX,motionXY,motionYX,motionYY};
	cv::Mat motion;
	cv::merge(in,4,motion);
	motionXX.release(); motionXY.release(); motionYX.release(); motionYY.release();
	return motion;
}
//==============================================================================
/** Propose the final flow prediction over overlapping neighborhoods.
 */
template <class P>
cv::Mat MotionPuzzle<P>::proposePredictionOverlapFlows(const std::vector<P*> \
&candidates,const cv::Size &featsize,unsigned motionW,unsigned motionH,unsigned step,\
MotionTreeClass::ENTROPY entropy,cv::Mat &appearance){
	std::cout<<"[MotionPuzzle<P>::proposePredictionOverlapFlows] motion prediction"<<std::endl;
	assert(motionW>step && motionH>step);
	unsigned scaleX = motionW-step;
	unsigned scaleY = motionH-step;
	cv::Mat motionX = cv::Mat::zeros(cv::Size(featsize.width*scaleX,\
		featsize.height*scaleY),CV_32FC1);
	cv::Mat motionY = cv::Mat::zeros(cv::Size(featsize.width*scaleX,\
		featsize.height*scaleY),CV_32FC1);
	appearance      = cv::Mat::zeros(cv::Size(featsize.width*scaleX,\
		featsize.height*scaleY),CV_8UC3);
	// [0] Loop over candidate patches for current tree
	for(typename std::vector<P*>::const_iterator c=candidates.begin();c!=\
	candidates.end();++c){
		cv::Mat motion      = (*c)->motion();
		cv::Mat appear      = (*c)->appearance();
		int patchpos        = c-candidates.begin();
		cv::Point initpoint = (*c)->center();
		cv::Point point;
		point.x = initpoint.x*(motionW-step);
		point.y = initpoint.y*(motionH-step);
		// [1.2] Loop over the pixels inside the patch
		cv::MatConstIterator_<cv::Vec3b> a=appear.begin<cv::Vec3b>();
		for(cv::MatConstIterator_<float> m=motion.begin<float>();m!=\
		motion.end<float>()-(motion.cols/2),a!=appear.end<cv::Vec3b>();++m,++a){
			// [6] For each point get the distance to its mean
			unsigned pos                 = m-(motion.begin<float>());
			cv::Point pt                 = (*c)->pos2pt(pos,motionW,motionH);
			pt.x                         = pt.x+point.x-initpoint.x;
			pt.y                         = pt.y+point.y-initpoint.y;
			float valX                   = (*m);
			float valY                   = (*(m+(motion.cols/2)));
			motionX.at<float>(pt)       += valX;
			motionY.at<float>(pt)       += valY;
			appearance.at<cv::Vec3b>(pt)+= (*a);
		} // inside the patch
	} // over patches
	cv::Mat in[] = {motionX,motionY};
	cv::Mat motion;
	cv::merge(in,2,motion);
	motionX.release(); motionY.release();
	return motion;
}
//==============================================================================
/** Propose the final prediction over overlapping neighborhoods.
 */
template <class P>
cv::Mat MotionPuzzle<P>::proposePredictionOverlap(const std::vector<P*> &candidates,\
const cv::Size &featsize,unsigned motionW,unsigned motionH,unsigned step,\
MotionTreeClass::ENTROPY entropy,cv::Mat &appearance,bool usederivatives){
	if(usederivatives){
		return MotionPuzzle<P>::proposePredictionOverlapDerivatives(candidates,\
			featsize,motionW,motionH,step,entropy,appearance);
	}else{
		return MotionPuzzle<P>::proposePredictionOverlapFlows(candidates,featsize,\
			motionW,motionH,step,entropy,appearance);
	}
}
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
template class MotionPuzzle<MotionPuzzlePatch>;



