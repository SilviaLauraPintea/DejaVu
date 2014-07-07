/* MotionEval.h
 * Author: Silvia-Laura Pintea
 */
#ifndef MOTIONEVAL_H_
#define MOTIONEVAL_H_
#include <Auxiliary.h>
//==============================================================================
/** Evaluating the End-Point-Error between predicted flow and estimated one.
 */
class MotionEval {
	public:
		enum METHOD {MEAN_EPE,FLOW,RAW4PYTHON,ALL,UNEXPECTED,UNEXPECTED_PREV};
		MotionEval(const std::string &config);
		virtual ~MotionEval();
		/** Load only the motion ground truth from the features.
		 */
		cv::Mat loadGTfromFeat(const std::string &path2feat);
		/** Gets the global motion error: sum-of-squares distance
		 * between motions.
		 */
		float meanEPE(bool fromfeat=false,bool display=false);
		/** Raw files to be used with python.
		 */
		void raw4python(bool fromfeat=false);
		/** Evaluates one or all of the above methods.
		 */
		void run(MotionEval::METHOD method);
		/** Show a flow image.
		 */
		void flow(bool fromfeat=false);
		/** Displays the flow as it is usually displayed.
		 */
		static cv::Mat flow2im(const cv::Mat &flow,bool resize=true);
		/** Gets unexpected event.
		 */
		float unexpectedEPE(bool fromfeat=false,bool display=false);
		/** Gets unexpected event looking at previous frame.
		 */
		float unexpectedEPEPrev(bool fromfeat=false,bool display=false);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		std::string path2predi() const {return this->path2predi_;}
		std::string path2gt() const {return this->path2gt_;};
		unsigned featW() const {return this->featW_;}
		unsigned featH() const {return this->featH_;}
		unsigned motionW() const {return this->motionW_;}
		unsigned motionH() const {return this->motionH_;}
		std::string ext() const {return this->ext_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void path2predi(const std::string &path2predi){this->path2predi_ = path2predi;}
		void path2gt(const std::string &path2gt){this->path2gt_ = path2gt;};
		void featW(unsigned featW){this->featW_ = featW;}
		void featH(unsigned featH){this->featH_ = featH;}
		void motionW(unsigned motionW){this->motionW_ = motionW;}
		void motionH(unsigned motionH){this->motionH_ = motionH;}
		void ext(const std::string &ext){this->ext_ = ext;}
		//----------------------------------------------------------------------
	private:
		/** @var ext_
		 * Extension of the images.
		 */
		std::string ext_;
		/** @var path2test_
		 * Path to the images.
		 */
		std::string path2test_;
		/** @var path2predi_
		 * Path to the predicted motion.
		 */
		std::string path2predi_;
		/** @var path2gt_
		 * Path to the ground truth motion (extracted features).
		 */
		std::string path2gt_;
		/** @var featW_
		 * The width of the feature patch.
		 */
		unsigned featW_;
		/** @var featH_
		 * The height of the feature patch.
		 */
		unsigned featH_;
		/** @var featW_
		 * The width of the feature patch.
		 */
		unsigned motionW_;
		/** @var featH_
		 * The height of the feature patch.
		 */
		unsigned motionH_;
	private:
		DISALLOW_COPY_AND_ASSIGN(MotionEval);
};
//==============================================================================
#endif /* MOTIONEVAL_H_ */
