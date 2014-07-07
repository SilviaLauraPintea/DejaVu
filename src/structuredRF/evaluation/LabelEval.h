/* LabelEval.h
 * Author: Silvia-Laura Pintea
 */
#ifndef LABELEVAL_H_
#define LABELEVAL_H_
#include <Auxiliary.h>
//==============================================================================
class LabelEval {
	public:
		enum METHOD {GLOBAL,AVG_CLASS,AVG_PASCAL,ALL};
		LabelEval(const std::string &config);
		virtual ~LabelEval();
		/** Gets the global scores: percentage of pixels correctly classified.
		 */
		float global(std::string &predi);
		/** Average recall of all classes: avg_cls TP[cls]/(TP[cls]+FN[cls]).
		 */
		float avgClass(std::string &predi);
		/** Average intersection vs. union: avg_cla TP[cls]/(TP[cls]+FN[cls]+FP[cls]).
		 */
		float avgPascal(std::string &predi);
		/** Converts the input matrix to label IDs.
		 */
		cv::Mat color2label(const cv::Mat &mat);
		/** Evaluates one or all of the above methods.
		 */
		void run(LabelEval::METHOD method);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		std::map<cv::Vec3b,unsigned,vec3bCompare> classinfo() const{
			return this->classinfo_;
		}
		std::string path2gt() const{return this->path2gt_;};
		std::string path2predi() const{return this->path2predi_;};
		std::string path2mpredi() const{return this->path2mpredi_;};
		std::string ext() const{return this->ext_;};
		unsigned consideredCls() const{return this->consideredCls_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void classinfo(const std::map<cv::Vec3b,unsigned,vec3bCompare> &classinfo){
			this->classinfo_ = classinfo;
		}
		void path2gt(const std::string &path2gt){this->path2gt_ = path2gt;};
		void path2predi(const std::string &path2predi){this->path2predi_ = path2predi;};
		void path2mpredi(const std::string &path2mpredi){this->path2mpredi_ = path2mpredi;};
		void ext(const std::string &ext){this->ext_ = ext;};
		void consideredCls(unsigned consideredCls){this->consideredCls_ = consideredCls;}
		//----------------------------------------------------------------------
	private:
		/** @var classinfo_
		 * The mapping from colors to class labels.
		 */
		std::map<cv::Vec3b,unsigned,vec3bCompare> classinfo_;
		/** @var path2gt_
		 * Path to the ground truth labels.
		 */
		std::string path2gt_;
		/** @var path2predi_
		 * Path to the prediction files.
		 */
		std::string path2predi_;
		/** @var path2mpredi_
		 * Path to the motion prediction files.
		 */
		std::string path2mpredi_;
		/** @var ext_
		 * The extension of the files to consider for both ground truth and
		 * predictions: e.g ".png".
		 */
		std::string ext_;
		/** @var labTerm_
		 * Termination concatenated to the end of label names: e.g. "_L"
		 */
		std::string labTerm_;
		/** @var consideredCls_
		 * Number of classes we consider.
		 */
		unsigned consideredCls_;
		/** @var labW_
		 * Width of label patch.
		 */
		unsigned labW_;
		/** @var labH_
		 * Height of label patch.
		 */
		unsigned labH_;
		/** @var featW_
		 * Width of feature patch.
		 */
		unsigned featW_;
		/** @var featH_
		 * Height of feature patch.
		 */
		unsigned featH_;
	private:
		DISALLOW_COPY_AND_ASSIGN(LabelEval);
};
//==============================================================================
#endif /* LABELEVAL_H_ */
