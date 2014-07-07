/* RunMotionRF.h
 * Author: Silvia-Laura Pintea
 */
#ifndef RUNMOTIONRF_H_
#define RUNMOTIONRF_H_
#include <RunRF.h>
#include "MotionRFdetector.h"
//==============================================================================
/** Performs all the administrative bits, calls and reading, writing, path settings.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
class RunMotionRF:public RunRF<L,M,T,F,N,U>{
	public:
		/** Modes of running the RF code.
		 */
		enum MODE {TRAIN_RF,TEST_RF,TRAIN_TEST_RF,EXTRACT,EXTRACT_OF,TRAIN1,\
			TEST1,EXTRACT_OF1};
		RunMotionRF(const char* config);
		virtual ~RunMotionRF(){};
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Initialize and start training.
		 */
		virtual void run(RunMotionRF::MODE mode,const std::vector<std::string> \
			&argv=std::vector<std::string>());
		/** Initialize and start training.
		 */
		virtual void runTrain(const std::vector<std::string> &argv);
		/** Initialize and start detector on test set.
		 */
		virtual void runTest(const std::vector<std::string> &argv);
		/** Extract the training/test features (what for?).
		 */
		virtual void runExtract();
		/** Extracts only OF features.
		 */
		virtual void runExtractOF(const std::vector<std::string> &argv);
		/** Extract features from 1 image only with jobrunners.
		 */
		virtual void runExtract1(const std::vector<std::string> &argv);
		/** Predicts on 1 image only with the jobrunners.
		 */
		virtual void runTest1(const std::vector<std::string> &argv);
		/** Starts the training for 1 tree only (with jobrunners).
		 */
		virtual void runTrain1(const std::vector<std::string> &argv);
		/** Performs the RF detection on test images.
		 */
		virtual void test(MotionRFdetector<L,M,T,F,N,U> &crDetect,\
			std::string &testpath,std::string &resultpath,std::string &featpath);
		/** Predicts on a set of test images in batch mode.
		 */
		virtual void batchTest();
		/** Trains the complete RF on the data set in a batch mode + threading.
		 */
		virtual void batchTrain();
		/** Extracts feature/label patches from all the images in a batch mode.
		 */
		virtual void batchExtractOF();
		/** Starts the jobrunner commands for testing 1 image at a time.
		 */
		virtual void jobrunnerTest(const std::vector<std::string> &argv);
		/** Trains each tree separately with a jobrunner.
		 */
		virtual void jobrunnerTrain(const std::vector<std::string> &argv);
		/** Generating commands to extract patches from every image separately.
		 */
		virtual void jobrunnerExtractOF(const std::vector<std::string> &argv);
		/** Recursively reads the images from the test/train folder[s] and generates the
		 * commands to extract them with jobrunners.
		 */
		virtual void generateExtractCommands(const std::vector<std::string> &argv,\
			std::string &path2ims,const std::string &cwd,std::string &featpath);
		/** Recursively reads the images from the test folder[s].
		 */
		void generateTestCommands(const std::vector<std::string> &argv,\
			std::string &path2ims,std::string &path2results,\
			const std::string &cwd,std::string &featpath);
		/** Recursively looks into the directories until it find the images it needs to
		 * extract OF features from.
		 */
		void extractOF(std::string &path2img,std::string &path2feat);
		/** Recursively looks into the directories until it find the images it needs to
		 * extract features from.
		 */
		void extract(std::string &path2img,std::string &path2feat);
		/** Generate action recognition configs for the action rwecognition part.
 		 */
		static std::map<unsigned,std::vector<std::string> > generateConfigsAR(const std::string &path2ar,\
			std::string &path2test,std::string &path2train,std::string &path2testLabs,\
			std::string &path2trainLabs,const std::string &addition,const std::map\
			<unsigned,std::string> &changes,const std::deque<std::string> &allClasses,\
			const std::vector<std::string> &confRF,const std::string &path2results);
		/** Generate random forest configs.
 		 */
		static std::vector<std::string> generateConfigsRF(const std::string &path2results,\
			const std::string &path2models,const std::string &path2train,\
			const std::string &path2test,const std::string &addition,const std::map\
			<unsigned,std::string> &changesms,std::vector<std::string> &confFiles,\
			const std::deque<std::string> &allClasses);
		/** Generate the RF and AR configs.
		 */
		static void generateConfigs(const std::string &arCmmd,const std::string &rfCmmd,\
			const char* config, bool run);
		//----------------------------------------------------------------------
		//---SVR baseline-------------------------------------------------------
		//----------------------------------------------------------------------
		/** Trains and saves the model.
		 */
		void justTrain(unsigned treeId,const M &train);
		void trainSVR(const M &train);
		/** Predicts at every pixel using the SRF.
		 */
		void testSRF(const std::string &justname,const std::string &testpath,\
			const std::string &featpath,const std::string &resultpath,const \
			std::string &motionstr,const std::string &testFile,const std::string \
			&arrowsstr,const std::string &appearancestr);
		/** Predicts at every pixel using the SVR.
		 */
		void testSVR(const std::string &justname,const std::string &testpath,\
			const std::string &featpath,const std::string &resultpath,\
			const std::string &motionstr,const std::string &testFile);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		/** Getters for the class members.
		 */
		unsigned motionWidth() const {return this->motionWidth_;}
		unsigned motionHeight() const {return this->motionHeight_;}
		std::vector<float> sigmas() const {return this->sigmas_;}
		bool warpping() const {return this->warpping_;}
		bool ofThresh() const {return this->ofThresh_;}
		bool leafavg() const {return this->leafavg_;}
		bool parentfreq() const {return this->parentfreq_;}
		bool leafparentfreq() const {return this->leafparentfreq_;}
		float entropythresh() const {return this->entropythresh_;}
		std::string configfile() const {return this->configfile_;}
		unsigned serverport() const {return this->serverport_;}
		bool dryrun() const {return this->dryrun_;}
		unsigned bins() const {return this->bins_;}
		bool multicls() const {return this->multicls_;}
		bool usederivatives() const {return this->usederivatives_;}
		bool usepick() const {return this->usepick_;}
		unsigned trainstep() const {return this->trainstep_;}
		unsigned teststep() const {return this->teststep_;}
		bool hogORsift() const {return this->hogORsift_;}
		unsigned pttype() const {return this->pttype_;}
		unsigned growthtype() const {return this->growthtype_;}
		unsigned maxleaves() const {return this->maxleaves_;}
		unsigned maxsize() const {return this->maxsize_;}
		std::string core() const {return this->core_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void motionWidth(unsigned motionWidth){this->motionWidth_ = motionWidth;}
		void motionHeight(unsigned motionHeight){this->motionHeight_ = motionHeight;}
		void sigmas(const std::vector<float> &sigmas){this->sigmas_ = sigmas;}
		void warpping(bool warpping){this->warpping_ = warpping;}
		void ofThresh(bool ofThresh){this->ofThresh_ = ofThresh;}
		void leafavg(bool leafavg){this->leafavg_= leafavg;}
		void parentfreq(bool parentfreq){this->parentfreq_ = parentfreq;}
		void leafparentfreq(bool leafparentfreq){this->leafparentfreq_ = leafparentfreq;}
		void entropythresh(float entropythresh){this->entropythresh_ = entropythresh;}
		void configfile(std::string configfile){this->configfile_ = configfile;};
		void serverport(unsigned serverport){this->serverport_ = serverport;}
		void dryrun (bool dryrun){this->dryrun_ = dryrun;}
		void bins(unsigned bins){this->bins_ = bins;}
		void multicls(bool multicls){this->multicls_ = multicls;}
		void usederivatives(bool usederivatives){this->usederivatives_ = usederivatives;}
		void usepick(bool usepick){this->usepick_ = usepick;}
		void trainstep(unsigned trainstep){this->trainstep_ = trainstep;}
		void teststep(unsigned teststep){this->teststep_ = teststep;}
		void hogORsift(bool hogORsift){this->hogORsift_ = hogORsift;}
		void pttype(unsigned pttype){this->pttype_ = pttype;}
		void growthtype(unsigned growthtype){this->growthtype_ = growthtype;}
		void maxleaves(unsigned maxleaves){this->maxleaves_ = maxleaves;}
		void core(const std::string &core){this->core_ = core;}
		void maxsize(unsigned maxsize){this->maxsize_ = maxsize;}
		//----------------------------------------------------------------------
	private:
		/** @var motionWidth_
		* The width of the motion patch.
		*/
		unsigned motionWidth_;
		/** @var motionHeight_
		* The width of the motion patch.
		*/
		unsigned motionHeight_;
		/** @var sigmas_
		* The ratio of binsize for the bandwidth for the kernel density estimation.
		*/
		std::vector<float> sigmas_;
		/** @var warpping_
		* If we warp the image or not.
		*/
		bool warpping_;
		/** @var ofThresh_
		* If we threshold the OF labels or not.
		*/
		bool ofThresh_;
		/** @var leafavg_
		* If we take the avg over patches in leaf.
		*/
		bool leafavg_;
		/** @var parentfreq_
		* The parent frequency or not.
		*/
		bool parentfreq_;
		/** @var leafparentfreq_
		* The parent frequency or not in the leaves.
		*/
		bool leafparentfreq_;
		/** @var entropythresh_
		 * The entropy threshold for making a leaf.
		 */
		float entropythresh_;
		/** @var configfile_
		 * The config file name from where we load stuff.
		 */
		std::string configfile_;
		/** @var serverport_
		 * The server port for the jobrunners.
		 */
		unsigned serverport_;
		/** @var dryrun_
		 * If we make a dry run or a full one (with the jobrunners).
		 */
		bool dryrun_;
		/** @var bins_
		 * the number of bins in the precomputed histograms.
		 */
		unsigned bins_;
		/** @var multicls_
		 * If the forest is trained on multiple classes or not.
		 */
		bool multicls_;
		/** @var usederivatives_
		 * If we use flow or flow derivatives as predictions.
		 */
		bool usederivatives_;
		/** @var usepick_
		 * If we use a random position or the full patch with independence assumption.
		 */
		bool usepick_;
		/** @var trainstep_
		 * The patch sampling step for training.
		 */
		unsigned trainstep_;
		/** @var teststep_
		 * The patch sampling step for test.
		 */
		unsigned teststep_;
		/** @var hogORsift_
		 * HOG - 1 and SIFT - 0.
		 */
		bool hogORsift_;
		/** @var pttype_
		 * The point types at which we get patches: canny, harris
		 */
		unsigned pttype_;
		/** @var growthtype_
		 * The tree growing type: depth, breadth, worst
		 */
		unsigned growthtype_;
		/** @var maxleaves_
		 * The maximum number of leaves for depth and worst.
		 */
		unsigned maxleaves_;
		/** @var maxsize_
		 * The maximum image size.
		 */
		unsigned maxsize_;
		/** @var core_
		 * The core type in the jobrunner so I can start runner per cores.
		 */
		std::string core_;
		/** @var usrRF_
		 * Use RF or use something else.
		 */
		bool useRF_;
		/** @var linearKernel_
		 * If we use linear kernel in SVR.
		 */
		bool linearKernel_;
	private:
		DISALLOW_COPY_AND_ASSIGN(RunMotionRF);
};
//==============================================================================
#endif /* RUNMOTIONRF_H_ */
