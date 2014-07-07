/* RunMotionRF.cpp
 * Author: Silvia-Laura Pintea
 */
#include "RunMotionRF.h"
#include "MotionRF.h"
#include "MotionTreeNode.h"
#include <boost/thread/thread.hpp>
#include <boost/type_traits.hpp>
#include <dlib/svm.h>
//==============================================================================
/** Generate action recognition configs for the action recognition part.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
std::map<unsigned,std::vector<std::string> > RunMotionRF<L,M,T,F,N,U>::generateConfigsAR\
(const std::string &path2ar,std::string &path2test,std::string &path2train,std::string &path2testLabs,\
std::string &path2trainLabs,const std::string &addition,const std::map<unsigned,std::string> \
&changes,const std::deque<std::string> &allClasses,const std::vector<std::string> &confRF,\
const std::string &path2results){
	std::map<unsigned,std::vector<std::string> > defconf;
	std::vector<std::string> vec0;
	vec0.push_back("# [0] Core type for the jobrunners");
	vec0.push_back(confRF[1]);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(0,vec0));

	std::vector<std::string> vec1;
	vec1.push_back("# [1] Run name for logging");
	vec1.push_back("Exp"+addition);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(1,vec1));
	
	std::vector<std::string> vec2;
	vec2.push_back("# [2] Path to results (predictions)");
	vec2.push_back(path2ar +"results"+PATH_SEP);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(2,vec2));

	std::vector<std::string> vec3;
	vec3.push_back("# [3] The motion prediction classes");
	vec3.push_back(Auxiliary<unsigned,1>::number2string(allClasses.size()));
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(3,vec3));
	
	std::vector<std::string> vec4;
	vec4.push_back("# [4] Path to the training labels file");
	std::vector<std::string> trainLabs = Auxiliary<char,1>::listDir\
		(path2trainLabs,std::string(".txt"));
	for(std::vector<std::string>::iterator trL=trainLabs.begin();\
	trL!=trainLabs.end();++trL){
		vec4.push_back(path2trainLabs+(*trL));
	}
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(4,vec4));

	std::vector<std::string> vec5;
	vec5.push_back("# [5] Path to the test labels file");
	std::vector<std::string> testLabs = Auxiliary<char,1>::listDir\
		(path2testLabs,std::string(".txt"));
	for(std::vector<std::string>::iterator teL=testLabs.begin();\
	teL!=testLabs.end();++teL){
		vec5.push_back(path2testLabs+(*teL));
	}
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(5,vec5));

	std::vector<std::string> vec6;
	vec6.push_back("# [6] Paths to train labels predictions");
	std::string trPrediLabs = path2ar+"train_labels"+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(trPrediLabs.c_str(),true);
	for(std::deque<std::string>::const_iterator c=allClasses.begin();c!=\
	allClasses.end();++c){
		vec6.push_back(trPrediLabs+"Predi_"+(*c)+addition+".txt");
	}
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(6,vec6));

	std::vector<std::string> vec7;
	vec7.push_back("# [7] Paths to test labels predictions");
	std::string tePrediLabs = path2ar+"test_labels"+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(tePrediLabs.c_str(),true);
	for(std::deque<std::string>::const_iterator c=allClasses.begin();c!=\
	allClasses.end();++c){
		vec7.push_back(tePrediLabs+"Predi_"+(*c)+addition+".txt");
	}
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(7,vec7));

	std::vector<std::string> vec8;
	vec8.push_back("# [8] Path to the training images");
	vec8.push_back(path2train);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(8,vec8));

	std::vector<std::string> vec9;
	vec9.push_back("# [9] Path to the test images");
	vec9.push_back(path2test);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(9,vec9));

	std::vector<std::string> vec10;
	vec10.push_back("# [10] Path to the training motion");
	std::string lastTrain = Auxiliary<char,1>::getStringSplit\
		(path2train,PATH_SEP,-1);
	for(std::deque<std::string>::const_iterator c=allClasses.begin();c!=\
	allClasses.end();++c){
		vec10.push_back(path2results+"exp"+addition+PATH_SEP+"prediction_"+\
			(*c)+PATH_SEP+lastTrain+PATH_SEP);
	}
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(10,vec10));

	std::vector<std::string> vec11;
	vec11.push_back("# [11] Path to the test motion");
	std::string lastTest = Auxiliary<char,1>::getStringSplit\
		(path2test,PATH_SEP,-1);
	for(std::deque<std::string>::const_iterator c=allClasses.begin();c!=\
	allClasses.end();++c){
		vec11.push_back(path2results+"exp"+addition+PATH_SEP+"prediction_"+\
			(*c)+PATH_SEP+lastTest+PATH_SEP);
	}
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(11,vec11));

	std::vector<std::string> vec12;
	vec12.push_back("# [12] Path to the trained SVM models");
	vec12.push_back(path2ar+"svm"+PATH_SEP);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(12,vec12));
		
	std::vector<std::string> vec13;
	vec13.push_back("# [13] Path to extraced HOG-HOF descriptors (word images) only");
	vec13.push_back(path2ar+"feat"+PATH_SEP);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(13,vec13));

	std::vector<std::string> vec14;
	vec14.push_back("# [14] Image extension for both train and test");
	vec14.push_back(confRF[15]);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(14,vec14));
		
	std::vector<std::string> vec15;
	vec15.push_back("# [15] Patch size to be used for featues");
	vec15.push_back("18");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(15,vec15));

	std::vector<std::string> vec16;
	vec16.push_back("# [16] Class mapping from names to ids (non-zero)");
	vec16.push_back(Auxiliary<unsigned,1>::number2string(allClasses.size()));
	for(std::deque<std::string>::const_iterator c=allClasses.begin();c!=\
	allClasses.end();++c){
		vec16.push_back((*c)+" "+Auxiliary<unsigned,1>::number2string(c-allClasses.begin()));
	}
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(16,vec16));
	
	if(confRF[59]=="1"){ // MBH
		std::vector<std::string> vec17;
		vec17.push_back("# [17] Pyramid types: 0 - Spatial13, 1 - Spatial22, 2 - Fg_Bg, 3 - None");
		vec17.push_back("1 3 3 3");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(17,vec17));

		std::vector<std::string> vec18;
		vec18.push_back("# [18] Number of cells on x,y");
		vec18.push_back("3 4 4 4");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(18,vec18));

		std::vector<std::string> vec19;
		vec19.push_back("# [19] Number of cells on the temporal");
		vec19.push_back("3 1 1 1");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(19,vec19));
		
		std::vector<std::string> vec20;
		vec20.push_back("# [20] Number of bins for descriptors");
		vec20.push_back("3 8 9 9");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(20,vec20));
		
		std::vector<std::string> vec21;
		vec21.push_back("# [21] Feature types: 0 - HOG, 1- HOF, 2 - MBHx, 3 - MBHy");
		vec21.push_back("3 0 2 3");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(21,vec21));
	}else{ // HOF
		std::vector<std::string> vec17;
		vec17.push_back("# [17] Pyramid types: 0 - Spatial13, 1 - Spatial22, 2 - Fg_Bg, 3 - None");
		vec17.push_back("2 3 3");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(17,vec17));

		std::vector<std::string> vec18;
		vec18.push_back("# [18] Number of cells on x,y");
		vec18.push_back("2 4 4");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(18,vec18));

		std::vector<std::string> vec19;
		vec19.push_back("# [19] Number of cells on the temporal");
		vec19.push_back("2 1 1");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(19,vec19));
		
		std::vector<std::string> vec20;
		vec20.push_back("# [20] Number of bins for descriptors");
		vec20.push_back("2 8 9");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(20,vec20));
		
		std::vector<std::string> vec21;
		vec21.push_back("# [21] Feature types: 0 - HOG, 1- HOF, 2 - MBHx, 3 - MBHy");
		vec21.push_back("2 0 1");
		defconf.insert(std::pair<unsigned,std::vector<std::string> >(21,vec21));
	}
	std::vector<std::string> vec22;
	vec22.push_back("#[22] Step for extracting descriptors (every step pixel)");
	vec22.push_back("5");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(22,vec22));

	std::vector<std::string> vec23;
	vec23.push_back("# [23] Size of codebooks (HOG and HOF)");
	vec23.push_back("4000");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(23,vec23));

	std::vector<std::string> vec24;
	vec24.push_back("# [24] Number of folds in SVM training");
	vec24.push_back("0");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(24,vec24));

	std::vector<std::string> vec25;
	vec25.push_back("# [25] Kernle type: 1 - L1, 2- L2, 3 - SYMMETRIC_CHI2,"\
		" 4 - SYMMETRIC_CHI2_ABS, 5 - HIST_INTERSECTION, 6 - DOT_PROD");
	vec25.push_back("5");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(25,vec25));

	std::vector<std::string> vec26;
	vec26.push_back("# [26] Estimated (1) or predicted (0)");
	vec26.push_back("0");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(26,vec26));

	std::vector<std::string> vec27;
	vec27.push_back("# [27] Multiclass (1) or not (0)");
	vec27.push_back("0");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(27,vec27));

	std::vector<std::string> vec28;
	vec28.push_back("# [28] Server port for the jobrunners (0 means no bactch processing)");
	vec28.push_back(confRF[51]);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(28,vec28));

	std::vector<std::string> vec29;
	vec29.push_back("# [29] Dry run -- 1 (just printing) for jobrunners or not -- 0");
	vec29.push_back(confRF[53]);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(29,vec29));

	std::vector<std::string> vec30;
	vec30.push_back("# [30] Per video BoW (1) or per image (0)");
	vec30.push_back("0");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(30,vec30));
	
	std::vector<std::string> vec31;
	vec31.push_back("# [31] Flow derivatives (1) or just flows (0)");
	vec31.push_back(confRF[59]);
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(31,vec31));

	std::vector<std::string> vec32;
	vec32.push_back("# [32] Keypoints: dense - 0 or canny - 1");
	vec32.push_back("0");
	defconf.insert(std::pair<unsigned,std::vector<std::string> >(32,vec32));
	// [2] Now replace all these with the changes
	for(std::map<unsigned,std::string>::const_iterator m=changes.begin();m!=\
	changes.end();++m){
		assert(defconf.size()>(m->first));
		std::vector<std::string> tmp; 
		std::vector<std::string> aconf = defconf[m->first];
		tmp.push_back(aconf[0]);
		tmp.push_back(m->second);
		defconf[m->first] = tmp;
	}
	std::string arconf = path2ar+"config"+addition+".txt";
	std::ofstream conf(arconf.c_str(),std::ofstream::out);
	if(conf.is_open()){
		for(std::map<unsigned,std::vector<std::string> >::iterator m=defconf.begin();\
		m!=defconf.end();++m){
			for(std::vector<std::string>::iterator co=m->second.begin();co!=\
			m->second.end();++co){
				conf<<(*co)<<std::endl;
			}
			if(m->first==2 || m->first==14){conf<<std::endl;}
		}
	}
	conf.close();
	return defconf;
}
//==============================================================================
/** Generate random forest configs
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
std::vector<std::string> RunMotionRF<L,M,T,F,N,U>::generateConfigsRF(const std::string \
&path2results,const std::string &path2models,const std::string &path2train,\
const std::string &path2test,const std::string &addition,const std::map\
<unsigned,std::string> &changes,std::vector<std::string> &confFiles,const std::deque\
<std::string> &allClasses){
	std::vector<std::string> defconf;
	defconf.push_back("# [0] Core type for the jobrunners");
	defconf.push_back("core");
	defconf.push_back("# [1] Experiment name");
	defconf.push_back("[]");
	defconf.push_back("# [2] Path to the results directory");
	defconf.push_back("[]");
	defconf.push_back("# [3] Path to the training data");
	defconf.push_back("[]");
	defconf.push_back("# [4] Path to the test data");
	defconf.push_back("[]");
	defconf.push_back("# [5] Path to motion trees");
	defconf.push_back("[]");
	defconf.push_back("# [6] Path to the extracted features");
	defconf.push_back(path2results+"mfeat"+PATH_SEP);
	defconf.push_back("# [7] Image extension for reading images");
	defconf.push_back("jpg");
	defconf.push_back("# [8] [Width, Height] for the feature patches");
	defconf.push_back("32 32");
	defconf.push_back("# [9] [Width, Height] for the motion patches");
	defconf.push_back("32 32");
	defconf.push_back("# [10] Number of trees to use");
	defconf.push_back("11");
	defconf.push_back("# [11] Pyramid size followed by scales");
	defconf.push_back("1 1");
	defconf.push_back("# [12] Number of images used for training");
	defconf.push_back("20");
	defconf.push_back("# [13] Number of iterations/node during training");
	defconf.push_back("50");
	defconf.push_back("# [14] Entropy type: 0 - CENTER, 1 - RANDOM,"\
		" 2 - CENTER_RANDOM, 3 - MEAN_DIFF, 4 - APPROX_MAGNI_KERNEL,"\
		" 5 - APPROX_ANGLE_KERNEL");
	defconf.push_back("3");
	defconf.push_back("# [15] Training Step of the grid for sampling patches");
	defconf.push_back("1");
	defconf.push_back("# [16] Test Step of the grid for sampling patches");
	defconf.push_back("1");
	defconf.push_back("# [17] Save the trees to binary (1) files or text (0)");
	defconf.push_back("1");
	defconf.push_back("# [18] Sigma for the KDE: 4/2 derivatives, 2/1 flows (ratio of std)");
	defconf.push_back("1 0");
	defconf.push_back("# [19] Warping of the patches (yes - 1, no - 0)");
	defconf.push_back("1");
	defconf.push_back("# [20] Thresholding of the magnitudes (yes - 1 "\
		"[only makes sense with warping], no  - 0)");
	defconf.push_back("0");
	defconf.push_back("# [21] Avg in leaf or not");
	defconf.push_back("1");
	defconf.push_back("# [22] Parent frequency weights in splits [0 | 1]");
	defconf.push_back("0");
	defconf.push_back("# [23] Parent frequency weights in leaves [0 | 1]");
	defconf.push_back("0");
	defconf.push_back("# [24] Entropy thresholding for making leaves");
	defconf.push_back("0.1");
	defconf.push_back("# [25] Server port for jobrunners (0 for no jobrunners)");
	defconf.push_back("64000");
	defconf.push_back("# [26] Dry run for the jobrunners");
	defconf.push_back("0");
	defconf.push_back("# [27] The number of bins in the RF");
	defconf.push_back("0");
	defconf.push_back("# [28] Multiclass (1) or not (0)");
	defconf.push_back("0");
	defconf.push_back("# [29] Use flow derivatives (1) or not (0)");
	defconf.push_back("0");
	defconf.push_back("# [30] Use random pick (1) or full patch with independence (0)");
	defconf.push_back("0");
	defconf.push_back("# [31] HOG - 1 or SIFT - 0");
	defconf.push_back("1");
	defconf.push_back("# [32] Interest points for patches: 0 - Harris,"\
		" 1 - Canny, 2 - Dense");
	defconf.push_back("1");
	defconf.push_back("# [33] Tree growing type: 0 - depth, 1 - breadth, 2 - worst");
	defconf.push_back("2");
	defconf.push_back("# [34] Maximum number of leaves for depth and worst");
	defconf.push_back("1000");
	defconf.push_back("# [35] Maximum image scale (kth: 160)");
	defconf.push_back("300");
	// [1] No replace with changes
	for(std::map<unsigned,std::string>::const_iterator m=changes.begin();m!=\
	changes.end();++m){
		assert(defconf.size()>(m->first*2+1));
		defconf[m->first*2+1] = m->second;
	}
	// [2] Now loop over classes and generate cofing files + AR config
	boost::filesystem::path full_path(boost::filesystem::current_path());
	std::string cwd = full_path.string()+PATH_SEP;
	cwd += "exp"+addition+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(cwd.c_str(),true);
	for(std::deque<std::string>::const_iterator cls=allClasses.begin();cls!=\
	allClasses.end();++cls){
		// [6.1] First generate the default lines
		std::vector<std::string> localdefconf = defconf;
		localdefconf[3]  = "Exp"+addition;
		Auxiliary<uchar,1>::file_exists(path2results.c_str(),true);
		Auxiliary<uchar,1>::file_exists((path2results+"exp"+addition+PATH_SEP).c_str(),true);
		localdefconf[5]  = path2results+"exp"+addition+PATH_SEP+"prediction_"+(*cls)+PATH_SEP;
		localdefconf[7]  = path2train+(*cls)+PATH_SEP;
		localdefconf[9]  = path2test;
		Auxiliary<uchar,1>::file_exists(path2models.c_str(),true);
		Auxiliary<uchar,1>::file_exists((path2models+"exp"+addition+PATH_SEP).c_str(),true);
		localdefconf[11] = path2models+"exp"+addition+PATH_SEP+"trees_"+(*cls)+PATH_SEP;
		localdefconf[13] = path2results+"mfeat"+PATH_SEP;
		// [6.2] Now write the settings
		std::string oneconf = "exp"+addition+PATH_SEP+"config_"+(*cls)+addition+".txt";
		confFiles.push_back(oneconf);
		std::ofstream conf(oneconf.c_str(),std::ofstream::out);
		if(conf.is_open()){
			for(std::vector<std::string>::iterator cf=localdefconf.begin();cf!=\
			localdefconf.end();++cf){
				conf<<(*cf)<<std::endl;
				if(cf-localdefconf.begin()==15){conf<<std::endl;} //needed
			}
		}
		conf.close();
	}
	return defconf;
}
//==============================================================================
/** Generate the RF and AR configs.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::generateConfigs(const std::string &arCmmd,const std::string &rfCmmd,\
const char* config, bool run){
	std::ifstream in(config);
	unsigned charsize = 1000;
	char *buffer      = new char[charsize]();
	if(in.is_open()){
		// [0] Read the class names;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string classesStr = std::string(buffer);
		std::deque<std::string> allClasses = Auxiliary<char,1>::splitLine\
			(const_cast<char*>(classesStr.c_str()),' ');
		// [1] read path to RF train;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2train = std::string(buffer);
		path2train = std::string(buffer);
		path2train = Auxiliary<uchar,1>::trim(path2train);
		Auxiliary<uchar,1>::fixPath(path2train);
		// [2] read path to RF test;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2test = std::string(buffer);
		path2test = std::string(buffer);
		path2test = Auxiliary<uchar,1>::trim(path2test);
		Auxiliary<uchar,1>::fixPath(path2test);
		// [3] read path to RF results;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2results = std::string(buffer);
		path2results = Auxiliary<uchar,1>::trim(path2results);
		Auxiliary<uchar,1>::fixPath(path2results);
		Auxiliary<uchar,1>::file_exists(path2results.c_str(),true);
		// [4] read path to RF models;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2models = std::string(buffer);
		path2models = Auxiliary<uchar,1>::trim(path2models);
		Auxiliary<uchar,1>::fixPath(path2models);
		Auxiliary<uchar,1>::file_exists(path2models.c_str(),true);
		// [5] path to AR stuff;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2actionreco = std::string(buffer);
		path2actionreco = Auxiliary<uchar,1>::trim(path2actionreco);
		Auxiliary<uchar,1>::fixPath(path2actionreco);
		Auxiliary<uchar,1>::file_exists(path2actionreco.c_str(),true);
		// [6] path to AR train;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2arTrain = std::string(buffer);
		path2arTrain = Auxiliary<uchar,1>::trim(path2arTrain);
		Auxiliary<uchar,1>::fixPath(path2arTrain);
		// [7] path to AR test;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2arTest = std::string(buffer);
		path2arTest = Auxiliary<uchar,1>::trim(path2arTest);
		Auxiliary<uchar,1>::fixPath(path2arTest);
		// [8] path to AR train labels;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2arTrainLabs = std::string(buffer);
		path2arTrainLabs = Auxiliary<uchar,1>::trim(path2arTrainLabs);
		Auxiliary<uchar,1>::fixPath(path2arTrainLabs);
		// [9] path to AR test labels;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string path2arTestLabs = std::string(buffer);
		path2arTestLabs = Auxiliary<uchar,1>::trim(path2arTestLabs);
		Auxiliary<uchar,1>::fixPath(path2arTestLabs);
		// [10] map of changes for RF;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::string addition = "";
		std::map<unsigned,std::string> rfChanges;
		unsigned mapsize; in>>mapsize;
		for(unsigned s=0;s<mapsize;++s){
			unsigned pos; in>>pos;
			std::string newval; in.getline(buffer,charsize); 
			newval = std::string(buffer);
			newval = Auxiliary<uchar,1>::trim(newval);
			rfChanges.insert(std::pair<unsigned,std::string>(pos,newval));
			// [9.1] add it to the name
			for(unsigned i=0;i<newval.size();++i){
				if(newval[i]==' '){newval[i]='_';}
			}
			addition += "_"+newval;
		}
		// [11] map of changes for AR;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		std::map<unsigned,std::string> arChanges;
		in>>mapsize;
		for(unsigned s=0;s<mapsize;++s){
			unsigned pos; in>>pos;
			std::string newval; in.getline(buffer,charsize); 
			newval = std::string(buffer);
			newval = Auxiliary<uchar,1>::trim(newval);
			arChanges.insert(std::pair<unsigned,std::string>(pos,newval));
		}
		// [12] map of changes;
		delete [] buffer;
		// [13] Now generate the config files
		std::vector<std::string> confFiles;
		std::vector<std::string> rfConfs = RunMotionRF<L,M,T,F,N,U>::generateConfigsRF\
			(path2results,path2models,path2train,path2test,addition,rfChanges,\
			confFiles,allClasses);
		// [14] Now generate the config files
		std::string initpath2ar = path2actionreco;
		path2actionreco        += "exp"+addition+PATH_SEP;
		Auxiliary<uchar,1>::file_exists(path2actionreco.c_str(),true);
		std::map<unsigned,std::vector<std::string> > arConfs =  RunMotionRF<L,M,T,F,N,U>::\
			generateConfigsAR(path2actionreco,path2arTest,path2arTrain,path2arTestLabs,\
			path2arTrainLabs,addition,arChanges,allClasses,rfConfs,path2results);
		// [15] start running and wait and then start predicting
		if(run){
			std::string ext;
			if(rfConfs[35]=="1"){ext = ".bin";
			}else{ ext = ".txt";}
			// Check for trees to be done
			bool done = false;
			bool once = 0;
			std::map<unsigned,unsigned> treesdone;
			while(!done){
				done = true;
				std::deque<std::string>::iterator cls = allClasses.begin();
				for(std::vector<std::string>::iterator cf=confFiles.begin();cf!=\
				confFiles.end(),cls!=allClasses.end();++cf,++cls){
					bool classdone          = true;
					std::string modelsSaved = path2models+"exp"+addition+PATH_SEP+\
						"trees_"+(*cls)+PATH_SEP;
					if(!Auxiliary<uchar,1>::file_exists(modelsSaved.c_str(),false)){
						classdone = false;
						done      = false; 
					}else{
						std::vector<std::string> trees = Auxiliary<char,1>::listDir\
							(modelsSaved,ext,"tree");
						std::cout<<"Check... "<<modelsSaved<<" "<<trees.size()<<std::endl;
						if(trees.size()!=atoi(rfConfs[21].c_str())){
							classdone = false;
							done      = false;
						}else{
							treesdone.insert(std::pair<unsigned,unsigned>(cls-allClasses.begin(),1));
							unsigned donesofar = 0;
							for(std::map<unsigned,unsigned>::iterator td=treesdone.begin();td!=\
							treesdone.end();++td){
								donesofar += td->second;
							}
							classdone = true;
							if(donesofar!=allClasses.size()){done = false;
							}else{done = true; break;}
						}
					}
					if(!once && !classdone){
						std::string startRFtrain = "./"+rfCmmd+" 0 0 "+(*cf);
							std::cout<<"[RunMotion::generateConfigs] run: "<<\
							startRFtrain.c_str()<<std::endl;
						system(startRFtrain.c_str());
					}
				}
				if(!done){	
					std::cout<<"Waiting for the tree training ..."<<std::endl;
					boost::this_thread::sleep(boost::posix_time::seconds(30));
				}
				once = 1;
			}
			// Now check if for predictions to be done
			done = false;
			once = 0;
			while(!done){
				done = true;
				std::vector<std::string> trainPreds = arConfs[10];
				std::vector<std::string> testPreds  = arConfs[11];
				std::vector<std::string>::iterator tr = trainPreds.begin()+1;
				std::vector<std::string>::iterator cf = confFiles.begin();
				for(std::vector<std::string>::iterator te=testPreds.begin()+1;\
				tr!=trainPreds.end(),te!=testPreds.end(),cf!=confFiles.end();++te,++tr,++cf){
					bool clasdone     = true;	
					std::string trRes = (*tr);
					Auxiliary<uchar,1>::fixPath(trRes);
					std::string teRes = (*te);
					Auxiliary<uchar,1>::fixPath(teRes);
					if(rfConfs[59]=="1"){ // MBH
						trRes = trRes+"deri_motion"+PATH_SEP;
						teRes = teRes+"deri_motion"+PATH_SEP;
					}else{ // HOF
						trRes = trRes+"flow_motion"+PATH_SEP;
						teRes = teRes+"flow_motion"+PATH_SEP;
					}
					if(!Auxiliary<uchar,1>::file_exists(teRes.c_str(),false) || \
					!Auxiliary<uchar,1>::file_exists(trRes.c_str(),false)){
						clasdone = false;
						done     = false;
					}else{ 
						std::vector<std::string> trainFeat = Auxiliary<char,1>::listDir(trRes,ext);
						std::vector<std::string> testFeat  = Auxiliary<char,1>::listDir(teRes,ext);
						std::vector<std::string> trainIms  = Auxiliary<char,1>::listDir(path2arTrain,"."+rfConfs[15]);
						std::vector<std::string> testIms   = Auxiliary<char,1>::listDir(path2arTest,"."+rfConfs[15]);
						std::cout<<"check.."<<trRes<<"=="<<path2arTrain<<" && "<<teRes<<"=="<<path2arTest<<std::endl;
						std::cout<<"check.."<<trainFeat.size()<<"!="<<trainIms.size()<<" || "<<testFeat.size()<<\
							"!="<<testIms.size()<<std::endl;
						if(trainFeat.size()!=trainIms.size() || testFeat.size()!=testIms.size()){
							done     = false;
							clasdone = false;
						}
					}
					if(!clasdone && !once){
						std::string startRFtest = "./"+rfCmmd+" 0 1 "+(*cf);
						std::cout<<"[RunMotion::generateConfigs] run: "<<\
						startRFtest.c_str()<<std::endl;
						system(startRFtest.c_str());
					}
					if(!done){
						std::cout<<"Waiting for predictions ..."<<std::endl;
						boost::this_thread::sleep(boost::posix_time::seconds(30));
					}
				}	
				once = 1;			
			}
			std::string arConfFile = path2actionreco+"config"+addition+".txt";
			std::string commandAR  = initpath2ar+arCmmd+" 9 "+arConfFile;
			std::cout<<"[RunMotion::generateConfigs] run: "<<commandAR<<std::endl;
			system(commandAR.c_str());
		}
	}else{
		std::cerr<<"File not found "<<config<<std::endl;
		std::exit(-1);
	}
	in.close();
}
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
RunMotionRF<L,M,T,F,N,U>::RunMotionRF(const char* config){
	this->configfile_ = std::string(config);
	std::ifstream in(config);
	unsigned charsize   = 1000;
	char *buffer        = new char[charsize]();
	this->useRF_        = true;
	this->linearKernel_ = false;
	if(in.is_open()){
		// [0] Core type in the jobrunners;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->core_ = std::string(buffer);
		// [1] Run name for logs: std::string runName_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->runName_ = std::string(buffer);
		this->runName_ = Auxiliary<uchar,1>::trim(this->runName_);
		// [2] Path to results: std::string path2results_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2results_ = std::string(buffer);
		this->path2results_ = Auxiliary<uchar,1>::trim(this->path2results_);
		Auxiliary<uchar,1>::fixPath(this->path2results_);
		Auxiliary<uchar,1>::file_exists(this->path2results_.c_str(),true);
		// [3] Path to training data: std::string path2train_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2train_ = std::string(buffer);
		this->path2train_ = Auxiliary<uchar,1>::trim(this->path2train_);
		Auxiliary<uchar,1>::fixPath(this->path2train_);
		// [4] Path to test data: std::string path2test_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2test_ = std::string(buffer);
		this->path2test_ = Auxiliary<uchar,1>::trim(this->path2test_);
		Auxiliary<uchar,1>::fixPath(this->path2test_);
		// [5] Path to motion model: std::string path2model_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2model_ = std::string(buffer);
		this->path2model_ = Auxiliary<uchar,1>::trim(this->path2model_);
		Auxiliary<uchar,1>::fixPath(this->path2model_);
		Auxiliary<uchar,1>::file_exists(this->path2model_.c_str(),true);
		// [6] Path to motion features: std::string path2feat_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->path2feat_ = std::string(buffer);
		this->path2feat_ = Auxiliary<uchar,1>::trim(this->path2feat_);
		Auxiliary<uchar,1>::fixPath(this->path2feat_);
		Auxiliary<uchar,1>::file_exists(this->path2feat_.c_str(),true);
		// [7] Image extension: std::string ext_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		this->ext_ = std::string(buffer);
		this->ext_ = "."+Auxiliary<uchar,1>::trim(this->ext_);
		// [8] Feature patch sizes: unsigned patchWidth_;unsigned patchHeigth_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		in>>this->patchWidth_;in>>this->patchHeight_;
		// [9] Motion patch sizes: unsigned motionWidth_;unsigned motionHeigth_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);
		in>>this->motionWidth_;in>>this->motionHeight_;
		// [10] Number of trees: nsigned noTrees_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->noTrees_;
		// [11] Pyramid scales for prediction: std::vector<unsigned> pyrScales_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);unsigned ssize; in>>ssize;
		for(unsigned s=0;s<ssize;++s){
			float val;in>>val;
			if(s==0 && val!=1){
				std::cerr<<"[RunMotionRF::RunMotionRF] first level of the pyramid"<<\
					" should be 1"<<std::endl;
				throw std::exception();
			}
			this->pyrScales_.push_back(val);
		}
		// [12] Training size: unsigned trainSize_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->trainSize_;
		// [13] Irerations per node: unsigned iterPerNode_;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->iterPerNode_;
		// [14] Entropy type: typename StructuredTree<M,T,F,N,U>::ENTROPY entropy_;
		unsigned dummy;
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>dummy;
		this->entropy_ = static_cast<typename StructuredTree<M,T,F,N,U>::ENTROPY>(dummy);
		// [15] Step for sampling patches: unsigned trainstep_
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->trainstep_;
		// [16] Step for sampling patches: unsigned teststep_
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->teststep_;
		// [17] The feature patch should be the largest at all times
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->binary_;
		// [18] Sigma_ for kernel density estimation
		in.getline(buffer,charsize);in.getline(buffer,charsize);unsigned sigmasize; in>>sigmasize;
		for(unsigned ss=0;ss<sigmasize;++ss){
			float val;in>>val;
			this->sigmas_.push_back(val);
		}
		// [19] The warpping.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->warpping_;
		// [20] The OF labels thresholding.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->ofThresh_;
		// [21] The leaf avg.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->leafavg_;
		// [22] The parent freq.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->parentfreq_;
		// [23] The leaf parent frequency.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->leafparentfreq_;
		// [24] Threshold for the entropy
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->entropythresh_;
		// [25] Port number for the server
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->serverport_;
		// [26] For the jobrunners (should it be a dry run or full)
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->dryrun_;
		// [27] the number of bins to be used in the histograms.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->bins_;
		if(this->entropy_==MotionTree<M,T,F,N,U>::MEAN_DIFF){assert(this->bins_==0);}
		// [28] if the svm is multiclass or 1-vs-all.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->multicls_;
		// [29] if we use derivatives of flow or not.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->usederivatives_;
		// [30] full patch entropy or random position.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->usepick_;
		// [31] Use HOG or SIFT descriptors.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->hogORsift_;
		// [32] Points to extract: Harris, Canny.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->pttype_;
		// [33] Tree growing style: depth-first, breadth-first, worst first.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->growthtype_;
		// [34] Maximum number of leaves for the trees.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->maxleaves_;
		// [35] Maximum image size.
		in.getline(buffer,charsize);in.getline(buffer,charsize);in>>this->maxsize_;
		delete [] buffer;
		this->predMethod_ = Puzzle<PuzzlePatch>::SIMPLE;
	}else{
		delete [] buffer;
		std::cerr<<"File not found "<<config<<std::endl;
		std::exit(-1);
	}
	in.close();
}
//==============================================================================
/** Initialize and start training.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::runTrain(const std::vector<std::string> &argv){
	assert(argv.size()==2);
	if(this->serverport_){
		this->jobrunnerTrain(argv);
	}else{
		this->batchTrain();
	}
}
//==============================================================================
/** Trains each tree separately with a jobrunner.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::jobrunnerTrain(const std::vector<std::string> &argv){
	// [0] Just shoot the command to start the training of 1 tree.
	boost::filesystem::path full_path(boost::filesystem::current_path());
	std::string cwd = full_path.string()+PATH_SEP;
	// [1] For each tree start a jobrunner task
	for(unsigned t=0;t<this->noTrees_;++t){
		std::string command = "jobmanager localhost:"+Auxiliary<unsigned,1>::\
			number2string(this->serverport_)+" schedule RF"+(this->runName_)+\
			" "+this->core_+" \""+cwd+argv[0]+" "+argv[1]+" "+Auxiliary<unsigned,1>::\
			number2string(RunMotionRF<L,M,T,F,N,U>::TRAIN1)+" "+cwd+\
			(this->configfile_)+" "+Auxiliary<unsigned,1>::number2string(t)+"\"";
		if(this->dryrun_){
			std::cout<<command<<std::endl;
		}else{
			int returnval = system(command.c_str());
		}
	}
}
//==============================================================================
/** Trains the complete RF on the data set in a batch mode + threading.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::batchTrain(){
	// [0] Init forest with number of trees
	MotionRF<L,M,T,F,N,U> forest(this->noTrees_,this->hogORsift_);
	// [1] Create directory for storing the models
	Auxiliary<uchar,1>::file_exists(this->path2model_.c_str(),true);
	std::vector<std::string> files = Auxiliary<uchar,1>::listDir(this->path2train_);
	if(files.empty()){
		std::cerr<<"[RunMotionRF::run_train] No folder in the training path "<<\
			this->path2train_<<std::endl;
		throw std::exception();
	}
	// [2] Train each tree on a different subset (if needed)
	#if DO_PRAGMA
		#pragma omp parallel for schedule(dynamic,1)
	#endif
	// [3] Loop over the number of trees
	for(unsigned t=0;t<this->noTrees_;++t){
		// [4] Initialize random number generator
		time_t times = time(NULL);
		int seed     = (int)times+t;
		CvRNG cvRNG(seed);
		// [5] Initialize training data set
		M train(&cvRNG,this->patchWidth_,this->patchHeight_,this->classInfo_.\
			size(),this->labWidth_,this->labHeight_,this->trainSize_,this->noPatches_,\
			this->consideredCls_,this->balance_,this->trainstep_,this->motionWidth_,\
			this->motionHeight_,this->warpping_,this->ofThresh_,this->entropy_,\
			this->sigmas_,this->bins_,this->multicls_,this->usederivatives_,\
			this->hogORsift_,this->pttype_,this->maxsize_);
		// [6] Extract training features and training patches
		std::cout<<"[RunMotionRF::run_train]: Extracting patches for tree "<<t\
			<<".."<<std::endl;
		train.extractPatches(this->path2train_,this->path2labs_,this->path2feat_,\
			files,this->classInfo_,this->labTerm_,this->ext_,false,\
			(this->entropy_!=MotionTree<M,T,F,N,U>::MEAN_DIFF));
		// [7] Train the forest on this set of images
		std::cout<<"[RunMotionRF::run_train] Forest training ..."<<std::endl;
		clock_t begin = clock();
		forest.trainForestTree(1,1e+3,&cvRNG,train,this->iterPerNode_,t,\
			this->path2model_.c_str(),this->runName_,this->entropy_,\
			this->consideredCls_,this->binary_,this->leafavg_,\
			this->parentfreq_,this->leafparentfreq_,this->runName_,\
			this->entropythresh_,this->usepick_,this->hogORsift_,\
			this->growthtype_,this->maxleaves_);
		clock_t end   = clock();
		std::cout<<"Train forest time elapsed: "<<double\
			(Auxiliary<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
		// [8] Save all forest trees in the files
		std::cout<<"[RunMotionRF::run_train] Saving forest ..."<<std::endl;
		forest.saveTree(this->path2model_.c_str(),t);
	}
}
//==============================================================================
/** Trains the one tree on the data set with the jobrunners.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::runTrain1(const std::vector<std::string> &argv){
	if(argv.size()!=3){
		std::cerr<<"[RunMotionRF::runTrain1] expects: <<cmmd what option "<<\
			"config_file.txt tree_number>>"<<std::endl;
		throw std::exception();
	}
	unsigned treeId = atoi(argv[2].c_str());
	// [1] Create directory for storing the models
	std::vector<std::string> files = Auxiliary<uchar,1>::listDir(this->path2train_);
	if(files.empty()){
		std::cerr<<"[RunMotionRF::run_train] No folder in the training path "<<\
			this->path2train_<<std::endl;
		throw std::exception();
	}
	// [4] Initialize random number generator
	time_t times = time(NULL);
	int seed     = (int)times+treeId;
	CvRNG cvRNG(seed);
	// [5] Initialize training data set
	M train(&cvRNG,this->patchWidth_,this->patchHeight_,this->classInfo_.\
		size(),this->labWidth_,this->labHeight_,this->trainSize_,this->noPatches_,\
		this->consideredCls_,this->balance_,this->trainstep_,this->motionWidth_,\
		this->motionHeight_,this->warpping_,this->ofThresh_,this->entropy_,\
		this->sigmas_,this->bins_,this->multicls_,this->usederivatives_,\
		this->hogORsift_,this->pttype_,this->maxsize_);
	// [6] Extract training features and training patches
	std::cout<<"[RunMotionRF::run_train]: Extracting patches for tree "<<treeId\
		<<".."<<std::endl;
	train.extractPatches(this->path2train_,this->path2labs_,this->path2feat_,\
		files,this->classInfo_,this->labTerm_,this->ext_,false,\
		(this->entropy_!=MotionTree<M,T,F,N,U>::MEAN_DIFF));
	this->justTrain(treeId,train);
}
//==============================================================================
/** Trains and saves the model.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::justTrain(unsigned treeId,const M &train){
	if(this->useRF_){
		// [7] Train the forest on this set of images
		std::cout<<"[RunMotionRF::run_train] Forest training ..."<<std::endl;
		time_t times = time(NULL);
		int seed     = (int)times+treeId;
		CvRNG cvRNG(seed);
		MotionRF<L,M,T,F,N,U> forest(this->noTrees_,this->hogORsift_);
		forest.trainForestTree(1,1e+3,&cvRNG,train,this->iterPerNode_,treeId,\
				this->path2model_.c_str(),this->runName_,this->entropy_,\
				this->consideredCls_,this->binary_,this->leafavg_,this->parentfreq_,\
				this->leafparentfreq_,this->runName_,this->entropythresh_,this->usepick_,\
				this->hogORsift_,this->growthtype_,this->maxleaves_);
		Auxiliary<uchar,1>::file_exists(this->path2model_.c_str(),true);
		clock_t begin = clock();
		forest.trainForestTree(1,1e+3,&cvRNG,train,this->iterPerNode_,treeId,\
			this->path2model_.c_str(),this->runName_,this->entropy_,\
			this->consideredCls_,this->binary_,this->leafavg_,this->parentfreq_,\
			this->leafparentfreq_,this->runName_,this->entropythresh_,this->usepick_,\
			this->hogORsift_,this->growthtype_,this->maxleaves_);
		clock_t end   = clock();
		std::cout<<"Train forest time elapsed: "<<double\
			(Auxiliary<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
		// [8] Save all forest trees in the files
		std::cout<<"[RunMotionRF::run_train] Saving forest ..."<<std::endl;
		forest.saveTree(this->path2model_.c_str(),treeId);
	}else{
		this->trainSVR(train);
	}
}
//==============================================================================
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::trainSVR(const M &train){
	std::vector<std::vector<const T*> > patches = train.patches();
	F *features                                 = train.features();
	assert(patches.size()==1); // no classes, why I kept it so difficult?
	typedef dlib::matrix<double,0,1> sample_type;
	std::vector<sample_type> samples;
	std::vector<double> targetsX;
	std::vector<double> targetsY;
	std::vector<cv::Mat> torelease;
	std::cout<<"[RunMotionRF::trainSSVM] #train samples: "<<patches[0].size()<<std::endl;
	for(typename std::vector<const T*>::iterator p=patches[0].begin();p!=\
	patches[0].end();++p){
		cv::Mat afeat  = (*p)->featRowHack(features);
		afeat.convertTo(afeat,cv::DataType<double>::type);
		dlib::matrix<double,0,1> onesample = dlib::mat(reinterpret_cast<double*>\
			(afeat.data),afeat.cols,afeat.rows);
		float mX,mY;
		(*p)->motionCenter(features,mX,mY);
		samples.push_back(onesample);
		targetsX.push_back(static_cast<double>(mX));
		targetsY.push_back(static_cast<double>(mY));
		torelease.push_back(afeat);
	}
	// Train the two SVRs and save them locally
	if(this->linearKernel_){
		typedef dlib::linear_kernel<sample_type> kernel_type;
		dlib::svr_linear_trainer<kernel_type> trainerX;
		dlib::svr_linear_trainer<kernel_type> trainerY;
		trainerX.set_c(1.0);
		trainerY.set_c(1.0);
		// Now do the training and save the results
		std::string path2dfx = this->path2model_+"svr_linearX.bin";
		dlib::decision_function<kernel_type> dfX = trainerX.train(samples,targetsX);
		dlib::decision_function<kernel_type> dfY = trainerY.train(samples,targetsY);
		std::ofstream foutX(path2dfx.c_str(),std::ios::binary);
		dlib::serialize(dfX,foutX);
		foutX.close();
		std::string path2dfy = this->path2model_+"svr_linearY.bin";
		std::ofstream foutY(path2dfy.c_str(),std::ios::binary);
		dlib::serialize(dfY,foutY);
		foutY.close();
	}else{
		dlib::rls trainerX;
		dlib::rls trainerY;
		// Now do the training and save the results
		std::string path2dfx = this->path2model_+"rlsX.bin";
		std::vector<double>::iterator tgx = targetsX.begin();
		std::vector<double>::iterator tgy = targetsY.begin();
		for(std::vector<sample_type>::iterator spl=samples.begin();spl!=\
		samples.end(),tgx!=targetsX.end(),tgy!=targetsY.end();++spl,++tgx,++tgy){
			trainerX.train<sample_type>(*spl,*tgx);
			trainerY.train<sample_type>(*spl,*tgy);
		}
		dlib::decision_function<dlib::linear_kernel<sample_type> > dfX = \
			trainerX.get_decision_function();
		dlib::decision_function<dlib::linear_kernel<sample_type> > dfY = \
			trainerY.get_decision_function();
		std::ofstream foutX(path2dfx.c_str(),std::ios::binary);
		dlib::serialize(dfX,foutX);
		foutX.close();
		std::string path2dfy = this->path2model_+"rlsY.bin";
		std::ofstream foutY(path2dfy.c_str(),std::ios::binary);
		dlib::serialize(dfY,foutY);
		foutY.close();
	}
	for(std::vector<cv::Mat>::iterator r=torelease.begin();r!=torelease.end();++r){
		r->release();
	}
}
//==============================================================================
/** Starts the prediction.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::runTest(const std::vector<std::string> &argv){
	assert(argv.size()==2);
	if(this->serverport_){
		this->jobrunnerTest(argv);
	}else{
		this->batchTest();
	}
}
//==============================================================================
/** Starts the jobrunner commands for testing 1 image at a time.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::jobrunnerTest(const std::vector<std::string> &argv){
	boost::filesystem::path full_path(boost::filesystem::current_path());
	std::string cwd = full_path.string()+PATH_SEP;
	this->generateTestCommands(argv,this->path2test_,this->path2results_,cwd,\
		this->path2feat_);
}
//==============================================================================
/** Recursively reads the images from the test folder[s].
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::generateTestCommands(const std::vector<std::string> \
&argv,std::string &path2ims,std::string &path2results,const std::string &cwd,\
std::string &featpath){
	std::cout<<"[RunMotionRF::readImages]: "<<path2ims<<" "<<this->ext_<<std::endl;
	std::vector<std::string> testFiles = Auxiliary<uchar,1>::listDir(path2ims,this->ext_);
	// [0] List the directories and descend into them
	if(testFiles.empty()){
		testFiles = Auxiliary<uchar,1>::listDir(path2ims);
		for(std::vector<std::string>::iterator t=testFiles.begin();t!=\
		testFiles.end();++t){
			if(t->find(".")==0){continue;}
			std::string newpath2results = path2results+(*t)+PATH_SEP;
			std::cout<<"[RunMotionRF::readImages]: mkdir "<<newpath2results<<std::endl;
			Auxiliary<uchar,1>::file_exists(newpath2results.c_str(),true);
			std::string newpath2ims     = path2ims+(*t)+PATH_SEP;
			std::string newfeatpath     = featpath+(*t)+PATH_SEP;
			this->generateTestCommands(argv,newpath2ims,newpath2results,cwd,newfeatpath);
		}
	}else{
		// [1] Loop over all images in current directory and generate commands.
		for(std::vector<std::string>::iterator t=testFiles.begin();t!=\
		testFiles.end();++t){
			// [2] First check if the training is done, if not wait for it
			std::string commandWait = "jobmanager localhost:"+Auxiliary<unsigned,1>::\
				number2string(this->serverport_)+" wait RF"+(this->runName_)+\
				" "+this->core_+" 0 0 0";
			if(this->dryrun_){
				std::cout<<commandWait<<std::endl;
			}else{
				int returnval = system(commandWait.c_str());
			}
			// [3] If training done than start testing
			std::string command = "jobmanager localhost:"+Auxiliary<unsigned,1>::\
				number2string(this->serverport_)+" schedule RF"+(this->runName_)+\
				" "+this->core_+" \""+cwd+argv[0]+" "+argv[1]+" "+Auxiliary<unsigned,1>::\
				number2string(RunMotionRF<L,M,T,F,N,U>::TEST1)+" "+cwd+\
				(this->configfile_)+" "+(*t)+" "+path2results+" "+featpath+" "+\
				path2ims+"\"";
			if(this->dryrun_){
				std::cout<<command<<std::endl;
			}else{
				int returnval = system(command.c_str());
			}
		}
	}	
}
//==============================================================================
/** Predicts on 1 image only with the jobrunners.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::runTest1(const std::vector<std::string> &argv){
	if(argv.size()!=6){
		std::cerr<<"[RunMotionRF::runTest1] expects: <<cmmd what option "<<\
			"config_file.txt image_name results_path feature_path image_path>>"<<std::endl;
		throw std::exception();
	}
	std::string testFile   = argv[2];
	std::string resultpath = argv[3];
	std::string featpath   = argv[4];
	std::string testpath   = argv[5];
	// [3] Run detection for all scales
	std::string justname = testFile.substr(0,testFile.size()-4);
	clock_t begin        = clock();
	// First check for prediction, if there, return
	std::string motionstr, appearancestr, arrowsstr;
	if(this->usederivatives_){
		Auxiliary<uchar,1>::file_exists((resultpath+std::string("deri_motion")+\
			std::string(PATH_SEP)).c_str(),true);
		Auxiliary<uchar,1>::file_exists((resultpath+std::string("deri_appearance")+\
			std::string(PATH_SEP)).c_str(),true);
		Auxiliary<uchar,1>::file_exists((resultpath+std::string("deri_arrows")+\
			std::string(PATH_SEP)).c_str(),true);
		arrowsstr     = "deri_arrows";
		appearancestr = "deri_appearance";
		motionstr     = "deri_motion";
	}else{
		Auxiliary<uchar,1>::file_exists((resultpath+std::string("flow_motion")+\
			std::string(PATH_SEP)).c_str(),true);
		Auxiliary<uchar,1>::file_exists((resultpath+std::string("flow_appearance")+\
			std::string(PATH_SEP)).c_str(),true);
		Auxiliary<uchar,1>::file_exists((resultpath+std::string("flow_arrows")+\
			std::string(PATH_SEP)).c_str(),true);
		arrowsstr     = "flow_arrows";
		appearancestr = "flow_appearance";
		motionstr     = "flow_motion";
	}
	bool isthere = false;	
	for(unsigned k=0;k<this->pyrScales_.size();++k){
		std::string motionOut = (resultpath+motionstr+std::string\
			(PATH_SEP)+Auxiliary<int,1>::number2string(k)+"_"+testFile.\
			substr(0,testFile.size()-4)+".bin").c_str();
		if(Auxiliary<char,1>::file_exists(motionOut.c_str(),false)){
			isthere = true;
			break;
		}
	}
	// [3] If leave average then for consistency, take mean in prediction:
	if(this->useRF_){
		this->testSRF(justname,testpath,featpath,resultpath,motionstr,testFile,\
			arrowsstr,appearancestr);
	}else{
		this->testSVR(justname,testpath,featpath,resultpath,motionstr,testFile);
	}
}
//==============================================================================
/** Predicts at every pixel using the SRF.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::testSRF(const std::string &justname,const std::string \
&testpath,const std::string &featpath,const std::string &resultpath,const std::string \
&motionstr,const std::string &testFile,const std::string &arrowsstr,const std::string \
&appearancestr){
	// [2] Initialize output over pyramid
	std::vector<cv::Mat> vMotionDetect(this->pyrScales_.size());
	std::vector<cv::Mat> vArrowsDetect(this->pyrScales_.size());
	std::vector<cv::Mat> vAppearDetect(this->pyrScales_.size());
	if(this->leafavg_){
		this->entropy_ = MotionTree<M,T,F,N,U>::MEAN_DIFF;
	}
	// [3] Initialize forest with number of trees
	MotionRF<L,M,T,F,N,U> forest(this->noTrees_,this->hogORsift_);
	// [3] Load the forest trees
	forest.loadForestBin(this->path2model_,this->binary_);
	// [3] Initialize detector
	MotionRFdetector<L,M,T,F,N,U> crDetect(&forest,this->patchWidth_,this->patchHeight_,\
		this->classInfo_.size(),this->labWidth_,this->labHeight_,this->motionWidth_,\
		this->motionHeight_,this->predMethod_,this->teststep_,static_cast\
		<typename MotionTree<M,T,F,N,U>::ENTROPY >(this->entropy_),\
		this->usederivatives_,this->hogORsift_,this->pttype_,this->path2results_,this->maxsize_);
	// [4] Run detector over test images
	crDetect.detectPyramid(justname,testpath,featpath,this->ext_,this->pyrScales_,\
		vMotionDetect,vArrowsDetect,vAppearDetect,this->path2model_,0);
	// [4] Store each result over the scales for this image
	for(unsigned int k=0;k<vMotionDetect.size();++k){
		if(this->usederivatives_){
			assert(vMotionDetect[k].channels()==4);
		}else{
			assert(vMotionDetect[k].channels()==2);
		}
		// [5] Write the temporary image out (with actual values)
		// [6] Define the output path for the arrows also
		std::string arrowsOut = (resultpath+arrowsstr+std::string\
			(PATH_SEP)+Auxiliary<int,1>::number2string(k)+"_"+testFile).c_str();
		// [7] Define the output path for the motion
		std::string motionOut = (resultpath+motionstr+std::string\
			(PATH_SEP)+Auxiliary<int,1>::number2string(k)+"_"+testFile.\
			substr(0,testFile.size()-4)+".bin").c_str();
		std::string appearOut = (resultpath+appearancestr+\
			std::string(PATH_SEP)+Auxiliary<int,1>::number2string(k)+"_"+\
			testFile).c_str();
		// [8] Define the parameters for storing the arrows
		std::vector<int> params;
		if(strcmp(this->ext_.c_str(),".jpg")){
			params.push_back(CV_IMWRITE_JPEG_QUALITY);
			params.push_back(100);
			cv::imwrite(arrowsOut,vArrowsDetect[k],params);
			cv::imwrite(appearOut,vAppearDetect[k],params);
		}else if(strcmp(this->ext_.c_str(),".png")){
			params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			params.push_back(4);
			cv::imwrite(arrowsOut,vArrowsDetect[k],params);
			cv::imwrite(appearOut,vAppearDetect[k],params);
		}else{
			cv::imwrite(arrowsOut,vArrowsDetect[k]);
			cv::imwrite(appearOut,vAppearDetect[k],params);
		}
		// [9] For motion we write a binary file:
		if(this->usederivatives_){
			Auxiliary<float,4>::mat2bin(vMotionDetect[k],motionOut.c_str(),false);
		}else{
			Auxiliary<float,2>::mat2bin(vMotionDetect[k],motionOut.c_str(),false);
		}
		vMotionDetect[k].release();
		vArrowsDetect[k].release();
		vAppearDetect[k].release();
	}
	vArrowsDetect.clear();
	vMotionDetect.clear();
	vAppearDetect.clear();
}
//==============================================================================
/** Predicts at every pixel using the SVR.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::testSVR(const std::string &justname,const std::string \
&testpath,const std::string &featpath,const std::string &resultpath,const std::string \
&motionstr,const std::string &testFile){
	// Useless matrices
	if(this->leafavg_){
		this->entropy_ = MotionTree<M,T,F,N,U>::MEAN_DIFF;
	}
	// Dummy forest
	MotionRF<L,M,T,F,N,U> forest(this->noTrees_,this->hogORsift_);
	MotionRFdetector<L,M,T,F,N,U> crDetect(&forest,this->patchWidth_,this->patchHeight_,\
		this->classInfo_.size(),this->labWidth_,this->labHeight_,this->motionWidth_,\
		this->motionHeight_,this->predMethod_,this->teststep_,static_cast\
		<typename MotionTree<M,T,F,N,U>::ENTROPY >(this->entropy_),\
		this->usederivatives_,this->hogORsift_,this->pttype_,this->path2results_,this->maxsize_);
	// Load features
	cv::Size imsize;
	F* features = new F(this->usederivatives_);
	std::vector<std::vector<const T*> > patches = crDetect.justfeatures(justname,\
		testpath,this->ext_,featpath,this->pyrScales_,features,imsize);
	assert(patches.size()==1); // no classes, why I kept it so difficult?
	typedef dlib::matrix<double,0,1> sample_type;
	typedef dlib::linear_kernel<sample_type> kernel_type;
	std::string path2dfx,path2dfy;
	if(this->linearKernel_){
		path2dfx = this->path2model_+"svr_linearX.bin";
		path2dfy = this->path2model_+"svr_linearY.bin";
	}else{
		path2dfx = this->path2model_+"rlsX.bin";
		path2dfy = this->path2model_+"rlsY.bin";
	}
	std::ifstream foutX(path2dfx.c_str(),std::ios::binary);
	dlib::decision_function<kernel_type> dfX;
	dlib::deserialize(dfX,foutX);
	foutX.close();
	dlib::decision_function<kernel_type> dfY;
	std::ifstream foutY(path2dfy.c_str(),std::ios::binary);
	dlib::deserialize(dfY,foutY);
	foutY.close();
	std::cout<<"[RunMotionRF::testSVR] #test samples: "<<patches[0].size()<<std::endl;
	cv::Rect imRoi = cv::Rect(this->motionWidth_/2,this->motionHeight_/2,\
		(imsize.width-this->motionWidth_),(imsize.height-this->motionHeight_));
	cv::Mat vMotionDetect = cv::Mat::zeros(imsize,CV_32FC2);
	for(typename std::vector<const T*>::iterator p=patches[0].begin();p!=\
	patches[0].end();++p){
		cv::Mat afeat = (*p)->featRowHack(features);
		afeat.convertTo(afeat,cv::DataType<double>::type);
		dlib::matrix<double,0,1> onesample = dlib::mat(reinterpret_cast<double*>\
			(afeat.data),afeat.cols,afeat.rows);
		cv::Vec2f motion;
		motion.val[0]      = static_cast<float>(dfX(onesample));
		motion.val[1]      = static_cast<float>(dfY(onesample));
		cv::Point centerpt = (*p)->point();
		cv::Rect roi  = cv::Rect(centerpt.x-this->motionWidth_,centerpt.y-\
			this->motionHeight_,this->motionWidth_,this->motionHeight_);
		roi.x      = std::max(0,roi.x);
		roi.y      = std::max(0,roi.y);
		roi.width  = std::min(roi.width,imsize.width-roi.x);
		roi.height = std::min(roi.height,imsize.height-roi.y);
		cv::Mat aroi = vMotionDetect(roi);
		aroi.setTo(motion);
	}
	std::string motionOut = (resultpath+motionstr+std::string\
		(PATH_SEP)+"0_"+testFile.substr(0,testFile.size()-4)+".bin").c_str();
	// [9] For motion we write a binary file:
	Auxiliary<float,2>::mat2bin(vMotionDetect,motionOut.c_str(),false);
	vMotionDetect.release();
	// Release stuff
	if(features){delete features; features=NULL;}
	for(typename std::vector<std::vector<const T*> >::iterator pa=patches.begin();pa!=\
	patches.end();++pa){
		for(typename std::vector<const T*>::iterator p=pa->begin();p!=pa->end();++p){
			if(*p){
				delete (*p); (*p) = NULL;
			}
		}
	}
}
//==============================================================================
/** Predicts on a set of test images in batch mode.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::batchTest(){
	// [-1] If leave average then for consistency, take mean in prediction:
	if(this->leafavg_){
		this->entropy_ = MotionTree<M,T,F,N,U>::MEAN_DIFF;
	}
	// [0] Initialize forest with number of trees
	MotionRF<L,M,T,F,N,U> forest(this->noTrees_,this->hogORsift_);
	// [1] Load the forest trees
	forest.loadForestBin(this->path2model_,this->binary_);
	// [2] Initialize detector
	MotionRFdetector<L,M,T,F,N,U> detect(&forest,this->patchWidth_,this->patchHeight_,\
		this->classInfo_.size(),this->labWidth_,this->labHeight_,\
		this->motionWidth_,this->motionHeight_,this->predMethod_,\
		this->teststep_,static_cast<typename MotionTree<M,T,F,N,U>::ENTROPY >\
		(this->entropy_),this->usederivatives_,this->hogORsift_,this->pttype_,\
		this->path2results_,this->maxsize_);
	// [3] Create directory for output
	Auxiliary<uchar,1>::file_exists(this->path2results_.c_str(),true);
	// [4] Run detector over test images
	this->test(detect,this->path2test_,this->path2results_,this->path2feat_);
}
//==============================================================================
/** Performs the RF detection on test images.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::test(MotionRFdetector<L,M,T,F,N,U> &crDetect,\
std::string &testpath,std::string &resultpath,std::string &featpath){
	// [0] List the current dir to see if it has images in it
	std::vector<std::string> testFiles = Auxiliary<uchar,1>::listDir(testpath,this->ext_);
	// [1] List the directories and descend into them
	if(testFiles.empty()){
		testFiles = Auxiliary<uchar,1>::listDir(testpath);
		for(std::vector<std::string>::iterator t=testFiles.begin();t!=\
		testFiles.end();++t){
			if(t->find(".")==0){continue;}
			std::string newresultpath = resultpath+(*t)+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(newresultpath.c_str(),true);
			std::string newfeatpath   = featpath+(*t)+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(newfeatpath.c_str(),true);
			std::string newtestpath   = testpath+(*t)+PATH_SEP;
			this->test(crDetect,newtestpath,newresultpath,newfeatpath);
		}
	// [2] We are there at the images, so process them
	}else{
		// [2] Loop over all the test images
		for(unsigned n=0;n<testFiles.size();++n){
			// [2] Initialize output over pyramid
			std::vector<cv::Mat> vMotionDetect(this->pyrScales_.size());
			std::vector<cv::Mat> vArrowsDetect(this->pyrScales_.size());
			std::vector<cv::Mat> vAppearDetect(this->pyrScales_.size());
			// [3] Run detection for all scales
			std::string justname = testFiles[n].substr(0,testFiles[n].size()-4);
			clock_t begin = clock();
			// First check for prediction, if there, return
			std::string arrowsstr,motionstr,appearancestr;
			if(this->usederivatives_){
				Auxiliary<uchar,1>::file_exists((resultpath+std::string("deri_motion")+\
					std::string(PATH_SEP)).c_str(),true);
				Auxiliary<uchar,1>::file_exists((resultpath+std::string("deri_appearance")+\
					std::string(PATH_SEP)).c_str(),true);
				Auxiliary<uchar,1>::file_exists((resultpath+std::string("deri_arrows")+\
					std::string(PATH_SEP)).c_str(),true);
				arrowsstr     = "deri_arrows";
				appearancestr = "deri_appearance";
				motionstr     = "deri_motion";
			}else{
				Auxiliary<uchar,1>::file_exists((resultpath+std::string("flow_motion")+\
					std::string(PATH_SEP)).c_str(),true);
				Auxiliary<uchar,1>::file_exists((resultpath+std::string("flow_appearance")+\
					std::string(PATH_SEP)).c_str(),true);
				Auxiliary<uchar,1>::file_exists((resultpath+std::string("flow_arrows")+\
					std::string(PATH_SEP)).c_str(),true);
				arrowsstr     = "flow_arrows";
				appearancestr = "flow_appearance";
				motionstr     = "flow_motion";
			}
			bool isthere = false;
			for(unsigned k=0;k<this->pyrScales_.size();++k){
				std::string motionOut = (resultpath+motionstr+std::string\
					(PATH_SEP)+Auxiliary<int,1>::number2string(k)+"_"+testFiles[n].\
					substr(0,testFiles[n].size()-4)+".bin").c_str();
				if(Auxiliary<char,1>::file_exists(motionOut.c_str(),false)){
					isthere = true;
					break;
				}
			}
			if(isthere){continue;}
			crDetect.detectPyramid(justname,testpath,featpath,this->ext_,\
				this->pyrScales_,vMotionDetect,vArrowsDetect,vAppearDetect,\
				this->path2model_,n);
			clock_t end = clock();
			std::cout<<"Prediction 1 img time elapsed: "<<double(Auxiliary\
				<uchar,1>::diffclock(end,begin))<<" sec"<<std::endl;
			// [4] Store each result over the scales for this image
			for(unsigned int k=0;k<vMotionDetect.size();++k){
				if(this->usederivatives_){
					assert(vMotionDetect[k].channels()==4);
				}else{
					assert(vMotionDetect[k].channels()==2);
				}
				// [6] Define the output path for the arrows also
				std::string arrowsOut = (resultpath+arrowsstr+\
					std::string(PATH_SEP)+Auxiliary<int,1>::number2string(k)+"_"+\
					testFiles[n]).c_str();
				// [7] Define the output path for the motion
				std::string motionOut = (resultpath+motionstr+\
					std::string(PATH_SEP)+Auxiliary<int,1>::number2string(k)+"_"+\
					testFiles[n].substr(0,testFiles[n].size()-4)+".bin").c_str();
				std::string appearOut = (resultpath+appearancestr+\
					std::string(PATH_SEP)+Auxiliary<int,1>::number2string(k)+"_"+\
					testFiles[n]).c_str();
				// [8] Define the parameters for storing the arrows
				std::vector<int> params;
				if(strcmp(this->ext_.c_str(),".jpg")){
					params.push_back(CV_IMWRITE_JPEG_QUALITY);
					params.push_back(100);
					cv::imwrite(arrowsOut,vArrowsDetect[k],params);
					cv::imwrite(appearOut,vAppearDetect[k],params);
				}else if(strcmp(this->ext_.c_str(),".png")){
					params.push_back(CV_IMWRITE_PNG_COMPRESSION);
					params.push_back(4);
					cv::imwrite(arrowsOut,vArrowsDetect[k],params);
					cv::imwrite(appearOut,vAppearDetect[k],params);
				}else{
					cv::imwrite(arrowsOut,vArrowsDetect[k]);
					cv::imwrite(appearOut,vAppearDetect[k],params);
				}
				// [9] For motion we write a binary file:
				if(this->usederivatives_){
					Auxiliary<float,4>::mat2bin(vMotionDetect[k],motionOut.c_str(),false);
				}else{
					Auxiliary<float,2>::mat2bin(vMotionDetect[k],motionOut.c_str(),false);
				}
				vMotionDetect[k].release();
				vArrowsDetect[k].release();
				vAppearDetect[k].release();
			}
			vArrowsDetect.clear();
			vMotionDetect.clear();
			vAppearDetect.clear();
		}
	}
}
//==============================================================================
/** Extracts feature/label patches from all the images (WHAT IS THIS FOR?).
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::runExtract(){
	std::cout<<"[RunMotionRF::extract]: Extracting patches training"<<std::endl;
	this->extract(this->path2test_,this->path2feat_);
	std::cout<<"[RunMotionRF::extract]: Extracting patches test"<<std::endl;
	this->extract(this->path2train_,this->path2feat_);
}
//==============================================================================
/** Recursively looks into the directories until it find the images it needs to
 * extract features from.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::extract(std::string &path2img,std::string \
&path2feat){
	// [0] Create directory for storing the models
	std::vector<std::string> files = Auxiliary<uchar,1>::listDir(path2img,this->ext_);
	// [1] List the directories and descend into them
	if(files.empty()){
		files = Auxiliary<uchar,1>::listDir(path2img);
		for(std::vector<std::string>::iterator f=files.begin();f!=files.end();++f){
			if(f->find(".")==0){continue;}
			std::string newpath2feat = path2feat+(*f)+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(newpath2feat.c_str(),true);
			std::string newpath2img  = path2img+(*f)+PATH_SEP;
			this->extract(newpath2img,newpath2feat);
		}
	// [2] We are there at the images, so process them
	}else{
		// [1] Initialize random number generator
		time_t times = time(NULL);
		int seed     = (int)times;
		CvRNG cvRNG(seed);
		// [2] Initialize training data set
		M train(&cvRNG,this->patchWidth_,this->patchHeight_,this->classInfo_.\
			size(),this->labWidth_,this->labHeight_,1e+5,0,this->consideredCls_,0,\
			this->trainstep_,this->motionWidth_,this->motionHeight_,this->warpping_,\
			this->ofThresh_,this->entropy_,this->sigmas_,this->bins_,\
			this->multicls_,this->usederivatives_,this->hogORsift_,this->pttype_,\
			this->maxsize_);
		// [3] Extract training features and test features
		train.trainingSize(1e+5);
		train.storefeat(true);
		train.extractPatches(path2img,this->path2labs_,path2feat,files,\
			this->classInfo_,this->labTerm_,this->ext_,true,\
			(this->entropy_!=MotionTree<M,T,F,N,U>::MEAN_DIFF));
	}
}
//==============================================================================
/** Extracts feature/label patches from all the images.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::runExtractOF(const std::vector<std::string> &argv){
	if(this->serverport_){
		this->jobrunnerExtractOF(argv);
	}else{
		this->batchExtractOF();
	}
}
//==============================================================================
/** Extracts feature/label patches from all the images in a batch mode.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::batchExtractOF(){
	Auxiliary<uchar,1>::file_exists(this->path2feat_.c_str(),true);
	Auxiliary<uchar,1>::file_exists(this->path2feat_.c_str(),true);
	std::cout<<"[RunMotionRF::batchExtractOF]: Extracting patches training"<<std::endl;
	this->extractOF(this->path2train_,this->path2feat_);
	std::cout<<"[RunMotionRF::batchExtractOF]: Extracting patches test"<<std::endl;
	this->extractOF(this->path2test_,this->path2feat_);
}
//==============================================================================
/** Recursively looks into the directories until it find the images it needs to
 * extract OF features from.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::extractOF(std::string &path2img,std::string &path2feat){
	// [0] Create directory for storing the models
	std::vector<std::string> files = Auxiliary<uchar,1>::listDir(path2img,this->ext_);
	// [1] List the directories and descend into them
	if(files.empty()){
		files = Auxiliary<uchar,1>::listDir(path2img);
		for(std::vector<std::string>::iterator f=files.begin();f!=files.end();++f){
			if(f->find(".")==0){continue;}
			std::string newpath2feat = path2feat+(*f)+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(newpath2feat.c_str(),true);
			std::string newpath2img  = path2img+(*f)+PATH_SEP;
			this->extractOF(newpath2img,newpath2feat);
		}
	// [2] We are there at the images, so process them
	}else{
		// [1] Initialize random number generator
		time_t times = time(NULL);
		int seed     = (int)times;
		CvRNG cvRNG(seed);
		// [2] Initialize training data set
		M train(&cvRNG,this->patchWidth_,this->patchHeight_,this->classInfo_.\
			size(),this->labWidth_,this->labHeight_,1e+5,0,this->consideredCls_,0,\
			this->trainstep_,this->motionWidth_,this->motionHeight_,this->warpping_,\
			this->ofThresh_,this->entropy_,this->sigmas_,this->bins_,\
			this->multicls_,this->usederivatives_,this->hogORsift_,this->pttype_,\
			this->maxsize_);
		// [3] Extract training features and test features
		train.trainingSize(1e+5);
		train.storefeat(true);
		typename M::Algorithm algo = train.algo();
		train.extractPatchesOF(path2img,path2feat,files,this->ext_,algo,true);
	}
}
//==============================================================================
/** Generating commands to extract patches from every image separately.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::jobrunnerExtractOF(const std::vector\
<std::string> &argv){
	boost::filesystem::path full_path(boost::filesystem::current_path());
	std::string cwd = full_path.string()+PATH_SEP;
	Auxiliary<uchar,1>::file_exists(this->path2feat_.c_str(),true);
	Auxiliary<uchar,1>::file_exists(this->path2feat_.c_str(),true);
	this->generateExtractCommands(argv,this->path2train_,cwd,this->path2feat_);
	this->generateExtractCommands(argv,this->path2test_,cwd,this->path2feat_);
}
//==============================================================================
/** Recursively reads the images from the test/train folder[s] and generates the
 * commands to extract them with jobrunners.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::generateExtractCommands(const std::vector\
<std::string> &argv,std::string &path2ims,const std::string &cwd,std::string \
&featpath){
	std::cout<<"[RunMotionRF::readImages]: "<<path2ims<<std::endl;
	std::vector<std::string> testFiles = Auxiliary<uchar,1>::listDir(path2ims,this->ext_);
	// [1] List the directories and descend into them
	if(testFiles.empty()){
		testFiles = Auxiliary<uchar,1>::listDir(path2ims);
		for(std::vector<std::string>::iterator t=testFiles.begin();t!=\
		testFiles.end();++t){
			if(t->find(".")==0){continue;}
			std::string newpath2ims = path2ims+(*t)+PATH_SEP;
			std::string newfeatpath = featpath+(*t)+PATH_SEP;
			Auxiliary<uchar,1>::file_exists(newfeatpath.c_str(),true);
			std::cout<<"[RunMotionRF::readImages]: mkdir "<<newfeatpath<<std::endl;
			this->generateExtractCommands(argv,newpath2ims,cwd,newfeatpath);
		}
	}else{
		// [0] Loop over all images in current directory and generate commands.
		for(std::vector<std::string>::iterator t=testFiles.begin();t!=\
		testFiles.end()-1;++t){
			std::string command = "jobmanager localhost:"+Auxiliary<unsigned,1>::\
				number2string(this->serverport_)+" schedule extract"+(this->runName_)+\
				" "+this->core_+" \""+cwd+argv[0]+" "+argv[1]+" "+Auxiliary<unsigned,1>::\
				number2string(RunMotionRF<L,M,T,F,N,U>::EXTRACT_OF1)+" "+cwd+\
				(this->configfile_)+" "+(*t)+" "+(*(t+1))+" "+path2ims+" "+\
				featpath+"\"";
			if(this->dryrun_){
				std::cout<<command<<std::endl;
			}else{
				int returnval = system(command.c_str());
			}
		}
	}
}
//==============================================================================
/** Extract features from 1 image only with jobrunners.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::runExtract1(const std::vector<std::string> &argv){
	if(argv.size()!=6){
		std::cerr<<"[RunMotionRF::runExtract1] not enough arguments."<<std::endl;
		throw std::exception();
	}
	// [1] Initialize random number generator
	time_t times = time(NULL);
	int seed     = (int)times;
	CvRNG cvRNG(seed);
	// [2] Initialize training data set
	M train(&cvRNG,this->patchWidth_,this->patchHeight_,this->classInfo_.\
		size(),this->labWidth_,this->labHeight_,1e+5,0,this->consideredCls_,0,\
		this->trainstep_,this->motionWidth_,this->motionHeight_,this->warpping_,\
		this->ofThresh_,this->entropy_,this->sigmas_,this->bins_,this->multicls_,\
		this->usederivatives_,this->hogORsift_,this->pttype_,this->maxsize_);
	// [3] Extract training features and test features
	std::vector<std::string> img;
	img.push_back(std::string(argv[2]));img.push_back(std::string(argv[3]));
	std::string path2img(argv[4]);
	std::string path2feat(argv[5]);
	typename M::Algorithm algo = train.algo();
	train.extractPatchesOF(path2img,path2feat,img,this->ext_,algo,true);
}
//==============================================================================
/** Initialize and start training.
 */
template <template <class M,class T,class F,class N,class U> class L,class M,class T,\
class F,class N,class U>
void RunMotionRF<L,M,T,F,N,U>::run(RunMotionRF::MODE mode,const std::vector\
<std::string> &argv){
	switch(mode){
		case RunMotionRF<L,M,T,F,N,U>::TRAIN_RF:
			// train forest
			this->runTrain(argv);
		break;
		case RunMotionRF<L,M,T,F,N,U>::TEST_RF:
			// test forest
			this->runTest(argv);
			break;
		case RunMotionRF<L,M,T,F,N,U>::EXTRACT:
			// extract features
			this->runExtract();
			break;
		case RunMotionRF<L,M,T,F,N,U>::EXTRACT_OF:
			// extract GT OF
			this->runExtractOF(argv);
			break;
		case RunMotionRF<L,M,T,F,N,U>::TRAIN1:
			// train 1 tree of the forest with jobrunners
			this->runTrain1(argv);
			break;
		case RunMotionRF<L,M,T,F,N,U>::TEST1:
			// test 1 image with jobrunners
			this->runTest1(argv);
			break;
		case RunMotionRF<L,M,T,F,N,U>::EXTRACT_OF1:
			// extract features from 1 image only
			this->runExtract1(argv);
			break;
		case RunMotionRF<L,M,T,F,N,U>::TRAIN_TEST_RF:
			this->runTrain(argv);
			this->runTest(argv);
			break;
		default:
			std::cerr<<"[RunMotionRF::run] option not implemented"<<std::endl;
			break;
	}
}
//==============================================================================
//==============================================================================
//==============================================================================
template class RunMotionRF<MotionTree,MotionPatch<MotionPatchFeature<FeaturesMotion>,\
	FeaturesMotion>,MotionPatchFeature<FeaturesMotion>,FeaturesMotion,MotionTreeNode\
	<MotionLeafNode,MotionPatchFeature<FeaturesMotion> >,MotionLeafNode>;















