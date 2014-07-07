/* main.cpp
 * Author: Silvia-Laura Pintea
 */
#include <RunMotionRF.h>
#include <MotionTree.h>
#include <MotionEval.h>
#include <MotionTreeNode.h>
#include <omp.h>
typedef RunMotionRF<MotionTree,MotionPatch<MotionPatchFeature<FeaturesMotion>,\
	FeaturesMotion>,MotionPatchFeature<FeaturesMotion>,FeaturesMotion,\
	MotionTreeNode<MotionLeafNode,MotionPatchFeature<FeaturesMotion> >,\
	MotionLeafNode> RunMotionRFClass;
int main(int argc,char* argv[]){
	// [0] Check argument list
	unsigned mode = 1;
	unsigned what = 0;
	if(argc!=4 && argc!=5 && argc!=8 && argc!=3){
		std::cout<<">>> Usage: demo [what] [mode] [config.txt]"<<std::endl<<std::endl;
		std::cout<<">>> [what]: 0 - motion; 1 - motion evaluation"<<std::endl<<std::endl;
		std::cout<<"[mode (0)]: 0 - train; 1 - test; 2 - train & test; "<<\
			"3 - extract; 4 - extract flow (only Motion); 5 - train with "<<\
			"jobrunners; 6 - test with jobrunners; 7 - extract OF with "<<\
			"jobrunners"<<std::endl;
		std::cout<<"[mode (1)]: 0 - ERROR, 1 - FLOW, 2 - RAW4PYTHON, 3 - ALL"<<std::endl;
		std::cout<<"[mode (2)]: generate config files"<<std::endl;
		std::exit(-1);
	}
	std::vector<std::string> args;
	std::string binary(argv[0]);
	std::size_t pos = binary.find("./");
	if(pos!=std::string::npos){ // the binary name
		std::string tmp = binary.substr(pos+2,binary.size()-pos-1);
		args.push_back(tmp);
	}else{
		args.push_back(std::string(argv[0])); // the binary name
	}
	args.push_back(std::string(argv[1])); // what to run
	if(argc>4){
		for(unsigned i=4;i<argc;++i){
			args.push_back(std::string(argv[i]));
		}
	}
	// [1] Cast the pointer to whatever class we need to.
	RunMotionRFClass *rfMotion = NULL;
	// [2] For motion evaluation
	MotionEval *motionEval     = NULL;
	// [3] For all
	what = atoi(argv[1]);
	mode = atoi(argv[2]);
	switch(what){
		case 0: // baseline stuff
			rfMotion = new RunMotionRFClass(argv[3]);
			rfMotion->run(static_cast<RunMotionRFClass::MODE>(mode),args);
			break;
		case 1: // motion evaluation
			motionEval = new MotionEval(argv[3]);
			motionEval->run(static_cast<MotionEval::METHOD>(mode));
			break;
		case 2:
			RunMotionRFClass::generateConfigs("actionReco","randomForest",argv[2],true);
			break;
		default:
			std::cerr<<"[main] option not implemented"<<std::endl;
			break;
	}
	if(rfMotion){ delete rfMotion;}
	if(motionEval){delete motionEval;}
	return 0;
}




