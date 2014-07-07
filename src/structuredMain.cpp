/* main.cpp
 * Author: Silvia-Laura Pintea
 */
#include <RunRF.h>
#include <Auxiliary.h>
#include <LabelEval.h>
#include <omp.h>
typedef RunRF<StructuredTree,StructuredPatch<LabelPatchFeature<Features>,Features>,\
	LabelPatchFeature<Features>,Features,StructuredTreeNode<LabelLeafNode>,\
	LabelLeafNode> RunRFClass;
int main(int argc,char* argv[]){
	// [0] Check argument list
	unsigned mode = 1;
	unsigned what = 0;
	if(argc!=4){
		std::cout<<">>> Usage: demo [what] [mode] [config.txt]"<<std::endl<<std::endl;
		std::cout<<">>> [what]: 0 - segmentation; 1 - label evaluation; "<<\
			"3 - motion evaluation"<<std::endl<<std::endl;
		std::cout<<"[mode (0)]: 0 - train; 1 - test; 2 - train & test; "<<\
			"3 - extract;"<<std::endl;
		std::cout<<"[mode (1)]: 0 - global; 1 - avg-class; 2 - avg-pascal; "<<\
			"3 - all"<<std::endl;
		std::exit(-1);
	}
	std::vector<const char*> args;
	// [1] Cast the pointer to whatever class we need to.
	RunRFClass *rfBase         = NULL;
	// [2] For label evaluation
	LabelEval *labEval         = NULL;
	// [4] For all
	what = atoi(argv[1]);
	mode = atoi(argv[2]);
	switch(what){
		case 0: // baseline stuff
			rfBase = new RunRFClass(argv[3]);
			rfBase->run(static_cast<RunRFClass::MODE>(mode));
			break;
		case 1: // label evaluation
			labEval = new LabelEval(argv[3]);
			labEval->run(static_cast<LabelEval::METHOD>(mode));
			break;
		default:
			std::cerr<<"[main] option not implemented"<<std::endl;
			break;
	}
	if(rfBase){ delete rfBase;}
	if(labEval){ delete labEval;}
	return 0;
}




