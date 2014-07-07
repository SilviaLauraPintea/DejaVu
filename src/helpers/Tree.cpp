/* Tree.cpp
 * Author: Silvia-Laura Pintea
 */
#ifndef TREE_CPP_
#define TREE_CPP_
#include "Tree.h"
//==============================================================================
//==============================================================================
//==============================================================================
/** Reads the tree from the file.
 */
template <class N,class U>
Tree<N,U>::Tree(const char *filename,bool binary){
	this->root_        = NULL;
	this->path2models_ = filename;
	this->binary_      = binary;
	this->noleaves_    = 0;
	this->treeId_      = 0;
	this->growthtype_  = 1;
	std::cout<<"[Tree<evalFct,T,U>::Tree] Load Tree "<<filename<<std::endl;
	std::ifstream in(filename);
	if(in.is_open()){
		in>>this->treeId_;
		if(this->binary_){
			this->readNodeBin(NULL,in,Tree<N,U>::ROOT);
		}else{
			this->readNodeTxt(NULL,in,Tree<N,U>::ROOT);
		}
	}else{
		std::cerr<<"Could not read tree: "<<filename<<std::endl;
	}
	in.close();
};
//==============================================================================
/** Recursively read tree from file.
 */
template <class N,class U>
void Tree<N,U>::readNodeTxt(N *parent,std::ifstream &in,SIDE side){
	// [0] Read the node info
	unsigned nodetype;in>>nodetype;
	long unsigned nodeid; in>>nodeid;
	N *newnode;
	// [1] Add the node on the proper side
	if(!nodetype){
		U* leaf = new U(this->path2models_,nodeid,this->treeId_,false);
		newnode = this->addNode(parent,side,nodeid,leaf);
		delete leaf;
		return;
	}else{
		newnode = this->addNode(parent,side,nodeid,NULL);
		this->readNodeTxt(newnode,in,Tree<N,U>::LEFT);
		this->readNodeTxt(newnode,in,Tree<N,U>::RIGHT);
	}
}
//==============================================================================
/** Recursively read tree from binary file.
 */
template <class N,class U>
void Tree<N,U>::readNodeBin(N *parent,std::ifstream &in,SIDE side){
	// [0] Read the node info
	unsigned nodetype; in.read(reinterpret_cast<char*>(&nodetype),sizeof(unsigned));
	long unsigned nodeid; in.read(reinterpret_cast<char*>(&nodeid),sizeof(long unsigned));
	N *newnode;
	// [1] Add the node on the proper side
	if(!nodetype){
		U* leaf = new U(this->path2models_,nodeid,this->treeId_,true);
		newnode = this->addNode(parent,side,nodeid,leaf);
		delete leaf;
		return;
	}else{
		newnode = this->addNode(parent,side,nodeid,NULL);
		this->readNodeBin(newnode,in,Tree<N,U>::LEFT);
		this->readNodeBin(newnode,in,Tree<N,U>::RIGHT);
	}
}
//==============================================================================
/** Recursively destroys the nodes in the tree.
 */
template <class N,class U>
void Tree<N,U>::destroyTree(N *anode){
	if(anode!=NULL){
		this->destroyTree(anode->left());
		this->destroyTree(anode->right());
		delete anode;
		anode = NULL;
	}
}
//==============================================================================
/** Adds a node to the tree given the parent node and the side.
 */
template <class N,class U>
N* Tree<N,U>::addNode(N *parent,SIDE side,long unsigned nodeid,const U *leaf){
	// [0] If we don't have a root yet, then we add it
	if(!this->root_ && side==Tree<N,U>::ROOT){
		this->root_ = new N(nodeid,leaf);
		return this->root_;
	// [1] Else we put it on the indicated side
	}else{
		N* newNode = new N(nodeid,leaf);
		if(side == Tree<N,U>::LEFT){
			parent->left(newNode);
		}else if(side == Tree<N,U>::RIGHT){
			parent->right(newNode);
		}
		return newNode;
	}
	std::cerr<<"[Tree<evalFct,U>::addNode]: no correct side to add on."<<std::endl;
}
//==============================================================================
/** Show tree --- traverses the tree in inorder LOR.
 */
template <class N,class U>
void Tree<N,U>::showTree(){
	this->preorderTxt(this->root_,std::cout);
}
//==============================================================================
/** Traverses the tree in preorder OLR and displays the tests.
 */
template <class N,class U>
void Tree<N,U>::preorderBin(N *node,std::ofstream &out) const {
	if(node!=NULL){
		node->showNode(out,this->path2models_,this->treeId_,true);
		this->preorderBin(node->left(),out);
		this->preorderBin(node->right(),out);
	}
}
//==============================================================================
/** Saves the tree to binary file.
 */
template <class N,class U>
bool Tree<N,U>::saveTreeBin() const{
	bool done = false;
	std::ofstream out;
	out.open(this->path2models_,std::ios::out|std::ios::binary);
	// FIRST WRITE THE DIMENSIONS OF THE MATRIX
	if(out.good()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		// [0] Write general stuff (sizes, trees, etc.)
		unsigned treeId = this->treeId_;
		out.write(reinterpret_cast<const char*>(&treeId),sizeof(unsigned));
		this->preorderBin(this->root_,out);
		out.close();
	}
	done = true;
	return done;
}
//==============================================================================
/** Traverses the tree in preorder OLR and displays the tests.
 */
template <class N,class U>
void Tree<N,U>::preorderTxt(N *node,std::ofstream &out) const {
	if(node!=NULL){
		node->showNode(out,this->path2models_,this->treeId_,false);
		this->preorderTxt(node->left(),out);
		this->preorderTxt(node->right(),out);
	}
}
//==============================================================================
template <class N,class U>
bool Tree<N,U>::saveTree() const{
	if(this->binary_){
		this->saveTreeBin();
	}else{
		this->saveTreeTxt();
	}
}
//==============================================================================
/** Saves the tree in a txt file.
 */
template <class N,class U>
bool Tree<N,U>::saveTreeTxt() const{
	bool done = false;
	std::ofstream out(this->path2models_);
	if(out.good()){
		out.precision(std::numeric_limits<double>::digits10);
		out.precision(std::numeric_limits<float>::digits10);
		out<<this->treeId_<<" "<<std::endl;
		this->preorderTxt(this->root_,out);
		out.close();
	}
	done = true;
	return done;
}
//==============================================================================
/** Grows the tree either breath-first or worst-first until a leaf is reached.
 */
template <class N,class U>
void Tree<N,U>::growLimit(unsigned maxleaves){
	float nodeval;
	std::cout<<"nodeval: "; std::cin>>nodeval;
	this->noleaves_      = 0;
	long unsigned nodeid = 0;
	U* leaf              = new U();
	N *root              = this->addNode(NULL,Tree<N,U>::ROOT,nodeid,NULL);
	this->queue_.push_back(std::pair<float,N* >(nodeval,root));
	++nodeid;
	while(!this->queue_.empty()){
		if(static_cast<Tree<N,U>::GROWTH_TYPE>(this->growthtype_)==Tree<N,U>::WORST_FIRST){
			std::sort(this->queue_.begin(),this->queue_.end(),Tree<N,U>::sortPairs);
		}
		std::pair<float,N* > parentnode = this->queue_[0];
		this->queue_.erase(this->queue_.begin());
		// read the left node
		std::cout<<"left nodeval: "; std::cin>>nodeval;
		if(nodeval && this->noleaves_<maxleaves){
			N *leftNode = this->addNode(parentnode.second,Tree<N,U>::LEFT,nodeid,NULL);
			this->queue_.push_back(std::pair<float,N* >(nodeval,leftNode));
		}else{
			++this->noleaves_;
			N *leftNode = this->addNode(parentnode.second,Tree<N,U>::LEFT,nodeid,leaf);
		}
		++nodeid;
		// read the right node
		std::cout<<"nodeval right: "; std::cin>>nodeval;
		if(nodeval && this->noleaves_<maxleaves){
			N *rightNode = this->addNode(parentnode.second,Tree<N,U>::RIGHT,nodeid,NULL);
			this->queue_.push_back(std::pair<float,N* >(nodeval,rightNode));
		}else{
			N *rightNode = this->addNode(parentnode.second,Tree<N,U>::RIGHT,nodeid,leaf);
			++this->noleaves_;
		}
		++nodeid;
	}
}
//==============================================================================
/** Sort the nodes in the queue by their value.
 */
template <class N,class U>
bool Tree<N,U>::sortPairs(const std::pair<float,N* > &p1,\
const std::pair<float,N* > &p2){
	return (p1.first>p2.first);
}
//==============================================================================
//==============================================================================
//==============================================================================
#endif // TREE_CPP_
