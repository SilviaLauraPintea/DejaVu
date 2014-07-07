/* Tree.h
 * Author: Silvia-Laura Pintea
 */
#ifndef TREE_H_
#define TREE_H_
#include <Auxiliary.h>
//==============================================================================
/** The tree node structure to store the info in it.
 */
template <class U>
class node{
	public:
		node(){
			this->nodeid_ = 0;
			this->leaf_   = NULL;
			this->left_   = NULL;
			this->right_  = NULL;
		}
		node(long unsigned nodeid,const U* leaf=NULL){
			this->nodeid_ = nodeid;
			if(leaf){ this->leaf_ = new U(*leaf);
			}else{ this->leaf_ = NULL; }
			this->right_  = NULL;
			this->left_   = NULL;
		};
		virtual ~node(){
			if(this->leaf_){
				delete this->leaf_;
				this->leaf_ = NULL;
			}
		};
		//----------------------------------------------------------------------
		//----VIRTUAL METHODS---------------------------------------------------
		//----------------------------------------------------------------------
		virtual void showNode(std::ofstream &out,const char *path2model,\
		unsigned treeid,bool binary){
			if(binary){
				this->showNodeBin(out,path2model,treeid);
			}else{
				this->showNodeTxt(out,path2model,treeid);
			}
		}
		/** Write node to text output stream.
		 */
		virtual void showNodeTxt(std::ofstream &out,const char* path2model,\
		unsigned treeid){
			out.precision(std::numeric_limits<double>::digits10);
			out.precision(std::numeric_limits<float>::digits10);
			if(this->leaf_){
				out<<0<<" "; // leaf-node
			}else{
				out<<1<<" "; // test-node
			}
			out<<this->nodeid_<<" ";
			if(this->leaf_ && !this->leaf_->isempty()){
				this->leaf_->showLeafTxt(path2model,this->nodeid_,treeid);
			}
		}
		/** Write node to binary output stream.
		 */
		virtual void showNodeBin(std::ofstream &out,const char *path2model,\
		unsigned treeid){
			out.precision(std::numeric_limits<double>::digits10);
			out.precision(std::numeric_limits<float>::digits10);
			if(this->leaf_){
				unsigned nodetype = 0;
				out.write(reinterpret_cast<const char*>(&nodetype),sizeof(unsigned));
			}else{
				unsigned nodetype = 1;
				out.write(reinterpret_cast<const char*>(&nodetype),sizeof(unsigned));
			}
			long unsigned nodeid = this->nodeid_;
			out.write(reinterpret_cast<const char*>(&nodeid),sizeof(long unsigned));
			if(this->leaf_ && !this->leaf_->isempty()){
				this->leaf_->showLeafBin(path2model,this->nodeid_,treeid);
			}
		}
		/** Use default Copy and assignment constructors.
		 */
		virtual node& clone(node const &rhs){
			this->nodeid(rhs.nodeid());
			this->leaf(rhs.leaf());
			this->left(rhs.left());
			this->right(rhs.right());
			return *this;
		}
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		long unsigned nodeid() const {return this->nodeid_;}
		U* leaf() const {return this->leaf_;};
		node<U> *left() const {return this->left_;};
		node<U> *right() const {return this->right_;};
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void nodeid(long unsigned nodeid){this->nodeid_ = nodeid;}
		void leaf(const U* leaf){
			if(this->leaf_){
				delete this->leaf_;
				this->leaf_ = NULL;
			}
			this->leaf_ = new U(*leaf);
		};
		void left(node<U> *left){
			this->left_ = left; // no deep copy here, just switch pointers
		};
		void right(node<U> *right){
			this->right_ = right; // no deep copy here, just switch pointers
		};
		//----------------------------------------------------------------------
	protected:
		/** @var leaf_
		 * Pointer to the leaf variable.
		 */
		U* leaf_;
		/** @var nodeid_
		 * Unique number identifying this node.
		 */
		long unsigned nodeid_;
	private:
		DISALLOW_COPY_AND_ASSIGN(node);
		/** @var left_
		 * Left child of the tree.
		 */
		node<U> *left_;
		/** @var right_
		 * Right child of the tree.
		 */
		node<U> *right_;
};
//==============================================================================
//==============================================================================
//==============================================================================
/** Standard binary tree class.
 */
template <class N,class U>
class Tree{
	public:
		/** Where to add the new node in the tree: 0 - left,1 - right,2 -root
		 */
		enum GROWTH_TYPE {DEPTH_FIRST,BREADTH_FIRST,WORST_FIRST};
		/** Where to add the new node in the tree: 0 - left,1 - right,2 -root
		 */
		enum SIDE {LEFT,RIGHT,ROOT};
		Tree(const char *filename,bool binary);
		Tree(unsigned growthtype=0){
			this->root_        = NULL;
			this->treeId_      = 0;
			this->path2models_ = '\0';
			this->binary_      = true;
			this->growthtype_  = growthtype;
			this->noleaves_    = 0;
		};
		virtual ~Tree(){
			this->destroyTree(this->root_);
		};
		/** Recursively destroys the nodes in the tree.
		 */
		void destroyTree(N *anode);
		/** Traverses the tree in inorder LOR and displays the tests.
		 */
		void preorderTxt(N *node,std::ofstream &out) const;
		/** Traverses the tree in preorder OLR and displays the tests.
		 */
		void preorderBin(N *node,std::ofstream &out) const;
		/** Show tree --- traverses the tree in inorder LOR.
		 */
		void showTree();
		/** Sort the nodes in the queue by their value.
		 */
		static bool sortPairs(const std::pair<float,N* > &p1,\
			const std::pair<float,N* > &p2);
		//----------------------------------------------------------------------
		//---VIRTUAL FUNCTIONS--------------------------------------------------
		//----------------------------------------------------------------------
		/** Recursively read tree from file.
		 */
		virtual void readNodeTxt(N *parent,std::ifstream &in,SIDE side);
		/** Recursively read tree from binary file.
		 */
		virtual void readNodeBin(N *parent,std::ifstream &in,SIDE side);
		/** Adds a node to the tree given the parent node and the side.
		 */
		virtual N* addNode(N *parent,SIDE side,long unsigned nodeid,const U *leaf);
		/** Saves the tree to file.
		 */
		virtual bool saveTree() const;
		/** Saves the tree to binary file.
		 */
		virtual bool saveTreeBin() const;
		/** Saves the tree in a txt file.
		 */
		virtual bool saveTreeTxt() const;
		/** Grows the tree either breath-first or worst-first until a leaf is reached.
		 */
		virtual void growLimit(unsigned maxleaves);
		//----------------------------------------------------------------------
		//---GETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		unsigned treeId() const {return this->treeId_;}
		N* root() const {return this->root_;}
		const char* path2models() const {return this->path2models_;}
		bool binary() const {return this->binary_;}
		std::vector<std::pair<float,N* > > queue() const {return this->queue_;}
		long unsigned noleaves() const {return this->noleaves_;}
		unsigned growthtype() const {return this->growthtype_;}
		//----------------------------------------------------------------------
		//---SETTERS------------------------------------------------------------
		//----------------------------------------------------------------------
		void treeId(unsigned treeId){this->treeId_ = treeId;}
		void root(const N *root){
			if(this->root_){
				delete this->root_;
				this->root_ = NULL;
			}
			this->root_->clone(root);
		}
		void path2models(const char *path2models){
			this->path2models_ = path2models;
		}
		void binary(bool binary){this->binary_ = binary;}
		void noleaves(long unsigned noleaves){this->noleaves_ = noleaves;}
		void growthtype(unsigned growthtype){this->growthtype_ = growthtype;}
		//----------------------------------------------------------------------
	protected:
		/** @var root_
		 * The root node of the tree.
		 */
		N *root_;
		/** @var treeId_
		 * Tree id to know which tree are we working on.
		 */
		unsigned treeId_;
		/** @var path2models_
		 * The path to the Trees.
		 */
		const char *path2models_;
		/** @var binary_
		 * If the tree should be binary or not
		 */
		bool binary_;
		/** @var queue_
		 * The queue in which to store the nodes to be processed.
		 */
		std::vector<std::pair<float,N* > > queue_;
		/** @var noleaves_
		 * The number of leaves already created.
		 */
		long unsigned noleaves_;
		/** @var growthtype_
		 * The the growth type for the tree.
		 */
		unsigned growthtype_;
	private:
		DISALLOW_COPY_AND_ASSIGN(Tree);
};
//==============================================================================
#endif /* TREE_H_ */
#include "Tree.cpp"











































