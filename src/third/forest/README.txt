// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
 
// You may use, copy, reproduce, and distribute this Software for any 
// non-commercial purpose, subject to the restrictions of the 
// Microsoft Research Shared Source license agreement ("MSR-SSLA"). 
// Some purposes which can be non-commercial are teaching, academic 
// research, public demonstrations and personal experimentation. You 
// may also distribute this Software with books or other teaching 
// materials, or publish the Software on websites, that are intended 
// to teach the use of the Software for academic or other 
// non-commercial purposes.
// You may not use or distribute this Software or any derivative works 
// in any form for commercial purposes. Examples of commercial 
// purposes would be running business operations, licensing, leasing, 
// or selling the Software, distributing the Software for use with 
// commercial products, using the Software in the creation or use of 
// commercial products or any other activity which purpose is to 
// procure a commercial gain to you or others.
// If the Software includes source code or data, you may create 
// derivative works of such portions of the Software and distribute 
// the modified Software for non-commercial purposes, as provided 
// herein.

// THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO 
// EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT 
// LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A 
// PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR 
// ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR 
// NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL 
// FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST 
// PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR 
// DERIVATIVE WORKS.

// NEITHER MICROSOFT NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE 
// LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS MSR-SSLA, 
// INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL 
// DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT 
// LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF 
// LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE 
// WORKS.

// When using this software, please acknowledge the effort that 
// went into development by referencing the paper:
//
// Gall J. and Lempitsky V., Class-Specific Hough Forests for 
// Object Detection, IEEE Conference on Computer Vision and Pattern 
// Recognition (CVPR'09), 2009.

// Note that this is not the original software that was used for 
// the paper mentioned above. It is a re-implementation for Linux. 



#compile (needs OpenCV)
make all

#clean
make clear

#run
./run.sh mode [config.txt] [tree_offset]
mode: 0 - train; 1 - show leafs; 2 - detect
config.txt: config file
tree_offset: output number for trees (treetable[index+offset].txt)

A config.txt example is given in the subdirectory 'example'

#example train
./run_train.sh

#example detect
./run_detect.sh

Config.txt:

Information for storing and loading trees:
# Path to trees + prefix 'treetable'
/scratch/tmp/forest/example/trees/treetable
# Number of trees
10

Image patch size (needs to be the same for training and learning):
# Patch width
16
# Patch height
16

Information about the test set:
# Path to images
/scratch/tmp/forest/example/testimages
# File with names of images
/scratch/tmp/forest/example/test.txt

Not used:
# Extract features
1

Specify scale and ratio if necessary:
# Scales (Number of scales - Scales)
3 0.5 1 2 // 3 scales with factors 0.5, 1, and 2
# Ratios (Number of ratios - ratio)
1 1 // 1 aspect ratio with factor 1 = fixed aspect ratio 

Information for storing the Hough images (scaled by a fixed factor such that max<=255):
# Output path
/scratch/tmp/forest/example/detect
# Scale factor for output image (default: 128)
50

Information about positive and negative examples:
# Path to positive examples
/scratch/tmp/forest/example/trainimages
# File with positive examples
/scratch/tmp/forest/example/train_pos.txt
# Subset of positive images -1: all images
-1
# Sample patches from pos. examples
50

# Path to negative examples
/scratch/tmp/forest/example/trainimages
# File with negative examples
/scratch/tmp/forest/example/train_neg.txt
# Subset of negative images -1: all images
-1
# Sample patches from neg. examples
50

train_neg.txt:
50 1 // number of images + dummy value (1)
neg0.png 0 0 100 40 // filename + boundingbox (top left - bottom right)

train_pos.txt:
50 1 // number of images + dummy value (1)
pos0.png 0 0 74 36 37 18 // filename + boundingbox (top left - bottom right) + center of bounding box

Output:
detect-[I]_sc[S]_c[R].png
I: image id
S: scale id
C: ratio id

The images are slides of the 4D (x,y,scale,ratio) voting space for an image I. 
For detection, the space needs to be smoothed and the maxima detected (or using mean shift). 
This is not part of the source code.



Juergen Gall

BIWI, ETH Zurich	17 July 2009