DejaVu: Motion Prediction in Static Images
============
This work proposes motion prediction in single still images by learning it from a set of videos. The building assumption is that similar motion is characterized by similar appearance. The proposed method learns local motion patterns given a specific appearance and adds the predicted motion in a number of applications. This work (i) introduces a novel method to predict motion from appearance in a single static image, (ii) to that end, extends of the Structured Random Forest with regression derived from first principles, and (iii) shows the value of adding motion predictions in different tasks such as: weak frame-proposals containing unexpected events, action recognition, motion saliency. Illustrative results indicate that motion prediction is not only feasible, but also provides valuable information for a number of applications.

When using this software, please reference the paper:

##### S.L. Pintea, J.C. van Gemert and A.W.M. Smeulders, Dejavu: Motion Prediction in Static Images, ECCV 2014.
 
>> Compile (needs OpenCV2.4.+, vlfeat-0.9.16, dlib-18.7 and Boost).
>> Edit the bin/CMakeLists.txt with the correct path towards the sources directory. 

cmake CMakeLists.txt
make

>> Run: edit the config file with the corresponding paths towards data (see bin/config_example.txt):

- Usage: ./dejavu [what] [mode] [config.txt]

- [what]: 0 - motion; 1 - motion evaluation

[mode (0)]: 0 - train; 1 - test; 2 - train & test; 3 - extract; 4 - extract flow (only Motion); 5 - train with jobrunners; 6 - test with    jobrunners; 7 - extract OF with jobrunners

[mode (1)]: 0 - segmentation error, 1 - motion error, 2 - raw values for python, 3 - all.

[mode (2)]: generate config files


>> Code documentation: 

- Motion prediction: 
https://github.com/SilviaLauraPintea/DejaVu/blob/master/motionRF.pdf

- Structured RF (inherited by the motion part): 
https://github.com/SilviaLauraPintea/DejaVu/blob/master/structuredRF.pdf
