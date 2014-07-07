DejaVu
============
When using this software, please reference the paper:

#####S. Pintea, J. van Gemert and A. Smeulders, Dejavu: Motion Prediction in Static Images, ECCV 2014.


 
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

- Motion prediction part: 
https://github.com/SilviaLauraPintea/DejaVu/blob/master/motionRF.pdf

- Structured RF part (inherited by the motion part): 
https://github.com/SilviaLauraPintea/DejaVu/blob/master/structuredRF.pdf



####Have loads of fun!
