DejaVu
============
When using this software, please reference the paper:

S. Pintea, J. van Gemert and A. Smeulders, Dejavu: Motion Prediction in Static Images, ECCV 2014.


 
>> Compile (needs OpenCV2.4.+ and Boost).
>> Edit the bin/CMakeLists.txt with the correct path towards the sources directory. 

cmake CMakeLists.txt
make



>> Run: edit the cofing file with the corresponding paths towards data (see bin/config_example.txt):

- Usage: ./dejavu [what] [mode] [config.txt]

- [what]: 0 - motion; 1 - motion evaluation

[mode (0)]: 0 - train; 1 - test; 2 - train & test; 3 - extract; 4 - extract flow (only Motion); 5 - train with jobrunners; 6 - test with    jobrunners; 7 - extract OF with jobrunners

[mode (1)]: 0 - ERROR, 1 - FLOW, 2 - RAW4PYTHON, 3 - ALL

[mode (2)]: generate config files



###Have loads of fun!
