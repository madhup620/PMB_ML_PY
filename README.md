# PMB_ML_PY
Pedestrian detection, distance estimation using Python, OpenCV, Stereo camera.
You can use an input file or a direct camera (stereo / depth camera) connected to the system.
In my work, I have used an Intel realsense D435 camera.
Algorithms:
1. Histogram of Oriented Gradients - feature descriptor
2. Support Vector Machine - Classifier for pedestrians in HOG
3. Non Maximum Suppression - A technique to reduce the number of overlapping bounding boxes of the same person
4. Haar cascade features for Face / Lower body / Upper body detection as extensions which can be controlled from CLI option or code
