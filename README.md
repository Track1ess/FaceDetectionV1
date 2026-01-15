# FaceDetectionV1

FaceDetectionV1 is a C/C++ (CMake-based) face detection project. This repository includes:
- A top-level `CMakeLists.txt` for building the project
- An `environment.yml` for anaconda environment setup
- A `build/` directory

## Setup

conda env create -f environment.yml
conda activate FaceDetectionV1

cd build
./FaceDetection

## Output

A result file for face detection named "results_mobile.csv" and an extracted face feature file "embeddings_mobile.npy" will be generated in the current folder.
