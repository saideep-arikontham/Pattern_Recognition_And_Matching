# Content based image retrieval - Project 2

## Overview
This project demonstrates various content based image retrieval techniques along with different evaluation metrics using OpenCV. It includes generating feature vectors, choosing a query image and getting top matches to the selected query image.

---

## Content based image retrieval techniques
- Baseline Matching
    - Using a 7x7 center pixels of the image
- Histogram Matching
    - Using RG chromaticity histogram
- Multi-histogram Matching
    - Using two RGB histograms 
- Texture and Color
    - Using RGB histogram and Texture histogram
- Deep Network Embeddings
    - Using DNN embeddings
- Extensions
    - Using depth information with RGB histogram
    - Generating DNN embeddings for small image database
    - Using four RGB histograms

## Evaluation Metrics
- Sum of squared distances
- Histogram intersection
- Chi-Squared distances
- Cosine distance
- Earth mover's distance
---

## Project Structure

```
├── bin/
│   ├── #Executable binaries
├── images/                                 # Self collected database
│   ├── collected image database
├── include/                                # Includes for external libraries (if any)
├── output/                                 # Output folder for filter effected images
├── src/                                    # Source files
│   ├── DA2Network.hpp
│   ├── csv_util.cpp
│   ├── csv_util.h
│   ├── generate_feature_vectors.cpp 
│   ├── model_fp16.onnx
│   ├── pattern_recognition.cpp
│   ├── resnet18-v2-7.onnx
│   ├── test.cpp
│   ├── utils.cpp
│   └── utils.h
├── .gitignore                              # Git ignore file
├── makefile                                # Build configuration
```

---

## Tools used
- `OS`: MacOS
- `C++ Compiler`: Apple clang version 16.0.0
- `IDE`: Visual Studio code

---

## Dependencies
- OpenCV
- ONNX Runtime

**Note:** Update the dependency paths in the makefile after installation.

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

3. Compile the project:
   ```bash
   make generate_feature_vectors
   make pattern_recognition
   ```

---

## Usage

Run the `generate_feature_vectors` followed by `pattern_recognition` program to generate feature vectors and top matches respectively:

### 1. Baseline Matching

```bash
./bin/generate_feature_vectors <image_database> 7x7_middle_square
./bin/pattern_recognition <query_image> 7x7_middle_square ./feature_vectors/feature_vectors_7x7.csv
```

### 2. Histogram matching
```bash
./bin/generate_feature_vectors <image_database> rg_histogram
./bin/pattern_recognition <query_image> rg_histogram ./feature_vectors/feature_vectors_rg_hist.csv
```

### 3. Multi-Histogram matching
```bash
./bin/generate_feature_vectors <image_database> two_rgb_histogram
./bin/pattern_recognition <query_image> two_rgb_histogram ./feature_vectors/feature_vectors_two_rgb_hist.csv
```

### 4. Color and Texture matching
```bash
./bin/generate_feature_vectors <image_database> rgb_texture_hist
./bin/pattern_recognition <query_image> rgb_texture_hist ./feature_vectors/feature_vectors_rgb_texture_hist.csv
```

### 5. DNN Embeddings
```bash
./bin/pattern_recognition <query_image> deep_network ./feature_vectors/ResNet18_olym.csv
```

### 6. RGB Histogram with depth
```bash
./bin/generate_feature_vectors <image_database> depth_rgb
./bin/pattern_recognition <query_image> depth_rgb ./feature_vectors/feature_vectors_depth_rgb_hist.csv
```

### 7. Generating DNN embeddings
```bash
./bin/generate_feature_vectors <image_database> dnn
./bin/pattern_recognition <query_image> dnn ./feature_vectors/feature_vectors_dnn.csv
```

### 8. Using depth with DNN embeddings
```bash
./bin/generate_feature_vectors <image_database> depth_dnn
./bin/pattern_recognition <query_image> depth_dnn ./feature_vectors/feature_vectors_depth_dnn.csv
```

### 9. Using 4 RGB histograms
```bash
./bin/generate_feature_vectors <image_database> four_rgb_histogram
./bin/pattern_recognition <query_image> four_rgb_histogram ./feature_vectors/feature_vectors_four_rgb_hist.csv
```

More information about the internal implementation along with outputs is included in **Project2_Report.pdf**

---

## Highlights
- The `utils.cpp` file includes multiple utility functions like 
    - Simple reading, displaying images and top matches functions
    - Histogram, embedding computing functions
    - Generating feature vector functions
    - Functions to write feature vectors to a file
    - Functions for evaluation metrics

- The file `test.cpp` is just for testing purposes.

---

## Contact
- **Name**: Saideep Arikontham
- **Email**: arikontham.s@northeastern,edu