# C++ Machine Learning Project: Digit Recognition with Support Vector Machine (SVM)

## Overview
This project implements a comprehensive digit recognition system using Support Vector Machine (SVM) in C++. It features advanced machine learning capabilities including cross-validation, hyperparameter optimization, and data augmentation.

## Features
### Core ML Features
- **SVM Classification:** Robust implementation using LIBSVM
- **Cross-validation:** K-fold cross-validation with shuffling support
- **Grid Search:** Automated hyperparameter optimization
- **Data Augmentation:** Multiple augmentation techniques:
  - Gaussian noise addition
  - Image rotation with interpolation
  - Translation with padding
- **Performance Metrics:**
  - Confusion matrix
  - Per-class precision and recall
  - F1 score
  - Accuracy metrics

### Image Preprocessing
- **Basic Operations:**
  - Normalization (min-max scaling)
  - Standardization (zero mean, unit variance)
  - Binarization with configurable threshold
- **Advanced Processing:**
  - Noise removal using median filtering
  - Contrast adjustment
  - Image sharpening
- **Batch Processing:**
  - Process multiple images with configurable pipeline
  - Parallel processing support
- **Analysis Tools:**
  - Statistical analysis (mean, stddev, min, max)
  - Entropy calculation
  - Image validation utilities

### User Interface & Utilities
- **Progress Tracking:** Visual progress bars for long operations
- **CLI Tools:**
  - Interactive prediction mode
  - Batch prediction support
  - Model training interface
- **Model Management:**
  - Save/load trained models
  - Model performance analysis
  - Visualization tools

## Prerequisites
- C++17 compatible compiler
- CMake (version 3.10 or higher)
- LIBSVM library
- OpenCV library
- (Optional) Python with matplotlib for visualization

### Installation

#### macOS
```bash
# Install dependencies
brew install cmake libsvm opencv

# Clone the repository
git clone https://github.com/yourusername/digit-recognition-svm.git
cd digit-recognition-svm

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make
```

#### Linux
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install cmake libsvm-dev libopencv-dev

# Follow the same build steps as macOS
```

#### Windows
```bash
# Using vcpkg
vcpkg install libsvm:x64-windows opencv:x64-windows

# Configure with CMake
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

## Usage

### 1. Download MNIST Dataset
```bash
# Make the script executable
chmod +x scripts/download_mnist.sh

# Download and extract dataset
./scripts/download_mnist.sh
```

### 2. Train the Model
```bash
# Basic training
./digit_recognition --train

# Training with grid search
./digit_recognition --train --grid-search

# Training with data augmentation
./digit_recognition --train --augment
```

### 3. Make Predictions
```bash
# Interactive mode
./digit_recognition --predict

# Batch prediction
./digit_recognition --predict-batch path/to/images/

# Load specific model
./digit_recognition --predict --model path/to/model.model
```

### 4. Analyze Performance
```bash
# Generate performance report
./digit_recognition --evaluate --model model.model

# Generate visualizations
./digit_recognition --visualize --model model.model
```

### 5. Image Preprocessing
```bash
# Basic preprocessing
./digit_recognition --preprocess input.png --normalize --output processed.png

# Advanced preprocessing
./digit_recognition --preprocess input.png \
                   --normalize \
                   --remove-noise \
                   --adjust-contrast 1.2 \
                   --output processed.png

# Batch preprocessing
./digit_recognition --preprocess-batch input_dir/ \
                   --normalize \
                   --remove-noise \
                   --output output_dir/
```

## Advanced Usage

### Grid Search Configuration
Modify the parameter grid in `include/utils.h`:
```cpp
struct ParamGrid {
    std::vector<std::string> kernel_types = {"linear", "rbf", "polynomial", "sigmoid"};
    std::vector<double> C_values = {0.1, 1.0, 10.0, 100.0};
    std::vector<double> gamma_values = {0.001, 0.01, 0.1, 1.0};
};
```

### Data Augmentation Settings
Configure augmentation parameters in your code:
```cpp
svm.augmentTrainingData(features, labels, 
                       3,      // num_augmented_per_sample
                       0.1,    // noise_stddev
                       15.0,   // max_rotation_angle
                       2);     // max_translation
```

### Preprocessing Configuration
```cpp
// Single image preprocessing
auto preprocessor = preprocessing::ImagePreprocessor();
auto image = preprocessor.loadImage("input.png");
image = preprocessor.normalize(image);
image = preprocessor.removeNoise(image, width, height);
image = preprocessor.adjustContrast(image, 1.2);
preprocessor.saveImage("output.png", image, width, height);

// Batch preprocessing with custom pipeline
std::vector<std::vector<double>> images = loadImages();
auto processed = preprocessor.batchPreprocess(
    images,
    true,   // normalize
    true,   // remove noise
    true,   // adjust contrast
    1.2     // contrast factor
);
```

### Cross-validation
```cpp
// Perform 5-fold cross-validation
double cv_accuracy = svm.crossValidate(features, labels, 5);
```

## Project Structure
```
digit-recognition-svm/
├── CMakeLists.txt           # Build configuration
├── README.md               # This file
├── data/                   # Dataset directory
│   └── mnist/             # MNIST dataset files
├── include/               # Header files
│   ├── dataset.h         # Dataset handling
│   ├── preprocessing.h   # Image preprocessing
│   ├── svm.h            # SVM implementation
│   └── utils.h          # Utility functions
├── src/                  # Source files
│   ├── dataset.cpp      # Dataset implementation
│   ├── main.cpp         # Main program
│   ├── preprocessing.cpp # Preprocessing implementation
│   ├── svm.cpp         # SVM implementation
│   └── utils.cpp       # Utility implementation
├── tests/               # Test files
│   └── test_preprocessing.cpp # Preprocessing tests
└── scripts/             # Utility scripts
    └── download_mnist.sh # Dataset download script
```

## Performance Tips
- Use `--grid-search` for optimal hyperparameters
- Enable data augmentation for better generalization
- Use appropriate kernel type for your data
- Adjust C and gamma parameters based on grid search results
- Apply appropriate preprocessing based on image quality:
  - Use noise removal for noisy images
  - Adjust contrast for low-contrast images
  - Normalize or standardize for consistent scaling

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- LIBSVM library
- OpenCV library
- MNIST dataset
- Contributors and maintainers