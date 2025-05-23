#!/bin/bash

# Example script demonstrating preprocessing functionality

# Create test directories if they don't exist
mkdir -p examples/input
mkdir -p examples/output

echo "Downloading sample MNIST images..."
# Download a few MNIST images for testing (you would need to implement this part)
# For now, we'll assume some images are in examples/input/

# Basic preprocessing example
echo -e "\n1. Basic preprocessing example"
./digit_recognition --preprocess examples/input/digit.png \
                   --normalize \
                   --output examples/output/digit_normalized.png

# Advanced preprocessing pipeline
echo -e "\n2. Advanced preprocessing pipeline"
./digit_recognition --preprocess examples/input/digit.png \
                   --normalize \
                   --remove-noise \
                   --adjust-contrast 1.2 \
                   --sharpen \
                   --output examples/output/digit_enhanced.png

# Batch processing example
echo -e "\n3. Batch processing example"
./digit_recognition --preprocess-batch examples/input \
                   --normalize \
                   --remove-noise \
                   --output examples/output/batch

# Binarization example
echo -e "\n4. Binarization example"
./digit_recognition --preprocess examples/input/digit.png \
                   --normalize \
                   --binarize 0.5 \
                   --output examples/output/digit_binary.png

# Standardization example
echo -e "\n5. Standardization example"
./digit_recognition --preprocess examples/input/digit.png \
                   --standardize \
                   --output examples/output/digit_standardized.png

# Rotation example
echo -e "\n6. Rotation example"
./digit_recognition --preprocess examples/input/digit.png \
                   --normalize \
                   --rotate 45 \
                   --output examples/output/digit_rotated.png

# Scaling example
echo -e "\n7. Scaling example"
./digit_recognition --preprocess examples/input/digit.png \
                   --normalize \
                   --scale 1.5 \
                   --output examples/output/digit_scaled.png

# Combined rotation and scaling
echo -e "\n8. Combined rotation and scaling"
./digit_recognition --preprocess examples/input/digit.png \
                   --normalize \
                   --rotate 30 \
                   --scale 1.2 \
                   --output examples/output/digit_transformed.png

echo -e "\nAll examples completed. Check examples/output/ for results." 