#!/bin/bash

# Create directories if they don't exist
mkdir -p data/mnist

# Download MNIST dataset files
wget -P data/mnist http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P data/mnist http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -P data/mnist http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P data/mnist http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Extract files
cd data/mnist
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

echo "MNIST dataset downloaded and extracted successfully!" 