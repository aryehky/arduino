#include "preprocessing.h"
#include "svm.h"
#include "dataset.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

namespace {

class PreprocessingSVMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load a small subset of MNIST data
        dataset.loadTrainingData();
        dataset.loadTestData();
        
        // Take a small subset for faster testing
        training_features = std::vector<std::vector<double>>(
            dataset.getTrainingFeatures().begin(),
            dataset.getTrainingFeatures().begin() + 1000
        );
        training_labels = std::vector<int>(
            dataset.getTrainingLabels().begin(),
            dataset.getTrainingLabels().begin() + 1000
        );
        
        test_features = std::vector<std::vector<double>>(
            dataset.getTestFeatures().begin(),
            dataset.getTestFeatures().begin() + 200
        );
        test_labels = std::vector<int>(
            dataset.getTestLabels().begin(),
            dataset.getTestLabels().begin() + 200
        );
    }
    
    // Helper function to add noise to images
    void addNoise(std::vector<std::vector<double>>& images, double stddev) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, stddev);
        
        for (auto& image : images) {
            for (auto& pixel : image) {
                pixel = std::clamp(pixel + d(gen), 0.0, 1.0);
            }
        }
    }
    
    Dataset dataset{"data/mnist/train-images.idx3-ubyte",
                   "data/mnist/train-labels.idx1-ubyte",
                   "data/mnist/t10k-images.idx3-ubyte",
                   "data/mnist/t10k-labels.idx1-ubyte"};
                   
    std::vector<std::vector<double>> training_features;
    std::vector<int> training_labels;
    std::vector<std::vector<double>> test_features;
    std::vector<int> test_labels;
    
    const int width = 28;
    const int height = 28;
};

TEST_F(PreprocessingSVMTest, TrainingWithPreprocessing) {
    // Create a copy of features and add noise
    auto noisy_features = training_features;
    addNoise(noisy_features, 0.1);
    
    // Create preprocessor and SVM
    preprocessing::ImagePreprocessor preprocessor;
    SVM svm;
    
    // Preprocess training data
    std::vector<std::vector<double>> processed_features;
    processed_features.reserve(noisy_features.size());
    
    for (const auto& image : noisy_features) {
        auto processed = preprocessor.normalize(image);
        processed = preprocessor.removeNoise(processed, width, height);
        processed = preprocessor.adjustContrast(processed, 1.2);
        processed_features.push_back(processed);
    }
    
    // Train SVM on processed data
    svm.train(processed_features, training_labels);
    
    // Preprocess test data
    std::vector<std::vector<double>> processed_test;
    processed_test.reserve(test_features.size());
    
    for (const auto& image : test_features) {
        auto processed = preprocessor.normalize(image);
        processed = preprocessor.removeNoise(processed, width, height);
        processed = preprocessor.adjustContrast(processed, 1.2);
        processed_test.push_back(processed);
    }
    
    // Evaluate on processed test data
    auto confusion_matrix = svm.getConfusionMatrix(processed_test, test_labels, 10);
    double accuracy = confusion_matrix.getAccuracy();
    
    // The accuracy should be reasonable even with noisy input
    EXPECT_GT(accuracy, 0.8);
}

TEST_F(PreprocessingSVMTest, BatchPreprocessingPipeline) {
    preprocessing::ImagePreprocessor preprocessor;
    
    // Add noise to training data
    auto noisy_features = training_features;
    addNoise(noisy_features, 0.1);
    
    // Process batch using the batch processing function
    auto processed_features = preprocessor.batchPreprocess(
        noisy_features,
        true,   // normalize
        true,   // remove noise
        true,   // adjust contrast
        1.2     // contrast factor
    );
    
    // Train and evaluate
    SVM svm;
    svm.train(processed_features, training_labels);
    
    // Process test data
    auto processed_test = preprocessor.batchPreprocess(
        test_features,
        true,   // normalize
        true,   // remove noise
        true,   // adjust contrast
        1.2     // contrast factor
    );
    
    auto confusion_matrix = svm.getConfusionMatrix(processed_test, test_labels, 10);
    double accuracy = confusion_matrix.getAccuracy();
    
    EXPECT_GT(accuracy, 0.8);
}

TEST_F(PreprocessingSVMTest, PreprocessingEffectiveness) {
    preprocessing::ImagePreprocessor preprocessor;
    SVM svm_raw, svm_processed;
    
    // Add significant noise to training data
    auto noisy_features = training_features;
    addNoise(noisy_features, 0.2);
    
    // Train on noisy data without preprocessing
    svm_raw.train(noisy_features, training_labels);
    auto raw_confusion = svm_raw.getConfusionMatrix(test_features, test_labels, 10);
    double raw_accuracy = raw_confusion.getAccuracy();
    
    // Train on preprocessed data
    auto processed_features = preprocessor.batchPreprocess(
        noisy_features,
        true,   // normalize
        true,   // remove noise
        true,   // adjust contrast
        1.2     // contrast factor
    );
    
    svm_processed.train(processed_features, training_labels);
    auto processed_confusion = svm_processed.getConfusionMatrix(test_features, test_labels, 10);
    double processed_accuracy = processed_confusion.getAccuracy();
    
    // Preprocessing should improve accuracy
    EXPECT_GT(processed_accuracy, raw_accuracy);
}

TEST_F(PreprocessingSVMTest, ImageTransformationPipeline) {
    preprocessing::ImagePreprocessor preprocessor;
    
    // Create a test image (simple pattern)
    std::vector<double> test_image(28 * 28, 0.0);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            test_image[i * 28 + j] = (i + j) % 2 == 0 ? 1.0 : 0.0;
        }
    }
    
    // Test rotation
    auto rotated = preprocessor.rotateImage(test_image, 28, 28, 45.0);
    EXPECT_EQ(rotated.size(), test_image.size());
    
    // Test scaling
    auto scaled = preprocessor.scaleImage(test_image, 28, 28, 1.5);
    EXPECT_EQ(scaled.size(), static_cast<size_t>(28 * 1.5 * 28 * 1.5));
    
    // Test combined transformation
    auto transformed = preprocessor.rotateAndScale(test_image, 28, 28, 30.0, 1.2);
    EXPECT_EQ(transformed.size(), static_cast<size_t>(28 * 1.2 * 28 * 1.2));
}

TEST_F(PreprocessingSVMTest, TransformationWithNoise) {
    preprocessing::ImagePreprocessor preprocessor;
    
    // Add noise to training data
    auto noisy_features = training_features;
    addNoise(noisy_features, 0.1);
    
    // Apply transformations to noisy data
    std::vector<std::vector<double>> transformed_features;
    transformed_features.reserve(noisy_features.size());
    
    for (const auto& image : noisy_features) {
        auto transformed = preprocessor.rotateAndScale(image, 28, 28, 15.0, 1.1);
        transformed = preprocessor.normalize(transformed);
        transformed = preprocessor.removeNoise(transformed, 28, 28);
        transformed_features.push_back(transformed);
    }
    
    // Train and evaluate
    SVM svm;
    svm.train(transformed_features, training_labels);
    
    auto confusion_matrix = svm.getConfusionMatrix(test_features, test_labels, 10);
    double accuracy = confusion_matrix.getAccuracy();
    
    // Should still maintain reasonable accuracy
    EXPECT_GT(accuracy, 0.7);
}

TEST_F(PreprocessingSVMTest, ImageFilteringPipeline) {
    preprocessing::ImagePreprocessor preprocessor;
    
    // Create a test image (simple pattern)
    std::vector<double> test_image(28 * 28, 0.0);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            test_image[i * 28 + j] = (i + j) % 2 == 0 ? 1.0 : 0.0;
        }
    }
    
    // Test Gaussian blur
    auto blurred = preprocessor.gaussianBlur(test_image, 28, 28, 1.0);
    EXPECT_EQ(blurred.size(), test_image.size());
    
    // Test edge detection
    auto edges_sobel = preprocessor.edgeDetection(test_image, 28, 28, true);
    auto edges_laplacian = preprocessor.edgeDetection(test_image, 28, 28, false);
    EXPECT_EQ(edges_sobel.size(), test_image.size());
    EXPECT_EQ(edges_laplacian.size(), test_image.size());
    
    // Test morphological operations
    auto eroded = preprocessor.morphologicalOperation(test_image, 28, 28, "erode", 3);
    auto dilated = preprocessor.morphologicalOperation(test_image, 28, 28, "dilate", 3);
    EXPECT_EQ(eroded.size(), test_image.size());
    EXPECT_EQ(dilated.size(), test_image.size());
}

TEST_F(PreprocessingSVMTest, FilteringWithNoise) {
    preprocessing::ImagePreprocessor preprocessor;
    
    // Add noise to training data
    auto noisy_features = training_features;
    addNoise(noisy_features, 0.1);
    
    // Apply filtering to noisy data
    std::vector<std::vector<double>> filtered_features;
    filtered_features.reserve(noisy_features.size());
    
    for (const auto& image : noisy_features) {
        auto filtered = preprocessor.gaussianBlur(image, 28, 28, 0.8);
        filtered = preprocessor.edgeDetection(filtered, 28, 28, true);
        filtered = preprocessor.morphologicalOperation(filtered, 28, 28, "dilate", 3);
        filtered = preprocessor.normalize(filtered);
        filtered_features.push_back(filtered);
    }
    
    // Train and evaluate
    SVM svm;
    svm.train(filtered_features, training_labels);
    
    auto confusion_matrix = svm.getConfusionMatrix(test_features, test_labels, 10);
    double accuracy = confusion_matrix.getAccuracy();
    
    // Should maintain reasonable accuracy with filtering
    EXPECT_GT(accuracy, 0.7);
}

TEST_CASE("ImageSegmentationPipeline", "[preprocessing]") {
    // Create a simple test image
    std::vector<double> img_data(100, 0.0);
    for (int i = 0; i < 100; ++i) {
        img_data[i] = (i % 10) / 10.0;  // Create some patterns
    }
    
    // Test threshold segmentation
    auto thresholded = preprocessing::ImagePreprocessor::thresholdSegmentation(img_data, 0.5, false);
    REQUIRE(thresholded.size() == img_data.size());
    
    // Test adaptive threshold
    auto adaptive_thresholded = preprocessing::ImagePreprocessor::thresholdSegmentation(img_data, 0.5, true);
    REQUIRE(adaptive_thresholded.size() == img_data.size());
    
    // Test watershed segmentation
    auto watershed = preprocessing::ImagePreprocessor::watershedSegmentation(img_data, 10, 10, 5);
    REQUIRE(watershed.size() == img_data.size());
    
    // Test k-means segmentation
    auto kmeans = preprocessing::ImagePreprocessor::kmeansSegmentation(img_data, 10, 10, 3);
    REQUIRE(kmeans.size() == img_data.size());
}

TEST_CASE("HistogramAnalysis", "[preprocessing]") {
    // Create a test image with known histogram properties
    std::vector<double> img_data(100, 0.0);
    for (int i = 0; i < 100; ++i) {
        img_data[i] = (i % 4) / 4.0;  // Create 4 distinct levels
    }
    
    // Test histogram statistics
    auto stats = preprocessing::ImagePreprocessor::computeHistogramStats(img_data);
    REQUIRE(stats.mean > 0.0);
    REQUIRE(stats.median > 0.0);
    REQUIRE(stats.entropy > 0.0);
    REQUIRE(stats.peak_count >= 4);  // Should detect the 4 distinct levels
    
    // Test histogram equalization
    auto equalized = preprocessing::ImagePreprocessor::histogramEqualization(img_data);
    REQUIRE(equalized.size() == img_data.size());
    
    // Test adaptive histogram equalization
    auto adaptive_equalized = preprocessing::ImagePreprocessor::adaptiveHistogramEqualization(img_data, 10, 10, 5);
    REQUIRE(adaptive_equalized.size() == img_data.size());
}

TEST_CASE("SegmentationWithNoise", "[preprocessing]") {
    // Create training data
    std::vector<std::vector<double>> training_data;
    std::vector<int> labels;
    
    // Add some noisy digits
    for (int digit = 0; digit < 10; ++digit) {
        for (int i = 0; i < 10; ++i) {
            std::vector<double> img_data(100, 0.0);
            // Add digit pattern
            for (int j = 0; j < 100; ++j) {
                img_data[j] = (j % 10 == digit) ? 0.8 : 0.2;
            }
            // Add noise
            for (int j = 0; j < 100; ++j) {
                img_data[j] += (rand() % 100) / 1000.0;
            }
            training_data.push_back(img_data);
            labels.push_back(digit);
        }
    }
    
    // Apply segmentation to noisy images
    for (auto& img : training_data) {
        img = preprocessing::ImagePreprocessor::thresholdSegmentation(img, 0.5, true);
        img = preprocessing::ImagePreprocessor::histogramEqualization(img);
    }
    
    // Train SVM
    SVM svm;
    svm.train(training_data, labels);
    
    // Test accuracy
    int correct = 0;
    for (size_t i = 0; i < training_data.size(); ++i) {
        int prediction = svm.predict(training_data[i]);
        if (prediction == labels[i]) {
            correct++;
        }
    }
    
    double accuracy = static_cast<double>(correct) / training_data.size();
    REQUIRE(accuracy > 0.7);  // Should maintain good accuracy after segmentation
}

} // namespace 