#include "preprocessing.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace {

class PreprocessingTest : public ::testing::Test {
protected:
    // Helper function to create a test image
    std::vector<double> createTestImage(int width, int height) {
        std::vector<double> image(width * height);
        for (int i = 0; i < width * height; ++i) {
            image[i] = static_cast<double>(i) / (width * height);
        }
        return image;
    }
    
    // Common test image dimensions
    const int width = 28;
    const int height = 28;
};

TEST_F(PreprocessingTest, NormalizeEmptyImage) {
    std::vector<double> empty;
    EXPECT_THROW(preprocessing::ImagePreprocessor::normalize(empty),
                 std::invalid_argument);
}

TEST_F(PreprocessingTest, NormalizeValidImage) {
    auto image = createTestImage(width, height);
    auto normalized = preprocessing::ImagePreprocessor::normalize(image);
    
    EXPECT_EQ(normalized.size(), image.size());
    EXPECT_NEAR(normalized.front(), 0.0, 1e-6);
    EXPECT_NEAR(normalized.back(), 1.0, 1e-6);
}

TEST_F(PreprocessingTest, StandardizeEmptyImage) {
    std::vector<double> empty;
    EXPECT_THROW(preprocessing::ImagePreprocessor::standardize(empty),
                 std::invalid_argument);
}

TEST_F(PreprocessingTest, StandardizeValidImage) {
    auto image = createTestImage(width, height);
    auto standardized = preprocessing::ImagePreprocessor::standardize(image);
    
    EXPECT_EQ(standardized.size(), image.size());
    
    // Calculate mean and stddev of standardized image
    double sum = 0.0;
    for (double pixel : standardized) {
        sum += pixel;
    }
    double mean = sum / standardized.size();
    
    double sq_sum = 0.0;
    for (double pixel : standardized) {
        sq_sum += (pixel - mean) * (pixel - mean);
    }
    double stddev = std::sqrt(sq_sum / standardized.size());
    
    EXPECT_NEAR(mean, 0.0, 1e-6);
    EXPECT_NEAR(stddev, 1.0, 1e-6);
}

TEST_F(PreprocessingTest, BinarizeInvalidThreshold) {
    auto image = createTestImage(width, height);
    EXPECT_THROW(preprocessing::ImagePreprocessor::binarize(image, -0.1),
                 std::invalid_argument);
    EXPECT_THROW(preprocessing::ImagePreprocessor::binarize(image, 1.1),
                 std::invalid_argument);
}

TEST_F(PreprocessingTest, BinarizeValidImage) {
    auto image = createTestImage(width, height);
    double threshold = 0.5;
    auto binarized = preprocessing::ImagePreprocessor::binarize(image, threshold);
    
    EXPECT_EQ(binarized.size(), image.size());
    for (double pixel : binarized) {
        EXPECT_TRUE(pixel == 0.0 || pixel == 1.0);
    }
}

TEST_F(PreprocessingTest, RemoveNoiseInvalidDimensions) {
    auto image = createTestImage(width, height);
    EXPECT_THROW(preprocessing::ImagePreprocessor::removeNoise(image, width + 1, height),
                 std::invalid_argument);
}

TEST_F(PreprocessingTest, RemoveNoiseValidImage) {
    auto image = createTestImage(width, height);
    auto denoised = preprocessing::ImagePreprocessor::removeNoise(image, width, height);
    
    EXPECT_EQ(denoised.size(), image.size());
    EXPECT_TRUE(preprocessing::validatePixelRange(denoised, 0.0, 1.0));
}

TEST_F(PreprocessingTest, AdjustContrastInvalidFactor) {
    auto image = createTestImage(width, height);
    EXPECT_THROW(preprocessing::ImagePreprocessor::adjustContrast(image, -1.0),
                 std::invalid_argument);
    EXPECT_THROW(preprocessing::ImagePreprocessor::adjustContrast(image, 0.0),
                 std::invalid_argument);
}

TEST_F(PreprocessingTest, AdjustContrastValidImage) {
    auto image = createTestImage(width, height);
    double factor = 1.5;
    auto adjusted = preprocessing::ImagePreprocessor::adjustContrast(image, factor);
    
    EXPECT_EQ(adjusted.size(), image.size());
    EXPECT_TRUE(preprocessing::validatePixelRange(adjusted, 0.0, 1.0));
}

TEST_F(PreprocessingTest, ComputeStatsEmptyImage) {
    std::vector<double> empty;
    EXPECT_THROW(preprocessing::ImagePreprocessor::computeStats(empty),
                 std::invalid_argument);
}

TEST_F(PreprocessingTest, ComputeStatsValidImage) {
    auto image = createTestImage(width, height);
    auto stats = preprocessing::ImagePreprocessor::computeStats(image);
    
    EXPECT_NEAR(stats.min, 0.0, 1e-6);
    EXPECT_NEAR(stats.max, 1.0, 1e-6);
    EXPECT_GT(stats.stddev, 0.0);
}

TEST_F(PreprocessingTest, BatchPreprocessEmptyBatch) {
    std::vector<std::vector<double>> empty;
    EXPECT_THROW(preprocessing::ImagePreprocessor::batchPreprocess(empty),
                 std::invalid_argument);
}

TEST_F(PreprocessingTest, BatchPreprocessValidBatch) {
    std::vector<std::vector<double>> batch;
    batch.push_back(createTestImage(width, height));
    batch.push_back(createTestImage(width, height));
    
    auto processed = preprocessing::ImagePreprocessor::batchPreprocess(
        batch, true, true, true, 1.2);
    
    EXPECT_EQ(processed.size(), batch.size());
    for (const auto& image : processed) {
        EXPECT_EQ(image.size(), width * height);
        EXPECT_TRUE(preprocessing::validatePixelRange(image, 0.0, 1.0));
    }
}

TEST_F(PreprocessingTest, ValidateImageDimensions) {
    auto image = createTestImage(width, height);
    EXPECT_TRUE(preprocessing::validateImageDimensions(image, width, height));
    EXPECT_FALSE(preprocessing::validateImageDimensions(image, width + 1, height));
}

TEST_F(PreprocessingTest, ValidatePixelRange) {
    auto image = createTestImage(width, height);
    EXPECT_TRUE(preprocessing::validatePixelRange(image, 0.0, 1.0));
    
    // Test with invalid range
    image[0] = 1.5;
    EXPECT_FALSE(preprocessing::validatePixelRange(image, 0.0, 1.0));
}

} // namespace 