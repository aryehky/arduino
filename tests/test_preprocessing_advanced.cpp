#include "../include/preprocessing.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace preprocessing;

class PreprocessingAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test image (2x2 RGB)
        test_rgb_image = {
            1.0, 0.0, 0.0,  // Red pixel
            0.0, 1.0, 0.0,  // Green pixel
            0.0, 0.0, 1.0,  // Blue pixel
            1.0, 1.0, 1.0   // White pixel
        };
    }

    std::vector<double> test_rgb_image;
};

TEST_F(PreprocessingAdvancedTest, RGBToGrayscale) {
    auto grayscale = ImagePreprocessor::rgbToGrayscale(test_rgb_image, 2, 2);
    
    // Check dimensions
    EXPECT_EQ(grayscale.size(), 4);
    
    // Check conversion values
    EXPECT_NEAR(grayscale[0], 0.299, 1e-5);  // Red to grayscale
    EXPECT_NEAR(grayscale[1], 0.587, 1e-5);  // Green to grayscale
    EXPECT_NEAR(grayscale[2], 0.114, 1e-5);  // Blue to grayscale
    EXPECT_NEAR(grayscale[3], 1.000, 1e-5);  // White to grayscale
}

TEST_F(PreprocessingAdvancedTest, RGBToHSVAndBack) {
    auto hsv = ImagePreprocessor::rgbToHSV(test_rgb_image, 2, 2);
    auto rgb_back = ImagePreprocessor::hsvToRGB(hsv, 2, 2);
    
    // Check dimensions
    EXPECT_EQ(hsv.size(), 12);  // 4 pixels * 3 channels
    EXPECT_EQ(rgb_back.size(), 12);
    
    // Check HSV values for red pixel
    EXPECT_NEAR(hsv[0], 0.0, 1e-5);    // Hue
    EXPECT_NEAR(hsv[1], 1.0, 1e-5);    // Saturation
    EXPECT_NEAR(hsv[2], 1.0, 1e-5);    // Value
    
    // Check round-trip conversion
    for (size_t i = 0; i < test_rgb_image.size(); ++i) {
        EXPECT_NEAR(rgb_back[i], test_rgb_image[i], 1e-5);
    }
}

TEST_F(PreprocessingAdvancedTest, FrequencyDomainOperations) {
    // Create a simple test image
    std::vector<double> test_image = {
        1.0, 0.0,
        0.0, 1.0
    };
    
    auto fft_result = ImagePreprocessor::applyFFT(test_image, 2, 2);
    auto ifft_result = ImagePreprocessor::applyIFFT(fft_result, 2, 2);
    
    // Check dimensions
    EXPECT_EQ(fft_result.size(), 4);
    EXPECT_EQ(ifft_result.size(), 4);
    
    // Note: FFT/IFFT tests are basic for now since implementation is pending
    // More comprehensive tests should be added once FFT is implemented
}

TEST_F(PreprocessingAdvancedTest, FrequencyFilter) {
    std::vector<double> test_image = {
        1.0, 0.0,
        0.0, 1.0
    };
    
    auto filtered = ImagePreprocessor::frequencyFilter(test_image, 2, 2, "lowpass", 0.5);
    
    // Check dimensions
    EXPECT_EQ(filtered.size(), 4);
    
    // Note: More comprehensive tests should be added once frequency filtering is implemented
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 