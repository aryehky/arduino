#include "preprocessing.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <fstream>

namespace preprocessing {

std::vector<double> ImagePreprocessor::loadImage(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open image file: " + filepath);
    }
    
    // Implementation will depend on image format - placeholder for now
    // TODO: Add support for common image formats (PNG, JPEG, etc.)
    return std::vector<double>();
}

void ImagePreprocessor::saveImage(const std::string& filepath, 
                                const std::vector<double>& image, 
                                int width, 
                                int height) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create output file: " + filepath);
    }
    
    // Implementation will depend on desired output format
    // TODO: Add support for saving in common image formats
}

std::vector<double> ImagePreprocessor::normalize(const std::vector<double>& image) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    
    auto stats = computeStats(image);
    if (stats.max == stats.min) {
        return std::vector<double>(image.size(), 0.0);
    }
    
    std::vector<double> normalized(image.size());
    std::transform(image.begin(), image.end(), normalized.begin(),
                  [&stats](double pixel) {
                      return (pixel - stats.min) / (stats.max - stats.min);
                  });
    
    return normalized;
}

std::vector<double> ImagePreprocessor::standardize(const std::vector<double>& image) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    
    auto stats = computeStats(image);
    if (stats.stddev == 0) {
        return std::vector<double>(image.size(), 0.0);
    }
    
    std::vector<double> standardized(image.size());
    std::transform(image.begin(), image.end(), standardized.begin(),
                  [&stats](double pixel) {
                      return (pixel - stats.mean) / stats.stddev;
                  });
    
    return standardized;
}

std::vector<double> ImagePreprocessor::binarize(const std::vector<double>& image, double threshold) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    if (threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("Threshold must be between 0 and 1");
    }
    
    std::vector<double> binarized(image.size());
    std::transform(image.begin(), image.end(), binarized.begin(),
                  [threshold](double pixel) {
                      return pixel >= threshold ? 1.0 : 0.0;
                  });
    
    return binarized;
}

std::vector<double> ImagePreprocessor::removeNoise(const std::vector<double>& image, 
                                                 int width, 
                                                 int height) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    // Implement median filter for noise removal
    const int kernel_size = 3;
    std::vector<double> kernel = {
        1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/9.0, 1.0/9.0, 1.0/9.0
    };
    
    return applyKernel(image, kernel, width, height, kernel_size);
}

std::vector<double> ImagePreprocessor::adjustContrast(const std::vector<double>& image, 
                                                    double factor) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    if (factor <= 0.0) {
        throw std::invalid_argument("Contrast factor must be positive");
    }
    
    auto stats = computeStats(image);
    std::vector<double> adjusted(image.size());
    
    std::transform(image.begin(), image.end(), adjusted.begin(),
                  [&stats, factor](double pixel) {
                      double centered = pixel - stats.mean;
                      double stretched = centered * factor;
                      return std::clamp(stretched + stats.mean, 0.0, 1.0);
                  });
    
    return adjusted;
}

std::vector<double> ImagePreprocessor::sharpen(const std::vector<double>& image, 
                                             int width, 
                                             int height) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    // Implement sharpening using Laplacian kernel
    const int kernel_size = 3;
    std::vector<double> kernel = {
        -1.0, -1.0, -1.0,
        -1.0,  9.0, -1.0,
        -1.0, -1.0, -1.0
    };
    
    return applyKernel(image, kernel, width, height, kernel_size);
}

ImageStats ImagePreprocessor::computeStats(const std::vector<double>& image) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    
    ImageStats stats;
    
    // Calculate mean
    stats.mean = std::accumulate(image.begin(), image.end(), 0.0) / image.size();
    
    // Calculate min and max
    auto [min_it, max_it] = std::minmax_element(image.begin(), image.end());
    stats.min = *min_it;
    stats.max = *max_it;
    
    // Calculate standard deviation
    double variance = std::accumulate(image.begin(), image.end(), 0.0,
        [&stats](double acc, double pixel) {
            double diff = pixel - stats.mean;
            return acc + (diff * diff);
        }) / image.size();
    
    stats.stddev = std::sqrt(variance);
    
    return stats;
}

double ImagePreprocessor::computeEntropy(const std::vector<double>& image) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    
    // Create histogram (assuming normalized values between 0 and 1)
    const int num_bins = 256;
    std::vector<int> histogram(num_bins, 0);
    
    for (double pixel : image) {
        int bin = static_cast<int>(std::clamp(pixel * (num_bins - 1), 0.0, num_bins - 1.0));
        histogram[bin]++;
    }
    
    // Calculate entropy
    double entropy = 0.0;
    double total_pixels = static_cast<double>(image.size());
    
    for (int count : histogram) {
        if (count > 0) {
            double probability = count / total_pixels;
            entropy -= probability * std::log2(probability);
        }
    }
    
    return entropy;
}

std::vector<std::vector<double>> ImagePreprocessor::batchPreprocess(
    const std::vector<std::vector<double>>& images,
    bool normalize_flag,
    bool remove_noise_flag,
    bool adjust_contrast_flag,
    double contrast_factor) {
    
    if (images.empty()) {
        throw std::invalid_argument("Empty image batch provided");
    }
    
    std::vector<std::vector<double>> processed_images;
    processed_images.reserve(images.size());
    
    for (const auto& image : images) {
        std::vector<double> processed = image;
        
        if (normalize_flag) {
            processed = normalize(processed);
        }
        if (remove_noise_flag) {
            // Assuming square images for simplicity
            int size = static_cast<int>(std::sqrt(processed.size()));
            processed = removeNoise(processed, size, size);
        }
        if (adjust_contrast_flag) {
            processed = adjustContrast(processed, contrast_factor);
        }
        
        processed_images.push_back(processed);
    }
    
    return processed_images;
}

std::vector<double> ImagePreprocessor::applyKernel(
    const std::vector<double>& image,
    const std::vector<double>& kernel,
    int width,
    int height,
    int kernel_size) {
    
    if (kernel.size() != kernel_size * kernel_size) {
        throw std::invalid_argument("Invalid kernel dimensions");
    }
    
    std::vector<double> padded = padImage(image, width, height, kernel_size / 2);
    std::vector<double> result(width * height);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int px = x + kx;
                    int py = y + ky;
                    sum += padded[py * (width + kernel_size - 1) + px] * 
                           kernel[ky * kernel_size + kx];
                }
            }
            
            result[y * width + x] = std::clamp(sum, 0.0, 1.0);
        }
    }
    
    return result;
}

std::vector<double> ImagePreprocessor::padImage(
    const std::vector<double>& image,
    int width,
    int height,
    int padding) {
    
    int padded_width = width + 2 * padding;
    int padded_height = height + 2 * padding;
    std::vector<double> padded(padded_width * padded_height, 0.0);
    
    // Copy original image to padded image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            padded[(y + padding) * padded_width + (x + padding)] = 
                image[y * width + x];
        }
    }
    
    return padded;
}

bool validateImageDimensions(const std::vector<double>& image, 
                           int expected_width, 
                           int expected_height) {
    return !image.empty() && 
           image.size() == static_cast<size_t>(expected_width * expected_height);
}

bool validatePixelRange(const std::vector<double>& image, 
                       double min, 
                       double max) {
    return std::all_of(image.begin(), image.end(),
                      [min, max](double pixel) {
                          return pixel >= min && pixel <= max;
                      });
}

} // namespace preprocessing 