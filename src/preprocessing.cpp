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

std::vector<double> ImagePreprocessor::rotateImage(const std::vector<double>& image,
                                                 int width,
                                                 int height,
                                                 double angle_degrees) {
    std::vector<double> rotated(width * height, 0.0);
    double angle_rad = angle_degrees * M_PI / 180.0;
    double cos_angle = std::cos(angle_rad);
    double sin_angle = std::sin(angle_rad);
    
    // Calculate center point
    double center_x = (width - 1) / 2.0;
    double center_y = (height - 1) / 2.0;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Calculate rotated coordinates
            double dx = x - center_x;
            double dy = y - center_y;
            double rotated_x = dx * cos_angle - dy * sin_angle + center_x;
            double rotated_y = dx * sin_angle + dy * cos_angle + center_y;
            
            // Use bilinear interpolation to get pixel value
            rotated[y * width + x] = bilinearInterpolation(image, width, height, rotated_x, rotated_y);
        }
    }
    
    return rotated;
}

std::vector<double> ImagePreprocessor::scaleImage(const std::vector<double>& image,
                                                int original_width,
                                                int original_height,
                                                double scale_factor) {
    int new_width = static_cast<int>(original_width * scale_factor);
    int new_height = static_cast<int>(original_height * scale_factor);
    std::vector<double> scaled(new_width * new_height, 0.0);
    
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            // Calculate corresponding position in original image
            double original_x = x / scale_factor;
            double original_y = y / scale_factor;
            
            // Use bilinear interpolation
            scaled[y * new_width + x] = bilinearInterpolation(image, original_width, original_height,
                                                            original_x, original_y);
        }
    }
    
    return scaled;
}

std::vector<double> ImagePreprocessor::rotateAndScale(const std::vector<double>& image,
                                                    int width,
                                                    int height,
                                                    double angle_degrees,
                                                    double scale_factor) {
    // First rotate, then scale
    auto rotated = rotateImage(image, width, height, angle_degrees);
    return scaleImage(rotated, width, height, scale_factor);
}

std::vector<double> ImagePreprocessor::bilinearInterpolation(const std::vector<double>& image,
                                                           int width,
                                                           int height,
                                                           double x,
                                                           double y) {
    // Get the four surrounding pixels
    int x1 = static_cast<int>(std::floor(x));
    int y1 = static_cast<int>(std::floor(y));
    int x2 = std::min(x1 + 1, width - 1);
    int y2 = std::min(y1 + 1, height - 1);
    
    // Calculate interpolation weights
    double wx = x - x1;
    double wy = y - y1;
    
    // Get pixel values
    double p11 = image[y1 * width + x1];
    double p12 = image[y1 * width + x2];
    double p21 = image[y2 * width + x1];
    double p22 = image[y2 * width + x2];
    
    // Perform bilinear interpolation
    double interpolated = (1 - wx) * (1 - wy) * p11 +
                         wx * (1 - wy) * p12 +
                         (1 - wx) * wy * p21 +
                         wx * wy * p22;
    
    return std::clamp(interpolated, 0.0, 1.0);
}

std::vector<double> ImagePreprocessor::createGaussianKernel(int size, double sigma) {
    std::vector<double> kernel(size * size);
    double sum = 0.0;
    int center = size / 2;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            double dx = x - center;
            double dy = y - center;
            double value = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            kernel[y * size + x] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (double& value : kernel) {
        value /= sum;
    }
    
    return kernel;
}

std::vector<double> ImagePreprocessor::gaussianBlur(const std::vector<double>& image,
                                                  int width,
                                                  int height,
                                                  double sigma) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    int kernel_size = static_cast<int>(std::ceil(6 * sigma)) | 1; // Ensure odd size
    auto kernel = createGaussianKernel(kernel_size, sigma);
    
    return applyKernel(image, kernel, width, height, kernel_size);
}

std::vector<double> ImagePreprocessor::applySobelOperator(const std::vector<double>& image,
                                                        int width,
                                                        int height,
                                                        bool horizontal) {
    const std::vector<double> sobel_x = {
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    };
    
    const std::vector<double> sobel_y = {
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    };
    
    return applyKernel(image, horizontal ? sobel_x : sobel_y, width, height, 3);
}

std::vector<double> ImagePreprocessor::edgeDetection(const std::vector<double>& image,
                                                   int width,
                                                   int height,
                                                   bool useSobel) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    if (useSobel) {
        auto grad_x = applySobelOperator(image, width, height, true);
        auto grad_y = applySobelOperator(image, width, height, false);
        
        // Combine gradients
        std::vector<double> edges(width * height);
        for (size_t i = 0; i < edges.size(); ++i) {
            edges[i] = std::sqrt(grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]);
        }
        
        // Normalize
        return normalize(edges);
    } else {
        // Simple edge detection using Laplacian
        const std::vector<double> laplacian = {
            0.0,  1.0, 0.0,
            1.0, -4.0, 1.0,
            0.0,  1.0, 0.0
        };
        
        auto edges = applyKernel(image, laplacian, width, height, 3);
        return normalize(edges);
    }
}

std::vector<double> ImagePreprocessor::applyMorphologicalKernel(const std::vector<double>& image,
                                                              int width,
                                                              int height,
                                                              const std::vector<double>& kernel,
                                                              const std::string& operation) {
    int kernel_size = static_cast<int>(std::sqrt(kernel.size()));
    auto padded = padImage(image, width, height, kernel_size / 2);
    std::vector<double> result(width * height);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<double> window;
            window.reserve(kernel_size * kernel_size);
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int px = x + kx;
                    int py = y + ky;
                    window.push_back(padded[py * (width + kernel_size - 1) + px]);
                }
            }
            
            if (operation == "erode") {
                result[y * width + x] = *std::min_element(window.begin(), window.end());
            } else if (operation == "dilate") {
                result[y * width + x] = *std::max_element(window.begin(), window.end());
            } else {
                throw std::invalid_argument("Invalid morphological operation");
            }
        }
    }
    
    return result;
}

std::vector<double> ImagePreprocessor::morphologicalOperation(const std::vector<double>& image,
                                                            int width,
                                                            int height,
                                                            const std::string& operation,
                                                            int kernel_size) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    if (kernel_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd");
    }
    
    // Create a simple square kernel
    std::vector<double> kernel(kernel_size * kernel_size, 1.0);
    
    return applyMorphologicalKernel(image, width, height, kernel, operation);
}

std::vector<double> ImagePreprocessor::computeLocalThreshold(const std::vector<double>& image,
                                                           int width,
                                                           int height,
                                                           int window_size) {
    std::vector<double> thresholds(width * height);
    int half_window = window_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<double> window;
            window.reserve(window_size * window_size);
            
            // Collect pixels in window
            for (int wy = -half_window; wy <= half_window; ++wy) {
                for (int wx = -half_window; wx <= half_window; ++wx) {
                    int px = std::clamp(x + wx, 0, width - 1);
                    int py = std::clamp(y + wy, 0, height - 1);
                    window.push_back(image[py * width + px]);
                }
            }
            
            // Compute local threshold (mean of window)
            thresholds[y * width + x] = std::accumulate(window.begin(), window.end(), 0.0) / window.size();
        }
    }
    
    return thresholds;
}

std::vector<double> ImagePreprocessor::thresholdSegmentation(const std::vector<double>& image,
                                                           double threshold,
                                                           bool adaptive) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    
    std::vector<double> segmented(image.size());
    
    if (adaptive) {
        // Assuming square image for simplicity
        int size = static_cast<int>(std::sqrt(image.size()));
        auto local_thresholds = computeLocalThreshold(image, size, size, 15);
        
        for (size_t i = 0; i < image.size(); ++i) {
            segmented[i] = image[i] > local_thresholds[i] ? 1.0 : 0.0;
        }
    } else {
        std::transform(image.begin(), image.end(), segmented.begin(),
                      [threshold](double pixel) {
                          return pixel > threshold ? 1.0 : 0.0;
                      });
    }
    
    return segmented;
}

std::vector<int> ImagePreprocessor::findPeaks(const std::vector<int>& histogram,
                                            int min_distance) {
    std::vector<int> peaks;
    for (size_t i = 1; i < histogram.size() - 1; ++i) {
        if (histogram[i] > histogram[i-1] && histogram[i] > histogram[i+1]) {
            // Check if this peak is far enough from previous peaks
            bool is_far_enough = true;
            for (int peak : peaks) {
                if (std::abs(static_cast<int>(i) - peak) < min_distance) {
                    is_far_enough = false;
                    break;
                }
            }
            if (is_far_enough) {
                peaks.push_back(i);
            }
        }
    }
    return peaks;
}

std::vector<double> ImagePreprocessor::computeDistanceTransform(const std::vector<double>& image,
                                                              int width,
                                                              int height) {
    std::vector<double> distance(width * height, std::numeric_limits<double>::max());
    
    // Forward pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image[y * width + x] > 0.5) {
                distance[y * width + x] = 0;
            } else {
                if (x > 0) {
                    distance[y * width + x] = std::min(distance[y * width + (x-1)] + 1,
                                                     distance[y * width + x]);
                }
                if (y > 0) {
                    distance[y * width + x] = std::min(distance[(y-1) * width + x] + 1,
                                                     distance[y * width + x]);
                }
            }
        }
    }
    
    // Backward pass
    for (int y = height-1; y >= 0; --y) {
        for (int x = width-1; x >= 0; --x) {
            if (x < width-1) {
                distance[y * width + x] = std::min(distance[y * width + (x+1)] + 1,
                                                 distance[y * width + x]);
            }
            if (y < height-1) {
                distance[y * width + x] = std::min(distance[(y+1) * width + x] + 1,
                                                 distance[y * width + x]);
            }
        }
    }
    
    return distance;
}

std::vector<double> ImagePreprocessor::watershedSegmentation(const std::vector<double>& image,
                                                           int width,
                                                           int height,
                                                           int min_distance) {
    // Compute gradient magnitude
    auto grad_x = applySobelOperator(image, width, height, true);
    auto grad_y = applySobelOperator(image, width, height, false);
    
    std::vector<double> gradient(width * height);
    for (size_t i = 0; i < gradient.size(); ++i) {
        gradient[i] = std::sqrt(grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]);
    }
    
    // Compute distance transform
    auto distance = computeDistanceTransform(gradient, width, height);
    
    // Find peaks in distance transform
    std::vector<int> histogram(256, 0);
    for (double d : distance) {
        int bin = static_cast<int>(std::clamp(d * 255, 0.0, 255.0));
        histogram[bin]++;
    }
    
    auto peaks = findPeaks(histogram, min_distance);
    
    // Create segmentation mask
    std::vector<double> segmented(width * height, 0.0);
    for (int peak : peaks) {
        double threshold = peak / 255.0;
        for (size_t i = 0; i < segmented.size(); ++i) {
            if (distance[i] <= threshold) {
                segmented[i] = 1.0;
            }
        }
    }
    
    return segmented;
}

std::vector<double> ImagePreprocessor::kmeansSegmentation(const std::vector<double>& image,
                                                        int width,
                                                        int height,
                                                        int k,
                                                        int max_iterations) {
    if (k < 2) {
        throw std::invalid_argument("Number of clusters must be at least 2");
    }
    
    // Initialize centroids randomly
    std::vector<double> centroids(k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < k; ++i) {
        centroids[i] = dis(gen);
    }
    
    std::vector<double> segmented(image.size());
    std::vector<int> assignments(image.size());
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Assign pixels to nearest centroid
        for (size_t i = 0; i < image.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            
            for (int j = 0; j < k; ++j) {
                double dist = std::abs(image[i] - centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            assignments[i] = best_cluster;
        }
        
        // Update centroids
        std::vector<double> new_centroids(k, 0.0);
        std::vector<int> counts(k, 0);
        
        for (size_t i = 0; i < image.size(); ++i) {
            new_centroids[assignments[i]] += image[i];
            counts[assignments[i]]++;
        }
        
        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                centroids[i] = new_centroids[i] / counts[i];
            }
        }
    }
    
    // Create segmentation mask
    for (size_t i = 0; i < segmented.size(); ++i) {
        segmented[i] = static_cast<double>(assignments[i]) / (k - 1);
    }
    
    return segmented;
}

HistogramStats ImagePreprocessor::computeHistogramStats(const std::vector<double>& image,
                                                      int num_bins) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    
    HistogramStats stats;
    stats.histogram.resize(num_bins, 0);
    
    // Compute histogram
    for (double pixel : image) {
        int bin = static_cast<int>(std::clamp(pixel * (num_bins - 1), 0.0, num_bins - 1.0));
        stats.histogram[bin]++;
    }
    
    // Compute statistics
    double total_pixels = static_cast<double>(image.size());
    
    // Mean
    stats.mean = std::accumulate(image.begin(), image.end(), 0.0) / total_pixels;
    
    // Median
    std::vector<double> sorted = image;
    std::sort(sorted.begin(), sorted.end());
    stats.median = sorted[sorted.size() / 2];
    
    // Mode
    auto max_it = std::max_element(stats.histogram.begin(), stats.histogram.end());
    stats.mode = static_cast<double>(std::distance(stats.histogram.begin(), max_it)) / (num_bins - 1);
    
    // Peak count
    stats.peak_count = findPeaks(stats.histogram, 5).size();
    
    // Entropy
    stats.entropy = 0.0;
    for (int count : stats.histogram) {
        if (count > 0) {
            double probability = count / total_pixels;
            stats.entropy -= probability * std::log2(probability);
        }
    }
    
    return stats;
}

std::vector<double> ImagePreprocessor::histogramEqualization(const std::vector<double>& image) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided");
    }
    
    const int num_bins = 256;
    std::vector<int> histogram(num_bins, 0);
    
    // Compute histogram
    for (double pixel : image) {
        int bin = static_cast<int>(std::clamp(pixel * (num_bins - 1), 0.0, num_bins - 1.0));
        histogram[bin]++;
    }
    
    // Compute cumulative distribution function
    std::vector<double> cdf(num_bins);
    cdf[0] = static_cast<double>(histogram[0]) / image.size();
    for (int i = 1; i < num_bins; ++i) {
        cdf[i] = cdf[i-1] + static_cast<double>(histogram[i]) / image.size();
    }
    
    // Apply equalization
    std::vector<double> equalized(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        int bin = static_cast<int>(std::clamp(image[i] * (num_bins - 1), 0.0, num_bins - 1.0));
        equalized[i] = cdf[bin];
    }
    
    return equalized;
}

std::vector<double> ImagePreprocessor::adaptiveHistogramEqualization(const std::vector<double>& image,
                                                                   int width,
                                                                   int height,
                                                                   int window_size) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> equalized(width * height);
    int half_window = window_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<double> window;
            window.reserve(window_size * window_size);
            
            // Collect pixels in window
            for (int wy = -half_window; wy <= half_window; ++wy) {
                for (int wx = -half_window; wx <= half_window; ++wx) {
                    int px = std::clamp(x + wx, 0, width - 1);
                    int py = std::clamp(y + wy, 0, height - 1);
                    window.push_back(image[py * width + px]);
                }
            }
            
            // Compute local histogram
            const int num_bins = 256;
            std::vector<int> histogram(num_bins, 0);
            
            for (double pixel : window) {
                int bin = static_cast<int>(std::clamp(pixel * (num_bins - 1), 0.0, num_bins - 1.0));
                histogram[bin]++;
            }
            
            // Compute local CDF
            std::vector<double> cdf(num_bins);
            cdf[0] = static_cast<double>(histogram[0]) / window.size();
            for (int i = 1; i < num_bins; ++i) {
                cdf[i] = cdf[i-1] + static_cast<double>(histogram[i]) / window.size();
            }
            
            // Apply local equalization
            int bin = static_cast<int>(std::clamp(image[y * width + x] * (num_bins - 1), 0.0, num_bins - 1.0));
            equalized[y * width + x] = cdf[bin];
        }
    }
    
    return equalized;
}

} // namespace preprocessing 