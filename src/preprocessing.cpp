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

std::vector<double> ImagePreprocessor::applyCLAHE(const std::vector<double>& image,
                                                int width,
                                                int height,
                                                int window_size,
                                                double clip_limit) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (window_size < 2 || window_size % 2 == 0) {
        throw std::invalid_argument("Window size must be odd and >= 3");
    }
    if (clip_limit <= 0.0) {
        throw std::invalid_argument("Clip limit must be positive");
    }

    std::vector<double> result(width * height);
    const int half_window = window_size / 2;
    const int num_bins = 256;

    // Process each window
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Create histogram for local window
            std::vector<int> histogram(num_bins, 0);
            int valid_pixels = 0;

            for (int wy = -half_window; wy <= half_window; ++wy) {
                for (int wx = -half_window; wx <= half_window; ++wx) {
                    int nx = x + wx;
                    int ny = y + wy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int bin = static_cast<int>(image[ny * width + nx] * (num_bins - 1));
                        histogram[bin]++;
                        valid_pixels++;
                    }
                }
            }

            // Clip histogram
            int excess = 0;
            for (int& count : histogram) {
                if (count > clip_limit) {
                    excess += count - clip_limit;
                    count = static_cast<int>(clip_limit);
                }
            }

            // Redistribute excess
            int increment = excess / num_bins;
            int remainder = excess % num_bins;
            for (int& count : histogram) {
                count += increment;
                if (remainder > 0) {
                    count++;
                    remainder--;
                }
            }

            // Calculate CDF
            std::vector<double> cdf(num_bins);
            cdf[0] = static_cast<double>(histogram[0]) / valid_pixels;
            for (int i = 1; i < num_bins; ++i) {
                cdf[i] = cdf[i-1] + static_cast<double>(histogram[i]) / valid_pixels;
            }

            // Apply transformation
            int bin = static_cast<int>(image[y * width + x] * (num_bins - 1));
            result[y * width + x] = cdf[bin];
        }
    }

    return result;
}

std::vector<double> ImagePreprocessor::applyBilateralFilter(const std::vector<double>& image,
                                                          int width,
                                                          int height,
                                                          double sigma_space,
                                                          double sigma_color) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (sigma_space <= 0.0 || sigma_color <= 0.0) {
        throw std::invalid_argument("Sigma values must be positive");
    }

    std::vector<double> result(width * height);
    const int window_size = static_cast<int>(6 * sigma_space);
    const double sigma_space_sq = 2 * sigma_space * sigma_space;
    const double sigma_color_sq = 2 * sigma_color * sigma_color;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            double weight_sum = 0.0;
            double center_pixel = image[y * width + x];

            for (int wy = -window_size; wy <= window_size; ++wy) {
                for (int wx = -window_size; wx <= window_size; ++wx) {
                    int nx = x + wx;
                    int ny = y + wy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        double pixel = image[ny * width + nx];
                        double space_weight = std::exp(-(wx*wx + wy*wy) / sigma_space_sq);
                        double color_weight = std::exp(-std::pow(pixel - center_pixel, 2) / sigma_color_sq);
                        double weight = space_weight * color_weight;

                        sum += weight * pixel;
                        weight_sum += weight;
                    }
                }
            }

            result[y * width + x] = sum / weight_sum;
        }
    }

    return result;
}

std::vector<double> ImagePreprocessor::morphologicalOperation(const std::vector<double>& image,
                                                            int width,
                                                            int height,
                                                            int kernel_size,
                                                            bool is_dilation) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (kernel_size < 3 || kernel_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd and >= 3");
    }

    std::vector<double> result(width * height);
    const int half_kernel = kernel_size / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double value = is_dilation ? 0.0 : 1.0;

            for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        double pixel = image[ny * width + nx];
                        if (is_dilation) {
                            value = std::max(value, pixel);
                        } else {
                            value = std::min(value, pixel);
                        }
                    }
                }
            }

            result[y * width + x] = value;
        }
    }

    return result;
}

// Texture Analysis Methods
std::vector<double> ImagePreprocessor::computeLocalBinaryPattern(const std::vector<double>& image,
                                                               int width,
                                                               int height,
                                                               int radius) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> lbp(width * height);
    const int num_neighbors = 8;
    
    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            double center = image[y * width + x];
            unsigned char pattern = 0;
            
            // Sample points in a circle around the center
            for (int i = 0; i < num_neighbors; i++) {
                double angle = 2 * M_PI * i / num_neighbors;
                int nx = x + radius * cos(angle);
                int ny = y + radius * sin(angle);
                
                if (image[ny * width + nx] >= center) {
                    pattern |= (1 << i);
                }
            }
            
            lbp[y * width + x] = static_cast<double>(pattern) / 255.0;
        }
    }
    
    return lbp;
}

std::vector<double> ImagePreprocessor::computeGLCM(const std::vector<double>& image,
                                                 int width,
                                                 int height,
                                                 int distance,
                                                 int angle) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    const int num_levels = 256;
    std::vector<std::vector<int>> glcm(num_levels, std::vector<int>(num_levels, 0));
    
    // Convert image to discrete levels
    std::vector<int> discrete_image(width * height);
    for (int i = 0; i < width * height; i++) {
        discrete_image[i] = static_cast<int>(image[i] * (num_levels - 1));
    }
    
    // Compute GLCM
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int nx = x + distance * cos(angle * M_PI / 180.0);
            int ny = y + distance * sin(angle * M_PI / 180.0);
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int i = discrete_image[y * width + x];
                int j = discrete_image[ny * width + nx];
                glcm[i][j]++;
            }
        }
    }
    
    // Convert GLCM to feature vector
    return computeGLCMFeatures(glcm);
}

std::vector<double> ImagePreprocessor::computeHaralickFeatures(const std::vector<double>& image,
                                                             int width,
                                                             int height) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    // Compute GLCM for different angles
    std::vector<std::vector<int>> glcm_0 = std::vector<std::vector<int>>(256, std::vector<int>(256, 0));
    std::vector<std::vector<int>> glcm_45 = std::vector<std::vector<int>>(256, std::vector<int>(256, 0));
    std::vector<std::vector<int>> glcm_90 = std::vector<std::vector<int>>(256, std::vector<int>(256, 0));
    std::vector<std::vector<int>> glcm_135 = std::vector<std::vector<int>>(256, std::vector<int>(256, 0));
    
    // Compute GLCMs for different angles
    // ... (implementation details omitted for brevity)
    
    // Compute Haralick features
    std::vector<double> features;
    features.push_back(computeHaralickContrast(glcm_0));
    features.push_back(computeHaralickEnergy(glcm_0));
    features.push_back(computeHaralickCorrelation(glcm_0));
    
    return features;
}

// Feature Detection Methods
std::vector<std::pair<int, int>> ImagePreprocessor::detectCorners(const std::vector<double>& image,
                                                                 int width,
                                                                 int height,
                                                                 double threshold) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> harris_response = computeHarrisResponse(image, width, height);
    std::vector<std::pair<int, int>> corners;
    
    // Non-maximum suppression
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double current = harris_response[y * width + x];
            if (current > threshold) {
                bool is_max = true;
                for (int dy = -1; dy <= 1 && is_max; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        if (harris_response[(y + dy) * width + (x + dx)] >= current) {
                            is_max = false;
                            break;
                        }
                    }
                }
                if (is_max) {
                    corners.emplace_back(x, y);
                }
            }
        }
    }
    
    return corners;
}

std::vector<std::pair<int, int>> ImagePreprocessor::detectBlobs(const std::vector<double>& image,
                                                               int width,
                                                               int height,
                                                               double min_sigma,
                                                               double max_sigma) {
    if (!validateImageDimensions(image, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<std::pair<int, int>> blobs;
    const int num_scales = 10;
    double sigma_step = (max_sigma - min_sigma) / (num_scales - 1);
    
    // Compute Laplacian of Gaussian at different scales
    std::vector<std::vector<double>> scale_space(num_scales);
    for (int i = 0; i < num_scales; i++) {
        double sigma = min_sigma + i * sigma_step;
        scale_space[i] = computeLaplacianOfGaussian(image, width, height, sigma);
    }
    
    // Find local maxima in scale space
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int s = 1; s < num_scales - 1; s++) {
                double current = scale_space[s][y * width + x];
                bool is_max = true;
                
                // Check 3x3x3 neighborhood
                for (int ds = -1; ds <= 1 && is_max; ds++) {
                    for (int dy = -1; dy <= 1 && is_max; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0 && ds == 0) continue;
                            if (scale_space[s + ds][(y + dy) * width + (x + dx)] >= current) {
                                is_max = false;
                                break;
                            }
                        }
                    }
                }
                
                if (is_max) {
                    blobs.emplace_back(x, y);
                }
            }
        }
    }
    
    return blobs;
}

// Image Registration Methods
std::pair<double, double> ImagePreprocessor::computeImageAlignment(const std::vector<double>& source,
                                                                 const std::vector<double>& target,
                                                                 int width,
                                                                 int height) {
    if (!validateImageDimensions(source, width, height) || 
        !validateImageDimensions(target, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    // Compute gradients
    std::vector<double> source_grad = computeGradient(source, width, height);
    std::vector<double> target_grad = computeGradient(target, width, height);
    
    // Compute translation using phase correlation
    double dx = 0.0, dy = 0.0;
    // ... (implementation details omitted for brevity)
    
    return std::make_pair(dx, dy);
}

std::vector<double> ImagePreprocessor::registerImages(const std::vector<double>& source,
                                                    const std::vector<double>& target,
                                                    int width,
                                                    int height,
                                                    int max_iterations) {
    if (!validateImageDimensions(source, width, height) || 
        !validateImageDimensions(target, width, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> registered = source;
    double best_mi = computeMutualInformation(source, target, width, height);
    
    for (int iter = 0; iter < max_iterations; iter++) {
        auto [dx, dy] = computeImageAlignment(registered, target, width, height);
        
        // Apply transformation
        std::vector<double> transformed(width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double nx = x + dx;
                double ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    transformed[y * width + x] = bilinearInterpolation(registered, width, height, nx, ny);
                }
            }
        }
        
        double mi = computeMutualInformation(transformed, target, width, height);
        if (mi > best_mi) {
            best_mi = mi;
            registered = transformed;
        } else {
            break;
        }
    }
    
    return registered;
}

// Helper Methods
std::vector<double> ImagePreprocessor::computeGLCMFeatures(const std::vector<std::vector<int>>& glcm) {
    std::vector<double> features;
    features.push_back(computeHaralickContrast(glcm));
    features.push_back(computeHaralickEnergy(glcm));
    features.push_back(computeHaralickCorrelation(glcm));
    return features;
}

double ImagePreprocessor::computeHaralickContrast(const std::vector<std::vector<int>>& glcm) {
    double contrast = 0.0;
    int size = glcm.size();
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            contrast += glcm[i][j] * (i - j) * (i - j);
        }
    }
    
    return contrast;
}

double ImagePreprocessor::computeHaralickEnergy(const std::vector<std::vector<int>>& glcm) {
    double energy = 0.0;
    int size = glcm.size();
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            energy += glcm[i][j] * glcm[i][j];
        }
    }
    
    return energy;
}

double ImagePreprocessor::computeHaralickCorrelation(const std::vector<std::vector<int>>& glcm) {
    double correlation = 0.0;
    int size = glcm.size();
    
    // Compute means
    double mean_i = 0.0, mean_j = 0.0;
    double sum = 0.0;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mean_i += i * glcm[i][j];
            mean_j += j * glcm[i][j];
            sum += glcm[i][j];
        }
    }
    
    mean_i /= sum;
    mean_j /= sum;
    
    // Compute correlation
    double var_i = 0.0, var_j = 0.0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            correlation += (i - mean_i) * (j - mean_j) * glcm[i][j];
            var_i += (i - mean_i) * (i - mean_i) * glcm[i][j];
            var_j += (j - mean_j) * (j - mean_j) * glcm[i][j];
        }
    }
    
    correlation /= std::sqrt(var_i * var_j);
    return correlation;
}

std::vector<double> ImagePreprocessor::computeHarrisResponse(const std::vector<double>& image,
                                                           int width,
                                                           int height) {
    std::vector<double> response(width * height);
    
    // Compute image gradients
    std::vector<double> Ix(width * height);
    std::vector<double> Iy(width * height);
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            Ix[y * width + x] = (image[y * width + (x + 1)] - image[y * width + (x - 1)]) / 2.0;
            Iy[y * width + x] = (image[(y + 1) * width + x] - image[(y - 1) * width + x]) / 2.0;
        }
    }
    
    // Compute Harris response
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double Ixx = 0.0, Iyy = 0.0, Ixy = 0.0;
            
            // Compute structure tensor
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int idx = (y + dy) * width + (x + dx);
                    Ixx += Ix[idx] * Ix[idx];
                    Iyy += Iy[idx] * Iy[idx];
                    Ixy += Ix[idx] * Iy[idx];
                }
            }
            
            // Compute Harris response
            double det = Ixx * Iyy - Ixy * Ixy;
            double trace = Ixx + Iyy;
            response[y * width + x] = det - 0.04 * trace * trace;
        }
    }
    
    return response;
}

std::vector<double> ImagePreprocessor::computeLaplacianOfGaussian(const std::vector<double>& image,
                                                                int width,
                                                                int height,
                                                                double sigma) {
    std::vector<double> log_response(width * height);
    
    // Create LoG kernel
    int kernel_size = static_cast<int>(6 * sigma);
    if (kernel_size % 2 == 0) kernel_size++;
    int radius = kernel_size / 2;
    
    std::vector<double> kernel(kernel_size * kernel_size);
    double sum = 0.0;
    
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            double r2 = x * x + y * y;
            double value = (1.0 - r2 / (2 * sigma * sigma)) * 
                          std::exp(-r2 / (2 * sigma * sigma));
            kernel[(y + radius) * kernel_size + (x + radius)] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (double& k : kernel) {
        k /= sum;
    }
    
    // Apply kernel
    return applyKernel(image, kernel, width, height, kernel_size);
}

std::vector<double> ImagePreprocessor::computeGradient(const std::vector<double>& image,
                                                     int width,
                                                     int height) {
    std::vector<double> gradient(width * height);
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double dx = (image[y * width + (x + 1)] - image[y * width + (x - 1)]) / 2.0;
            double dy = (image[(y + 1) * width + x] - image[(y - 1) * width + x]) / 2.0;
            gradient[y * width + x] = std::sqrt(dx * dx + dy * dy);
        }
    }
    
    return gradient;
}

double ImagePreprocessor::computeMutualInformation(const std::vector<double>& source,
                                                 const std::vector<double>& target,
                                                 int width,
                                                 int height) {
    const int num_bins = 256;
    std::vector<std::vector<int>> joint_hist(num_bins, std::vector<int>(num_bins, 0));
    std::vector<int> source_hist(num_bins, 0);
    std::vector<int> target_hist(num_bins, 0);
    
    // Compute histograms
    for (int i = 0; i < width * height; i++) {
        int s_bin = static_cast<int>(source[i] * (num_bins - 1));
        int t_bin = static_cast<int>(target[i] * (num_bins - 1));
        joint_hist[s_bin][t_bin]++;
        source_hist[s_bin]++;
        target_hist[t_bin]++;
    }
    
    // Compute mutual information
    double mi = 0.0;
    double n = static_cast<double>(width * height);
    
    for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < num_bins; j++) {
            if (joint_hist[i][j] > 0) {
                double p_joint = joint_hist[i][j] / n;
                double p_source = source_hist[i] / n;
                double p_target = target_hist[j] / n;
                mi += p_joint * std::log2(p_joint / (p_source * p_target));
            }
        }
    }
    
    return mi;
}

} // namespace preprocessing 