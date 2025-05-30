#include "color_spaces.h"
#include <algorithm>
#include <random>
#include <map>
#include <queue>
#include <cmath>

namespace preprocessing {

// Helper function for RGB to XYZ conversion
std::vector<double> rgbToXYZ(const std::vector<double>& rgb_image, int width, int height) {
    std::vector<double> xyz_image(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        double r = rgb_image[i * 3];
        double g = rgb_image[i * 3 + 1];
        double b = rgb_image[i * 3 + 2];
        
        // Convert to linear RGB if needed (assuming input is sRGB)
        r = r <= 0.04045 ? r / 12.92 : std::pow((r + 0.055) / 1.055, 2.4);
        g = g <= 0.04045 ? g / 12.92 : std::pow((g + 0.055) / 1.055, 2.4);
        b = b <= 0.04045 ? b / 12.92 : std::pow((b + 0.055) / 1.055, 2.4);
        
        // Convert to XYZ
        xyz_image[i * 3] = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;     // X
        xyz_image[i * 3 + 1] = r * 0.2126729 + g * 0.7151522 + b * 0.0721750; // Y
        xyz_image[i * 3 + 2] = r * 0.0193339 + g * 0.1191920 + b * 0.9503041; // Z
    }
    
    return xyz_image;
}

// Helper function for XYZ to LAB conversion
std::vector<double> xyzToLab(const std::vector<double>& xyz_image, int width, int height) {
    std::vector<double> lab_image(width * height * 3);
    
    // D65 illuminant reference values
    const double Xn = 0.95047;
    const double Yn = 1.00000;
    const double Zn = 1.08883;
    
    for (int i = 0; i < width * height; i++) {
        double x = xyz_image[i * 3] / Xn;
        double y = xyz_image[i * 3 + 1] / Yn;
        double z = xyz_image[i * 3 + 2] / Zn;
        
        // Apply non-linear transformation
        x = x > 0.008856 ? std::pow(x, 1.0/3.0) : (7.787 * x + 16.0/116.0);
        y = y > 0.008856 ? std::pow(y, 1.0/3.0) : (7.787 * y + 16.0/116.0);
        z = z > 0.008856 ? std::pow(z, 1.0/3.0) : (7.787 * z + 16.0/116.0);
        
        // Calculate LAB values
        lab_image[i * 3] = (116.0 * y) - 16.0;                    // L
        lab_image[i * 3 + 1] = 500.0 * (x - y);                   // a
        lab_image[i * 3 + 2] = 200.0 * (y - z);                   // b
    }
    
    return lab_image;
}

std::vector<double> ColorSpaces::rgbToLab(const std::vector<double>& rgb_image,
                                        int width,
                                        int height) {
    if (!validateImageDimensions(rgb_image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    auto xyz_image = rgbToXYZ(rgb_image, width, height);
    return xyzToLab(xyz_image, width, height);
}

// Helper function for LAB to XYZ conversion
std::vector<double> labToXYZ(const std::vector<double>& lab_image, int width, int height) {
    std::vector<double> xyz_image(width * height * 3);
    
    // D65 illuminant reference values
    const double Xn = 0.95047;
    const double Yn = 1.00000;
    const double Zn = 1.08883;
    
    for (int i = 0; i < width * height; i++) {
        double l = lab_image[i * 3];
        double a = lab_image[i * 3 + 1];
        double b = lab_image[i * 3 + 2];
        
        // Convert LAB to XYZ
        double y = (l + 16.0) / 116.0;
        double x = a / 500.0 + y;
        double z = y - b / 200.0;
        
        // Apply inverse non-linear transformation
        x = std::pow(x, 3) > 0.008856 ? std::pow(x, 3) : (x - 16.0/116.0) / 7.787;
        y = std::pow(y, 3) > 0.008856 ? std::pow(y, 3) : (y - 16.0/116.0) / 7.787;
        z = std::pow(z, 3) > 0.008856 ? std::pow(z, 3) : (z - 16.0/116.0) / 7.787;
        
        xyz_image[i * 3] = x * Xn;     // X
        xyz_image[i * 3 + 1] = y * Yn; // Y
        xyz_image[i * 3 + 2] = z * Zn; // Z
    }
    
    return xyz_image;
}

// Helper function for XYZ to RGB conversion
std::vector<double> xyzToRGB(const std::vector<double>& xyz_image, int width, int height) {
    std::vector<double> rgb_image(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        double x = xyz_image[i * 3];
        double y = xyz_image[i * 3 + 1];
        double z = xyz_image[i * 3 + 2];
        
        // Convert XYZ to linear RGB
        double r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
        double g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560;
        double b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252;
        
        // Convert linear RGB to sRGB
        r = r <= 0.0031308 ? 12.92 * r : 1.055 * std::pow(r, 1.0/2.4) - 0.055;
        g = g <= 0.0031308 ? 12.92 * g : 1.055 * std::pow(g, 1.0/2.4) - 0.055;
        b = b <= 0.0031308 ? 12.92 * b : 1.055 * std::pow(b, 1.0/2.4) - 0.055;
        
        // Clamp values to [0, 1]
        rgb_image[i * 3] = std::max(0.0, std::min(1.0, r));
        rgb_image[i * 3 + 1] = std::max(0.0, std::min(1.0, g));
        rgb_image[i * 3 + 2] = std::max(0.0, std::min(1.0, b));
    }
    
    return rgb_image;
}

std::vector<double> ColorSpaces::labToRGB(const std::vector<double>& lab_image,
                                        int width,
                                        int height) {
    if (!validateImageDimensions(lab_image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    auto xyz_image = labToXYZ(lab_image, width, height);
    return xyzToRGB(xyz_image, width, height);
}

std::vector<double> ColorSpaces::quantizeColors(const std::vector<double>& image,
                                              int width,
                                              int height,
                                              int num_colors) {
    if (!validateImageDimensions(image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (num_colors < 2 || num_colors > 256) {
        throw std::invalid_argument("Number of colors must be between 2 and 256");
    }

    // Convert image to LAB color space for better quantization
    auto lab_image = rgbToLab(image, width, height);
    
    // Initialize k-means clustering
    std::vector<std::vector<double>> centroids(num_colors, std::vector<double>(3));
    std::vector<int> assignments(width * height);
    
    // Randomly initialize centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < num_colors; i++) {
        int idx = static_cast<int>(dis(gen) * width * height);
        centroids[i][0] = lab_image[idx * 3];
        centroids[i][1] = lab_image[idx * 3 + 1];
        centroids[i][2] = lab_image[idx * 3 + 2];
    }
    
    // K-means clustering
    const int max_iterations = 100;
    bool changed;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        changed = false;
        
        // Assign pixels to nearest centroid
        for (int i = 0; i < width * height; i++) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            
            for (int j = 0; j < num_colors; j++) {
                double dist = 0;
                for (int k = 0; k < 3; k++) {
                    double diff = lab_image[i * 3 + k] - centroids[j][k];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            if (assignments[i] != best_cluster) {
                assignments[i] = best_cluster;
                changed = true;
            }
        }
        
        if (!changed) break;
        
        // Update centroids
        std::vector<std::vector<double>> new_centroids(num_colors, std::vector<double>(3, 0.0));
        std::vector<int> counts(num_colors, 0);
        
        for (int i = 0; i < width * height; i++) {
            int cluster = assignments[i];
            for (int k = 0; k < 3; k++) {
                new_centroids[cluster][k] += lab_image[i * 3 + k];
            }
            counts[cluster]++;
        }
        
        for (int i = 0; i < num_colors; i++) {
            if (counts[i] > 0) {
                for (int k = 0; k < 3; k++) {
                    centroids[i][k] = new_centroids[i][k] / counts[i];
                }
            }
        }
    }
    
    // Create quantized image
    std::vector<double> quantized(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        int cluster = assignments[i];
        for (int k = 0; k < 3; k++) {
            quantized[i * 3 + k] = centroids[cluster][k];
        }
    }
    
    // Convert back to RGB
    return labToRGB(quantized, width, height);
}

bool ColorSpaces::validateImageDimensions(const std::vector<double>& image,
                                        int expected_width,
                                        int expected_height) {
    return !image.empty() && 
           image.size() == static_cast<size_t>(expected_width * expected_height);
}

std::vector<double> ColorSpaces::computeColorMeans(const std::vector<double>& image,
                                                 int width,
                                                 int height) {
    std::vector<double> means(3, 0.0);
    
    for (int i = 0; i < width * height; i++) {
        for (int k = 0; k < 3; k++) {
            means[k] += image[i * 3 + k];
        }
    }
    
    for (int k = 0; k < 3; k++) {
        means[k] /= (width * height);
    }
    
    return means;
}

std::vector<double> ColorSpaces::computeColorCovariance(const std::vector<double>& image,
                                                      int width,
                                                      int height,
                                                      const std::vector<double>& means) {
    std::vector<double> covariance(9, 0.0);  // 3x3 matrix stored as vector
    
    for (int i = 0; i < width * height; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                double diff_j = image[i * 3 + j] - means[j];
                double diff_k = image[i * 3 + k] - means[k];
                covariance[j * 3 + k] += diff_j * diff_k;
            }
        }
    }
    
    for (int i = 0; i < 9; i++) {
        covariance[i] /= (width * height);
    }
    
    return covariance;
}

std::vector<double> ColorSpaces::rgbToHSV(const std::vector<double>& rgb_image,
                                        int width,
                                        int height) {
    if (!validateImageDimensions(rgb_image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> hsv_image(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        double r = rgb_image[i * 3];
        double g = rgb_image[i * 3 + 1];
        double b = rgb_image[i * 3 + 2];
        
        double max_val = std::max({r, g, b});
        double min_val = std::min({r, g, b});
        double delta = max_val - min_val;
        
        // Calculate Value
        hsv_image[i * 3 + 2] = max_val;
        
        // Calculate Saturation
        hsv_image[i * 3 + 1] = max_val == 0.0 ? 0.0 : delta / max_val;
        
        // Calculate Hue
        if (delta == 0.0) {
            hsv_image[i * 3] = 0.0;
        } else if (max_val == r) {
            hsv_image[i * 3] = 60.0 * std::fmod((g - b) / delta + 6.0, 6.0);
        } else if (max_val == g) {
            hsv_image[i * 3] = 60.0 * ((b - r) / delta + 2.0);
        } else {
            hsv_image[i * 3] = 60.0 * ((r - g) / delta + 4.0);
        }
        
        // Normalize hue to [0, 1]
        hsv_image[i * 3] /= 360.0;
    }
    
    return hsv_image;
}

std::vector<double> ColorSpaces::hsvToRGB(const std::vector<double>& hsv_image,
                                        int width,
                                        int height) {
    if (!validateImageDimensions(hsv_image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> rgb_image(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        double h = hsv_image[i * 3] * 360.0;  // Convert hue back to degrees
        double s = hsv_image[i * 3 + 1];
        double v = hsv_image[i * 3 + 2];
        
        double c = v * s;
        double x = c * (1.0 - std::abs(std::fmod(h / 60.0, 2.0) - 1.0));
        double m = v - c;
        
        double r, g, b;
        if (h >= 0.0 && h < 60.0) {
            r = c; g = x; b = 0.0;
        } else if (h >= 60.0 && h < 120.0) {
            r = x; g = c; b = 0.0;
        } else if (h >= 120.0 && h < 180.0) {
            r = 0.0; g = c; b = x;
        } else if (h >= 180.0 && h < 240.0) {
            r = 0.0; g = x; b = c;
        } else if (h >= 240.0 && h < 300.0) {
            r = x; g = 0.0; b = c;
        } else {
            r = c; g = 0.0; b = x;
        }
        
        rgb_image[i * 3] = r + m;
        rgb_image[i * 3 + 1] = g + m;
        rgb_image[i * 3 + 2] = b + m;
    }
    
    return rgb_image;
}

std::vector<double> ColorSpaces::rgbToYUV(const std::vector<double>& rgb_image,
                                        int width,
                                        int height) {
    if (!validateImageDimensions(rgb_image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> yuv_image(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        double r = rgb_image[i * 3];
        double g = rgb_image[i * 3 + 1];
        double b = rgb_image[i * 3 + 2];
        
        // Convert RGB to YUV using BT.601 coefficients
        yuv_image[i * 3] = 0.299 * r + 0.587 * g + 0.114 * b;                    // Y
        yuv_image[i * 3 + 1] = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5;    // U
        yuv_image[i * 3 + 2] = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5;     // V
        
        // Clamp U and V to [0, 1]
        yuv_image[i * 3 + 1] = std::max(0.0, std::min(1.0, yuv_image[i * 3 + 1]));
        yuv_image[i * 3 + 2] = std::max(0.0, std::min(1.0, yuv_image[i * 3 + 2]));
    }
    
    return yuv_image;
}

std::vector<double> ColorSpaces::yuvToRGB(const std::vector<double>& yuv_image,
                                        int width,
                                        int height) {
    if (!validateImageDimensions(yuv_image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> rgb_image(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        double y = yuv_image[i * 3];
        double u = yuv_image[i * 3 + 1] - 0.5;  // Center U around 0
        double v = yuv_image[i * 3 + 2] - 0.5;  // Center V around 0
        
        // Convert YUV to RGB using BT.601 coefficients
        rgb_image[i * 3] = y + 1.13983 * v;                     // R
        rgb_image[i * 3 + 1] = y - 0.39465 * u - 0.58060 * v;  // G
        rgb_image[i * 3 + 2] = y + 2.03211 * u;                // B
        
        // Clamp RGB values to [0, 1]
        rgb_image[i * 3] = std::max(0.0, std::min(1.0, rgb_image[i * 3]));
        rgb_image[i * 3 + 1] = std::max(0.0, std::min(1.0, rgb_image[i * 3 + 1]));
        rgb_image[i * 3 + 2] = std::max(0.0, std::min(1.0, rgb_image[i * 3 + 2]));
    }
    
    return rgb_image;
}

std::vector<double> ColorSpaces::adjustHue(const std::vector<double>& image,
                                         int width,
                                         int height,
                                         double hue_shift) {
    if (!validateImageDimensions(image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    // Convert to HSV, adjust hue, and convert back to RGB
    auto hsv_image = rgbToHSV(image, width, height);
    
    for (int i = 0; i < width * height; i++) {
        // Shift hue and wrap around if necessary
        hsv_image[i * 3] = std::fmod(hsv_image[i * 3] + hue_shift + 1.0, 1.0);
    }
    
    return hsvToRGB(hsv_image, width, height);
}

std::vector<double> ColorSpaces::adjustSaturation(const std::vector<double>& image,
                                                int width,
                                                int height,
                                                double factor) {
    if (!validateImageDimensions(image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (factor < 0.0) {
        throw std::invalid_argument("Saturation factor must be non-negative");
    }
    
    // Convert to HSV, adjust saturation, and convert back to RGB
    auto hsv_image = rgbToHSV(image, width, height);
    
    for (int i = 0; i < width * height; i++) {
        hsv_image[i * 3 + 1] = std::min(1.0, hsv_image[i * 3 + 1] * factor);
    }
    
    return hsvToRGB(hsv_image, width, height);
}

std::vector<double> ColorSpaces::adjustBrightness(const std::vector<double>& image,
                                                int width,
                                                int height,
                                                double factor) {
    if (!validateImageDimensions(image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    std::vector<double> result(width * height * 3);
    
    for (int i = 0; i < width * height * 3; i++) {
        result[i] = std::max(0.0, std::min(1.0, image[i] * factor));
    }
    
    return result;
}

std::vector<double> ColorSpaces::adjustContrast(const std::vector<double>& image,
                                              int width,
                                              int height,
                                              double factor) {
    if (!validateImageDimensions(image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    // Calculate mean luminance
    double mean = 0.0;
    for (int i = 0; i < width * height; i++) {
        mean += (image[i * 3] * 0.299 + image[i * 3 + 1] * 0.587 + image[i * 3 + 2] * 0.114);
    }
    mean /= (width * height);
    
    std::vector<double> result(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            // Apply contrast adjustment around the mean
            result[i * 3 + c] = mean + factor * (image[i * 3 + c] - mean);
            // Clamp to [0, 1]
            result[i * 3 + c] = std::max(0.0, std::min(1.0, result[i * 3 + c]));
        }
    }
    
    return result;
}

std::vector<double> ColorSpaces::whiteBalance(const std::vector<double>& image,
                                            int width,
                                            int height) {
    if (!validateImageDimensions(image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    
    // Calculate mean values for each channel
    std::vector<double> means = computeColorMeans(image, width, height);
    
    // Find the maximum mean value
    double max_mean = std::max({means[0], means[1], means[2]});
    
    // Calculate scaling factors
    std::vector<double> scale_factors(3);
    for (int i = 0; i < 3; i++) {
        scale_factors[i] = max_mean / means[i];
    }
    
    // Apply white balance
    std::vector<double> result(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            result[i * 3 + c] = std::min(1.0, image[i * 3 + c] * scale_factors[c]);
        }
    }
    
    return result;
}

std::vector<double> ColorSpaces::colorCorrection(const std::vector<double>& image,
                                               int width,
                                               int height,
                                               const std::vector<double>& color_matrix) {
    if (!validateImageDimensions(image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (color_matrix.size() != 9) {
        throw std::invalid_argument("Color matrix must be 3x3 (9 elements)");
    }
    
    std::vector<double> result(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            double new_value = 0.0;
            for (int j = 0; j < 3; j++) {
                new_value += image[i * 3 + j] * color_matrix[c * 3 + j];
            }
            result[i * 3 + c] = std::max(0.0, std::min(1.0, new_value));
        }
    }
    
    return result;
}

} // namespace preprocessing 