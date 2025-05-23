#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include <string>
#include <optional>

namespace preprocessing {

struct ImageStats {
    double mean;
    double stddev;
    double min;
    double max;
};

class ImagePreprocessor {
public:
    // Image loading and conversion
    static std::vector<double> loadImage(const std::string& filepath);
    static void saveImage(const std::string& filepath, const std::vector<double>& image, int width, int height);
    
    // Basic preprocessing
    static std::vector<double> normalize(const std::vector<double>& image);
    static std::vector<double> standardize(const std::vector<double>& image);
    static std::vector<double> binarize(const std::vector<double>& image, double threshold = 0.5);
    
    // Advanced preprocessing
    static std::vector<double> removeNoise(const std::vector<double>& image, int width, int height);
    static std::vector<double> adjustContrast(const std::vector<double>& image, double factor);
    static std::vector<double> sharpen(const std::vector<double>& image, int width, int height);
    
    // Image analysis
    static ImageStats computeStats(const std::vector<double>& image);
    static double computeEntropy(const std::vector<double>& image);
    
    // Batch processing
    static std::vector<std::vector<double>> batchPreprocess(
        const std::vector<std::vector<double>>& images,
        bool normalize = true,
        bool remove_noise = true,
        bool adjust_contrast = true,
        double contrast_factor = 1.2
    );

    // New methods for image transformation
    std::vector<double> rotateImage(const std::vector<double>& image, 
                                  int width, 
                                  int height, 
                                  double angle_degrees);
                                  
    std::vector<double> scaleImage(const std::vector<double>& image,
                                 int original_width,
                                 int original_height,
                                 double scale_factor);
                                 
    std::vector<double> rotateAndScale(const std::vector<double>& image,
                                     int width,
                                     int height,
                                     double angle_degrees,
                                     double scale_factor);

    // New filtering methods
    static std::vector<double> gaussianBlur(const std::vector<double>& image, 
                                          int width, 
                                          int height, 
                                          double sigma = 1.0);
                                          
    static std::vector<double> edgeDetection(const std::vector<double>& image,
                                           int width,
                                           int height,
                                           bool useSobel = true);
                                           
    static std::vector<double> morphologicalOperation(const std::vector<double>& image,
                                                    int width,
                                                    int height,
                                                    const std::string& operation,
                                                    int kernel_size = 3);

private:
    // Helper functions
    static std::vector<double> applyKernel(
        const std::vector<double>& image,
        const std::vector<double>& kernel,
        int width,
        int height,
        int kernel_size
    );
    
    static std::vector<double> padImage(
        const std::vector<double>& image,
        int width,
        int height,
        int padding
    );

    // Helper methods for transformations
    std::vector<double> bilinearInterpolation(const std::vector<double>& image,
                                            int width,
                                            int height,
                                            double x,
                                            double y);

    // New helper methods for filtering
    static std::vector<double> createGaussianKernel(int size, double sigma);
    static std::vector<double> applySobelOperator(const std::vector<double>& image,
                                                int width,
                                                int height,
                                                bool horizontal);
    static std::vector<double> applyMorphologicalKernel(const std::vector<double>& image,
                                                      int width,
                                                      int height,
                                                      const std::vector<double>& kernel,
                                                      const std::string& operation);
};

// Validation utilities
bool validateImageDimensions(const std::vector<double>& image, int expected_width, int expected_height);
bool validatePixelRange(const std::vector<double>& image, double min = 0.0, double max = 1.0);

} // namespace preprocessing

#endif // PREPROCESSING_H 