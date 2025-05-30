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

struct HistogramStats {
    std::vector<int> histogram;
    double entropy;
    double mean;
    double median;
    double mode;
    int peak_count;
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

    // New segmentation methods
    static std::vector<double> thresholdSegmentation(const std::vector<double>& image,
                                                   double threshold,
                                                   bool adaptive = false);
                                                   
    static std::vector<double> watershedSegmentation(const std::vector<double>& image,
                                                   int width,
                                                   int height,
                                                   int min_distance = 10);
                                                   
    static std::vector<double> kmeansSegmentation(const std::vector<double>& image,
                                                int width,
                                                int height,
                                                int k = 2,
                                                int max_iterations = 100);
    
    // New histogram analysis methods
    static HistogramStats computeHistogramStats(const std::vector<double>& image,
                                              int num_bins = 256);
                                              
    static std::vector<double> histogramEqualization(const std::vector<double>& image);
    
    static std::vector<double> adaptiveHistogramEqualization(const std::vector<double>& image,
                                                           int width,
                                                           int height,
                                                           int window_size = 8);

    // New advanced preprocessing methods
    static std::vector<double> applyCLAHE(const std::vector<double>& image,
                                        int width,
                                        int height,
                                        int window_size = 8,
                                        double clip_limit = 2.0);
                                        
    static std::vector<double> applyBilateralFilter(const std::vector<double>& image,
                                                  int width,
                                                  int height,
                                                  double sigma_space = 3.0,
                                                  double sigma_color = 0.1);
                                                  
    static std::vector<double> morphologicalOperation(const std::vector<double>& image,
                                                    int width,
                                                    int height,
                                                    int kernel_size = 3,
                                                    bool is_dilation = true);

    // New frequency domain operations
    static std::vector<double> applyFFT(const std::vector<double>& image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> applyIFFT(const std::vector<double>& fft_image,
                                       int width,
                                       int height);
                                       
    static std::vector<double> frequencyFilter(const std::vector<double>& image,
                                             int width,
                                             int height,
                                             const std::string& filter_type,
                                             double cutoff_frequency);

    // New color space conversion methods
    static std::vector<double> rgbToGrayscale(const std::vector<double>& rgb_image,
                                            int width,
                                            int height);
                                            
    static std::vector<double> rgbToHSV(const std::vector<double>& rgb_image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> hsvToRGB(const std::vector<double>& hsv_image,
                                      int width,
                                      int height);

    static std::vector<double> rgbToLab(const std::vector<double>& rgb_image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> labToRGB(const std::vector<double>& lab_image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> rgbToYUV(const std::vector<double>& rgb_image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> yuvToRGB(const std::vector<double>& yuv_image,
                                      int width,
                                      int height);

    // New color manipulation methods
    static std::vector<double> adjustHue(const std::vector<double>& image,
                                       int width,
                                       int height,
                                       double hue_shift);
                                       
    static std::vector<double> adjustSaturation(const std::vector<double>& image,
                                              int width,
                                              int height,
                                              double factor);
                                              
    static std::vector<double> adjustBrightness(const std::vector<double>& image,
                                              int width,
                                              int height,
                                              double factor);

    // New texture analysis methods
    static std::vector<double> computeLocalBinaryPattern(const std::vector<double>& image,
                                                       int width,
                                                       int height,
                                                       int radius = 1);
                                                       
    static std::vector<double> computeGLCM(const std::vector<double>& image,
                                         int width,
                                         int height,
                                         int distance = 1,
                                         int angle = 0);
                                         
    static std::vector<double> computeHaralickFeatures(const std::vector<double>& image,
                                                     int width,
                                                     int height);

    // New feature detection methods
    static std::vector<std::pair<int, int>> detectCorners(const std::vector<double>& image,
                                                         int width,
                                                         int height,
                                                         double threshold = 0.01);
                                                         
    static std::vector<std::pair<int, int>> detectBlobs(const std::vector<double>& image,
                                                       int width,
                                                       int height,
                                                       double min_sigma = 1.0,
                                                       double max_sigma = 3.0);
                                                       
    static std::vector<std::pair<int, int>> detectEdges(const std::vector<double>& image,
                                                       int width,
                                                       int height,
                                                       double low_threshold = 0.1,
                                                       double high_threshold = 0.3);

    // New image registration methods
    static std::pair<double, double> computeImageAlignment(const std::vector<double>& source,
                                                         const std::vector<double>& target,
                                                         int width,
                                                         int height);
                                                         
    static std::vector<double> registerImages(const std::vector<double>& source,
                                            const std::vector<double>& target,
                                            int width,
                                            int height,
                                            int max_iterations = 100);

    // New image enhancement methods
    static std::vector<double> unsharpMasking(const std::vector<double>& image,
                                            int width,
                                            int height,
                                            double amount = 1.0,
                                            double radius = 1.0);
                                            
    static std::vector<double> toneMapping(const std::vector<double>& image,
                                         int width,
                                         int height,
                                         const std::string& method = "reinhard");
                                         
    static std::vector<double> denoiseNonLocalMeans(const std::vector<double>& image,
                                                  int width,
                                                  int height,
                                                  double h = 10.0,
                                                  int template_size = 7,
                                                  int search_size = 21);

    // New feature matching methods
    static std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> matchFeatures(
        const std::vector<double>& image1,
        const std::vector<double>& image2,
        int width1,
        int height1,
        int width2,
        int height2,
        const std::string& method = "sift");
        
    static std::vector<double> computeOpticalFlow(const std::vector<double>& image1,
                                                const std::vector<double>& image2,
                                                int width,
                                                int height,
                                                int window_size = 15);

    // New advanced filtering methods
    static std::vector<double> anisotropicDiffusion(const std::vector<double>& image,
                                                  int width,
                                                  int height,
                                                  int iterations = 10,
                                                  double kappa = 30.0,
                                                  double lambda = 0.25);
                                                  
    static std::vector<double> guidedFilter(const std::vector<double>& image,
                                          const std::vector<double>& guide,
                                          int width,
                                          int height,
                                          int radius = 4,
                                          double epsilon = 0.01);
                                          
    static std::vector<double> rollingGuidanceFilter(const std::vector<double>& image,
                                                   int width,
                                                   int height,
                                                   int iterations = 4,
                                                   double sigma_s = 3.0,
                                                   double sigma_r = 0.1);

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

    // New helper methods for segmentation
    static std::vector<double> computeLocalThreshold(const std::vector<double>& image,
                                                   int width,
                                                   int height,
                                                   int window_size);
                                                   
    static std::vector<int> findPeaks(const std::vector<int>& histogram,
                                    int min_distance);
                                    
    static std::vector<double> computeDistanceTransform(const std::vector<double>& image,
                                                      int width,
                                                      int height);

    // New helper methods for texture analysis
    static std::vector<double> computeGLCMFeatures(const std::vector<std::vector<int>>& glcm);
    static double computeHaralickContrast(const std::vector<std::vector<int>>& glcm);
    static double computeHaralickEnergy(const std::vector<std::vector<int>>& glcm);
    static double computeHaralickCorrelation(const std::vector<std::vector<int>>& glcm);
    
    // New helper methods for feature detection
    static std::vector<double> computeHarrisResponse(const std::vector<double>& image,
                                                   int width,
                                                   int height);
    static std::vector<double> computeLaplacianOfGaussian(const std::vector<double>& image,
                                                        int width,
                                                        int height,
                                                        double sigma);
                                                        
    // New helper methods for image registration
    static std::vector<double> computeGradient(const std::vector<double>& image,
                                             int width,
                                             int height);
    static double computeMutualInformation(const std::vector<double>& source,
                                         const std::vector<double>& target,
                                         int width,
                                         int height);

    // New helper methods for image enhancement
    static std::vector<double> computeGaussianPyramid(const std::vector<double>& image,
                                                    int width,
                                                    int height,
                                                    int levels);
                                                    
    static std::vector<double> computeLaplacianPyramid(const std::vector<double>& image,
                                                     int width,
                                                     int height,
                                                     int levels);
                                                     
    static std::vector<double> blendPyramids(const std::vector<double>& pyramid1,
                                           const std::vector<double>& pyramid2,
                                           int width,
                                           int height,
                                           int levels);
    
    // New helper methods for feature matching
    static std::vector<double> computeSIFTFeatures(const std::vector<double>& image,
                                                 int width,
                                                 int height);
                                                 
    static std::vector<double> computeORBFeatures(const std::vector<double>& image,
                                                int width,
                                                int height);
                                                
    static double computeFeatureDistance(const std::vector<double>& feature1,
                                       const std::vector<double>& feature2);
    
    // New helper methods for advanced filtering
    static std::vector<double> computeStructureTensor(const std::vector<double>& image,
                                                    int width,
                                                    int height,
                                                    double sigma);
                                                    
    static std::vector<double> computeEigenvalues(const std::vector<double>& tensor,
                                                int width,
                                                int height);
                                                
    static std::vector<double> computeDiffusionTensor(const std::vector<double>& eigenvalues,
                                                    int width,
                                                    int height,
                                                    double kappa);
};

// Validation utilities
bool validateImageDimensions(const std::vector<double>& image, int expected_width, int expected_height);
bool validatePixelRange(const std::vector<double>& image, double min = 0.0, double max = 1.0);

} // namespace preprocessing

#endif // PREPROCESSING_H 