#ifndef COLOR_SPACES_H
#define COLOR_SPACES_H

#include <vector>
#include <cmath>
#include <stdexcept>

namespace preprocessing {

class ColorSpaces {
public:
    // RGB to other color spaces
    static std::vector<double> rgbToLab(const std::vector<double>& rgb_image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> rgbToYUV(const std::vector<double>& rgb_image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> rgbToHSV(const std::vector<double>& rgb_image,
                                      int width,
                                      int height);
    
    // Other color spaces to RGB
    static std::vector<double> labToRGB(const std::vector<double>& lab_image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> yuvToRGB(const std::vector<double>& yuv_image,
                                      int width,
                                      int height);
                                      
    static std::vector<double> hsvToRGB(const std::vector<double>& hsv_image,
                                      int width,
                                      int height);
    
    // Color manipulation
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
                                              
    static std::vector<double> adjustContrast(const std::vector<double>& image,
                                            int width,
                                            int height,
                                            double factor);
    
    // Color quantization
    static std::vector<double> quantizeColors(const std::vector<double>& image,
                                            int width,
                                            int height,
                                            int num_colors);
                                            
    // Color correction
    static std::vector<double> whiteBalance(const std::vector<double>& image,
                                          int width,
                                          int height);
                                          
    static std::vector<double> colorCorrection(const std::vector<double>& image,
                                             int width,
                                             int height,
                                             const std::vector<double>& color_matrix);
                                             
    // Color analysis
    static std::vector<double> computeColorHistogram(const std::vector<double>& image,
                                                   int width,
                                                   int height,
                                                   int num_bins = 256);
                                                   
    static std::vector<double> computeDominantColors(const std::vector<double>& image,
                                                   int width,
                                                   int height,
                                                   int num_colors = 5);
    
private:
    static bool validateImageDimensions(const std::vector<double>& image,
                                      int expected_width,
                                      int expected_height);
                                      
    static std::vector<double> computeColorMeans(const std::vector<double>& image,
                                               int width,
                                               int height);
                                               
    static std::vector<double> computeColorCovariance(const std::vector<double>& image,
                                                    int width,
                                                    int height,
                                                    const std::vector<double>& means);
};

} // namespace preprocessing

#endif // COLOR_SPACES_H 