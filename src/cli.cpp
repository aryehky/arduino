#include "cli.h"
#include "svm.h"
#include "dataset.h"
#include "utils.h"
#include "preprocessing.h"
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <opencv2/opencv.hpp>

CLI::CLI(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                options[arg.substr(2)] = argv[++i];
            } else {
                flags.push_back(arg.substr(2));
            }
        }
    }
}

bool CLI::hasFlag(const std::string& flag) const {
    return std::find(flags.begin(), flags.end(), flag) != flags.end();
}

std::string CLI::getOption(const std::string& option) const {
    auto it = options.find(option);
    return it != options.end() ? it->second : "";
}

void CLI::validateOptions() const {
    if (flags.empty() && options.empty()) {
        showHelp();
        throw std::runtime_error("No options provided");
    }

    if (hasFlag("train") && hasFlag("predict")) {
        throw std::runtime_error("Cannot train and predict simultaneously");
    }

    if (hasFlag("preprocess") && hasFlag("preprocess-batch")) {
        throw std::runtime_error("Cannot use single and batch preprocessing simultaneously");
    }
}

void CLI::showHelp() const {
    std::cout << "Usage: digit_recognition [options]\n\n"
              << "Options:\n"
              << "  --train              Train a new model\n"
              << "  --grid-search        Perform grid search during training\n"
              << "  --augment            Use data augmentation during training\n"
              << "  --predict            Run in prediction mode\n"
              << "  --predict-batch DIR  Run batch prediction on directory\n"
              << "  --evaluate           Evaluate model performance\n"
              << "  --visualize          Generate visualizations\n"
              << "  --model PATH         Specify model path\n"
              << "\nPreprocessing Options:\n"
              << "  --preprocess FILE    Preprocess a single image\n"
              << "  --preprocess-batch DIR Preprocess all images in directory\n"
              << "  --output PATH        Output path for processed image(s)\n"
              << "  --normalize          Apply normalization\n"
              << "  --standardize        Apply standardization\n"
              << "  --remove-noise       Apply noise removal\n"
              << "  --adjust-contrast FACTOR Apply contrast adjustment\n"
              << "  --sharpen            Apply sharpening\n"
              << "  --binarize THRESHOLD Apply binarization\n"
              << "  --rotate ANGLE       Rotate image by specified angle (-180 to 180 degrees)\n"
              << "  --scale FACTOR       Scale image by specified factor (0 to 2)\n"
              << "  --help               Show this help message\n";
}

void CLI::handleTraining() {
    std::cout << "Loading dataset...\n";
    Dataset dataset("data/mnist/train-images.idx3-ubyte",
                   "data/mnist/train-labels.idx1-ubyte",
                   "data/mnist/t10k-images.idx3-ubyte",
                   "data/mnist/t10k-labels.idx1-ubyte");
    
    dataset.loadTrainingData();
    dataset.loadTestData();

    auto& training_features = dataset.getTrainingFeatures();
    auto& training_labels = dataset.getTrainingLabels();
    
    SVM svm;

    if (hasFlag("grid-search")) {
        std::cout << "Performing grid search...\n";
        utils::ParamGrid param_grid;
        auto best_params = svm.gridSearch(training_features, training_labels, param_grid);
        
        std::cout << "Best parameters:\n"
                  << "Kernel: " << best_params.best_kernel << "\n"
                  << "C: " << best_params.best_C << "\n"
                  << "Gamma: " << best_params.best_gamma << "\n"
                  << "Validation accuracy: " << best_params.best_accuracy * 100 << "%\n";
    }

    if (hasFlag("augment")) {
        std::cout << "Augmenting training data...\n";
        svm.augmentTrainingData(training_features, training_labels);
    }

    std::cout << "Training model...\n";
    svm.train(training_features, training_labels);

    std::string model_path = getOption("model");
    if (model_path.empty()) {
        model_path = "digit_recognition_model.model";
    }
    
    std::cout << "Saving model to " << model_path << "\n";
    svm.saveModel(model_path);
}

void CLI::handlePrediction() {
    std::string model_path = getOption("model");
    if (model_path.empty()) {
        model_path = "digit_recognition_model.model";
    }

    SVM svm;
    std::cout << "Loading model from " << model_path << "\n";
    svm.loadModel(model_path);

    std::cout << "Enter digit image data (28x28 pixels, space-separated values):\n";
    std::vector<double> features;
    double pixel;
    while (features.size() < 784 && std::cin >> pixel) {
        features.push_back(pixel / 255.0);
    }

    if (features.size() == 784) {
        int prediction = svm.predict(features);
        std::cout << "Predicted digit: " << prediction << "\n";
    } else {
        throw std::runtime_error("Invalid input size");
    }
}

void CLI::handleBatchPrediction() {
    std::string dir_path = getOption("predict-batch");
    if (dir_path.empty()) {
        throw std::runtime_error("No directory specified for batch prediction");
    }

    std::string model_path = getOption("model");
    if (model_path.empty()) {
        model_path = "digit_recognition_model.model";
    }

    SVM svm;
    std::cout << "Loading model from " << model_path << "\n";
    svm.loadModel(model_path);

    // Implementation for batch prediction would go here
    // This would involve reading all images from the specified directory
    std::cout << "Batch prediction not yet implemented\n";
}

void CLI::handleEvaluation() {
    std::string model_path = getOption("model");
    if (model_path.empty()) {
        model_path = "digit_recognition_model.model";
    }

    Dataset dataset("data/mnist/train-images.idx3-ubyte",
                   "data/mnist/train-labels.idx1-ubyte",
                   "data/mnist/t10k-images.idx3-ubyte",
                   "data/mnist/t10k-labels.idx1-ubyte");
    
    dataset.loadTestData();
    auto& test_features = dataset.getTestFeatures();
    auto& test_labels = dataset.getTestLabels();

    SVM svm;
    svm.loadModel(model_path);

    auto confusion_matrix = svm.getConfusionMatrix(test_features, test_labels, 10);
    
    std::cout << "\nEvaluation Results:\n"
              << "Accuracy: " << confusion_matrix.getAccuracy() * 100 << "%\n"
              << "F1 Score: " << confusion_matrix.getF1Score() * 100 << "%\n\n"
              << confusion_matrix.toString() << "\n";
}

void CLI::handleVisualization() {
    // This would be implemented if we add visualization features
    std::cout << "Visualization features not yet implemented\n";
}

void CLI::validateRotationOptions() const {
    if (hasFlag("rotate")) {
        try {
            double angle = std::stod(getOption("rotate"));
            if (angle < -180.0 || angle > 180.0) {
                throw std::runtime_error("Rotation angle must be between -180 and 180 degrees");
            }
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid rotation angle");
        }
    }
}

void CLI::validateScalingOptions() const {
    if (hasFlag("scale")) {
        try {
            double factor = std::stod(getOption("scale"));
            if (factor <= 0.0 || factor > 2.0) {
                throw std::runtime_error("Scale factor must be between 0 and 2");
            }
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid scale factor");
        }
    }
}

void CLI::validatePreprocessingOptions() const {
    if (hasFlag("preprocess")) {
        if (getOption("preprocess").empty()) {
            throw std::runtime_error("No input file specified for preprocessing");
        }
        if (getOption("output").empty()) {
            throw std::runtime_error("No output path specified");
        }
    }
    
    if (hasFlag("preprocess-batch")) {
        if (getOption("preprocess-batch").empty()) {
            throw std::runtime_error("No input directory specified for batch preprocessing");
        }
        if (getOption("output").empty()) {
            throw std::runtime_error("No output directory specified");
        }
    }
    
    if (hasFlag("adjust-contrast")) {
        try {
            double factor = std::stod(getOption("adjust-contrast"));
            if (factor <= 0.0) {
                throw std::runtime_error("Contrast factor must be positive");
            }
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid contrast adjustment factor");
        }
    }
    
    if (hasFlag("binarize")) {
        try {
            double threshold = std::stod(getOption("binarize"));
            if (threshold < 0.0 || threshold > 1.0) {
                throw std::runtime_error("Binarization threshold must be between 0 and 1");
            }
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid binarization threshold");
        }
    }
    
    validateRotationOptions();
    validateScalingOptions();
}

void CLI::applyPreprocessingPipeline(const std::string& input_path,
                                   const std::string& output_path,
                                   int width,
                                   int height) const {
    // Load image using OpenCV
    cv::Mat image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + input_path);
    }
    
    // Convert to our format
    std::vector<double> img_data;
    img_data.reserve(width * height);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            img_data.push_back(image.at<uchar>(i, j) / 255.0);
        }
    }
    
    preprocessing::ImagePreprocessor preprocessor;
    
    // Apply preprocessing steps based on flags
    if (hasFlag("normalize")) {
        img_data = preprocessor.normalize(img_data);
    }
    if (hasFlag("standardize")) {
        img_data = preprocessor.standardize(img_data);
    }
    if (hasFlag("remove-noise")) {
        img_data = preprocessor.removeNoise(img_data, width, height);
    }
    if (hasFlag("adjust-contrast")) {
        double factor = std::stod(getOption("adjust-contrast"));
        img_data = preprocessor.adjustContrast(img_data, factor);
    }
    if (hasFlag("sharpen")) {
        img_data = preprocessor.sharpen(img_data, width, height);
    }
    if (hasFlag("binarize")) {
        double threshold = std::stod(getOption("binarize"));
        img_data = preprocessor.binarize(img_data, threshold);
    }
    
    // Apply new transformation options
    if (hasFlag("rotate") || hasFlag("scale")) {
        double angle = hasFlag("rotate") ? std::stod(getOption("rotate")) : 0.0;
        double scale = hasFlag("scale") ? std::stod(getOption("scale")) : 1.0;
        
        if (hasFlag("rotate") && hasFlag("scale")) {
            img_data = preprocessor.rotateAndScale(img_data, width, height, angle, scale);
        } else if (hasFlag("rotate")) {
            img_data = preprocessor.rotateImage(img_data, width, height, angle);
        } else {
            img_data = preprocessor.scaleImage(img_data, width, height, scale);
        }
    }
    
    // Convert back to OpenCV format
    cv::Mat processed(height, width, CV_8UC1);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            processed.at<uchar>(i, j) = static_cast<uchar>(img_data[i * width + j] * 255);
        }
    }
    
    // Save processed image
    if (!cv::imwrite(output_path, processed)) {
        throw std::runtime_error("Failed to save processed image: " + output_path);
    }
}

void CLI::handlePreprocessing() {
    validatePreprocessingOptions();
    
    std::string input_path = getOption("preprocess");
    std::string output_path = getOption("output");
    
    // Load image to get dimensions
    cv::Mat image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + input_path);
    }
    
    std::cout << "Processing image: " << input_path << "\n";
    applyPreprocessingPipeline(input_path, output_path, image.cols, image.rows);
    std::cout << "Saved processed image to: " << output_path << "\n";
}

void CLI::handleBatchPreprocessing() {
    validatePreprocessingOptions();
    
    std::string input_dir = getOption("preprocess-batch");
    std::string output_dir = getOption("output");
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    int processed_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string input_path = entry.path().string();
            std::string filename = entry.path().filename().string();
            std::string output_path = (std::filesystem::path(output_dir) / filename).string();
            
            try {
                cv::Mat image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
                if (!image.empty()) {
                    std::cout << "Processing: " << filename << "\n";
                    applyPreprocessingPipeline(input_path, output_path, image.cols, image.rows);
                    processed_count++;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to process " << filename << ": " << e.what() << "\n";
            }
        }
    }
    
    std::cout << "Processed " << processed_count << " images\n"
              << "Output directory: " << output_dir << "\n";
}

void CLI::run() {
    try {
        validateOptions();

        if (hasFlag("help")) {
            showHelp();
            return;
        }

        if (hasFlag("train")) {
            handleTraining();
        } else if (hasFlag("predict")) {
            handlePrediction();
        } else if (hasFlag("predict-batch")) {
            handleBatchPrediction();
        } else if (hasFlag("evaluate")) {
            handleEvaluation();
        } else if (hasFlag("visualize")) {
            handleVisualization();
        } else if (hasFlag("preprocess")) {
            handlePreprocessing();
        } else if (hasFlag("preprocess-batch")) {
            handleBatchPreprocessing();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        throw;
    }
} 