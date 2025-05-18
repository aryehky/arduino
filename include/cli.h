#ifndef CLI_H
#define CLI_H

#include <string>
#include <vector>
#include <map>
#include <functional>

class CLI {
public:
    CLI(int argc, char** argv);
    
    bool hasFlag(const std::string& flag) const;
    std::string getOption(const std::string& option) const;
    void run();

private:
    std::map<std::string, std::string> options;
    std::vector<std::string> flags;

    void handleTraining();
    void handlePrediction();
    void handleBatchPrediction();
    void handleEvaluation();
    void handleVisualization();
    
    void showHelp() const;
    void validateOptions() const;
};

#endif // CLI_H 