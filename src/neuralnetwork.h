#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

#include "layer.h"
#include "dataset.h"
#include "util.h"

namespace LightNet {

class NeuralNetwork
{

public:
    NeuralNetwork(std::vector<size_t> architecture, Dataset dataset);

    struct Prediction{
        double predictedEncodedTarget;
        std::string predictedUnencodedTarget;

        double actualEncodedTarget;
        std::string actualUnencodedTarget;

        double confidence;
    };

    std::vector<Layer> getLayers() const;

    std::vector<size_t> getArchitecture() const;

    void train(size_t epochs = 5, double learningRate = 0.1);

    void printOutputs();

    std::vector<Prediction> predict(Dataset predictionDataset);

    bool save(std::string filename);

private:
    std::vector<Layer> layers;
    std::vector<size_t> architecture;

    Dataset dataset;

    void buildNetwork();

    void feedForward(std::vector<double> activations);

    double computeLoss(double target);

    void backPropagation(double target, double learningRate);

    int getTargetNeuronIndex(double target);

};

}

#endif // NEURALNETWORK_H
