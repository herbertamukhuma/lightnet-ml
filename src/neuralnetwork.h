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

    std::vector<Layer> getLayers() const;

    std::vector<size_t> getArchitecture() const;

    void train(size_t iterations = 5);

private:
    const double LEARNING_RATE = 0.5;

    std::vector<Layer> layers;
    std::vector<size_t> architecture;

    Dataset dataset;

    void buildNetwork();

    void feedForward(std::vector<double> activations);

    double computeLoss(double target);

    void backPropagation(double target);

    int getTargetNeuronIndex(double target);

};

}

#endif // NEURALNETWORK_H
