#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>

#include "neuron.h"
#include "mathutil.h"

namespace LightNet {

class Layer
{
public:

    Layer(size_t numberOfNeurons, size_t numberOfActivationsPerNeuron, Neuron::ActivationFunction activationFunction);

    Layer();

    void print();

    bool setActivations(std::vector<double> activations, size_t positionInNet);

    std::vector<double> compute();

    std::vector<LightNet::Neuron> getNeurons() const;

    void addNeuron(Neuron neuron);

    void updateWeight(size_t neuronIndex, size_t weightIndex, double newWeight);

    void updateBias(size_t neuronIndex, double newBias);

protected:
    void init(size_t numberOfNeurons, size_t numberOfActivationsPerNeuron, Neuron::ActivationFunction activationFunction);

    std::vector<LightNet::Neuron> neurons;
};

}

#endif // LAYER_H
