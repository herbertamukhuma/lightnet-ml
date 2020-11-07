#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>

#include "neuron.h"

namespace LightNet {

class Layer
{
public:
    Layer(size_t numberOfNeurons, size_t numberOfInputsPerNeuron);

    void print();

protected:
    void init(size_t numberOfNeurons, size_t numberOfInputsPerNeuron);

    std::vector<LightNet::Neuron> neurons;
};

}

#endif // LAYER_H
