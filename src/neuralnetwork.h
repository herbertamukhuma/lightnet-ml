#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

#include "layer.h"

namespace LightNet {

class NeuralNetwork
{

public:
    NeuralNetwork(std::vector<size_t> architecture);

    std::vector<Layer> getLayers() const;

    std::vector<size_t> getArchitecture() const;

private:
    std::vector<Layer> layers;
    std::vector<size_t> architecture;

    void buildNetwork();
};

}

#endif // NEURALNETWORK_H
