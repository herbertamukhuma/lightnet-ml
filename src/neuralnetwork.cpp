#include "neuralnetwork.h"

namespace LightNet {

NeuralNetwork::NeuralNetwork(std::vector<size_t> architecture) : architecture(architecture)
{
    buildNetwork();
}

std::vector<Layer> NeuralNetwork::getLayers() const
{
    return layers;
}

std::vector<size_t> NeuralNetwork::getArchitecture() const
{
    return architecture;
}

void NeuralNetwork::buildNetwork()
{
    size_t counter = 0;

    for(size_t &numberOfNeurons : architecture){

        if(counter == 0){
            // in the first layer, each neuron has only one input
            Layer layer(numberOfNeurons,1);
            layers.push_back(layer);
        }else {
            // in subsequent layers, the number of inputs for each
            // neuron equals the number of neurons in prevoius layer
            Layer layer(numberOfNeurons, architecture[counter-1]);
            layers.push_back(layer);
        }

        counter++;
    }
}

// end of namespace
}
