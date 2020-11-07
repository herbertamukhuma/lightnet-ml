#include "layer.h"

namespace LightNet {

Layer::Layer(size_t numberOfNeurons, size_t numberOfInputsPerNeuron)
{
    init(numberOfNeurons, numberOfInputsPerNeuron);
}

void Layer::print()
{
    std::cout << "### LAYER ###" << std::endl;

    int n = 1;

    for(Neuron& neuron: neurons){

        std::cout << "Neuron #" << n << ":" << std::endl
                  << "Input Weights:" << std::endl;

        std::vector<double> weights = neuron.getInputWeights();

        for(double weight : weights){
            std::cout << weight << std::endl;
        }

        n++;
    }
}

void Layer::init(size_t numberOfNeurons, size_t numberOfInputsPerNeuron)
{
    for(size_t i = 0; i < numberOfNeurons; i++)
    {
        Neuron neuron;

        for(size_t i2 = 0; i2 < numberOfInputsPerNeuron; i2++){
            neuron.addInputWeight(neuron.generateRandomWeight());
        }

        neurons.push_back(neuron);
    }
}

// end of namespace
}
