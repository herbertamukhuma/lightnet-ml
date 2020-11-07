#include "neuron.h"

namespace LightNet {

Neuron::Neuron()
{

}

double Neuron::generateRandomWeight()
{
    return ((double)rand()) / RAND_MAX;
}

void Neuron::addInputWeight(double weight)
{
    inputWeights.push_back(weight);
}

std::vector<double> Neuron::getInputWeights() const
{
    return inputWeights;
}

// end of namespace
}
