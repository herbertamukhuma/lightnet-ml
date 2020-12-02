#include "neuron.h"

namespace LightNet {

Neuron::Neuron()
{

}

double Neuron::generateRandomWeight()
{
    return ((double)rand()) / RAND_MAX;
}

void Neuron::addWeight(double weight)
{
    weights.push_back(weight);
}

void Neuron::addActivation(double activation)
{
    inputActivations.push_back(activation);
}

std::vector<double> Neuron::getWeights() const
{
    return weights;
}

std::vector<double> Neuron::getActivations() const
{
    return inputActivations;
}

void Neuron::clearActivations()
{
    inputActivations.clear();
}

double Neuron::compute()
{
    std::vector<double> activationWeightProducts;

    size_t counter = 0;

    for(double weight : weights){
        double activationWeightProduct = weight * inputActivations[counter];
        activationWeightProducts.push_back(activationWeightProduct);
        counter++;
    }

    double sum = MathUtil::sum(activationWeightProducts); // + bias;

    output = MathUtil::sigmoid(sum);

    return output;

}

double Neuron::getOutput() const
{
    return output;
}

void Neuron::updateWeight(size_t index, double newWeight)
{
    weights[index] = newWeight;
}

void Neuron::setBias(double value)
{
    bias = value;
}

double Neuron::getBias() const
{
    return bias;
}

void Neuron::setWeights(const std::vector<double> value)
{
    weights = value;
}

// end of namespace
}
