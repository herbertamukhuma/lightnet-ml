#include "layer.h"

namespace LightNet {

Layer::Layer(size_t numberOfNeurons, size_t numberOfActivationsPerNeuron)
{
    init(numberOfNeurons, numberOfActivationsPerNeuron);
}

Layer::Layer()
{

}

void Layer::print()
{
    std::cout << "### LAYER ###" << std::endl;

    int n = 1;

    for(Neuron& neuron: neurons){

        std::cout << "Neuron #" << n << ":" << std::endl
                  << "Input Weights:" << std::endl;

        std::vector<double> weights = neuron.getWeights();

        for(double weight : weights){
            std::cout << weight << std::endl;
        }

        n++;
    }
}

bool Layer::setActivations(std::vector<double> activations, size_t positionInNet)
{
    if(positionInNet == 0){

        // this is the input layer

        if(activations.size() != neurons.size()){
            std::cerr << " -- the number of activations in the input layer must be equal to the number of neurons" <<
                         " inside this layer." << std::endl <<
                         " Activations count: " << activations.size() << std::endl <<
                         " Neurons count: " << neurons.size() << std::endl;
            return  false;
        }

        size_t counter = 0;

        for(Neuron &neuron : neurons){
            neuron.clearActivations();
            neuron.addActivation(activations[counter]);
            counter++;
        }

    }else {

        // this is either a hidden layer or the output layer

        for(Neuron &neuron : neurons){

            if(neuron.getWeights().size() != activations.size()){
                std::cerr << " -- the number of weights per neuron in Layer(#" << positionInNet << ") must be equal to" <<
                             " the number of activations." << std::endl <<
                             " Weights per neuron: " << neuron.getWeights().size() << std::endl <<
                             " Activations count: " << activations.size() << std::endl;
                return false;
            }

            neuron.clearActivations();

            for(double activation : activations){
                neuron.addActivation(activation);
            }
        }
    }

    return true;
}

std::vector<double> Layer::compute()
{
    std::vector<double> computedNeuronOutputs;

    for(Neuron &neuron : neurons){
        computedNeuronOutputs.push_back(neuron.compute());
    }

    return  computedNeuronOutputs;
}

void Layer::init(size_t numberOfNeurons, size_t numberOfActivationsPerNeuron)
{
    for(size_t i = 0; i < numberOfNeurons; i++)
    {
        Neuron neuron;

        for(size_t i2 = 0; i2 < numberOfActivationsPerNeuron; i2++){
            neuron.addWeight(neuron.generateRandomWeight());
        }

        neuron.setBias(neuron.generateRandomWeight());

        neurons.push_back(neuron);
    }
}

std::vector<LightNet::Neuron> Layer::getNeurons() const
{
    return neurons;
}

void Layer::addNeuron(Neuron neuron)
{
    neurons.push_back(neuron);
}

void Layer::updateWeight(size_t neuronIndex, size_t weightIndex, double newWeight)
{
    neurons.at(neuronIndex).updateWeight(weightIndex, newWeight);
}

void Layer::updateBias(size_t neuronIndex, double newBias)
{
    neurons[neuronIndex].setBias(newBias);
}

// end of namespace
}
