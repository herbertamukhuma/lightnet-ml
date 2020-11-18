#include "neuralnetwork.h"

namespace LightNet {

NeuralNetwork::NeuralNetwork(std::vector<size_t> architecture, Dataset dataset) :
    architecture(architecture),
    dataset(dataset)
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

void NeuralNetwork::train(size_t iterations)
{
    for(size_t i = 0; i < iterations; i++){

        std::vector<double> losses;

        for(size_t x = 0; x < dataset.getRowCount(); x++){

            feedForward(dataset.getInputs(x));
            losses.push_back(computeLoss(dataset.getTarget(x)));
            backPropagation(dataset.getTarget(x));
        }

        std::cout << " -- loss #" << i + 1 << ": " << MathUtil::mean(losses) << std::endl;
    }

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

void NeuralNetwork::feedForward(std::vector<double> activations)
{

    size_t counter = 0;

    for(Layer &layer : layers){

        layer.setActivations(activations, counter);

        // compute the outputs for this layer, which will be the activations for the next layer
        if(counter == layers.size() - 1){
            // for the last layer, we use sigmoid as the activation funtion to obtain activations from 0 to 1
            activations = layer.compute(MathUtil::Sigmoid);
        }else {
            activations = layer.compute(MathUtil::ReLU);
        }

        counter++;
    }
}

double NeuralNetwork::computeLoss(double target)
{
    int targetNeuronIndex = getTargetNeuronIndex(target);

    if(targetNeuronIndex == -1){
        std::cerr << " -- could not find the target neuron for the specified target: " << target << std::endl;
        return 0.0;
    }

    Layer outputLayer = layers[layers.size() - 1];

    size_t counter = 0;

    std::vector<std::tuple<double, double>> activationPairs;

    for(Neuron &neuron : outputLayer.getNeurons()){

        double expectedActivation;
        double actualActivation = neuron.getOutput();

        if((size_t)targetNeuronIndex == counter){
            expectedActivation = 1.0;
        }else {
            expectedActivation = 0.0;
        }

        activationPairs.push_back(std::make_tuple(expectedActivation, actualActivation));

        counter++;
    }

    return MathUtil::mse(activationPairs);
}

void NeuralNetwork::backPropagation(double target)
{
    int targetNeuronIndex = getTargetNeuronIndex(target);

    if(targetNeuronIndex == -1){
        std::cerr << " -- could not find the target neuron for the specified target: " << target << std::endl;
        return;
    }

    Layer &outputLayer = layers[layers.size() - 1];
    double outputNeuronCount = outputLayer.getNeurons().size();

    std::vector<double> costFuncDerivs;

    for(size_t i = 0; i < (size_t)outputNeuronCount; i++){

        double costFuncDeriv;

        if((size_t)targetNeuronIndex == i){
            costFuncDeriv = (-2 / outputNeuronCount) * (1.0 - outputLayer.getNeurons()[i].getOutput());
        }else {
            costFuncDeriv = (-2 / outputNeuronCount) * (0.0 - outputLayer.getNeurons()[i].getOutput());
        }

        costFuncDerivs.push_back(costFuncDeriv);
    }

    std::vector<std::vector<double>> prevNeuronsCumulativeWeightDerivs;

    for(size_t l = layers.size(); l > 0; l--){

        Layer &layer = layers[l-1];

        if(l == layers.size()){

            // this is the last/output layer

            std::vector<Neuron> neurons = layer.getNeurons();

            size_t neuronCounter = 0;

            std::vector<std::vector<double>> currentNeuronsCumulativeWeightDerivs(neurons.size());

            for(Neuron neuron : neurons){

                // update weight
                std::vector<double> weights = neuron.getWeights();
                std::vector<double> activations = neuron.getActivations();

                size_t weightCounter = 0;

                for(double weight : weights){

                    double weightDeriv = costFuncDerivs[neuronCounter] * MathUtil::sigmoidDeriv(neuron.getOutput()) * activations[weightCounter];

                    double newWeight = weight - (LEARNING_RATE * weightDeriv);

                    layer.updateWeight(neuronCounter, weightCounter, newWeight);

                    double cumulativeDeriv = costFuncDerivs[neuronCounter] * MathUtil::sigmoidDeriv(neuron.getOutput()) * weight;
                    currentNeuronsCumulativeWeightDerivs[neuronCounter].push_back(cumulativeDeriv);

                    weightCounter++;
                }

                neuronCounter++;
            }

            prevNeuronsCumulativeWeightDerivs = currentNeuronsCumulativeWeightDerivs;

        }else {

            // this is either a hidden layer or the input layer

            std::vector<Neuron> neurons = layer.getNeurons();

            size_t neuronCounter = 0;

            std::vector<std::vector<double>> currentNeuronsCumulativeWeightDerivs(neurons.size());

            for(Neuron &neuron : neurons){

                std::vector<double> weights = neuron.getWeights();
                std::vector<double> activations = neuron.getActivations();

                size_t weightCounter = 0;

                for(double weight : weights){

                    double cumulativeDeriv = 0.0;

                    for(std::vector<double> cumulativeWeightDerivs : prevNeuronsCumulativeWeightDerivs){
                        cumulativeDeriv += cumulativeWeightDerivs[neuronCounter];
                    }

                    double weightDeriv = cumulativeDeriv * MathUtil::reluDeriv(neuron.getOutput()) * activations[weightCounter];

                    cumulativeDeriv = cumulativeDeriv * MathUtil::reluDeriv(neuron.getOutput()) * weight;
                    currentNeuronsCumulativeWeightDerivs[neuronCounter].push_back(cumulativeDeriv);

                    double newWeight = weight - (LEARNING_RATE * weightDeriv);
                    layer.updateWeight(neuronCounter, weightCounter, newWeight);

                    weightCounter++;
                }

                neuronCounter++;
            }

            prevNeuronsCumulativeWeightDerivs = currentNeuronsCumulativeWeightDerivs;

        }

        // end of layers for loop
    }
}

int NeuralNetwork::getTargetNeuronIndex(double target)
{
    std::vector<double> targets = dataset.getUniqueTargets();

    // get the neuron index representing 'target'
    return Util::find(targets, target);
}

// end of namespace
}
