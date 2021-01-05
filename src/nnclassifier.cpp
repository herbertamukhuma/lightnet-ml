#include "nnclassifier.h"

namespace LightNet {

NNClassifier::NNClassifier(std::vector<size_t> architecture, Dataset dataset) :
    architecture(architecture),
    dataset(dataset)
{
    buildNetwork();
}

NNClassifier::NNClassifier()
{
    dataset = Dataset();
}

std::vector<Layer> NNClassifier::getLayers() const
{
    return layers;
}

std::vector<size_t> NNClassifier::getArchitecture() const
{
    return architecture;
}

void NNClassifier::train(size_t epochs, double learningRate)
{
    for(size_t i = 0; i < epochs; i++){

        std::vector<double> losses;

        for(size_t x = 0; x < dataset.getRowCount(); x++){
            feedForward(dataset.getInputs(x));
            losses.push_back(computeLoss(dataset.getTarget(x)));
            backPropagation(dataset.getTarget(x), learningRate);
        }

        std::cout << " -- loss #" << i + 1 << ": " << MathUtil::mean(losses) << std::endl;
    }

}

void NNClassifier::printOutputs()
{
    Layer &outputLayer = layers[layers.size() - 1];

    std::cout << " -- Outputs: ";

    for(Neuron &neuron : outputLayer.getNeurons()){
        std::cout << neuron.getOutput() << " ";
    }

    std::cout << std::endl;

}

std::vector<NNClassifier::Prediction> NNClassifier::predict(Dataset predictionDataset)
{
    std::vector<Prediction> predictions;

    for(size_t x = 0; x < predictionDataset.getRowCount(); x++){

        feedForward(predictionDataset.getInputs(x));

        Layer outputLayer = layers[layers.size() - 1];

        std::vector<double> outputs;

        for(Neuron neuron : outputLayer.getNeurons()){
            outputs.push_back(neuron.getOutput());
        }

        double max = MathUtil::maxElement(outputs);
        int maxIndex = Util::findMax(outputs);

        Prediction prediction;
        prediction.predictedUnencodedTarget = dataset.getUniqueUnencodedTargets()[maxIndex];
        prediction.predictedEncodedTarget = dataset.getUniqueEncodedTargets()[maxIndex];
        prediction.actualUnencodedTarget = predictionDataset.getUnencodedTarget(x);
        prediction.actualEncodedTarget = predictionDataset.getTarget(x);
        prediction.confidence = max;

        predictions.push_back(prediction);
    }

    return predictions;
}

bool NNClassifier::save(std::string filename)
{

    QJsonObject model;

    // save weights
    QJsonArray nnWeights;

    for(Layer layer : layers){

        QJsonArray layerNeuronWeights;

        for(Neuron neuron : layer.getNeurons()){

            QJsonArray neuronWeights;

            for(double weight : neuron.getWeights()){
                neuronWeights.append(weight);
            }

            layerNeuronWeights.append(neuronWeights);
        }

        nnWeights.append(layerNeuronWeights);
    }

    model.insert("weights", nnWeights);

    // dataset metadata

    QJsonObject datasetMetadata;

    QJsonArray encodedTargets;

    for(double target : dataset.getUniqueEncodedTargets()){
        encodedTargets.append(target);
    }

    QJsonArray unencodedTargets;

    for(std::string target : dataset.getUniqueUnencodedTargets()){
        unencodedTargets.append(QString::fromStdString(target));
    }

    datasetMetadata.insert("encoded_targets", encodedTargets);
    datasetMetadata.insert("unencoded_targets", unencodedTargets);

    model.insert("dataset_metadata", datasetMetadata);

    // save to file
    QFile modelFile(QString::fromStdString(filename));

    if(!modelFile.open(QIODevice::WriteOnly | QIODevice::Truncate)){
        std::cerr << " -- unable to open file at: " << filename << std::endl;
        return false;
    }

    modelFile.write(QJsonDocument(model).toJson());

    modelFile.close();

    return true;
}

NNClassifier NNClassifier::loadModel(std::string filename)
{
    NNClassifier net;

    QFile modelFile(QString::fromStdString(filename));

    if(!modelFile.open(QIODevice::ReadOnly)){
        std::cerr << " -- unable to open file at: " << filename << std::endl;
        return net;
    }

    QJsonObject model = QJsonDocument::fromJson(modelFile.readAll()).object();

    modelFile.close();

    // populate weights
    QJsonArray nnWeights = model["weights"].toArray();

    for(QJsonValueRef layerNeuronWeightsRef : nnWeights){

        QJsonArray layerNeuronWeights = layerNeuronWeightsRef.toArray();

        Layer layer;

        for(QJsonValueRef neuronWeightsRef : layerNeuronWeights){

            QJsonArray neuronWeights = neuronWeightsRef.toArray();

            Neuron neuron(Neuron::Sigmoid);

            for(QJsonValueRef weightRef : neuronWeights){
                neuron.addWeight(weightRef.toDouble());
            }

            layer.addNeuron(neuron);
        }

        net.layers.push_back(layer);
    }

    // populate dataset
    QJsonObject datasetMetadata = model["dataset_metadata"].toObject();

    QJsonArray encodedTargets = datasetMetadata["encoded_targets"].toArray();

    std::vector<double> encodedTargetsVec;

    for(QJsonValueRef targetRef : encodedTargets){
        encodedTargetsVec.push_back(targetRef.toDouble());
    }

    net.dataset.setUniqueEncodedTargets(encodedTargetsVec);

    QJsonArray unencodedTargets = datasetMetadata["unencoded_targets"].toArray();

    std::vector<std::string> unencodedTargetsVec;

    for(QJsonValueRef targetRef : encodedTargets){
        unencodedTargetsVec.push_back(targetRef.toString().toStdString());
    }

    net.dataset.setUniqueUnencodedTargets(unencodedTargetsVec);

    return net;
}

void NNClassifier::buildNetwork()
{
    size_t counter = 0;

    for(size_t &numberOfNeurons : architecture){

        if(counter == 0){
            // in the first layer, each neuron has only one input
            Layer layer(numberOfNeurons, 1, Neuron::Sigmoid);
            layers.push_back(layer);
        }else {
            // in subsequent layers, the number of inputs for each
            // neuron equals the number of neurons in prevoius layer
            Layer layer(numberOfNeurons, architecture[counter-1], Neuron::Sigmoid);
            layers.push_back(layer);
        }

        counter++;
    }
}

void NNClassifier::feedForward(std::vector<double> activations)
{

    size_t counter = 0;

    for(Layer &layer : layers){

        layer.setActivations(activations, counter);

        // compute the outputs for this layer, which will be the activations for the next layer
        activations = layer.compute();

        counter++;
    }
}

double NNClassifier::computeLoss(double target)
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

void NNClassifier::backPropagation(double target, double learningRate)
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
            costFuncDeriv = (-2.0 / outputNeuronCount) * (1.0 - outputLayer.getNeurons()[i].getOutput());
        }else {
            costFuncDeriv = (-2.0 / outputNeuronCount) * (0.0 - outputLayer.getNeurons()[i].getOutput());
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

                    double newWeight = weight - (learningRate * weightDeriv);

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

                double cumulativeDeriv = 0.0;

                for(std::vector<double> cumulativeWeightDerivs : prevNeuronsCumulativeWeightDerivs){
                    cumulativeDeriv += cumulativeWeightDerivs[neuronCounter];
                }

                std::vector<double> weights = neuron.getWeights();
                std::vector<double> activations = neuron.getActivations();

                size_t weightCounter = 0;

                for(double weight : weights){

                    double weightDeriv = cumulativeDeriv * MathUtil::sigmoidDeriv(neuron.getOutput()) * activations[weightCounter];

                    double newWeight = weight - (learningRate * weightDeriv);
                    layer.updateWeight(neuronCounter, weightCounter, newWeight);

                    double nextCumulativeDeriv = cumulativeDeriv * MathUtil::sigmoidDeriv(neuron.getOutput()) * weight;
                    currentNeuronsCumulativeWeightDerivs[neuronCounter].push_back(nextCumulativeDeriv);

                    weightCounter++;
                }

                neuronCounter++;
            }

            prevNeuronsCumulativeWeightDerivs = currentNeuronsCumulativeWeightDerivs;

        }

        // end of layers for loop
    }
}

int NNClassifier::getTargetNeuronIndex(double target)
{
    std::vector<double> targets = dataset.getUniqueEncodedTargets();

    // get the neuron index representing 'target'
    return Util::find(targets, target);
}

// end of namespace
}
