#include <iostream>

#include "src/neuralnetwork.h"
#include "src/dataset.h"
#include "src/mathutil.h"

using namespace std;
using namespace LightNet;

int main()
{

    Dataset dataset("/Users/user/GitHub/lightnet-ml/data/iris_flowers.csv", true);
    dataset.scale();

    Dataset testData = dataset.splitTestData(5);

//    NeuralNetwork net({dataset.getInputCount(), 4, dataset.getUniqueTargetCount()}, dataset);
//    net.train(500);

    NeuralNetwork net = NeuralNetwork::loadModel("/Users/user/GitHub/lightnet-ml/data/model.json");

    std::vector<NeuralNetwork::Prediction> predictions = net.predict(testData);

    for(NeuralNetwork::Prediction prediction : predictions){
        cout << "Predicted: " << prediction.predictedEncodedTarget << " Actual: " << prediction.actualEncodedTarget << " Conf: " << prediction.confidence << endl;
    }

//    if(net.save("/Users/user/GitHub/lightnet-ml/data/model.json")){
//        std::cout << "saved!" << std::endl;
//    }else {
//        std::cout << "failed to save!" << std::endl;
//    }

    return 0;
}
