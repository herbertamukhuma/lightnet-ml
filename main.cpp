#include <iostream>

#include "src/nnclassifier.h"
#include "src/dataset.h"
#include "src/mathutil.h"

using namespace std;
using namespace LightNet;

void trainAndSave();
void loadAndPredict();

int main()
{
    trainAndSave();
    return 0;
}

void trainAndSave(){

    Dataset dataset("D:/GitHub/lightnet-ml/data/iris_flowers.csv", true);
    dataset.scale();

    Dataset testData = dataset.splitTestData(5);

    NNClassifier net({dataset.getInputCount(), 10, dataset.getUniqueTargetCount()}, dataset);
    net.train(1000);

    std::vector<NNClassifier::Prediction> predictions = net.predict(testData);

    for(NNClassifier::Prediction prediction : predictions){
        cout << "Predicted: " << prediction.predictedEncodedTarget << " Actual: " << prediction.actualEncodedTarget << " Conf: " << prediction.confidence << endl;
    }

    if(net.save("D:/GitHub/lightnet-ml/data/model.json")){
        std::cout << "saved!" << std::endl;
    }else {
        std::cout << "failed to save!" << std::endl;
    }

}

void loadAndPredict(){

    Dataset dataset("D:/GitHub/lightnet-ml/data/iris_flowers.csv", true);
    dataset.scale();

    Dataset testData = dataset.splitTestData(5);

    NNClassifier net = NNClassifier::loadModel("D:/GitHub/lightnet-ml/data/model.json");

    std::vector<NNClassifier::Prediction> predictions = net.predict(testData);

    for(NNClassifier::Prediction prediction : predictions){
        cout << "Predicted: " << prediction.predictedEncodedTarget << " Actual: " << prediction.actualEncodedTarget << " Conf: " << prediction.confidence << endl;
    }

}
