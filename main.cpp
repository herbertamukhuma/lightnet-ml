#include <iostream>

#include "src/neuralnetwork.h"
#include "src/dataset.h"
#include "src/mathutil.h"

using namespace std;
using namespace LightNet;

int main()
{

    Dataset dataset("/Users/user/GitHub/lightnet-ml/data/embeddings.csv", true);

    dataset.scale();

    dataset.print();

//    NeuralNetwork net({dataset.getInputCount(), dataset.getUniqueTargetCount()}, dataset);
//    net.train(100);

    std::cout << "done" << std::endl;

    return 0;
}
