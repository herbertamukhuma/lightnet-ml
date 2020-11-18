#include <iostream>

#include "src/neuralnetwork.h"
#include "src/dataset.h"
#include "src/mathutil.h"

using namespace std;
using namespace LightNet;

int main()
{

    Dataset dataset("/Users/user/GitHub/lightnet-ml/data/embeddings.csv");

    NeuralNetwork net({dataset.getInputCount(), 1, 4}, dataset);
    net.train(10);

    cout << "done" << std::endl;

    return 0;
}
