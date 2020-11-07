#include <iostream>

#include "src/neuralnetwork.h"
#include "src/dataset.h"

using namespace std;
using namespace LightNet;

int main()
{

//    NeuralNetwork net({128, 6, 4});

//    Layer layer = net.getLayers()[1];

//    layer.print();

    Dataset dataset("F:/GitHub/lightnet-ml/data/embeddings.csv");
    dataset.print();
    return 0;
}
