#ifndef NEURON_H
#define NEURON_H

#include <stdlib.h>
#include <vector>

namespace LightNet {

class Neuron
{
public:
    Neuron();

    static double generateRandomWeight();

    void addInputWeight(double weight);

    std::vector<double> getInputWeights() const;

protected:
    std::vector<double> inputWeights;

    double outputValue;

    std::vector<double> inputValues;
};

}

#endif // NEURON_H
