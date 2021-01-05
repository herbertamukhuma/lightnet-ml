#ifndef NEURON_H
#define NEURON_H

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <ctime>

#include "mathutil.h"
#include "util.h"

namespace LightNet {

class Neuron
{
public:

    enum ActivationFunction{
        Sigmoid = 0,
        Relu = 1
    };

    Neuron(ActivationFunction activationFunction);

    double generateRandomWeight();

    void addWeight(double weight);

    void addActivation(double output);

    std::vector<double> getWeights() const;

    std::vector<double> getActivations() const;

    void clearActivations();

    double compute();

    double getOutput() const;

    void updateWeight(size_t index, double newWeight);

    void setBias(double value);

    double getBias() const;

    void setWeights(const std::vector<double> value);

protected:

    ActivationFunction activationFunction;

    std::vector<double> weights;

    std::vector<double> inputActivations;

    double output = 0;

    double bias;

    bool seeded = false;

};

}

#endif // NEURON_H
