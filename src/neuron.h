#ifndef NEURON_H
#define NEURON_H

#include <stdlib.h>
#include <vector>
#include <iostream>

#include "mathutil.h"
#include "util.h"

namespace LightNet {

class Neuron
{
public:
    Neuron();

    enum WeightUpdateDirection{
        Up = 0,
        Down = 1
    };

    static double generateRandomWeight();

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
    std::vector<double> weights;

    std::vector<double> inputActivations;

    double output = 0;

    double bias;

};

}

#endif // NEURON_H
