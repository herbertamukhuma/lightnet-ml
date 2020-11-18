#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <vector>
#include <algorithm>
#include <cmath>

namespace LightNet {

class MathUtil
{
public:
    static double sum(std::vector<double> inputs);

    static double relu(double input);

    static double sigmoid(double input);

    static double fastSigmoid(double input);

    static double mse(std::vector<std::tuple<double, double>> inputPairs);

    static double mean(std::vector<double> inputs);

    static double maxElement(std::vector<double> elements);

    static double sigmoidDeriv(double value);

    static double reluDeriv(double value);

    enum Activation_Func {
        ReLU = 0,
        Sigmoid = 1,
        FastSigmoid = 2
    };
};

}

#endif // MATHUTIL_H
