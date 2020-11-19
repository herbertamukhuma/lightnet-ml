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

    static double sigmoid(double input);

    static double mse(std::vector<std::tuple<double, double>> inputPairs);

    static double mean(std::vector<double> inputs);

    static double maxElement(std::vector<double> elements);

    static double minElement(std::vector<double> elements);

    static double sigmoidDeriv(double value);

    static double minMaxNormalization(double inputElement, double minElement, double maxElement);
};

}

#endif // MATHUTIL_H
