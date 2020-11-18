#include "mathutil.h"

namespace LightNet {

double MathUtil::sum(std::vector<double> inputs)
{
    double sum = 0.0;

    for(double input : inputs){
        sum += input;
    }

    return sum;
}

double MathUtil::relu(double input)
{
    return std::max(0.0, input);
}

double MathUtil::sigmoid(double input)
{
    return 1 / (1 + exp(-input));
}

double MathUtil::fastSigmoid(double input)
{
    return input / (1 + abs(input));
}

double MathUtil::mse(std::vector<std::tuple<double, double> > inputPairs)
{

    std::vector<double> squaredErrors;

    for(std::tuple<double, double> &pair : inputPairs){
        double error = std::get<0>(pair) - std::get<1>(pair);
        squaredErrors.push_back(std::pow(error, 2.0));
    }

    return mean(squaredErrors);
}

double MathUtil::mean(std::vector<double> inputs)
{
    double sum = MathUtil::sum(inputs);
    return sum / (double)inputs.size();
}

double MathUtil::maxElement(std::vector<double> elements)
{
    double max = elements[0];

    for(double element : elements){
        if(element > max){
            max = element;
        }
    }

    return max;
}

double MathUtil::sigmoidDeriv(double value)
{
    return value * (1.0 - value);
}

double MathUtil::reluDeriv(double value)
{
    if(value > 0.0){
        return 1.0;
    }else {
        return 0.0;
    }
}

// end of namespace
}
