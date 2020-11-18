#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

namespace LightNet {

class Dataset
{
public:
    Dataset(std::string filename);

    void load(std::string filename);

    bool isLoaded() const;

    void print();

    void printTargets();

    void printUniqueTargets();

    std::vector<double> getTargets();

    std::vector<double> getUniqueTargets();

    std::vector<double> getInputs(size_t rowIndex);

    double getTarget(size_t rowIndex);

    size_t getInputCount();

    size_t getRowCount();

    size_t getColumnCount();

private:
    bool loaded = false;

    std::vector<std::vector<double>> data;

    bool validateDataset();
};

}

#endif // DATASET_H
