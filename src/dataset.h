#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

#include "util.h"
#include "mathutil.h"

namespace LightNet {

class Dataset
{
public:
    Dataset(std::string filename, bool hasHeadings = false, char delimiter = ',');

    bool isLoaded() const;

    void print();

    void printTargets();

    void printUniqueTargets();

    std::vector<double> getTargets();

    std::vector<double> getUniqueTargets();

    std::vector<std::string> getUniqueUnencodedTargets();

    std::vector<double> getInputs(size_t rowIndex);

    double getTarget(size_t rowIndex);

    size_t getInputCount();

    size_t getUniqueTargetCount();

    size_t getRowCount();

    size_t getColumnCount();

    void scale();

private:
    bool loaded = false;

    bool scaled = false;

    std::vector<std::vector<std::string>> rawData;

    std::vector<std::vector<double>> data;

    void encodeTargets();

    void load(std::string filename, bool hasHeadings, char delimiter);

    bool validateDataset();
};

}

#endif // DATASET_H
