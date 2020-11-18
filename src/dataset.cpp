#include "dataset.h"

namespace LightNet {

Dataset::Dataset(std::string filename)
{
    load(filename);
}

void Dataset::load(std::string filename)
{
    std::fstream fin;

    fin.open(filename, std::ios::in);

    if(!fin.is_open()){
        std::cerr << " -- unable to open file at: " << filename << std::endl;
        return;
    }

    std::vector<double> row;

    std::string line, word;

    while (fin.good()) {

        row.clear();

        std::getline(fin, line);

        std::stringstream s(line);

        while(std::getline(s, word, ',')){

            if(word.empty()){
                break;
            }

            row.push_back(std::stod(word));
        }

        if(!row.empty()){
            data.push_back(row);
        }

    }

    loaded = validateDataset();
}

bool Dataset::isLoaded() const
{
    return loaded;
}

void Dataset::print()
{
    std::cout << "Dataset => rows: " << getRowCount() << " columns: " << getColumnCount() << "\n" << std::endl;

    for(std::vector<double> &row : data){

        size_t counter = 0;

        for(double &cellData : row){

            if(counter < row.size() - 1){
                std::cout << cellData << ", ";
            }else {
                std::cout << cellData;
            }

            counter++;
        }

        std::cout << std::endl;
    }
}

void Dataset::printTargets()
{
    std::vector<double> targets = getTargets();

    for(double &target : targets){
        std::cout << target << std::endl;
    }
}

void Dataset::printUniqueTargets()
{
    std::vector<double> uniqueTargets = getUniqueTargets();

    for(double &uniqueTarget : uniqueTargets){
        std::cout << uniqueTarget << std::endl;
    }
}

std::vector<double> Dataset::getTargets()
{
    std::vector<double> targets;

    // targets are in the last column
    for(std::vector<double> &row : data){
        targets.push_back(row[row.size()-1]);
    }

    return targets;
}

std::vector<double> Dataset::getUniqueTargets()
{
    std::vector<double> uniqueTargets;

    // targets are in the last column
    for(std::vector<double> &row : data){

        double target = row[row.size()-1];

        if(std::count(uniqueTargets.begin(), uniqueTargets.end(), target) < 1){
            uniqueTargets.push_back(target);
        }

    }

    return uniqueTargets;
}

std::vector<double> Dataset::getInputs(size_t rowIndex)
{
    std::vector<double> row = data[rowIndex];
    row.pop_back();
    return row;
}

double Dataset::getTarget(size_t rowIndex)
{
    std::vector<double> row = data[rowIndex];
    return row[row.size()-1];
}

size_t Dataset::getInputCount()
{
    return getColumnCount() - 1;
}

size_t Dataset::getRowCount()
{
    return data.size();
}

size_t Dataset::getColumnCount()
{
    return data[0].size();
}

bool Dataset::validateDataset()
{

    // check that the rows are of equal sizes
    size_t previousRowSize = 0;
    size_t counter = 0;

    for(std::vector<double> &row : data){

        if(counter == 0){
            // first iteration
        }else {

            if(row.size() != previousRowSize){
                std::cerr << " -- all rows must be of the same size" << std::endl
                          << "current row size: " << row.size() << std::endl
                          << "previous row size: " << previousRowSize << std::endl;
                return false;
            }
        }

        previousRowSize = row.size();
        counter++;
    }

    return true;
}

// end of namespace
}
