#include "dataset.h"

namespace LightNet {

Dataset::Dataset(std::string filename, bool hasHeadings, char delimiter)
{
    load(filename, hasHeadings, delimiter);
}

Dataset::Dataset(MatrixS data)
{
    rawData = data;

    encodeTargets();

    loaded = validateDataset();
}

Dataset::Dataset()
{

}

Dataset::Dataset(Matrix data, MatrixS rawData) : rawData(rawData), data(data)
{

}

void Dataset::setUniqueEncodedTargets(const std::vector<double> &value)
{
    uniqueEncodedTargets = value;
}

void Dataset::setUniqueUnencodedTargets(const std::vector<std::string> &value)
{
    uniqueUnencodedTargets = value;
}

void Dataset::load(std::string filename, bool hasHeadings, char delimiter)
{

    std::fstream fin;

    fin.open(filename, std::ios::in);

    if(!fin.is_open()){
        std::cerr << " -- unable to open file at: " << filename << std::endl;
        return;
    }

    std::vector<std::string> row;

    std::string line, word;

    size_t rowCounter = 0;

    while (fin.good()) {

        row.clear();

        std::getline(fin, line);

        if(hasHeadings && rowCounter == 0){
            // skip first row which has the headings
        }else {

            std::stringstream s(line);

            while(std::getline(s, word, delimiter)){

                if(word.empty()){
                    break;
                }

                row.push_back(word);
            }

            if(!row.empty()){
                rawData.push_back(row);
            }
        }

        rowCounter++;
    }

    encodeTargets();

    loaded = validateDataset();
}

void Dataset::encodeTargets()
{
    std::vector<std::string> uniqueUnencodedTargets = getUniqueUnencodedTargets();

    for(std::vector<std::string> row : rawData){

        std::vector<double> encodedRow;

        size_t columnCounter = 0;

        for(std::string column : row){

            if(columnCounter == row.size() - 1){
                double index = (double)Util::find(uniqueUnencodedTargets, column) + 1.0;
                encodedRow.push_back(index);
            }else {
                encodedRow.push_back(std::stod(column));
            }

            columnCounter++;
        }

        data.push_back(encodedRow);

    }
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
    std::vector<double> targets = getEncodedTargets();

    for(double &target : targets){
        std::cout << target << std::endl;
    }
}

void Dataset::printUniqueTargets()
{
    std::vector<double> uniqueTargets = getUniqueEncodedTargets();

    for(double &uniqueTarget : uniqueTargets){
        std::cout << uniqueTarget << std::endl;
    }
}

std::vector<double> Dataset::getEncodedTargets()
{
    std::vector<double> targets;

    // targets are in the last column
    for(std::vector<double> &row : data){
        targets.push_back(row[row.size()-1]);
    }

    return targets;
}

std::vector<double> Dataset::getUniqueEncodedTargets()
{
    if(!uniqueEncodedTargets.empty()){
        return uniqueEncodedTargets;
    }

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

std::vector<std::string> Dataset::getUniqueUnencodedTargets()
{
    if(!uniqueUnencodedTargets.empty()){
        return uniqueUnencodedTargets;
    }

    std::vector<std::string> uniqueTargets;

    // targets are in the last column
    for(std::vector<std::string> &row : rawData){

        std::string target = row[row.size()-1];

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

std::string Dataset::getUnencodedTarget(size_t rowIndex)
{
    std::vector<std::string> row = rawData[rowIndex];
    return row[row.size()-1];
}

size_t Dataset::getInputCount()
{
    return getColumnCount() - 1;
}

size_t Dataset::getUniqueTargetCount()
{
    return getUniqueEncodedTargets().size();
}

size_t Dataset::getRowCount()
{
    return data.size();
}

size_t Dataset::getColumnCount()
{
    return data[0].size();
}

void Dataset::scale()
{
    if(scaled) return;

    std::vector<std::vector<double>> columns(getColumnCount(), std::vector<double>(getRowCount(), 0));

    size_t rowCounter = 0;

    for(std::vector<double> row : data){

        size_t columnCounter = 0;

        for(double cellValue : row){
            columns[columnCounter][rowCounter] = cellValue;
            columnCounter++;
        }

        rowCounter++;
    }

    //data = columns;

    size_t columnCounter = 0;

    for(std::vector<double> column : columns){

        double minElement = MathUtil::minElement(column);
        double maxElement = MathUtil::maxElement(column);

        size_t rowCounter = 0;

        for(double cellValue : column){

            if(columnCounter == columns.size()-1){
                data[rowCounter][columnCounter] = cellValue;
            }else {
                data[rowCounter][columnCounter] = MathUtil::minMaxNormalization(cellValue, minElement, maxElement);
            }

            rowCounter++;
        }

        columnCounter++;
    }

    scaled = true;
}

Dataset Dataset::splitTestData(size_t ratio)
{
    size_t splitSize = (ratio / 100.0) * data.size();

    Matrix testData;
    MatrixS unencodedtestData;

    srand((unsigned int)time(NULL));

    while (testData.size() < splitSize) {        
        size_t index = rand() % data.size();

        testData.push_back(data[index]);
        unencodedtestData.push_back(rawData[index]);

        data.erase(data.begin() + index);
        rawData.erase(rawData.begin() + index);
    }

    return Dataset(testData, unencodedtestData);
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
