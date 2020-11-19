#include "dataset.h"

namespace LightNet {

Dataset::Dataset(std::string filename, bool hasHeadings, char delimiter)
{
    load(filename, hasHeadings, delimiter);
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

std::vector<std::string> Dataset::getUniqueUnencodedTargets()
{
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

size_t Dataset::getInputCount()
{
    return getColumnCount() - 1;
}

size_t Dataset::getUniqueTargetCount()
{
    return getUniqueTargets().size();
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
                std::cout << cellValue << " " << minElement << " " << maxElement << std::endl;
                data[rowCounter][columnCounter] = MathUtil::minMaxNormalization(cellValue, minElement, maxElement);
            }

            rowCounter++;
        }

        columnCounter++;
    }

    scaled = true;
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
