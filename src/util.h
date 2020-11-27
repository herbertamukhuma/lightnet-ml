#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <algorithm>
#include <string>

#include "mathutil.h"

namespace LightNet {

class Util
{
public:
    Util();

    static int find(std::vector<double> haystack, double needle);

    static int find(std::vector<std::string> haystack, std::string needle);

    static int findMax(std::vector<double> haystack);
};

}

#endif // UTIL_H
