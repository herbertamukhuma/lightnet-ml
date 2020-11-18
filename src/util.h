#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <algorithm>

namespace LightNet {

class Util
{
public:
    Util();

    static int find(std::vector<double> haystack, double needle);
};

}

#endif // UTIL_H
