#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <algorithm>
#include <string>

namespace LightNet {

class Util
{
public:
    Util();

    static int find(std::vector<double> haystack, double needle);

    static int find(std::vector<std::string> haystack, std::string needle);

};

}

#endif // UTIL_H
