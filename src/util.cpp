#include "util.h"

namespace LightNet {

Util::Util()
{

}

int Util::find(std::vector<double> haystack, double needle)
{
    auto it = std::find(haystack.begin(), haystack.end(), needle);

    if(it != haystack.end()){
        return it - haystack.begin();
    }else {
        return -1;
    }
}

int Util::find(std::vector<std::string> haystack, std::string needle)
{
    auto it = std::find(haystack.begin(), haystack.end(), needle);

    if(it != haystack.end()){
        return it - haystack.begin();
    }else {
        return -1;
    }
}

// end of namespace
}
