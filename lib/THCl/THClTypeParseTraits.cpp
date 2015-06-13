#include "THClTypeParseTraits.h"

#define REGISTER_PARSE_TYPE(X) template <> struct TypeParseTraits<X> \
    { static const char* name; } ; const char* TypeParseTraits<X>::name = #X


//REGISTER_PARSE_TYPE(unsigned int);
//REGISTER_PARSE_TYPE(unsigned long);

REGISTER_PARSE_TYPE(unsigned int);
REGISTER_PARSE_TYPE(unsigned long);


