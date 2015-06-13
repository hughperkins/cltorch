#include "THClTypeParseTraits.h"

#define REGISTER_PARSE_TYPE(X) template <> struct TypeParseTraits<X> \
    { static const char* name; } ; const char* TypeParseTraits<X>::name = #X

#define REGISTER_PARSE_TYPE_DEFINITION(X) \
  const char* TypeParseTraits<X>::name = #X


REGISTER_PARSE_TYPE_DEFINITION(unsigned int);
REGISTER_PARSE_TYPE_DEFINITION(unsigned long);


