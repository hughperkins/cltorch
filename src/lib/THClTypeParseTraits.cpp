#include "THClTypeParseTraits.h"

//#define REGISTER_PARSE_TYPE(X) template <> struct TypeParseTraits<X> 
//    { static const char* name; } ; const char* TypeParseTraits<X>::name = #X

//#define REGISTER_PARSE_TYPE_DEFINITION(X, opencltypename) \
  const char* TypeParseTraits<X>::name = #X \
  const char* TypeParseTraits<X>::openClTypeName = #opencltypename

//const char *TypeParseTraits<uint32_t>::name = "uint32_t";
//const char *TypeParseTraits<uint64_t>::name = "uint64_t";
const char *TypeParseTraits<uint32_t>::openClTypeName = "unsigned int";
const char *TypeParseTraits<uint64_t>::openClTypeName = "unsigned long";

//REGISTER_PARSE_TYPE_DEFINITION(unsigned int);
//REGISTER_PARSE_TYPE_DEFINITION(uint64_t, unsigned long);
//REGISTER_PARSE_TYPE_DEFINITION(uint32_t, unsigned int);

//template<> struct TypeParseTraits<uint64_t>
//{
//    static const char *name;
//};
//const char * TypeParseTraits<uint64_t>::name = "uint64_t"

//template<> struct TypeParseTraits<uint32_t>
//{
//    static const char *name;
//};
//const char * TypeParseTraits<uint32_t>::name = "uint32_t"

