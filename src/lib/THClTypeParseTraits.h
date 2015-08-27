#pragma once

#include "mystdint.h"

// adapted from http://stackoverflow.com/questions/1055452/c-get-name-of-type-in-template
template<typename T>
struct TypeParseTraits;

#define REGISTER_PARSE_TYPE_DECLARATION(X) template <> struct TypeParseTraits<X> \
    { static const char *openClTypeName; } ; 


//REGISTER_PARSE_TYPE_DECLARATION(unsigned int);
//REGISTER_PARSE_TYPE_DECLARATION(unsigned long);
REGISTER_PARSE_TYPE_DECLARATION(uint64_t);
REGISTER_PARSE_TYPE_DECLARATION(uint32_t);

