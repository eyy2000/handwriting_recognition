#ifndef STRINGHELPER_H
#define STRINGHELPER_H
#include <string>
#include <sstream>
// Inspect string 'a'.
// If 'b' is a prefix of 'a', remove the prefix and write the remaining substring in 'param'.
bool stripPrefix(std::string const& a, std::string const& b, std::string & param);

std::string stripWhitespacePrefix(std::string const& a);

#endif
