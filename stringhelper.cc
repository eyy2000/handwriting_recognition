#include "stringhelper.hh"

bool stripPrefix(std::string const& a, std::string const& b, std::string & param) {
	if (a.find(b) == 0) {
		param = a.substr(b.size());
		return true;
	} else {
		return false;
	}
}

std::string stripWhitespacePrefix(std::string const& a) {
	// find position of first character that is not whitespace
	size_t pos = a.find_first_not_of(" \t");
	if (pos != std::string::npos) {
		// return the substring starting from pos
		return a.substr(pos);
	} else {
		// return empty string if the whole string is whitespace
		return "";
	}
}
