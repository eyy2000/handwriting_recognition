#ifndef MNIST_READER_H
#define MNIST_READER_H
#include <vector>
#include <string>
#include <cstdint>
struct IDX {
	std::vector<uint32_t> sizes;
	std::vector<uint8_t> bytes;
};
struct MNIST {
	IDX data, labels;
	unsigned numberOfImages() const;
	unsigned pixelsPerImage() const;
	char labelAt(unsigned i) const;
	std::vector<double> imageAt(unsigned i) const;
	std::vector<uint8_t>::const_iterator beginAt(unsigned i) const;
	std::vector<uint8_t>::const_iterator endAt(unsigned i) const;
};

MNIST readMNIST(std::string dataPath, std::string labelPath);
#endif
