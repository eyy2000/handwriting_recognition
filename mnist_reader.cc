#include "mnist_reader.hh"
#include <fstream>
#include <algorithm>
uint32_t msb_to_uint32(uint8_t * b) {
	return b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3];
}

IDX readIDX(std::string path) {
	std::ifstream file(path, std::ios_base::in | std::ios_base::binary);
	IDX idx;
	char buf[4];
	if (!file) {
		throw path + ": failed to open file";
	}
	if (!file.read(buf, 4)) {
		throw path + ": failed to read magic number";
	}
	if (buf[0] != 0){
		throw path + ": error: expected magic bytes 0 to be zero";
	}
	if (buf[1] != 0){
		throw path + ": error: expected magic bytes 1 to be zero";
	}
	if (buf[2] != 8) {
		throw path + ": error: expected magic byte 2 to be 8";
	}
	uint8_t dimension = buf[3];

	uint32_t totalBytes = 1;
	for (unsigned i = 0; i < dimension; i++) {
		if (!file.read(buf, 4)) {
			throw path + ": failed to read sizes";

		}
		uint8_t buf8[4];
		std::copy(std::begin(buf), std::end(buf), std::begin(buf8));
		uint32_t size = msb_to_uint32(buf8);
		idx.sizes.push_back(size);
		totalBytes *= size;
	}

	std::vector<char> dataBuf;
	dataBuf.resize(totalBytes);
	idx.bytes.resize(totalBytes);
	if (!file.read(dataBuf.data(), totalBytes)) {
		throw path + ": error: failed to read data bytes";
	}
	std::copy(dataBuf.begin(), dataBuf.end(), idx.bytes.begin());
	return idx;
}

MNIST readMNIST(std::string dataPath, std::string labelPath) {
	MNIST img;
	img.data = readIDX(dataPath);
	img.labels = readIDX(labelPath);
	if (img.data.sizes.size() != 3) {
		throw dataPath + ": error: expected data dimension to be 3";
	}
	if (img.labels.sizes.size() != 1) {
		throw labelPath + ": error: expected labels dimension to be 1";
	}
	if (img.data.sizes[0] != img.labels.sizes[0]) {
		throw dataPath + ": error: expected data dimension 0 to match labels dimension 0";
	}

	return img;
}
unsigned MNIST::numberOfImages() const {
	return labels.sizes[0];
}
unsigned MNIST::pixelsPerImage() const {
	return data.sizes[1] * data.sizes[2];
}
char MNIST::labelAt(unsigned i) const {
	return labels.bytes.at(i);
}
std::vector<double> MNIST::imageAt(unsigned i) const {
	std::vector<double> pixels;
	pixels.resize(pixelsPerImage());
	std::copy(beginAt(i), endAt(i), pixels.begin());
	for (double& pixel: pixels) {
		pixel /= 255;
	}
	return pixels;
}
std::vector<uint8_t>::const_iterator MNIST::beginAt(unsigned i) const {
	return data.bytes.begin() + i * pixelsPerImage();
}
std::vector<uint8_t>::const_iterator MNIST::endAt(unsigned i) const {
	return beginAt(i + 1);
}
