#ifndef SIGMOID_H
#define SIGMOID_H
#include <vector>
double sigmoid(double x);
double sigmoidD(double x);
inline std::vector<double> sigmoid(std::vector<double> values) {
	for (double& v: values) {
		v = sigmoid(v);
	}
	return values;
}
#endif
