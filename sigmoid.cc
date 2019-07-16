#include "sigmoid.hh"
#include <cmath>
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}
double sigmoidD(double x) {
	double y = sigmoid(x);
	return y * (1 - y);
}
