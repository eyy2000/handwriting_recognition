#ifndef NETWORK2_H
#define NETWORK2_H
#include "matrix.hh"
#include "sigmoid.hh"
#include <random>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <typeinfo>
#include <cmath>
int maxOf(std::vector<double> const & a) {
	double winner = a[0];
	size_t winner_i = 0;
	for (size_t i = 1; i < a.size(); i++) {
		if (winner < a[i]) {
			winner = a[i];
			winner_i = i;
		}
	}
	return winner_i;
}

int secondMaxOf(std::vector<double> const & a) {
	int index = maxOf(a);
	double winner = a[0];
	size_t winner_i = 0;
	if(index == 0 && a.size() >=2){
		winner = a[1];
		winner_i = 1;
	}
	
	for (size_t i = 1; i < a.size(); i++) {
		if (winner < a[i] && (int)i != index) {
			winner = a[i];
			winner_i = i;
		}
	}
	return winner_i;
}

int minOf(std::vector<double> const & a) {
	double winner = a[0];
	size_t winner_i = 0;
	for (size_t i = 1; i < a.size(); i++) {
		if (winner > a[i]) {
			winner = a[i];
			winner_i = i;
		}
	}
	return winner_i;
}
class Network2 {
	// Each layer's weights are stored in a matrix.
	// Each matrix row is a list of weights for one neuron, + one bias weight.
	std::vector<Matrix> layerWeights;

	// Previous Weight storage;
	std::vector<Matrix> previousWeights;

	// Outputs of most recent forward propagation pass
	std::vector<std::vector<double>> layerOutputs;

	// Inputs of the most recent forward propagation pass
	std::vector<double> initialInputs;

	// Number of neuron in each layer, including the input layer.
	// Begins with input layer and ends with output layer.
	std::vector<unsigned> const shape;

	// error storage
	std::vector<double> error;

public:
	Network2(std::vector<unsigned> const& shape_) : shape(shape_) {
		if (shape.size() <= 2) {
			throw "shape must have more than 1 layer";
		}
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<> d(0,1);

		for (unsigned i = 1; i < shape.size(); i++) {
			// number of input neurons on layer i-1 == number of weights per neuron on layer i
			unsigned weightsPerNeuron = shape[i-1];
			unsigned neuronsPerLayer = shape[i];

			Matrix weights(neuronsPerLayer, weightsPerNeuron + 1); // +1 for bias weight
			weights.randomize(d, gen);
			layerWeights.push_back(weights);
		}
	}
	int prediction() const {
		return maxOf(layerOutputs.back());
	}

	int secondPrediction() const {
		return secondMaxOf(layerOutputs.back());
	}
	void forward(std::vector<double> input) {
		initialInputs = input;
		std::vector<std::vector<double>> outputList;

		for (unsigned i = 0; i < layerWeights.size(); i++) {
			assert(input.size() == shape[i]);
			input.push_back(1); // add bias node for next layer

			std::vector<double> output = sigmoid(layerWeights[i].multiply(input));

			outputList.push_back(output);
			input = output;
		}
		layerOutputs = outputList;
	}

	void backward(std::vector<double> desired, double g, double a) {
		double gain = g;
		double alpha = a;
		unsigned NL = layerWeights.size();
		assert(NL == layerOutputs.size());
		std::vector<double> curError;
		std::vector<double> preError;
		error.clear();
		for (int i = NL-1; i>=0; i--){
			unsigned NN = layerOutputs[i].size();
			unsigned r = layerWeights[i].getNumberOfRows();
			unsigned c = layerWeights[i].getNumberOfColumns();
			assert(NN == r);
			for(int j = 0; j < (int)r; j++){
				double delta = 0;
				if(i == (int)NL-1){
					delta = layerOutputs[i][j]*(1-layerOutputs[i][j])*(desired[j]-layerOutputs[i][j]);
					curError.push_back(delta);
				}
				else{
					double sum = 0;
					assert (preError.size() == layerOutputs[i+1].size());
					for (int u = 0; u < (int)preError.size(); u++){
						sum += preError[u]*layerWeights[i+1].get(u,j);
					}
					delta = layerOutputs[i][j]*(1-layerOutputs[i][j])*sum;
					curError.push_back(delta);
				}

				for (int k = 0; k <(int) c; k++){
					double xi;
					if(k == (int)c-1){
						xi = 1;
					}
					else{
						if(i > 0){
							xi = layerOutputs[i-1][k];
						}
						else{
							xi = initialInputs[k];
						}
					}
					double diff;
					if(previousWeights.size() == 0){
						diff = 0;
					}
					else{
						diff = layerWeights[i].get(j,k) - previousWeights[i].get(j,k);
					}
					layerWeights[i].set(j,k, layerWeights[i].get(j,k) + gain*delta*xi + alpha*(diff));
				}
			}
				

				preError = curError;
				curError.clear();
		}
		previousWeights = layerWeights;
	}
	void printError() {
		std::vector<double> realError;
		double totalError =0 ;
		for (unsigned i = 0; i < error.size(); i++) {
			realError.push_back(error[i]);
			
		}
		for (auto e: realError) {
			totalError += fabs(e);
		}
		std::cout << " ] " << totalError << " : ";
		for (auto e: realError) {
			std::cout << e << ' ';
		}
		std::cout << '\n';
	}
	void print(){
		int nM = layerWeights.size();
		int r;
		int c;
		for(int i =nM-1; i>=0; i--){
			Matrix M =layerWeights[i];
			r = M.getNumberOfRows();
			c = M.getNumberOfColumns();
			for(int j = 0; j<r; j++){
				std::cout << '[' << i << ' ' << j << "] ";
				std::cout << std::fixed << std::setprecision(6)<< layerOutputs[i][j]<<" ";
				std::cout<<"|";
				for(int k = 0; k < c; k++){
					std::cout << std::fixed << std::setprecision(6)<<M.get(j,k)<<" ";
				}
				std::cout<<"\b";
				std::cout<<"| "<<std::endl;
			}
		}

		for (unsigned i = 0; i < initialInputs.size(); i++) {
			std::cout << initialInputs[i] << ' ';
		}
		std::cout << '\n';
	}
};
#endif
