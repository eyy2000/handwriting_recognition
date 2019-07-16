#include "network.hh"
#include "stringhelper.hh"
#include "sigmoid.hh"
#include <fstream>
#include <iostream>
#include <cassert>
#include <random>

Neuron::Neuron(unsigned layer, unsigned neuron){
	this->value = 1;
	this->position.layer = layer;
	this->position.neuron = neuron;
}
NeuronConnection& Neuron::createOutputConnection(unsigned outputPos, double weight) {
	NeuronConnection* connection = new NeuronConnection;
	connection->weight = weight;
	connection->input = this->position;
	connection->output.neuron = outputPos;
	connection->output.layer = this->position.layer + 1;

	outputs.push_back(connection);
	return *connection;
}

Layer::Layer(unsigned layerPos) {
	this->position = layerPos;
}
Neuron& Layer::addNeuron() {
	neurons.emplace_back(this->position, neurons.size());
	return neurons[neurons.size()-1];
}

Layer& Network::addLayer() {
	layers.emplace_back(layers.size());
	return layers[layers.size()-1];
}
Network::Network() {

}
Network::Network(unsigned hiddenLayerCount, unsigned inputWidth, unsigned hiddenWidth, unsigned outputWidth) {
	// Setup some random number generator for the weights.
	// It will be a normal distribution with a mean of 0 and deviation of 1.
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(0,1);
	// Create all the layers
	for (unsigned i = 0; i < 2 + hiddenLayerCount; i++) {
		this->addLayer();
	}
	// Fill up the input layer
	for (unsigned i = 0; i < inputWidth; i++) {
		getInputLayer().addNeuron();
	}
	// Fill up all the hidden layers
	for (unsigned i = 0; i < hiddenLayerCount; i++) {
		Layer& layer = layers[i + 1];
		for (unsigned j = 0; j < hiddenWidth; j++) {
			layer.addNeuron();
		}
	}
	// Fill up the output layer
	for (unsigned i = 0; i < outputWidth; i++) {
		getOutputLayer().addNeuron();
	}
	// Make all the bipartite connections
	for (unsigned i = 0; i < layers.size() -1; i++) {
		Layer& layer1 = layers[i];
		Layer& layer2 = layers[i+1];



		// Make m:n connections from layer1 to layer2
		for (unsigned j = 0; j < layer1.neurons.size(); j++) {
			Neuron& neuron = layer1.neurons[j];
			for (unsigned k = 0; k < layer2.neurons.size(); k++) {
				neuron.createOutputConnection(k, d(gen));
			}
		}

		// Add bias neurons on layer1 and make 1:1 connections to layer2 neurons
		for (unsigned j = 0; j < layer2.neurons.size(); j++) {
			Neuron& neuron = layer1.addNeuron();
			neuron.createOutputConnection(j, d(gen));
		}
	}
	// make the opposite connections to become bidirectional
	connectInputs();
}

Network::Network(std::string path) {
	load(path);
}

void Network::connectInputs() {
	for (unsigned i = 0; i < layers.size() -1; i++) {
		Layer& layer1 = layers[i];
		for (unsigned j = 0; j < layer1.neurons.size(); j++) {
			Neuron& neuron1 = layer1.neurons[j];
			for (unsigned k = 0; k < neuron1.outputs.size(); k++) {
				NeuronConnection* connection = neuron1.outputs[k];
				assert(connection->input.layer == i);
				assert(connection->input.neuron == j);
				Layer& layer2 = layers.at(connection->output.layer);
				Neuron& neuron2 = layer2.neurons.at(connection->output.neuron);
				neuron2.inputs.push_back(connection);
			}
		}
	}
}

void Network::load(std::string path) {
	std::ifstream file(path, std::ios_base::in | std::ios_base::binary);
	std::string line;
	Layer * currentLayer = NULL;
	Neuron * currentNeuron = NULL;
	while (std::getline(file, line, '\n')) {
		std::istringstream iss(line);
		std::string word;

		// skip lines that don't have content
		if (!(iss >> word)) {
			continue;
		} else if (word == "layer") {
			currentLayer = &(this->addLayer());
			std::cout << "new layer created\n";
		} else if (word == "neuron") {
			currentNeuron = &(currentLayer->addNeuron());
			std::cout << "new neuron created\n";
		} else if (word == "output") {
			unsigned pos;
			double weight;
			if (currentNeuron == NULL) {
				std::cout << "error: no neuron to add connection to\n";
			} else if (iss >> pos >> weight) {
				currentNeuron->createOutputConnection(pos, weight);
				std::cout << "new connection created\n";
			} else {
				std::cout << "error: connection missing node position or weight\n";
			}
		}
	}
	connectInputs();
}

void Network::save(std::string path) {
	std::ofstream file(path, std::ios_base::out | std::ios_base::binary);
	for (unsigned i = 0; i < layers.size(); i++) {
		Layer& layer = layers[i];
		file << "layer\n";
		for (unsigned j = 0; j < layer.neurons.size(); j++) {
			Neuron& neuron = layer.neurons[j];
			file << " neuron # " << j << "\n";
			for (unsigned k = 0; k < neuron.inputs.size(); k++) {
				NeuronConnection* connection = neuron.inputs[k];
				file << "  # input " << connection->input.neuron << ' ' << connection->weight << '\n';
			}
			for (unsigned k = 0; k < neuron.outputs.size(); k++) {
				NeuronConnection* connection = neuron.outputs[k];
				file << "  output " << connection->output.neuron << ' ' << connection->weight << '\n';
			}
		}
	}
}

Layer& Network::getInputLayer() {
	return layers.at(0);
}

Layer& Network::getOutputLayer() {
	return layers.at(layers.size()-1);
}

Neuron& Network::neuronAt(NeuronPosition pos) {
	return layers.at(pos.layer).neurons.at(pos.neuron);
}

void Network::forward(std::vector<double> const& inputs) {
	// First feed inputs directly to input nodes (no activation function)
	for (unsigned i = 0; i < inputs.size() && i < getInputLayer().neurons.size(); i++) {
		getInputLayer().neurons[i].value = inputs[i];
	}

	// Then go through each neuron in order.
	// For every input connection of the neuron, sum all the value * weight, then apply activation function.
	for (unsigned i = 1; i < layers.size(); i++) {
		Layer& layer = layers[i];
		for (Neuron& neuron: layer.neurons) {
			// set to 0 before summing
			neuron.value = 0;
			for (NeuronConnection* connection: neuron.inputs) {
				Neuron& inputNeuron = this->neuronAt(connection->input);
				neuron.value += inputNeuron.value * connection->weight;
			}
			neuron.value = sigmoid(neuron.value);
		}
	}

}
