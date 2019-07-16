#ifndef NETWORK_H
#define NETWORK_H
#include <vector>
#include <string>
struct NeuronPosition {
	unsigned layer;
	unsigned neuron;
};
struct NeuronConnection {
	NeuronPosition input;
	NeuronPosition output;
	double weight;
};
class Neuron {
public:
	// The neuron's own position within the network.
	// This should not be changed and the neuron itself should not be moved or copied.
	NeuronPosition position;

	// List of output connections.
	std::vector<NeuronConnection*> outputs;

	// List of input connections. This list is only built after all the output connections are connected.
	// It uses the same connection objects as the outputs, that's why they are pointers.
	std::vector<NeuronConnection*> inputs;

	// Computed value of the neuron
	double value;

	Neuron(unsigned layerPos, unsigned neuronPos);

	NeuronConnection& createOutputConnection(unsigned i, double weight);
};

class Layer {
public:
	// The layer's own position within the network.
	// This should not be changed and the layer itself should not be moved or copied.
	unsigned position;

	// list of Neurons in the layer.
	std::vector<Neuron> neurons;

	// Create a layer. It must know what position it will be in.
	Layer(unsigned layerPos);

	// Add a new neuron to the layer.
	Neuron& addNeuron();
};

class Network {
	// The network is a list of list of neurons.
	std::vector<Layer> layers;
public:
	// Creates an empty network to be constructed later.
	Network();

	// Create a bipartite network.
	Network(unsigned hiddenLayerCount, unsigned inputWidth, unsigned hiddenWidth, unsigned outputWidth);

	// This constructor builds the network and neurons according to a description file.
	Network(std::string path);

	// Read a .nn file to construct a neural network
	void load(std::string path);

	// Save the current network into a .nn file
	void save(std::string path);

	// Forward propagate input values through the network
	void forward(std::vector<double> const& inputs);

	// Shorthands to get input and output layers
	Layer& getInputLayer();
	Layer& getOutputLayer();
private:
	// Get a neuron by layer and neuron position.
	Neuron& neuronAt(NeuronPosition pos);

	// Add a new layer
	Layer& addLayer();



	// Part of the network construction. After all the outputs are hooked up,
	// visit each connections and hook up the inputs.
	void connectInputs();
};
#endif
