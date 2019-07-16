#include <iostream>
#include "stringhelper.hh"
#include "network.hh"
#include "network2.hh"
#include "mnist_reader.hh"
void test() {
	MNIST training;
	try {
		training = readMNIST("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
	} catch (std::string& e) {
		std::cout << e << '\n';
	}
	Network nn(1, 25, 10, 1);
	nn.save("network.nn");
	Network nn2("network.nn");
	nn.save("network2.nn");
}


int main (int argc, char ** argv) {
	MNIST training;
	MNIST testing;
	try {
		testing = readMNIST("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
		training = readMNIST("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");
		//training = readMNIST("testdata/images.idx3-ubyte", "testdata/labels.idx1-ubyte");

		double score;
		double scoreX;
		double scoreY;
		double maxScore = 0;
		double bestX;
		double bestY;
		double bestPrimaryError;
		double bestSecondaryError;
		unsigned progress = training.numberOfImages()/100;
		unsigned curProgress = 0;
		double increment = 0.2;
		double gain = 0.6;
		double alpha = 0;
		std::vector<unsigned> shape = {training.pixelsPerImage(), 40, 25, 20, 15, 10};
		Network2 nn(shape);
		score = 0;
		scoreX = 0;
		scoreY = 0;
		curProgress = 0;
		progress = training.numberOfImages()/100;
		std::cout<<"Training"<<std::endl;
		for (unsigned i = 0; i < training.numberOfImages(); i++) {
			if ( i%progress == 0){
				curProgress += 1;
				if(i != 0){
					std::cout<<'\b'<<std::flush;
					std::cout<<'\b'<<std::flush;
					if(i-10 >= 10){
						std::cout<<'\b'<<std::flush;
					}
				}
				std::cout<<curProgress<<"%"<<std::flush;
			}
			int label = training.labelAt(i);
			std::vector<double> inputPixels = training.imageAt(i);
			nn.forward(inputPixels);
			//nn.print();
			std::vector<double> expected(10);
			expected[label] = 1.0;
			nn.backward(expected,gain,alpha);
			//std::cout << "[ " << label << " , "<<nn.prediction()<< " ] "<<std::endl;
	
			//std::cout<<"BackProp: "<<std::endl;
			//nn.print();
		}
		std::cout<<std::endl;
		std::cout<<"Training Complete"<<std::endl<<std::endl;
		std::cout<<"Testing"<<std::endl;

		progress = testing.numberOfImages()/100;
		curProgress = 0;
		double error = 0;
		double secondError = 0;
		for (unsigned i = 0; i < testing.numberOfImages(); i++) {
			/*if ( i%progress == 0){
				curProgress += 1;
				if(i != 0){
					std::cout<<'\b'<<std::flush;
					std::cout<<'\b'<<std::flush;
					if(i-10 >= 10){
						std::cout<<'\b'<<std::flush;
					}
				}
				std::cout<<curProgress<<"%"<<std::flush;
			}*/
			int label = testing.labelAt(i);
			std::vector<double> inputPixels = testing.imageAt(i);
			nn.forward(inputPixels);
			//nn.print();
			std::vector<double> expected(10);
			expected[label] = 1.0;
			//std::cout << "[ " << label << " , "<<nn.prediction()<< " ] ";
			if (label != nn.prediction()){
				error += 1;
				//std::cout<<" ---------ERROR---------"<<nn.secondPrediction();
				if(label != nn.secondPrediction()){
					secondError +=1;
				}
			}
			//std::cout<<std::endl;
		}

		//std::cout<<"Error rate: "<< error/testing.numberOfImages()*100<<"%"<<std::endl;
		//std::cout<<"Error rate second best: "<< secondError/error*100<<"%"<<std::endl;
		std::cout<<"Testing Complete"<<std::endl;
		//scoreX = 100 - error/testing.numberOfImages()*100;
		//scoreY = 0.5*(100 - secondError/error*100);
		//score = scoreX + scoreY;
		

		bestPrimaryError = error/testing.numberOfImages()*100;
		bestSecondaryError = secondError/error*100;

		//std::cout << "Best [gain,alpha]: [" <<bestX<< "," << bestY<<"]"<<std::endl;
		std::cout << "Primary Error: " <<bestPrimaryError<<std::endl;
		std::cout << "Secondary Error: " <<bestSecondaryError<<std::endl;
		
	} catch (std::string& e) {
		std::cout << e << '\n';
	} catch (char const* e) {
		std::cout << e << '\n';
	}

	return 0;
}
