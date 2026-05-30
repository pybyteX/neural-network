#include <chrono>
#include <cmath>
#include <iostream>
#include "matrix.h"
#include "nn.h"
#include <vector>

int main() {
	namespace ch = std::chrono;

	Matrix2D inputs;
	Matrix2D outputs;

	int j{};
	for (int i{}; i < 1000; ++i) {
		inputs.addRow({static_cast<double>(i) / 1000.0, static_cast<double>(j) / 1000.0});
		outputs.addRow({std::max(i, j) / 1000.0});
		++j;
	}

	Sequential seq(
		LinearLayer(2, 4), ActivationLayer(NeuralNet::ApplyTanh, NeuralNet::DTanh),
		LinearLayer(4, 1)
	);

	auto start{ ch::system_clock::now() };
	seq.train(inputs, outputs, 1000, 0.1);
	auto end{ ch::system_clock::now() };
	auto dur{ ch::duration_cast<ch::seconds>(end - start) };

	std::cout << "\nTook " << dur.count() << "s to run.\n";
	return 0;
}
