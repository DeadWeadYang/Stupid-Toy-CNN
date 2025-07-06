#pragma once
#include<fstream>
#include<sstream>
#include "util.hpp"
#include "istream_warp.hpp"
using namespace toyCNN;

inline void input_mnist_csv(unsigned int size,std::ifstream& file,std::vector<vec_t >& input,std::vector< vec_t>& labels) {
	csv_reader wrap_f(file);
	input.resize(size); labels.resize(size);
	for (int i = 0; i < size; ++i) {
		vec_t& in = input[i];
		vec_t& la = labels[i];
		in.resize(1 * 28 * 28);
		la.resize(10);
		int num=0; wrap_f >> num;
		std::fill(la.begin(), la.end(), float_t(0));
		la[num] = float_t(1);
		for (int j = 0,pix; j < 1 * 28 * 28; ++j) {
			wrap_f >> pix; in[j] = float_t(pix) / float_t(255);
		}

	}
}
inline void import_mnist(unsigned int train_size,unsigned  int test_size, std::vector< vec_t>& input_train, std::vector< vec_t>& labels_train, std::vector<vec_t>& input_test, std::vector<vec_t>& labels_test) {
	train_size = std::min(train_size, 60000u); test_size = std::min(test_size, 10000u);
	std::ifstream data_train("../mnist/mnist_train.csv");
	input_mnist_csv(train_size, data_train, input_train, labels_train);
	data_train.close();
	std::ifstream data_test("../mnist/mnist_test.csv");
	input_mnist_csv(test_size, data_test, input_test, labels_test);
	data_test.close();

}