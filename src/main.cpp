#include<iostream>
#include "toy_cnn.hpp"
#include "mnist.hpp"
using namespace toyCNN;
int main() {
	network net;
	net.setInputShape(28, 28, 1);
	net.setOutputShape(10, 1, 1);
	net.append(new CNN_layers::batch_norm(net.last_outshape(),false));
	net.append(new CNN_layers::std_conv(net.last_outshape(), 3, 4, 1, PaddingType::same_default_padding));
	net.append(new CNN_layers::batch_norm(net.last_outshape(), false));
	net.append(new CNN_activation::relu(net.last_outshape()));
	net.append(new CNN_layers::max_pool(net.last_outshape(),2,PaddingType::same_default_padding));
	net.append(new CNN_layers::batch_norm(net.last_outshape(), false));
	net.append(new CNN_layers::fully_c(net.last_outshape(), 10));
	//net.append(new CNN_activation::softmax(net.last_outshape()));
	std::ifstream iwnet("../weight.out",std::ios::binary);
	net.reconstruct();
	net.load(iwnet);
	iwnet.close();
	std::vector<vec_t>train_input,train_label,test_input,test_labels;
	import_mnist(40000, 8000, train_input, train_label, test_input, test_labels);
	std::ofstream fout("../trace.txt");
	learn_strategy stgy;
	stgy.bottom_rate = 1e-9;
	stgy.decay_epoch = 1;
	stgy.decay_rate = 0.95;
	stgy.learning_rate_decay = 0.01;
	stgy.learning_rate_start = 0.01;
	stgy.start_epoch = 10;
	int btc = 0,etc=0,eee=0;
	vec_t res;
	float_t train_loss_tot = 0;
	try {
		net.train(CrossEntropyWithSoftmax(), train_input, train_label, 5, 1,50, stgy, true, 
			[&]() {
				++etc; 
				{
					float_t acc = 0; 
					net.test(one_hot_Comparer(), test_input, test_labels, acc,/*&res*/nullptr);
					{
						train_loss_tot /= btc; btc = 0;
						fout << "mean train loss: " << train_loss_tot << '\n';
						std::cout << "mean train loss: " << train_loss_tot << '\n';
						//for (auto iii : res) {
						//	std::cout << iii << ' ';
						//}
						//std::cout << '\n';
					}
					net.save("../weight");
					fout << "epoch:" << etc << " accuracy:" << acc << '\n'; 
					std::cout << etc<<' '<<acc << '\n'; 
				} 
			}, 
			[&](float_t los) {
				++btc; train_loss_tot += los;
				//fout << "batch:" << btc << " loss:" << los << std::endl; 
				//std::cout << "batch:" << btc << " loss:" << los << std::endl;
			}
		);
	}
	catch (nn_error nn_err) {
		std::cout << nn_err.what() << std::endl;
	}
	net.save("../weight");
	return 0;
}