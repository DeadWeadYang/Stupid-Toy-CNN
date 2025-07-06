#pragma once
#ifndef __TOY_CNN_H__
#define __TOY_CNN_H__

#include "config.hpp"
#include "network.hpp"
#include "all_layers.h"
#include "loss_function.hpp"
namespace toyCNN {
	namespace CNN_layers {
		using fully_c = FullyConnectedParams;
		using std_conv = STDConvolutionalParams;
		using depthwise_conv = DepthwiseConvolutionalParams;
		using max_pool = MaxPoolingParams;
		using batch_norm = BatchNormalizationParams;
	}
	namespace CNN_activation {
		using identity = af_identity_Params;
		using relu = af_relu_Params;
		using leaky_relu = af_leaky_relu_Params;
		using elu = af_elu_Params;
		using sigmoid = af_sigmoid_Params;
		using tan_h = af_tanh_Params;
		using softmax = af_softmax_Params;
	}
}
#endif 
