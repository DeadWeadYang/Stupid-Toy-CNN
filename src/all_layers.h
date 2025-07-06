#pragma once
#ifndef __ALL_LAYERS_H__
#define __ALL_LAYERS_H__
#include "activation_function.hpp"
#include "fully_connected_layer.hpp"
#include "std_convolutional_layer.hpp"
#include "depthwise_convolutional_layer.hpp"
#include "batch_normalization_layer.hpp"
#include "maxpooling_layer.hpp"
namespace toyCNN {
	worker* LayerConstruct(Params* params) {
		switch (params->kind) {
		case KindOfWork::ACTIVATION:
			return ActivationConstruct(params);
		case KindOfWork::FULLY_CONNECTED:
			return FullyConnectedConstruct(params);
		case KindOfWork::STD_CONVOLUTIONAL:
			return  STDConvolutionalConstruct(params);
		case KindOfWork::DEPTHWISE_CONVOLUTIONAL:
			return DepthwiseConvolutionalConstruct(params);
		case KindOfWork::BATCH_NORMALIZATION:
			return BatchNormalizationConstruct(params);
		case KindOfWork::MAX_POOLING:
			return MaxPoolingConstruct(params);
		}
		return nullptr;
	}
}
#endif