#pragma once
#ifndef __ACTIVATION_FUNCTION_H__
#define __ACTIVATION_FUNCTION_H__
#include "worker.hpp"
namespace toyCNN {
	enum class ActivationType
	{
		IDENTITY, RELU, LEAKY_RELU, ELU, SIGMOID, TANH, SOFTMAX
	};
	struct ActivationParams :Params {
		ActivationType af_type;
		void save(std::ostream& os) override {
			Params::save(os);
			os << with_w("activation_func_type", 25) << '\n';
			os << with_w((int&)af_type,25) << '\n';
		}
		void load(std::istream& is) override {
			Params::load(is);
			is.ignore(25 + 1);
			is >> (int&)af_type;
			is.ignore(1);
		}
		ActivationParams(ActivationType type_,shape3d inshape_) :Params(KindOfWork::ACTIVATION){
			inShape = inshape_; outShape = inShape; af_type = type_;
		}
	};
	struct Activation :worker {
		ActivationType af_type;
		Activation(const ActivationParams& params) :worker(params) {
			inShape = params.inShape; outShape = params.outShape; af_type = params.af_type;
		}
		virtual float_t f(float_t x)const = 0;
		virtual float_t dfx(float_t x)const = 0;
		virtual float_t dfy(float_t y)const = 0;
		void forward_propagation(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT) override{
			//assert(dataIN.size() == dataOUT.size());
			for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
				const vec_t& in = dataIN[s];
				vec_t& out = dataOUT[s];
				//assert(in.size() == out.size());
				for (size_t i = 0, ied = in.size(); i < ied; ++i) {
					out[i] = f(in[i]);
				}
			}
		}
		void back_propagation_usingX(
			const std::vector<vec_t>& dataIN,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		) {
			//assert(dataIN.size() == gradOUT.size());
			//assert(gradIN.size() == gradOUT.size());
			for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
				const vec_t& in = dataIN[s];
				vec_t& prv_delta = gradOUT[s];
				const vec_t& cur_delta = gradIN[s];
				//assert(in.size() == cur_delta.size());
				//assert(cur_delta.size() == prv_delta.size());
				for (size_t i = 0, ied = in.size(); i < ied; ++i) {
					prv_delta[i] = cur_delta[i] * dfx(in[i]);
				}
			}
		}
		void back_propagation_usingY(	const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		) {
			//assert(dataOUT.size() == gradOUT.size());
			//assert(gradIN.size() == gradOUT.size());
			for (size_t s = 0, sed = dataOUT.size(); s < sed; ++s) {
				const vec_t& out = dataOUT[s];
				vec_t& prv_delta = gradOUT[s];
				const vec_t& cur_delta = gradIN[s];
				//assert(out.size() == cur_delta.size());
				//assert(cur_delta.size() == prv_delta.size());
				for (size_t i = 0, ied = out.size(); i < ied; ++i) {
					prv_delta[i] = cur_delta[i] * dfy(out[i]);
				}
			}
		}
	};
	struct af_identity_Params :ActivationParams {
		af_identity_Params(shape3d inshape_):ActivationParams(ActivationType::IDENTITY,inshape_){}
	};
	struct af_identity :Activation {
		using Activation::Activation;
		float_t f(float_t x)const override { return x; }
		float_t dfx(float_t x)const override { return float_t(1); }
		float_t dfy(float_t y)const override { return float_t(1); }
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			back_propagation_usingY(dataOUT, gradOUT, gradIN);
		}
	};
	struct af_relu_Params :ActivationParams {
		af_relu_Params(shape3d inshape_) :ActivationParams(ActivationType::RELU, inshape_) {}
	};
	struct af_relu :Activation {
		using Activation::Activation;
		float_t f(float_t x)const override { return x> float_t(0) ?x:0; }
		float_t dfx(float_t x)const override { return x> float_t(0) ?float_t(1):float_t(0); }
		float_t dfy(float_t y)const override { return y> float_t(0) ?float_t(1):float_t(0); }
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			back_propagation_usingY(dataOUT, gradOUT, gradIN);
		}
	};
	struct af_leaky_relu_Params :ActivationParams {
		af_leaky_relu_Params(shape3d inshape_) :ActivationParams(ActivationType::LEAKY_RELU, inshape_) {}
	};
	struct af_leaky_relu :Activation {
		using Activation::Activation;
		float_t f(float_t x)const override { return x>float_t(0) ? x : x * float_t(0.01); }
		float_t dfx(float_t x)const override { return x> float_t(0) ?float_t(1):float_t(0.01); }
		float_t dfy(float_t y)const override { return y> float_t(0) ?float_t(1):float_t(0.01); }
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			back_propagation_usingY(dataOUT, gradOUT, gradIN);
		}
	};
	struct af_elu_Params :ActivationParams {
		af_elu_Params(shape3d inshape_) :ActivationParams(ActivationType::ELU, inshape_) {}
	};
	struct af_elu :Activation {
		using Activation::Activation;
		float_t f(float_t x)const override { return x>float_t(0)?x:(std::exp(x)-float_t(1)); }
		float_t dfx(float_t x)const override { return x>float_t(0)?float_t(1):std::exp(x); }
		float_t dfy(float_t y)const override { return y>float_t(0)?float_t(1):(y+float_t(1)); }
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			back_propagation_usingY(dataOUT, gradOUT, gradIN);
		}
	};
	struct af_sigmoid_Params :ActivationParams {
		af_sigmoid_Params(shape3d inshape_) :ActivationParams(ActivationType::SIGMOID, inshape_) {}
	};
	struct af_sigmoid :Activation {
		using Activation::Activation;
		float_t f(float_t x)const override { return float_t(1)/(float_t(1)+std::exp(-x)); }
		float_t dfx(float_t x)const override { return float_t(1)/(std::exp(x)+std::exp(-x)+float_t(2)); }
		float_t dfy(float_t y)const override { return y*(float_t(1)-y); }
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			back_propagation_usingY(dataOUT, gradOUT, gradIN);
		}
	};
	struct af_tanh_Params :ActivationParams {
		af_tanh_Params(shape3d inshape_) :ActivationParams(ActivationType::TANH, inshape_) {}
	};
	struct af_tanh :Activation {
		using Activation::Activation;
		float_t f(float_t x)const override { return std::tanh(x); }
		float_t dfx(float_t x)const override { return sqr(float_t(2)/(std::exp(x)+std::exp(-x))); }
		float_t dfy(float_t y)const override { return float_t(1)-sqr(y); }
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			back_propagation_usingY(dataOUT, gradOUT, gradIN);
		}
	};
	struct af_softmax_Params :ActivationParams {
		af_softmax_Params(shape3d inshape_) :ActivationParams(ActivationType::SOFTMAX, inshape_) {}
	};
	struct af_softmax :Activation {
		using Activation::Activation;
		float_t f(float_t x)const override { return float_t(0); }
		float_t dfx(float_t x)const override { return float_t(0); }
		float_t dfy(float_t y)const override { return float_t(0); }
		void forward_propagation(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT) override {
			//assert(dataIN.size() == dataOUT.size());
			for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
				const vec_t& in = dataIN[s];
				vec_t& out = dataOUT[s];
				//assert(in.size() == out.size());
				float_t alpha = *std::max_element(in.begin(), in.end()),denom=float_t(0),numer;
				for (const auto& it : in) {
					denom += std::exp(it - alpha);
				}
				for (size_t i = 0, ied = in.size(); i < ied; ++i) {
					numer = std::exp(in[i] - alpha);
					out[i] = numer/denom;
				}
			}
		}
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			//assert(dataIN.size() == dataOUT.size());
			//assert(dataOUT.size() == gradIN.size());
			//assert(dataIN.size() == gradOUT.size());
			for (size_t s = 0, sed = gradIN.size(); s < sed; ++s) {
				const vec_t& out = dataOUT[s];
				const vec_t& gin = gradIN[s];
				vec_t& gout = gradOUT[s];
				//assert(out.size() == gin.size());
				//assert(gin.size() == gout.size());
				float_t ygsum = float_t(0);
				for (int i = 0; i < gin.size();i++) {
					ygsum += out[i]*gin[i];
				}
				for (size_t i = 0, ied = gin.size(); i < ied; ++i) {
					gout[i] = out[i] * (gin[i] - ygsum);
				}
			}
		}
	};
	inline worker* ActivationConstruct(Params* params) {
		auto& real_p = *dynamic_cast<ActivationParams*>(params);
		switch (real_p.af_type) {
		case ActivationType::IDENTITY:
			return new af_identity(real_p);
		case ActivationType::RELU:
			return new af_relu(real_p);
		case ActivationType::LEAKY_RELU:
			return new af_leaky_relu(real_p);
		case ActivationType::ELU:
			return new af_elu(real_p);
		case ActivationType::SIGMOID:
			return new af_sigmoid(real_p);
		case ActivationType::TANH:
			return new af_tanh(real_p);
		case ActivationType::SOFTMAX:
			return new af_softmax(real_p);
		}
		return nullptr;
	}
}

#endif