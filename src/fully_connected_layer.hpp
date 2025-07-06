#pragma once
#ifndef __FULLY_CONNECTED_LAYER_H__
#define __FULLY_CONNECTED_LAYER_H__
#include "worker.hpp"
namespace toyCNN {
	struct FullyConnectedParams :Params {
		bool has_bias;
		FullyConnectedParams(shape3d inshape_,size_t out_size,bool has_bias_=true) :Params(KindOfWork::FULLY_CONNECTED) {
			inShape = inshape_; outShape.reshape(out_size, 1, 1); has_bias = has_bias_;
		}
		void save(std::ostream& os) override {
			Params::save(os);
			os << with_w("has_bias", 10) << '\n';
			os <<  with_w(has_bias, 10) << '\n';
		}
		void load(std::istream& is) override {
			Params::load(is);
			is.ignore( 10 + 1);
			is >> has_bias;
			is.ignore(1);
		}
	};
	struct FullyConnected :worker {
		FullyConnected(const FullyConnectedParams& params):worker(params){
			has_bias = params.has_bias;
			weight.resize(inShape.size() * outShape.size());
			acc_gW.resize(weight.size(), float_t(0));
			if (has_bias) {
				bias.resize(outShape.size()); 
				acc_gB.resize(bias.size(), float_t(0));
			}
		}
		void forward_propagation(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT)override {
			//assert(dataIN.size() == dataOUT.size());
			for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
				const vec_t& in = dataIN[s];
				vec_t& out = dataOUT[s];
				//assert(weight.size() == in.size() * out.size());
				for (size_t i = 0, ied = out.size(); i < ied; ++i) {
					float_t tmp(0); size_t ib = i * in.size();
					for (size_t j = 0, jed = in.size(); j < jed; ++j) {
						tmp += weight[ib + j] * in[j];
					}
					out[i] = tmp;
				}
				if (has_bias) {
					for (size_t i = 0, ied = out.size(); i < ied; ++i) {
						out[i] += bias[i];
					}
				}
			}
		}
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		) {
			//assert(dataIN.size() == dataOUT.size());
			//assert(dataOUT.size() == gradIN.size());
			//assert(dataIN.size() == gradOUT.size());
			if (gradW.size() != gradIN.size()) {
				gradW.resize(gradIN.size());
			}
			if (has_bias && gradB.size() != gradIN.size()) {
				gradB.resize(gradIN.size());
			}
			for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
				const vec_t& in = dataIN[s];
				const vec_t& gin = gradIN[s];
				vec_t& gout = gradOUT[s];
				vec_t& gW = gradW[s];
				if (gW.size() != weight.size()) {
					gW.resize(weight.size());
				}
				if (has_bias) {
					vec_t& gB = gradB[s];
					gB.assign(gin.begin(), gin.end());
				}
				//assert(gout.size() == in.size());
				//assert(gW.size() == gout.size() * gin.size());
				//assert(gB.size() == gin.size());
				std::fill(gout.begin(), gout.end(), float_t(0));
				for (size_t i = 0, ied = gin.size(); i < ied; ++i) {
					float_t gii = gin[i];  size_t ib = i * gout.size();
					for (size_t j = 0, jed = gout.size(); j < jed; ++j) {
						gW[ib + j] = in[j] * gii;
						gout[j] += weight[ib + j] * gii;
					}
				}
			}

		}
		void update_weights(size_t accumulate_tot,float_t alpha)override {
			for (size_t s = 0, sed = gradW.size(); s < sed; ++s) {
				const vec_t& gWs = gradW[s];
				for (size_t i = 0; i < acc_gW.size(); ++i) {
					acc_gW[i] += gWs[i];
				}
				if (has_bias) {
					const vec_t& gBs = gradB[s];
					for (size_t i = 0; i < acc_gB.size(); ++i) {
						acc_gB[i] += gBs[i];
					}
				}
			}
			if (accumulate_tot) {
				float_t div = alpha / float_t(accumulate_tot);
				for (size_t i = 0; i < weight.size(); ++i) {
					weight[i] -= div * acc_gW[i];
					acc_gW[i] = 0.0;
				}
				if (has_bias) {
					for (size_t i = 0; i < bias.size(); ++i) {
						bias[i] -= div * acc_gB[i];
						acc_gB[i] = 0.0;
					}
				}
			}
		}
		void reset_weights()override {
			float_t fanin_factor = std::sqrt(2.0 / inShape.size());
			gaussian_rand(weight.begin(), weight.end(), 0, 1);
			std::for_each(weight.begin(), weight.end(), [=](auto& it) {it *= fanin_factor; });
			std::fill(bias.begin(), bias.end(), float_t(0));
		}
		 void save(std::ostream& os) override{
			//auto ttt = os.tellp();
			for ( auto& it : weight) {
				os.write((char*)&it, sizeof(it));
			}
			for ( auto& it : bias) {
				os.write((char*)&it, sizeof(it));
			}
		}
		 void load(std::istream& is) override {
			 //auto ttt = is.tellg();
			for ( auto& it : weight) {
				is.read((char*)&it, sizeof(it));
			}
			for ( auto& it : bias) {
				is.read((char*)&it, sizeof(it));
			}
		}
		vec_t weight, bias;
		std::vector<vec_t>gradW;
		std::vector<vec_t>gradB;
		vec_t acc_gW, acc_gB;
		bool has_bias{true};
	};
	inline worker* FullyConnectedConstruct(Params* params) {
		auto& real_p = *dynamic_cast<FullyConnectedParams*>(params);
		return new FullyConnected(real_p);
	}
}
#endif