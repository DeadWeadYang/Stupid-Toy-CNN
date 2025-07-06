#pragma once
#ifndef __BATCH_NORMALIZATION_LAYER_H__
#define __BATCH_NORMALIZATION_LAYER_H__
#include "worker.hpp"
#include "depthwise_convolutional_layer.hpp"
namespace toyCNN {
	struct BatchNormalizationParams :Params {
		bool has_gamma_beta;
		float_t safe_epsilon, momentum;
		bool take_last,moving_stddev;
		void save(std::ostream& os) override {
			Params::save(os);
			os << with_w("has_gamma_beta", 20) << with_w("safe_eps", 12) << with_w("momentum", 12) << with_w("take_last", 12) << with_w("to_move_stddev", 20) << '\n';
			os << with_w(has_gamma_beta, 20) << with_w(safe_epsilon, 12) << with_w(momentum, 12) << with_w(take_last, 12) << with_w(moving_stddev, 20) << '\n';
		}
		void load(std::istream& is) override {
			Params::load(is);
			is.ignore(20 + 12 + 12 + 12 + 20 + 1);
			is >> has_gamma_beta >> safe_epsilon >> momentum >> take_last >> moving_stddev;
			is.ignore(1);
		}
		BatchNormalizationParams(shape3d inshape_,bool has_gamma_beta_=true,float_t eps_=0.05,bool take_last_=false,float_t momentum_=0.99,bool moving_stddev_=false) :Params(KindOfWork::BATCH_NORMALIZATION) {
			inShape = inshape_; outShape = inShape;
			has_gamma_beta = has_gamma_beta_;
			safe_epsilon = eps_; 
			take_last = take_last_;
			momentum = momentum_;
			moving_stddev = moving_stddev_;
		}
	};
	struct BatchNormalization:worker {
		BatchNormalization(const BatchNormalizationParams& params) :worker(params), gamma_beta(params.has_gamma_beta?DepthwiseConvolutionalParams(params.inShape, 1): DepthwiseConvolutionalParams(shape3d(0,0,0),0)) {
			has_gamma_beta = params.has_gamma_beta;
			safe_epsilon = params.safe_epsilon;
			take_last = params.take_last;
			momentum = params.momentum;
			moving_stddev = params.moving_stddev;
			if (!take_last) {
				running_mean.resize(inShape.depth_,float_t(0));
				running_stddev.resize(inShape.depth_,float_t(0));
				if (!moving_stddev) {
					running_variance.resize(inShape.depth_, float_t(0));
				}
			}
			current_mean.resize(inShape.depth_);
			current_variance.resize(inShape.depth_);
			current_stddev.resize(inShape.depth_);
		}
		void calc_dst(const std::vector<vec_t>&datas, size_t data_area, size_t data_channels, vec_t* mean, vec_t* var) {
			if (!mean && !var)return;
			bool tmp_mean = false;
			if (!mean) {
				mean = new vec_t(data_channels);
				tmp_mean = true;
			}
			std::fill(mean->begin(), mean->end(), float_t(0));
			for (size_t s = 0, sed = datas.size(); s < sed; ++s) {
				const vec_t& in = datas[s];
				for (size_t c = 0, ced = data_channels; c < ced; ++c) {
					size_t ib = c * data_area; float_t& it = (*mean)[c];
					for (size_t i = 0; i < data_area; ++i) {
						it += in[ib + i];
					}
				}
			}
			float_t divs = float_t(1)/datas.size() / data_area;
			std::for_each(mean->begin(), mean->end(), [divs](float_t& it) {it *= divs; });
			if (!var)goto return_part;
			std::fill(var->begin(), var->end(), float_t(0));
			for (size_t s = 0, sed = datas.size(); s < sed; ++s) {
				const vec_t& in = datas[s];
				for (size_t c = 0, ced = data_channels; c < ced; ++c) {
					size_t ib = c * data_area;
					float_t& it = (*var)[c];
					float_t mis = (*mean)[c];
					for (size_t i = 0; i < data_area; ++i) {
						it+=sqr(in[ib + i]-mis);
					}
				}
			}
			std::for_each(var->begin(), var->end(), [divs](float_t& it) {it *= divs; });
		return_part:
			if (tmp_mean)delete mean;
			return;
		}
		void calc_stddev(const vec_t& var, vec_t&stddev) {
			stddev.resize(var.size());
			std::transform(var.begin(), var.end(), stddev.begin(), [eps = this->safe_epsilon](float_t it) {return sqrt(it + eps); });
		}
		void forward_propagation_(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT) {
			vec_t& mean = current_mean; vec_t& var = current_variance; vec_t& stddev = current_stddev;
			if (cur_phase == net_phase::train) {
				calc_dst(dataIN, inShape.area(), inShape.depth_, &mean, &var);
				calc_stddev(var, stddev);
			}
			else if (!take_last) {
				if (!moving_stddev && !runned_stddev) {
					calc_stddev(running_variance, running_stddev);
					runned_stddev = true;
				}
				mean = running_mean;
				var = running_variance;
				stddev = running_stddev;
			}
			for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
				const vec_t& in = dataIN[s];
				vec_t& out = dataOUT[s];
				for (size_t c = 0; c < inShape.depth_; ++c) {
					size_t ib = c * inShape.area();
					float_t c_mean = mean[c];
					float_t c_stddev = stddev[c];
					for (size_t i = 0, ied = inShape.area(); i < ied; ++i) {
						out[ib + i] = (in[ib + i] - c_mean) / c_stddev;
					}
				}
			}
		}
		void forward_propagation(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT)override {
			if (has_gamma_beta) {
				if (x_hat.size() != dataIN.size()) {
					x_hat.resize(dataIN.size());
				}
				for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
					if (x_hat[s].size() != inShape.size()) {
						x_hat[s].resize(inShape.size());
					}
				}
				forward_propagation_(dataIN, x_hat);
				gamma_beta.forward_propagation(x_hat, dataOUT);
			}
			else {
				forward_propagation_(dataIN, dataOUT);
			}
		}
		std::vector<vec_t> delta_times_hat;
		vec_t mean_delta_times_hat, mean_delta;
		void back_propagation_(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		) {
			if (delta_times_hat.size() != dataOUT.size()) {
				delta_times_hat.resize(dataOUT.size(),vec_t(inShape.size()));
			}
			for (auto& iter : delta_times_hat) {
				if (iter.size() != inShape.size()) {
					iter.resize(inShape.size());
				}
			}
			if (mean_delta_times_hat.size() != inShape.depth_) {
				mean_delta_times_hat.resize(inShape.depth_);
			}
			if (mean_delta.size() != inShape.depth_) {
				mean_delta.resize(inShape.depth_);
			}
			for (size_t s = 0, sed = dataOUT.size(); s < sed; ++s) {
				const vec_t& s_xhat = dataOUT[s];
				const vec_t& s_ghat = gradIN[s];
				vec_t& s_dth = delta_times_hat[s];
				for (size_t i = 0, ied = inShape.size(); i < ied; ++i) {
					s_dth[i] = s_xhat[i] * s_ghat[i];
				}
			}
			calc_dst(delta_times_hat, inShape.area(), inShape.depth_, &mean_delta_times_hat, nullptr);
			calc_dst(gradIN, inShape.area(), inShape.depth_, &mean_delta, nullptr);
			for (size_t s = 0, sed = dataOUT.size(); s < sed; ++s) {
				const vec_t& s_xhat = dataOUT[s];
				const vec_t& s_ghat = gradIN[s];
				vec_t& gout = gradOUT[s];
				for (size_t c = 0; c < inShape.depth_; ++c) {
					//size_t ib = c * inShape.area();
					float_t c_stddev = current_stddev[c];
					float_t c_mdth = mean_delta_times_hat[c];
					float_t c_md = mean_delta[c];
					for (size_t i = c*inShape.area(), ied =i+ inShape.area(); i < ied; ++i) {
						gout[i] = (s_ghat[i] - c_mdth - s_xhat[i] * c_mdth) / c_stddev;
					}
				}
			}
		}
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			if (has_gamma_beta) {
				if (grad_hat.size() != x_hat.size()) {
					grad_hat.resize(x_hat.size());
				}
				for (size_t s = 0, sed = x_hat.size(); s < sed; ++s) {
					if (grad_hat[s].size() != inShape.size()) {
						grad_hat[s].resize(inShape.size());
					}
				}
				gamma_beta.back_propagation(x_hat, dataOUT, grad_hat, gradIN);
				back_propagation_(dataIN, x_hat, gradOUT, grad_hat);
			}
			else {
				back_propagation_(dataIN, dataOUT, gradOUT, gradIN);
			}
		}
		void update_weights(size_t accumulate_tot,float_t alpha)override {
			if (has_gamma_beta) {
				gamma_beta.update_weights(accumulate_tot,alpha);
			}
			if (!take_last) {
				const vec_t& current_sth = moving_stddev ? current_stddev : current_variance;
				 vec_t& running_sth = moving_stddev ? running_stddev : running_variance;
				for (size_t c = 0, ced = inShape.depth_; c < ced; ++c) {
					running_mean[c] += (current_mean[c] - running_mean[c]) * (1 - momentum);
					running_sth[c] += (current_sth[c] - running_sth[c]) * (1 - momentum);
				}
				if (!moving_stddev)runned_stddev = false;
			}
		}
		void reset_weights()override {
			if (has_gamma_beta) {
				gamma_beta.reset_weights();
			}
		}
		 void save(std::ostream& os) override{
			 //auto ttt = os.tellp();
			 if (!take_last) {
				for (const auto& it : running_mean) {
					os.write((char*)&it, sizeof(it));
				}
				if(!moving_stddev)
				for (const auto& it : running_variance) {
					os.write((char*)&it, sizeof(it));
				}
				for (const auto& it : running_stddev) {
					os.write((char*)&it, sizeof(it));
				}

			}
			for (const auto& it : current_mean) {
				os.write((char*)&it, sizeof(it));
			}
			for (const auto& it : current_variance) {
				os.write((char*)&it, sizeof(it));
			}
			for (const auto& it : current_stddev) {
				os.write((char*)&it, sizeof(it));
			}
			if (has_gamma_beta) {
				gamma_beta.save(os);
			}
		}
		 void load(std::istream& is) override{
			 //auto ttt = is.tellg();
			 if (!take_last) {
				for (auto& it : running_mean) {
					is.read((char*)&it, sizeof(it));
				}
				if(!moving_stddev)
				for (auto& it : running_variance) {
					is.read((char*)&it, sizeof(it));
				}
				for (auto& it : running_stddev) {
					is.read((char*)&it, sizeof(it));
				}
			 }
			for (auto& it : current_mean) {
				is.read((char*)&it, sizeof(it));
			}
			for (auto& it : current_variance) {
				is.read((char*)&it, sizeof(it));
			}
			for (auto& it : current_stddev) {
				is.read((char*)&it, sizeof(it));
			}
			if (has_gamma_beta) {
				gamma_beta.load(is);
			}
		}
		net_phase cur_phase{ net_phase::train };
		bool has_gamma_beta;
		float_t safe_epsilon, momentum;
		bool take_last, moving_stddev, runned_stddev{false};
		vec_t running_mean, running_variance,running_stddev;
		vec_t current_mean, current_variance, current_stddev;
		std::vector<vec_t>x_hat, grad_hat;
		DepthwiseConvolutional gamma_beta;
	};
	inline worker* BatchNormalizationConstruct(Params* params) {
		auto& real_p = *dynamic_cast<BatchNormalizationParams*>(params);
		return new BatchNormalization(real_p);
	}
}
#endif