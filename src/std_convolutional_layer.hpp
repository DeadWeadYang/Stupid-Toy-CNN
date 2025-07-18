#pragma once
#ifndef __STD_CONVOLUTIONAL_LAYER_H__
#define __STD_CONVOLUTIONAL_LAYER_H__
#include "worker.hpp"
namespace toyCNN {
	struct STDConvolutionalParams :Params {
		size_t win_w, win_h, out_channels;
		PaddingType pad_type;
		bool has_bias;
		size_t stride_w, stride_h;
		void save(std::ostream& os) override {
			Params::save(os);
			os << with_w("pool_w", 8) << with_w("pool_h", 8)<<with_w("out_channels",15) << with_w("padding", 10) << with_w("stride_w", 10) << with_w("stride_h", 10) << '\n';
			os << with_w(win_w, 8) << with_w(win_h, 8) <<with_w(out_channels,15)<< with_w((int)pad_type, 10) << with_w(stride_w, 10) << with_w(stride_h, 10)  << '\n';
		}
		void load(std::istream& is) override {
			Params::load(is);
			is.ignore(8 + 8 + 15 + 10 + 10 + 10 + 1);
			is >> win_w >> win_h>>out_channels >> (int&)pad_type >> stride_w >> stride_h ;
			is.ignore(1);
		}
		STDConvolutionalParams(shape3d inshape_, size_t win_w_, size_t win_h_, size_t out_channels_, size_t stride_w_ = 1, size_t stride_h_ = 1, PaddingType pad_type_ = PaddingType::vaild_padding, bool has_bias_ = true) :Params(KindOfWork::STD_CONVOLUTIONAL){
			inShape = inshape_;
			win_w = win_w_; 
			win_h = win_h_;
			out_channels = out_channels_;
			pad_type = pad_type_;
			has_bias = has_bias_;
			stride_w = stride_w_;
			stride_h = stride_h_;
			switch (pad_type) {
			case PaddingType::vaild_padding:
				outShape.reshape((inShape.width_ - win_w) / stride_w + 1, (inShape.height_ - win_h) / stride_h + 1, out_channels_);
				break;
			case PaddingType::same_default_padding:
				outShape.reshape((inShape.width_ - 1) / stride_w + 1, (inShape.height_ - 1) / stride_h + 1, out_channels_);
				break;
			}
		}
		STDConvolutionalParams(shape3d inshape_, size_t win_side, size_t out_channels_, size_t stride_ = 1, PaddingType pad_type_ = PaddingType::vaild_padding, bool has_bias_ = true) :STDConvolutionalParams( inshape_,  win_side,  win_side,  out_channels_,  stride_,  stride_,  pad_type_ ,  has_bias_ ){}

	};
	struct STDConvolutional :worker {
		STDConvolutional(const STDConvolutionalParams& params) :worker(params) {
			conv_w = params.win_w;
			conv_h = params.win_h;
			padding = params.pad_type;
			has_bias = params.has_bias;
			stride_w = params.stride_w;
			stride_h = params.stride_h;
			conv_filter.resize(outShape.depth_, vec_t(inShape.depth_ * conv_w * conv_h));
			acc_gW.resize(outShape.depth_, vec_t(inShape.depth_ * conv_w * conv_h, float_t(0)));
			switch (padding) {
			case PaddingType::vaild_padding:
				padded_w = inShape.width_;
				padded_h = inShape.height_;
				break;
			case PaddingType::same_default_padding:
				padded_w = inShape.width_ + conv_w - 1;
				padded_h = inShape.height_ + conv_h - 1;
				break;
			}
			if (has_bias) {
				bias.resize(outShape.size());
				acc_gB.resize(bias.size(), float_t(0));
			}
		}
		void forward_propagation_(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT) {
			for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
				const vec_t& in = dataIN[s];
				vec_t& out = dataOUT[s];
				std::fill(out.begin(), out.end(), float_t(0));
				for (size_t oc = 0, oced = outShape.depth_, o_area = outShape.area(); oc < oced; ++oc) {
					size_t ocb = oc * o_area; const vec_t& conv_f = conv_filter[oc];
					for (size_t ic = 0, iced = inShape.depth_, i_area = padded_w * padded_h; ic < iced; ++ic) {
						size_t icb = ic * i_area,conv_b=ic*conv_w*conv_h;
						for (size_t conv_row = 0; conv_row < conv_h; ++conv_row) {
							for (size_t y = 0, iy = 0; iy + conv_h <= padded_h; ++y, iy += stride_h) {
								for (size_t x = 0, ix = 0; ix + conv_w <= padded_w; ++x, ix += stride_w) {
									for (size_t conv_col = 0; conv_col < conv_w; ++conv_col) {
										out[ocb + y * outShape.width_ + x] += in[icb + (iy + conv_row) * padded_w + (ix + conv_col)] * conv_f[conv_b + conv_row * conv_w + conv_col];
									}
								}
							}
						}
					}
				}
				if (has_bias) {
					for (size_t i = 0, ied = out.size(); i < ied; ++i) {
						out[i] += bias[i];
					}
				}
			}
		}
		void back_propagation_(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		) {
			for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
				const vec_t& in = dataIN[s];
				const vec_t& gin = gradIN[s];
				vec_t& gout = gradOUT[s];
				auto& gW = gradW[s];
				if (has_bias) {
					vec_t& gB = gradB[s];
					gB.assign(gin.begin(), gin.end());
				}
				std::fill(gout.begin(), gout.end(), float_t(0));
				for (size_t oc = 0, oced = outShape.depth_, o_area = outShape.area(); oc < oced; ++oc) {
					size_t ocb = oc * o_area; const vec_t& conv_f = conv_filter[oc]; 
					vec_t& gWc = gW[oc]; std::fill(gWc.begin(), gWc.end(), float_t(0));
					for (size_t ic = 0, iced = inShape.depth_, i_area = padded_w * padded_h; ic < iced; ++ic) {
						size_t icb = ic * i_area, conv_b = ic * conv_w * conv_h;
						for (size_t conv_row = 0; conv_row < conv_h; ++conv_row) {
							for (size_t y = 0, iy = 0; iy + conv_h <= padded_h; ++y, iy += stride_h) {
								for (size_t x = 0, ix = 0; ix + conv_w <= padded_w; ++x, ix += stride_w) {
									for (size_t conv_col = 0; conv_col < conv_w; ++conv_col) {
										float_t gii = gin[ocb + y * outShape.width_ + x];
										gWc[conv_b + conv_row * conv_w + conv_col] += gii * in[icb + (iy + conv_row) * padded_w + (ix + conv_col)];
										gout[icb + (iy + conv_row) * padded_w + (ix + conv_col)]+= gii* conv_f[conv_b + conv_row * conv_w + conv_col];
									}
								}
							}
						}
					}
				}
			}
		}
		void make_padding(const std::vector<vec_t>&dataIN) {
			switch (padding) {
			case PaddingType::same_default_padding:
				for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
					const vec_t& in = dataIN[s];
					vec_t& padded_in = padded_dataIN[s];
					for (size_t c = 0, ced = inShape.depth_; c < ced; ++c) {
						size_t ib = c * inShape.area(), pib = c * padded_w * padded_h;
						std::fill(padded_in.begin() + pib, padded_in.begin() + pib + (conv_h - 1) / 2 * padded_w, float_t(0));
						for (size_t y = 0; y < inShape.height_; ++y) {
							auto it = in.begin() + ib + y * inShape.width_;
							auto p_it = padded_in.begin() + pib + (conv_h - 1) / 2 * padded_w + y * padded_w;
							for (size_t i = (conv_w - 1) / 2; i; --i)*(p_it++) = float_t(0);
							for (size_t i = inShape.width_; i; --i)*(p_it++) = *(it++);
							for (size_t i = (conv_w - 1) / 2+((conv_w&1)^1); i; --i)*(p_it++) = float_t(0);
						}
						std::fill(padded_in.begin() + pib + (inShape.height_ + (conv_h - 1) / 2) * padded_w, padded_in.begin() + pib + padded_h * padded_w, float_t(0));
					}
				}
				break;
			}
		}
		void forward_propagation(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT)override {
			if (padding == PaddingType::vaild_padding) {
				forward_propagation_(dataIN, dataOUT); return;
			}

			if (padded_dataIN.size() != dataIN.size()) {
				padded_dataIN.resize(dataIN.size());
			}
			for (size_t s = 0, sed = padded_dataIN.size(); s < sed; ++s) {
				if (padded_dataIN[s].size() !=inShape.depth_*padded_w*padded_h) {
					padded_dataIN[s].resize(inShape.depth_ * padded_w * padded_h);
				}
			}
			make_padding(dataIN);
			forward_propagation_(padded_dataIN, dataOUT);
		}
		void make_unpadding(std::vector<vec_t>& gradOUT) {
			switch (padding) {
			case PaddingType::same_default_padding:
				for (size_t s = 0, sed = gradOUT.size(); s < sed; ++s) {
					const vec_t& padded_gout = padded_gradOUT[s];
					vec_t& gout = gradOUT[s];
					for (size_t c = 0, ced = inShape.depth_; c < ced; ++c) {
						size_t gob = c * inShape.area(), pgob = c * padded_w * padded_h +(conv_h-1)/2* padded_w;
						for (size_t y = 0; y < inShape.height_; ++y) {
							auto it = gout.begin() + gob + y * inShape.width_;
							auto p_it = padded_gout.begin() + pgob + y * padded_w +(conv_w-1)/2;
							for (size_t i = 0; i < inShape.width_; ++i)*(it++) = *(p_it++);
						}
					}
				}
				break;
			}
		}
		void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		)override {
			if (gradW.size() != gradIN.size()) {
				gradW.resize(gradIN.size());
			}
			if (gradB.size() != gradIN.size()) {
				gradB.resize(gradIN.size());
			}
			for (size_t s = 0, sed = gradIN.size(); s < sed; ++s) {
				if (gradW[s].size() != outShape.depth_) {
					gradW[s].resize(outShape.depth_);
				}
				if (gradB[s].size() != bias.size()) {
					gradB[s].resize(bias.size());
				}
				auto& gWs = gradW[s];
				for (size_t oc = 0, oced = outShape.depth_; oc < oced; ++oc) {
					if (gWs[oc].size() != inShape.depth_ * conv_w * conv_h) {
						gWs[oc].resize(inShape.depth_ * conv_w * conv_h);
					}
				}
			}
			if (padding == PaddingType::vaild_padding) {
				back_propagation_(dataIN, dataOUT, gradOUT, gradIN); return;
			}
			if (padded_gradOUT.size() != padded_dataIN.size()) {
				padded_gradOUT.resize(padded_dataIN.size());
			}
			for (size_t s = 0, sed = padded_dataIN.size(); s < sed; ++s) {
				if (padded_gradOUT[s].size() != inShape.depth_ * padded_w * padded_h) {
					padded_gradOUT[s].resize(inShape.depth_ * padded_w * padded_h);
				}
			}
			back_propagation_(padded_dataIN, dataOUT, padded_gradOUT, gradIN);
			make_unpadding(gradOUT);
		}
		void update_weights(size_t accumulate_tot,float_t alpha)override {
			for (size_t s = 0, sed = gradW.size(); s < sed; ++s) {
				const auto& gWs = gradW[s];
				for (size_t oc = 0; oc < acc_gW.size(); ++oc) {
					for (size_t i = 0, ied = acc_gW[oc].size(); i < ied; ++i) {
						acc_gW[oc][i] += gWs[oc][i];
					}
				}
				if(has_bias){
					const vec_t& gBs = gradB[s];
					for (size_t i = 0; i < acc_gB.size(); ++i) {
						acc_gB[i] += gBs[i];
					}
				}
			}
			if (accumulate_tot) {
				float_t div = alpha / float_t(accumulate_tot);
				for (size_t i = 0,jed=inShape.depth_*conv_w*conv_h; i < conv_filter.size(); ++i) {
					for (size_t j = 0; j < jed; ++j) {
						conv_filter[i][j] -= div * acc_gW[i][j];
						acc_gW[i][j] = 0.0;
					}
					
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
			float_t fanin_factor = std::sqrt(2.0 / (inShape.depth_*conv_h*conv_w));
			for (auto& iter : conv_filter) {
				gaussian_rand(iter.begin(), iter.end(), 0, fanin_factor);
				//gaussian_rand(iter.begin(), iter.end(), 0, 1);
				//std::for_each(iter.begin(), iter.end(), [=](auto& it) {it *= fanin_factor; });
			}
			std::fill(bias.begin(), bias.end(), float_t(0));
		}
		 void save(std::ostream& os) override{
			 //auto ttt = os.tellp();
			for ( auto& iter : conv_filter) {
				for ( auto& it : iter) {
					os.write((char*)&it, sizeof(it));
				}
			}
			for ( auto& it : bias) {
				os.write((char*)&it, sizeof(it));
			}
		}
		 void load(std::istream& is) override{
			 //auto ttt = is.tellg();
			for ( auto& iter : conv_filter) {
				for ( auto& it : iter) {
					is.read((char*)&it, sizeof(it));
				}
			}
			for (auto& it : bias) {
				is.read((char*)&it, sizeof(it));
			}
		}
		std::vector<vec_t> conv_filter; vec_t bias;
		std::vector<std::vector<vec_t>>gradW;
		std::vector<vec_t>acc_gW;vec_t acc_gB;
		std::vector<vec_t>gradB;
		std::vector<vec_t>padded_dataIN;
		std::vector<vec_t>padded_gradOUT;
		bool has_bias{ true };
		size_t conv_w, conv_h, stride_w,stride_h;
		size_t padded_w, padded_h;
		PaddingType padding;
	};
	inline worker* STDConvolutionalConstruct(Params* params) {
		auto& real_p = *dynamic_cast<STDConvolutionalParams*>(params);
		return new STDConvolutional(real_p);
	}
}
#endif