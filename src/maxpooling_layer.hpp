#pragma once
#ifndef __MAXPOOLING_LAYER_H__
#define __MAXPOOLING_LAYER_H__
#include "worker.hpp"
namespace toyCNN {
	struct MaxPoolingParams :Params {
		size_t win_w, win_h;
		PaddingType pad_type;
		size_t stride_w, stride_h;
		bool only_one;

		void save(std::ostream& os) override {
			Params::save(os);
			os << with_w("pool_w", 8) << with_w("pool_h", 8) << with_w("padding", 10) << with_w("stride_w", 10) << with_w("stride_h", 10) << with_w("only_one", 10) << '\n';
			os << with_w(win_w, 8) << with_w(win_h, 8) << with_w((int)pad_type, 10) << with_w(stride_w, 10) << with_w(stride_h, 10) << with_w(only_one, 10) << '\n';
		 }
		void load(std::istream& is) override {
			Params::load(is);
			is.ignore(8 + 8 + 10 + 10 + 10 + 10+1);
			is >> win_w >> win_h >> (int&)pad_type >> stride_w >> stride_h >> only_one;
			is.ignore(1);
		 }
		MaxPoolingParams(shape3d inshape_, size_t win_w_, size_t win_h_,  size_t stride_w_ = 1, size_t stride_h_ = 1, PaddingType pad_type_ = PaddingType::vaild_padding,bool only_one_=true) :Params(KindOfWork::MAX_POOLING) {
			inShape = inshape_;
			win_w = win_w_;
			win_h = win_h_;
			pad_type = pad_type_;
			stride_w = stride_w_;
			stride_h = stride_h_;
			only_one = only_one_;
			switch (pad_type) {
			case PaddingType::vaild_padding:
				outShape.reshape((inShape.width_ - win_w) / stride_w + 1, (inShape.height_ - win_h) / stride_h + 1, inShape.depth_);
				break;
			case PaddingType::same_default_padding:
				outShape.reshape((inShape.width_ - 1) / stride_w + 1, (inShape.height_ - 1) / stride_h + 1, inShape.depth_);
				break;
			}
		}
		MaxPoolingParams(shape3d inshape_, size_t win_side, PaddingType pad_type_ = PaddingType::vaild_padding,bool only_one_=true) :MaxPoolingParams(inshape_, win_side, win_side, win_side, win_side, pad_type_,only_one_) {}

	};
	struct MaxPooling:worker {
		MaxPooling(const MaxPoolingParams& params) :worker(params) {
			pool_w = params.win_w;
			pool_h = params.win_h;
			padding = params.pad_type;
			stride_w = params.stride_w;
			stride_h = params.stride_h;
			only_one = params.only_one;
			switch (padding) {
			case PaddingType::vaild_padding:
				padded_w = inShape.width_;
				padded_h = inShape.height_;
				break;
			case PaddingType::same_default_padding:
				padded_w = inShape.width_ + pool_w - 1;
				padded_h = inShape.height_ + pool_h - 1;
				break;
			}
		}
		void forward_propagation_(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT) {
			if (only_one) {
				size_t padded_insize = inShape.depth_ * padded_w * padded_h;
				if (chosen.size() != dataIN.size()) {
					chosen.resize(dataIN.size(),std::vector<size_t>(padded_insize));
				}
				if(chosen[0].size()!= padded_insize)
				for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
					if (chosen[s].size() != padded_insize) {
						chosen[s].resize(padded_insize);
					}
				}
				for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
					const vec_t& in = dataIN[s];
					vec_t& out = dataOUT[s];
					auto& chosen_s = chosen[s];
					std::fill(out.begin(), out.end(), std::numeric_limits<float_t>::lowest());
					for (size_t c = 0, ced = inShape.depth_; c < ced; ++c) {
						size_t icb = c * padded_w * padded_h, ocb = c * outShape.area();
						for (size_t pool_row = 0; pool_row < pool_h; ++pool_row) {
							for (size_t y = 0, iy = 0; iy + pool_h <= padded_h; ++y, iy += stride_h) {
								for (size_t x = 0, ix = 0; ix + pool_w <= padded_w; ++x, ix += stride_w) {
									float_t& it = out[ocb + y * outShape.width_ + x];
									for (size_t pool_col = 0; pool_col < pool_w; ++pool_col) {
										float_t that = in[icb + (iy + pool_row) * padded_w + (ix + pool_col)];
										that > it ? chosen_s[ocb + y * outShape.width_ + x]= icb + (iy + pool_row) * padded_w + (ix + pool_col), it = that : it;
									}
								}
							}
						}
					}
				}
			}else{
				for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
					const vec_t& in = dataIN[s];
					vec_t& out = dataOUT[s];
					std::fill(out.begin(), out.end(), std::numeric_limits<float_t>::lowest());
					for (size_t c = 0, ced = inShape.depth_; c < ced; ++c) {
						size_t icb = c * padded_w * padded_h, ocb = c * outShape.area();
						for (size_t pool_row = 0; pool_row < pool_h; ++pool_row) {
							for (size_t y = 0, iy = 0; iy + pool_h <= padded_h; ++y, iy += stride_h) {
								for (size_t x = 0, ix = 0; ix + pool_w <= padded_w; ++x, ix += stride_w) {
									float_t& it = out[ocb + y * outShape.width_ + x];
									for (size_t pool_col = 0; pool_col < pool_w; ++pool_col) {
										float_t that = in[icb + (iy + pool_row) * padded_w + (ix + pool_col)];
										that > it ? it = that : it;
									}
								}
							}
						}
					}
				}
			}
		}
		void back_propagation_(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT,
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN
		) {
			if (only_one) {
				for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
					const vec_t& in = dataIN[s];
					const vec_t& gin = gradIN[s];
					vec_t& gout = gradOUT[s];
					const auto& chosen_s = chosen[s];
					std::fill(gout.begin(), gout.end(), float_t(0));
					for (size_t c = 0, ced = inShape.depth_; c < ced; ++c) {
						size_t icb = c * padded_w * padded_h, ocb = c * outShape.area();
						//for (size_t pool_row = 0; pool_row < pool_h; ++pool_row) {
							for (size_t y = 0, iy = 0; iy + pool_h <= padded_h; ++y, iy += stride_h) {
								for (size_t x = 0, ix = 0; ix + pool_w <= padded_w; ++x, ix += stride_w) {
									size_t it = chosen_s[ocb + y * outShape.width_ + x], gii = gin[ocb + y * outShape.width_ + x];
									//for (size_t pool_col = 0; pool_col < pool_w; ++pool_col) {
										//gout[icb + (iy + pool_row) * padded_w + (ix + pool_col)] +=
										//	((it == icb + (iy + pool_row) * padded_w + (ix + pool_col)) ? gii : float_t(0));
									//}
									gout[it] += gii;
								}
							}
						//}
					}
				}

			}else{
				for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
					const vec_t& in = dataIN[s];
					const vec_t& out = dataOUT[s];
					const vec_t& gin = gradIN[s];
					vec_t& gout = gradOUT[s];
					std::fill(gout.begin(), gout.end(), float_t(0));
					for (size_t c = 0, ced = inShape.depth_; c < ced; ++c) {
						size_t icb = c * padded_w * padded_h, ocb = c * outShape.area();
						for (size_t pool_row = 0; pool_row < pool_h; ++pool_row) {
							for (size_t y = 0, iy = 0; iy + pool_h <= padded_h; ++y, iy += stride_h) {
								for (size_t x = 0, ix = 0; ix + pool_w <= padded_w; ++x, ix += stride_w) {
									float_t it = out[ocb + y * outShape.width_ + x], gii = gin[ocb + y * outShape.width_ + x];
									for (size_t pool_col = 0; pool_col < pool_w; ++pool_col) {
										gout[icb + (iy + pool_row) * padded_w + (ix + pool_col)] +=
											((it == in[icb + (iy + pool_row) * padded_w + (ix + pool_col)]) ? gii : float_t(0));
									}
								}
							}
						}
					}
				}
			}
		}
		void make_padding(const std::vector<vec_t>& dataIN) {
			switch (padding) {
			case PaddingType::same_default_padding:
				for (size_t s = 0, sed = dataIN.size(); s < sed; ++s) {
					const vec_t& in = dataIN[s];
					vec_t& padded_in = padded_dataIN[s];
					for (size_t c = 0, ced = inShape.depth_; c < ced; ++c) {
						size_t ib = c * inShape.area(), pib = c * padded_w * padded_h;
						std::fill(padded_in.begin() + pib, padded_in.begin() + pib + (pool_h - 1) / 2 * padded_w, std::numeric_limits<float_t>::lowest());
						for (size_t y = 0; y < inShape.height_; ++y) {
							auto it = in.begin() + ib + y * inShape.width_;
							auto p_it = padded_in.begin() + pib + (pool_h - 1) / 2 * padded_w + y * padded_w;
							for (size_t i = (pool_w - 1) / 2; i; --i)*(p_it++) = std::numeric_limits<float_t>::lowest();
							for (size_t i = inShape.width_; i; --i)*(p_it++) = *(it++);
							for (size_t i = (pool_w - 1) / 2 + ((pool_w & 1) ^ 1); i; --i)*(p_it++) = std::numeric_limits<float_t>::lowest();
						}
						std::fill(padded_in.begin() + pib + (inShape.height_ + (pool_h - 1) / 2 ) * padded_w, padded_in.begin() + pib + padded_h * padded_w, std::numeric_limits<float_t>::lowest());
					}
				}
				break;
			}
		}
		void forward_propagation(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT)override {
			if (padding == PaddingType::vaild_padding) {
				forward_propagation_(dataIN, dataOUT); return;
			}
			size_t padded_insize = inShape.depth_ * padded_w * padded_h;
			if (padded_dataIN.size() != dataIN.size()) {
				padded_dataIN.resize(dataIN.size(),vec_t(padded_insize));
			}
			if(padded_dataIN[0].size()!= padded_insize)
			for (size_t s = 0, sed = padded_dataIN.size(); s < sed; ++s) {
				if (padded_dataIN[s].size() != padded_insize) {
					padded_dataIN[s].resize(padded_insize);
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
						size_t gob = c * inShape.area(), pgob = c * padded_w * padded_h+(pool_h-1)/2*padded_w;
						for (size_t y = 0; y < inShape.height_; ++y) {
							auto it = gout.begin() + gob + y * inShape.width_;
							auto p_it = padded_gout.begin() + pgob + y * padded_w+(pool_w-1)/2;
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
			if (padding == PaddingType::vaild_padding) {
				back_propagation_(dataIN, dataOUT, gradOUT, gradIN); return;
			}
			size_t padded_insize = inShape.depth_ * padded_w * padded_h;
			if (padded_gradOUT.size() != padded_dataIN.size()) {
				padded_gradOUT.resize(padded_dataIN.size(),vec_t(padded_insize));
			}
			if (padded_gradOUT[0].size() != padded_insize) {
				for (size_t s = 0, sed = padded_dataIN.size(); s < sed; ++s) {
					if (padded_gradOUT[s].size() != padded_insize) {
						padded_gradOUT[s].resize(padded_insize);
					}
				}
			}
			back_propagation_(padded_dataIN, dataOUT, padded_gradOUT, gradIN);
			make_unpadding(gradOUT);
		}
		bool only_one;
		std::vector<std::vector<size_t>>chosen;
		std::vector<vec_t>padded_dataIN;
		std::vector<vec_t>padded_gradOUT;
		size_t padded_w, padded_h;
		size_t pool_w, pool_h, stride_w, stride_h;
		PaddingType padding;
	};
	inline worker* MaxPoolingConstruct(Params* params) {
		auto& real_p = *dynamic_cast<MaxPoolingParams*>(params);
		return new MaxPooling(real_p);
	}
}
#endif