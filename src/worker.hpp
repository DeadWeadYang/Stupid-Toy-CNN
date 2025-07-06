#pragma once
#ifndef __WORKER_H__
#define __WORKER_H__
#include "util.hpp"
namespace toyCNN{
		enum class KindOfWork
		{
			ACTIVATION,FULLY_CONNECTED,STD_CONVOLUTIONAL,DEPTHWISE_CONVOLUTIONAL,BATCH_NORMALIZATION,MAX_POOLING
		};
		enum class PaddingType {
			vaild_padding,
			same_default_padding,
			//same_circular_padding,
			//full_default_padding,
			//full_circular_padding
		};
	struct Params {
		Params(KindOfWork kind_):kind(kind_){}
		virtual ~Params()=default;
		virtual void save(std::ostream& os) {
			os << '\n' << with_w("kind", 10) << '\n';
			os << with_w((int&)kind, 10) << '\n';
			os << with_w("in_shape", 30) << '\n';
			os << with_w(inShape.width_, 10) << with_w(inShape.height_, 10) << with_w(inShape.depth_, 10) << '\n';
			os<< with_w("out_shape", 30) <<'\n';
			os << with_w(outShape.width_, 10) << with_w(outShape.height_, 10) << with_w(outShape.depth_, 10) << '\n';
		}
		virtual void load(std::istream& is) {
			is.ignore(1 + 10 + 1);
			is >> (int&)kind;
			is.ignore(1);
			is.ignore(30+1);
			is >> inShape.width_ >> inShape.height_ >> inShape.depth_;
			is.ignore(1);
			is.ignore(30+1);
			is >> outShape.width_ >> outShape.height_ >> outShape.depth_;
			is.ignore(1);
		}
		KindOfWork kind;
		shape3d inShape, outShape;
	};
	struct worker {
		KindOfWork kind;
		worker(const Params& params) :kind(params.kind),inShape(params.inShape),outShape(params.outShape){}
		virtual ~worker() = default;
		virtual void forward_propagation(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT){}
		//virtual void forward_propagation(const std::vector<vec_t>& dataIN, std::vector<vec_t>& dataOUT, const std::vector<vec_t>&PdataIN){}
		virtual void back_propagation(
			const std::vector<vec_t>& dataIN,const std::vector<vec_t>& dataOUT, 
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>&gradIN
		){}
		virtual void update_weights(size_t accumulate_tot,float_t alpha){}
		virtual void reset_weights(){}
		virtual void save(std::ostream& os){}
		virtual void load(std::istream& is){}
		/*virtual void back_propagation(
			const std::vector<vec_t>& dataIN, const std::vector<vec_t>& dataOUT, 
			std::vector<vec_t>& gradOUT, const std::vector<vec_t>& gradIN,const std::vector<vec_t>&SgradIN
		) {}*/
		shape3d inShape, outShape;
	};
}
#endif