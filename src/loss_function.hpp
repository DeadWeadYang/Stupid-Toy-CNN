#pragma once
#ifndef __LOSS_FUNCTION_H__
#define __LOSS_FUNCTION_H__
#include "util.hpp"
namespace toyCNN {
	float_t clipping(float_t x,float_t minimum_bound, float_t maximum_bound) {
		constexpr float_t clip_eps = 1e-15;
		return std::max(minimum_bound + clip_eps, std::min(x, maximum_bound - clip_eps));
	}
	struct LossFunction {
		virtual float_t f(const vec_t& y_hat, const vec_t& goal)const { return float_t(0); }
		virtual vec_t df(const vec_t& y_hat, const vec_t& goal)const { return vec_t(); }
	};
	struct MeanSquaredError :LossFunction {
		float_t f(const vec_t& y_hat, const vec_t& goal)const override{
			float_t res_loss(0);
			for (size_t i = 0, ied = goal.size();i<ied; ++i) {
				res_loss += sqr(y_hat[i] - goal[i]);
			}
			res_loss /= goal.size();
			return res_loss;
		}
		vec_t df(const vec_t& y_hat, const vec_t& goal)const override{
			vec_t grad(goal.size());
			float_t div = float_t(2) / float_t(goal.size());
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				grad[i] = div * (y_hat[i] - goal[i]);
			}
			return grad;
		}
	};
	struct BinaryCrossEntropy:LossFunction {

		float_t f(const vec_t& y_hat, const vec_t& goal)const override {
			float_t res_loss(0);
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				float_t clipped_y_hat = clipping(y_hat[i], 0.0, 1.0);
				res_loss += goal[i] * std::log(clipped_y_hat) + (float_t(1) - goal[i]) * std::log(float_t(1) - clipped_y_hat);
			}
			res_loss /= goal.size(); res_loss = -res_loss;
			return res_loss;
		}
		vec_t df(const vec_t& y_hat, const vec_t& goal)const override {
			vec_t grad(goal.size());
			float_t div = float_t(-1) / float_t(goal.size());
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				float_t clipped_y_hat = clipping(y_hat[i], 0.0, 1.0);
				grad[i] = div * (goal[i]/ clipped_y_hat -(float_t(1)-goal[i])/(float_t(1)- clipped_y_hat));
			}
			return grad;
		}
	};
	struct BinaryCrossEntropyWithSigmoid :LossFunction{

		float_t f(const vec_t& y_hat, const vec_t& goal)const override {
			float_t res_loss(0);
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				res_loss += goal[i]*y_hat[i]-std::log(float_t(1)+std::exp(y_hat[i]));
			}
			res_loss /= goal.size(); res_loss = -res_loss;
			return res_loss;
		}
		vec_t df(const vec_t& y_hat, const vec_t& goal)const override {
			vec_t grad(goal.size());
			float_t div = float_t(-1) / float_t(goal.size());
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				grad[i] = div * (goal[i]-float_t(1)/(float_t(1)+std::exp(-y_hat[i])) );
			}
			return grad;
		}
	};
	struct CrossEntropy :LossFunction {

		float_t f(const vec_t& y_hat, const vec_t& goal)const override {
			float_t res_loss(0);
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				float_t clipped_y_hat = clipping(y_hat[i], 0.0, 1.0);
				res_loss += goal[i] * std::log(clipped_y_hat);
			}
			res_loss =-res_loss;
			return res_loss;
		}
		vec_t df(const vec_t& y_hat, const vec_t& goal)const override {
			vec_t grad(goal.size());
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				float_t clipped_y_hat = clipping(y_hat[i], 0.0, 1.0);
				grad[i] = -goal[i] /(clipped_y_hat);
			}
			return grad;
		}
	};
	struct CrossEntropyWithSoftmax :LossFunction {

		float_t f(const vec_t& y_hat, const vec_t& goal)const override {
			float_t alpha = *std::max_element(y_hat.begin(), y_hat.end());
			float_t denom(0);
			for (const auto& it : y_hat) {
				denom += std::exp(it - alpha);
			}
			denom = std::log(denom);
			denom += alpha;
			float_t res_loss(0);
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				res_loss += goal[i] * (y_hat[i] - denom);
			}
			res_loss = -res_loss;
			return res_loss;
		}
		vec_t df(const vec_t& y_hat, const vec_t& goal)const override {
			vec_t grad(goal.size());
			float_t should_be_one(0);
			for (const auto& it : goal) {
				should_be_one += it;
			}
			float_t alpha = *std::max_element(y_hat.begin(), y_hat.end());
			float_t denom(0);
			for (const auto& it : y_hat) {
				denom += std::exp(it - alpha);
			}
			for (size_t i = 0, ied = goal.size(); i < ied; ++i) {
				grad[i] = should_be_one * std::exp(y_hat[i] - alpha) / denom - goal[i];
			}
			return grad;
		}
	};
}
#endif