#pragma once
#ifndef __NETWORK_H__
#define __NETWORK_H__
#include "util.hpp"
#include "all_layers.h"
#include "loss_function.hpp"
namespace toyCNN {
	struct learn_strategy {
		float_t learning_rate_start{0.1};
		float_t learning_rate_decay{0.1};
		float_t decay_rate{0.1};
		float_t bottom_rate{0.1};
		size_t start_epoch{ 1 }, decay_epoch{1};
		size_t cur_epoch{ 0 };
		inline float_t learning_rate() {
			++cur_epoch;
			return cur_epoch >= start_epoch ?
				(
					learning_rate_decay > bottom_rate ?
					(
						cur_epoch%decay_epoch==0?
						learning_rate_decay *= decay_rate
						:learning_rate_decay
					)
					: bottom_rate
				)
				: learning_rate_start;

		}
	};
	struct Comparer {
		virtual float_t compare(const vec_t& output, const vec_t& labels)const { return 0; }
	};
	struct one_hot_Comparer :Comparer {
		float_t compare(const vec_t& output, const vec_t& labels)const override{

			int predict = std::max_element(output.begin(), output.end()) - output.begin();
			int wanted = std::max_element(labels.begin(), labels.end()) - labels.begin();
			return predict == wanted ? 1 : 0;
		}
	};
	struct network {
		~network() {
			clear();
		}
		
		std::vector<Params*>ParamsOfLayers;
		std::vector<worker*>ActualLayers;
		std::vector<std::vector<vec_t> >dataStation;
		std::vector<std::vector<vec_t> >gradStation;
		std::vector<vec_t>for_labels;
		bool edited{true};
		shape3d inputShape;
		shape3d outputShape;
		void setInputShape(size_t w, size_t h, size_t d) {
			inputShape.reshape(w, h, d);
		}
		void setOutputShape(size_t w, size_t h, size_t d) {
			outputShape.reshape(w, h, d);
		}
		size_t real_size{ 0 };
		void clear() {
			for (auto& it : ParamsOfLayers)delete it;
			ParamsOfLayers.clear();
			for (auto& it : ActualLayers)delete it;
			ActualLayers.clear();
			dataStation.clear();
			real_size = 0;
		}
		shape3d last_outshape() {
			return real_size ? ParamsOfLayers[real_size - 1]->outShape : inputShape;
		}
		Params* top() { return real_size ? ParamsOfLayers[real_size - 1] : nullptr; }
		bool append(Params* l_p) {
			if (l_p->inShape != last_outshape())return false;
			edited = true;
			if (real_size == ParamsOfLayers.size()) {
				ParamsOfLayers.emplace_back(l_p); ++real_size;
			}
			else {
				delete ParamsOfLayers[real_size];
				ParamsOfLayers[real_size++] = l_p;
			}
			return edited; 
		}
		void drop() {
			edited = true;
			if (real_size)--real_size;
		}
		bool reconstruct() {
			if (last_outshape() != outputShape)return false;
			edited = false;
			for (auto& it : ActualLayers)delete it;
			ActualLayers.resize(real_size);
			for (size_t i = 0; i < real_size; ++i) {
				ActualLayers[i] = LayerConstruct(ParamsOfLayers[i]);
			}
			test_built = false;
			return true;
		}
		void rebuildStation(size_t batch_size) {
			if (dataStation.size() != real_size + 1) {
				dataStation.resize(real_size + 1);
			}
			if (gradStation.size() != real_size + 1) {
				gradStation.resize(real_size + 1);
			}
			size_t cur_size;
			for (size_t i = 0; i < real_size; ++i) {
				cur_size = ParamsOfLayers[i]->inShape.size();
				if (dataStation[i].size() != batch_size) {
					dataStation[i].resize(batch_size, vec_t(cur_size));
				}
				if (batch_size && dataStation[i][0].size() != cur_size) {
					for (auto& it : dataStation[i]) {
						it.resize(cur_size);
					}
				}
				if (gradStation[i].size() != batch_size) {
					gradStation[i].resize(batch_size, vec_t(cur_size));
				}
				if (batch_size && gradStation[i][0].size() != cur_size) {
					for (auto& it : gradStation[i]) {
						it.resize(cur_size);
					}
				}
			}
			cur_size = outputShape.size();
			if (dataStation[real_size].size() != batch_size) {
				dataStation[real_size].resize(batch_size, vec_t(cur_size));
			}
			if (batch_size && dataStation[real_size][0].size() != cur_size) {
				for (auto& it : dataStation[real_size]) {
					it.resize(cur_size);
				}
			}
			if (gradStation[real_size].size() != batch_size) {
				gradStation[real_size].resize(batch_size, vec_t(cur_size));
			}
			if (batch_size && gradStation[real_size][0].size() != cur_size) {
				for (auto& it : gradStation[real_size]) {
					it.resize(cur_size);
				}
			}
		}
		void reset_layers() {
			for (size_t i = 0; i < real_size; ++i) {
				ActualLayers[i]->reset_weights();
			}
		}
		void apply_grad(size_t acced_tot,float_t learning_rate) {
			for (size_t i = 0; i < real_size; ++i) {
				ActualLayers[i]->update_weights(acced_tot,learning_rate);
			}
		}
		typedef std::vector<vec_t>::const_iterator const_iter;
		void train_batch(const LossFunction& LossF, const_iter input_begin,const_iter labels_begin, size_t batch_size, size_t& accumulated, std::function<void(float_t)>callback_batch = nullptr ) {
			rebuildStation(batch_size);
			dataStation[0].assign(input_begin, input_begin + batch_size);
			for (size_t i = 0; i < real_size; ++i) {
				if (ActualLayers[i]->kind == KindOfWork::BATCH_NORMALIZATION) {
					dynamic_cast<BatchNormalization*>(ActualLayers[i])->cur_phase = net_phase::train;
				}
				ActualLayers[i]->forward_propagation(dataStation[i], dataStation[i + 1]);
			}
			for_labels.assign(labels_begin, labels_begin + batch_size);
			const auto& lastOutput = dataStation[real_size];
			if (callback_batch) {
				float_t batch_loss(0);
				for (size_t s = 0; s < batch_size; ++s) {
					batch_loss += LossF.f(lastOutput[s], for_labels[s]);
				}
				batch_loss /= batch_size;
				callback_batch(batch_loss);
			}
			for (size_t s = 0; s < batch_size; ++s) {
				gradStation[real_size][s] = LossF.df(lastOutput[s], for_labels[s]);
			}
			for (int i = real_size - 1; i >= 0; --i) {
				ActualLayers[i]->back_propagation(dataStation[i], dataStation[i + 1], gradStation[i], gradStation[i + 1]);
			}
			accumulated += batch_size;
		}
		bool train(const LossFunction&LossF,const std::vector<vec_t>& input, const std::vector<vec_t>& labels, size_t batch_size, size_t accumulate_times, size_t epoch,learn_strategy strategy, bool reset_weights = false, std::function<void()>callback_epoch = nullptr, std::function<void(float_t)>callback_batch = nullptr) {
			if (input.size() != labels.size()) {
				throw nn_error("input size not match labels size");
			}
			if (!input.size())return false;
			size_t sample_cnt = input.size();
			for (size_t s = 1; s < sample_cnt; ++s) {
				if (input[s].size() != input[0].size() || labels[s].size() != labels[0].size()) {
					throw nn_error("shape of input or labels not keep consistent");
				}
			}
			if (outputShape.size() != labels[0].size()) {
				throw nn_error("current output shape not match label shape");
			}

			if (edited) {
				bool constructflag=reconstruct();
				if (!constructflag)return false;
				reset_layers();
			}
			if (reset_weights)reset_layers();
			if (!accumulate_times)accumulate_times = 1;
			size_t acced_tot = 0, acced_cnt = 0;
			for (size_t epoch_i = 1; epoch_i <= epoch; ++epoch_i) {
				float_t learning_rate = strategy.learning_rate();
				assert(learning_rate > 0);
				for (size_t s = 0; s < sample_cnt; s += batch_size) {
					train_batch(LossF,input.begin() + s, labels.begin() + s, std::min(batch_size, sample_cnt - s),acced_tot,callback_batch);
					++acced_cnt; apply_grad(acced_cnt == accumulate_times ? acced_tot : 0,learning_rate);
					if (acced_cnt == accumulate_times) {
						acced_cnt = acced_tot = 0;
					}
				}
				if (acced_cnt) {
					apply_grad(acced_tot,learning_rate);
					acced_cnt = acced_tot = 0;
				}
				if (callback_epoch) {
					callback_epoch();
				}
				
			}
			return true;
		}
		std::vector<std::vector<vec_t>>testStation;
		bool test_built{ false };
		void buildTest() {
			if (test_built)return;
			test_built = true;
			if (testStation.size() != real_size + 1) {
				testStation.resize(real_size + 1,std::vector<vec_t>(1));
			}
			size_t cur_size;
			for (size_t i = 0; i < real_size; ++i) {
				cur_size = ParamsOfLayers[i]->inShape.size();
				if (testStation[i][0].size() != cur_size) {
					testStation[i][0].resize(cur_size);
				}
			}
			cur_size = outputShape.size();
			if (testStation[real_size][0].size() != cur_size) {
				testStation[real_size][0].resize(cur_size);
			}
		}
		float_t test_once(const Comparer&comp,const vec_t&input,const vec_t&labels) {
			testStation[0][0] = input;
			for (size_t i = 0; i < real_size; ++i) {
				if (ActualLayers[i]->kind == KindOfWork::BATCH_NORMALIZATION) {
					dynamic_cast<BatchNormalization*>(ActualLayers[i])->cur_phase = net_phase::test;
				}
				ActualLayers[i]->forward_propagation(testStation[i], testStation[i + 1]);
			}
			const auto& lastOutput = testStation[real_size][0];
			return comp.compare(lastOutput, labels);
		}
		bool test(const Comparer& comp, const std::vector<vec_t>& input, const std::vector<vec_t>& labels, float_t& accuracy, std::vector<float_t>* result=nullptr) {
			if (input.size() != labels.size()) {
				throw nn_error("input size not match labels size");
			}
			if (!input.size())return false;
			size_t sample_cnt = input.size();
			for (size_t s = 1; s < sample_cnt; ++s) {
				if (input[s].size() != input[0].size() || labels[s].size() != labels[0].size()) {
					throw nn_error("shape of input or labels not keep consistent");
				}
			}
			if (outputShape.size() != labels[0].size()) {
				throw nn_error("current output shape not match label shape");
			}

			if (edited) {
				throw nn_error("edited or never trained");
			}
			buildTest();
			if(result)result->resize(input.size());
			accuracy = 0;
			for (size_t s = 0; s < input.size(); ++s) {
				float_t res=test_once(comp,input[s], labels[s]);
				accuracy += res;
				if (result)(*result)[s] = res;
			}
			accuracy /= input.size();
			return true;
		}
		void save(std::ostream& os) {
			if (edited)return;
			for (size_t i = 0; i < real_size; ++i) {
				ActualLayers[i]->save(os);
			}
		}
		void load(std::istream& is) {
			if (edited)return;
			for (size_t i = 0; i < real_size; ++i) {
				ActualLayers[i]->load(is);
			}
		}
		void save(const std::string& file_name) {
			if (edited)return;
			std::ofstream file(file_name + ".out", std::ios::binary);
			std::ofstream file_params(file_name + "_params.out");
			for (size_t i = 0; i < real_size; ++i) {
				ParamsOfLayers[i]->save(file_params);
				ActualLayers[i]->save(file);
			}
			file.close();
			file_params.close();
		}
	};
}
#endif