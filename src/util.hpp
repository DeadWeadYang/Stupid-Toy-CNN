//this part is copied from tiny-dnn
//Copyright(c) 2013, Taiga Nomi
//All rights reserved.

#pragma once
#ifndef __MYCNN_UTIL_H__
#define __MYCNN_UTIL_H__
#include <vector>
#include <functional>
#include <random>
#include <type_traits>
#include <limits>
#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <sstream>
#include<fstream>
#include<iomanip>
#include<cmath>
#include "config.hpp"
#include "nn_error.hpp"

namespace toyCNN{
    template<typename T>
    struct with_w {
        const T& value_; const unsigned int& width_;
        //template<typename TT>
        //constexpr with_w( TT&& value,const unsigned int& width=0):value_(std::forward<TT>(value)),width_(width){}
        with_w(const T& value, const unsigned int& width = 0) :value_(value), width_(width) {}
        friend std::ostream& operator << (std::ostream& os, const with_w& I) {
            os << std::setw(I.width_) << I.value_; return os;
        }
        //friend std::istream& operator >> (std::istream& is, with_w<T>& I){
        //    is >> std::setw(I.width_) >> I.value_; return is;
        //}
    };
    //template<typename T>
    //with_w(const T&) -> with_w<T>;
    //template<typename T>
    //with_w(const T&, const unsigned int&)-> with_w<T>;
    //template<typename TT>with_w(TT&&, const unsigned int&) -> with_w<TT&&>;
	typedef std::vector<float_t> vec_t;
	//typedef std::vector<vec_t> tensor_t;
	enum class net_phase {
		train,
		test
	};
    class random_generator {
    public:
        static random_generator& get_instance() {
            static random_generator instance;
            return instance;
        }

        std::mt19937& operator()() {
            return gen_;
        }

        void set_seed(unsigned int seed) {
            gen_.seed(seed);
        }
    private:
        random_generator() : gen_(5489u/* standard default seed*/) {}
        std::mt19937 gen_;
    };
    template<typename T> inline
        typename std::enable_if<std::is_integral<T>::value, T>::type
        uniform_rand(T min, T max) {
        std::uniform_int_distribution<T> dst(min, max);
        return dst(random_generator::get_instance()());
    }

    template<typename T> inline
        typename std::enable_if<std::is_floating_point<T>::value, T>::type
        uniform_rand(T min, T max) {
        std::uniform_real_distribution<T> dst(min, max);
        return dst(random_generator::get_instance()());
    }

    template<typename T> inline
        typename std::enable_if<std::is_floating_point<T>::value, T>::type
        gaussian_rand(T mean, T sigma) {
        std::normal_distribution<T> dst(mean, sigma);
        return dst(random_generator::get_instance()());
    }

    inline void set_random_seed(unsigned int seed) {
        random_generator::get_instance().set_seed(seed);
    }

    template<typename Container>
    inline int uniform_idx(const Container& t) {
        return uniform_rand(0, int(t.size() - 1));
    }

    inline bool bernoulli(float_t p) {
        return uniform_rand(float_t(0), float_t(1)) <= p;
    }

    template<typename Iter>
    void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
        for (Iter it = begin; it != end; ++it)
            *it = uniform_rand(min, max);
    }

    template<typename Iter>
    void gaussian_rand(Iter begin, Iter end, float_t mean, float_t sigma) {
        for (Iter it = begin; it != end; ++it)
            *it = gaussian_rand(mean, sigma);
    }

    template<typename T>
    T* reverse_endian(T* p) {
        std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
        return p;
    }

    inline bool is_little_endian() {
        int x = 1;
        return *(char*)&x != 0;
    }


    template<typename T>
    size_t max_index(const T& vec) {
        auto begin_iterator = std::begin(vec);
        return std::max_element(begin_iterator, std::end(vec)) - begin_iterator;
    }

    template<typename T, typename U>
    U rescale(T x, T src_min, T src_max, U dst_min, U dst_max) {
        U value = static_cast<U>(((x - src_min) * (dst_max - dst_min)) / (src_max - src_min) + dst_min);
        return std::min(dst_max, std::max(value, dst_min));
    }

    inline void nop()
    {
        // do nothing
    }
    template <typename Container>
    size_t max_size(const Container& c) {
        typedef typename Container::value_type value_t;
        return std::max_element(c.begin(), c.end(),
            [](const value_t& left, const value_t& right) { return left.size() < right.size(); })->size();
    }

    inline std::string format_str(const char* fmt, ...) {
        static char buf[2048];

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
#ifdef _MSC_VER
#pragma warning(default:4996)
#endif
        return std::string(buf);
    }

    template <typename T>
    struct index3d {
        index3d(T width, T height, T depth) {
            reshape(width, height, depth);
        }

        index3d() : width_(0), height_(0), depth_(0) {}
        void reshape(T width, T height, T depth) {
            width_ = width;
            height_ = height;
            depth_ = depth;

            if ((long long)width * height * depth > std::numeric_limits<T>::max())
                throw nn_error(
                    format_str("error while constructing layer: layer size too large for tiny-cnn\nWidthxHeightxChannels=%dx%dx%d >= max size of [%s](=%d)",
                        width, height, depth, typeid(T).name(), std::numeric_limits<T>::max()));
        }

        T get_index(T x, T y, T channel) const {
            assert(x >= 0 && x < width_);
            assert(y >= 0 && y < height_);
            assert(channel >= 0 && channel < depth_);
            return (height_ * channel + y) * width_ + x;
        }

        T area() const {
            return width_ * height_;
        }

        T size() const {
            return width_ * height_ * depth_;
        }

        T width_;
        T height_;
        T depth_;
    };

    typedef index3d<size_t> shape3d;

    template <typename T>
    bool operator == (const index3d<T>& lhs, const index3d<T>& rhs) {
        return (lhs.width_ == rhs.width_) && (lhs.height_ == rhs.height_) && (lhs.depth_ == rhs.depth_);
    }

    template <typename T>
    bool operator != (const index3d<T>& lhs, const index3d<T>& rhs) {
        return !(lhs == rhs);
    }

    template <typename Stream, typename T>
    Stream& operator << (Stream& s, const index3d<T>& d) {
        s << d.width_ << "x" << d.height_ << "x" << d.depth_;
        return s;
    }

    template <typename Stream, typename T>
    Stream& operator << (Stream& s, const std::vector<index3d<T>>& d) {
        s << "[";
        for (size_t i = 0; i < d.size(); i++) {
            if (i) s << ",";
            s << "[" << d[i] << "]";
        }
        s << "]";
        return s;
    }
    template <typename T> inline T sqr(T value) { return value * value; }
}

#endif

