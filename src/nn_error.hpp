//this part is copied from tiny-dnn
//Copyright(c) 2013, Taiga Nomi
//All rights reserved.
#pragma once
#ifndef __NN_ERROR_H__
#define __NN_ERROR_H__
#include <exception>
#include <string>

namespace toyCNN {

    /**
     * basic exception class for tiny-cnn
     **/
    class nn_error : public std::exception {
    public:
        explicit nn_error(const std::string& msg) : msg_(msg) {}
        const char* what() const throw() override { return msg_.c_str(); }
    private:
        std::string msg_;
    };

    class nn_not_implemented_error : public nn_error {
    public:
        explicit nn_not_implemented_error(const std::string& msg = "not implemented") : nn_error(msg) {}
    };

}
#endif