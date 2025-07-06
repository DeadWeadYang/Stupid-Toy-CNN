#pragma once
#ifndef __ISTREAM_WRAP_H__
#define __ISTREAM_WRAP_H__
#include<fstream>
#include<iostream>
#include<sstream>
#include<set>
struct seperators_customized_stream {
	seperators_customized_stream(std::istream& real_stream_) :real_stream(real_stream_) {}
	void reset_seperators() { seperators.clear(); }
	void add_to_seperators(char c) { seperators.insert(c); }
	void remove_from_seperators(char c) {
		auto it = seperators.find(c);
		if (it != seperators.end()) {
			seperators.erase(it);
		}
	}
	template<typename T>
	seperators_customized_stream& operator >>(T& sth) {
		while ( seperators.find(real_stream.peek())!=seperators.end())real_stream.ignore();
		real_stream >> sth;
		return *this;
	}
protected:
	std::istream& real_stream;
	std::set<char>seperators;
};
struct seperator_customized_stream {
	seperator_customized_stream(std::istream& real_stream_,char seperator_=' ') :real_stream(real_stream_), seperator(seperator_) {}
	void set_seperator(char c) { seperator = c; }
	template<typename T>
	seperator_customized_stream& operator >>(T& sth) {
		while (real_stream.peek() == seperator)real_stream.ignore();
		real_stream >> sth;
		return *this;
	}
protected:
	std::istream& real_stream; char seperator;
};
struct csv_reader :seperator_customized_stream{
	void set_seperator(char c) = delete;
	csv_reader(std::istream& real_stream_) :seperator_customized_stream(real_stream_, ',')  {};
	using seperator_customized_stream::operator>>;
};
#endif // !__ISTREAM_WRAP_H__
