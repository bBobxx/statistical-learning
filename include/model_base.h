//
// Created by wyb on 18-12-13.
// Update by wyb on 18-12-15, add operator for vector.

#ifndef MACHINE_LEARNING_MODEL_BASE_H
#define MACHINE_LEARNING_MODEL_BASE_H

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
using std::vector;
using std::cout;
using std::endl;
//this base class is for run
class Base{
protected:
    std::vector<std::vector<double>> inData;//从文件都的数据
    std::vector<std::vector<double>> trainData;//分割后的训练数据，里面包含真值
    std::vector<std::vector<double>> testData;
    unsigned long indim = 0;
    std::vector<std::vector<double>> trainDataF;//真正的训练数据，特征
    std::vector<std::vector<double>> testDataF;
    std::vector<double> trainDataGT;//真值
    std::vector<double> testDataGT;
public:
    virtual void getData(const std::string& filename)=0;
    virtual void run()=0;
    virtual ~Base(){};
    template <class T1, class T2>
    friend auto operator + (const vector<T1>& v1, const vector<T2>& v2)->vector<decltype(v1[0] + v2[0])>;
    template <class T1, class T2>
    friend auto operator - (const vector<T1>& v1, const vector<T2>& v2)->vector<decltype(v1[0] + v2[0])>;
    template <class T1, class T2>
    friend double operator * (const vector<T1>& v1, const vector<T2>& v2);
    template <class T1, class T2>
    friend auto operator / (const vector<T1>& v1, const vector<T2>& v2)->vector<decltype(v1[0] + v2[0])>;
    template <class T1, class T2>
    friend auto operator + (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 + v2[0])>;
    template <class T1, class T2>
    friend auto operator - (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 + v2[0])>;
    template <class T1, class T2>
    friend auto operator * (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 + v2[0])>;
    template <class T1, class T2>
    friend auto operator / (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 + v2[0])>;
    template <class T1, class T2>
    friend auto operator + (const vector<T1>& v1, const T2& arg2)->vector<decltype(v1[0] + arg2)>;
    template <class T1, class T2>
    friend auto operator - (const vector<T1>& v1, const T2& arg2)->vector<decltype(v1[0] + arg2)>;
    template <class T1, class T2>
    friend auto operator * (const vector<T1>& v1, const T2& arg2)->vector<decltype(v1[0] + arg2)>;
    template <class T1, class T2>
    friend auto operator / (const vector<T1>& v1, const T2& arg2)->vector<decltype(v1[0] + arg2)>;
};
template <class T1, class T2>
auto operator + (const vector<T1>& v1, const vector<T2>& v2) ->vector<decltype(v1[0] + v2[0])> {

    if (v1.size() != v2.size()) {
        cout << "two vector must have same size." << endl;
        throw v1.size() != v2.size();
    }
    if (v1.empty()) {
        cout << "vector must not empty." << endl;
        throw v1.empty();
    }
    vector<decltype(v1[0] + v2[0])> re(v1.size());
    for (int i = 0; i < v1.size(); ++i) {
        re[i] = v1[i] + v2[i];
    }
    return re;
}
template <class T1, class T2>
auto operator - (const vector<T1>& v1, const vector<T2>& v2)->vector<decltype(v1[0] + v2[0])> {
    if (v1.size() != v2.size()) {
        cout << "two vector must have same size." << endl;
        throw v1.size() != v2.size();
    }
    if (v1.empty()){
        cout << "vector must not empty." << endl;
        throw v1.empty();
    }
    vector<decltype(v1[0] - v2[0])> re(v1.size());
    for (int i = 0; i < v1.size(); ++i) {
        re[i] = v1[i] - v2[i];
    }
    return re;
}


template <class T1, class T2>
double operator * (const vector<T1>& v1, const vector<T2>& v2) {
    if (v1.size() != v2.size()) {
        cout << "two vector must have same size." << endl;
        throw v1.size() != v2.size();
    }
    if (v1.empty()){
        cout << "vector must not empty." << endl;
        throw v1.empty();
    }
    decltype(v1[0] * v2[0]) re = 0;
    for (int i = 0; i < v1.size(); ++i) {
        re += v1[i] * v2[i];
    }
    return re;
}




template <class T1, class T2>
auto operator / (const vector<T1>& v1, const vector<T2>& v2)->vector<decltype(v1[0] / v2[0])> {
    if (v1.size() != v2.size()) {
        cout << "two vector must have same size." << endl;
        throw v1.size() != v2.size();
    }
    if (v1.empty()){
        cout << "vector must not empty." << endl;
        throw v1.empty();
    }
    vector<decltype(v1[0] / v2[0])> re(v1.size());
    for (int i = 0; i < v1.size(); ++i) {
        re[i] = v1[i] / v2[i];
    }
    return re;
}


template <class T1, class T2>
auto operator + (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 + v2[0])>{


    if (v2.empty()){
        cout << "vector must not empty." << endl;
        throw v2.empty();
    }
    vector<decltype(arg1 + v2[0])> re(v2.size());
    for (int i = 0; i < v2.size(); ++i) {
        re[i] = arg1 + v2[i];
    }
    return re;
}


template <class T1, class T2>
auto operator - (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 - v2[0])>{


    if (v2.empty()){
        cout << "vector must not empty." << endl;
        throw v2.empty();
    }
    vector<decltype(arg1 - v2[0])> re(v2.size());
    for (int i = 0; i < v2.size(); ++i) {
        re[i] = arg1 - v2[i];
    }
    return re;
}


template <class T1, class T2>
auto operator * (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 * v2[0])>{


    if (v2.empty()){
        cout << "vector must not empty." << endl;
        throw v2.empty();
    }
    vector<decltype(arg1 * v2[0])> re(v2.size());
    for (int i = 0; i < v2.size(); ++i) {
        re[i] = arg1 * v2[i];
    }
    return re;
}


template <class T1, class T2>
auto operator / (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 / v2[0])>{


    if (v2.empty()){
        cout << "vector must not empty." << endl;
        throw v2.empty();
    }
    vector<decltype(arg1 / v2[0])> re(v2.size());
    for (int i = 0; i < v2.size(); ++i) {
        re[i] = arg1 / v2[i];
    }
    return re;
}


template <class T1, class T2>
auto operator + (const vector<T1>& v1, const T2& arg2)->vector<decltype(v1[0] + arg2)>{
    return arg2+v1;
}


template <class T1, class T2>
auto operator - (const vector<T1>& v1, const T2& arg2)->vector<decltype(v1[0] - arg2)>{
    return arg2-v1;
}

template <class T1, class T2>
auto operator * (const vector<T1>& v1, const T2& arg2)->vector<decltype(v1[0] * arg2)>{
    return arg2+v1;
}

template <class T1, class T2>
auto operator / (const vector<T1>& v1, const T2& arg2)->vector<decltype(v1[0] / arg2)>{
    return arg2+v1;
}

#endif //MACHINE_LEARNING_MODEL_BASE_H
