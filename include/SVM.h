//
// Created by wyb on 19-2-17.
//

#ifndef MACHINE_LEARNING_SVM_H
#define MACHINE_LEARNING_SVM_H
#include <vector>
#include <utility>
#include <iostream>
#include "model_base.h"
using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::string;
class SVM : public Base{
private:
    vector<double> w;
    vector<double> alpha;
    double b;
    vector<double> E;
    double tol=0.001;
    double eps=0.0005;
    double C=1.0;
public:
    virtual void getData(const string &filename);
    virtual void run();
    void createTrainTest();
    void SMO();
    int SMOTakeStep(int& i1, int& i2);
    int SMOExamineExample(int i2);
    double kernel(vector<double>& , vector<double>&);
    double computeE(int& i);
    pair<double, double> SMOComputeOB(int& i1, int& i2, double&L, double& H);
    void initialize();
    void train();
    double predict(const vector<double>& inputData, const double& GT);
};

#endif //MACHINE_LEARNING_SVM_H
