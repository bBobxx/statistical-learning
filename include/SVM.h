//
// Created by wyb on 19-2-17.
//

#ifndef MACHINE_LEARNING_SVM_H
#define MACHINE_LEARNING_SVM_H
#include <vector>
#include <utility>
#include <iostream>
#include "model_base.h"

class SVM : public Base{
private:
    std::vector<double> w;
    std::vector<double> alpha;
    double b;
    std::vector<double> E;
    double tol=0.001;
    double eps=0.0005;
    double C=1.0;
public:
    virtual void getData(const std::string &filename);
    virtual void run();
    void createTrainTest();
    void SMO();
    int SMOTakeStep(int& i1, int& i2);
    int SMOExamineExample(int i2);
    double kernel(std::vector<double>& , std::vector<double>&);
    double computeE(int& i);
    std::pair<double, double> SMOComputeOB(int& i1, int& i2, double&L, double& H);
    void initialize();
    void train();
    double predict(const std::vector<double>& inputData);
};

#endif //MACHINE_LEARNING_SVM_H
