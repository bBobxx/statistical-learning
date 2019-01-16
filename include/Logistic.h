//
// Created by wyb on 19-1-16.
//

#ifndef MACHINE_LEARNING_LOGISTIC_H
#define MACHINE_LEARNING_LOGISTIC_H

#include <cmath>
#include <vector>
#include "model_base.h"
using std::vector;
using std::string;
using std::cout;
using std::endl;

class Logistic: public Base{
private:
    vector<double> w;
public:
    virtual void getData(const string &filename);
    virtual void run();
    double logistic(const vector<double>& data);
    void createTrainTest();
    void initialize(const vector<double>& );
    void train(const int& step, const double& lr);
    vector<double> computeGradient(const vector<double>& trainFeature, double trainGrT);
    double predict(const vector<double>& inputData, const double& GT);
};


#endif //MACHINE_LEARNING_LOGISTIC_H
