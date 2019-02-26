//
// Created by wyb on 19-1-16.
//

#ifndef MACHINE_LEARNING_LOGISTIC_H
#define MACHINE_LEARNING_LOGISTIC_H

#include <cmath>
#include <vector>
#include "model_base.h"

class Logistic: public Base{
private:
    vector<double> w;
public:
    virtual void getData(const std::string &filename);
    virtual void run();
    double logistic(const std::vector<double>& data);
    void createTrainTest();
    void initialize(const std::vector<double>& );
    void train(const int& step, const double& lr);
    std::vector<double> computeGradient(const std::vector<double>& trainFeature, double trainGrT);
    double predict(const std::vector<double>& inputData, const double& GT);
};


#endif //MACHINE_LEARNING_LOGISTIC_H
