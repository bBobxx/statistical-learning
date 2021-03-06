//
// Created by wyb on 19-2-21.
//

#ifndef MACHINE_LEARNING_ADABOOST_H
#define MACHINE_LEARNING_ADABOOST_H

#include <vector>
#include "model_base.h"
#include "perceptron.h"

class AdaBoost: public Base {
private:
    vector<double> clsfWeight;
    vector<double> featrWeight;
    vector<Perceptron* > classifiers;
public:
    virtual void getData(const std::string &filename);
    virtual void run();
    void createTrainTest();
    int computeWeights(Perceptron* classifier);
    int predict(vector<double>& testF);
};


#endif //MACHINE_LEARNING_ADABOOST_H
