//
// Created by wyb on 18-12-20.
//

#ifndef MACHINE_LEARNING_NAVIEBAYES_H
#define MACHINE_LEARNING_NAVIEBAYES_H

#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include "model_base.h"


class NavieBayes: public Base{
private:
    std::vector<std::vector<double>> xVal;
    std::vector<double> yVal;
    std::vector<std::map<std::pair<std::string,std::string>, double>> condProb;
    std::map<std::string, double> priProb;
public:
    virtual void getData(const std::string &filename);
    virtual void run();
    void setInVal(std::vector<std::vector<double>>& in ){xVal = in;}
    void setOutVal(std::vector<double>& out){yVal = out;}
    void initialize();
    void createTrainTest();
    void predict();
    void train(const std::string&);
    void maxLikeEstim();
    void bayesEstim(const double& );
};


#endif //MACHINE_LEARNING_NAVIEBAYES_H
