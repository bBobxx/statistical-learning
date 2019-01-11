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

using std::vector;
using std::string;
using std::map;
using std::set;
using std::pair;
using std::cout;
using std::endl;


class NavieBayes: public Base{
private:
    vector<vector<double>> xVal;
    vector<double> yVal;
    vector<map<pair<string,string>, double>> condProb;
    map<string, double> priProb;
public:
    virtual void getData(const std::string &filename);
    virtual void run();
    void setInVal(vector<vector<double>>& in ){xVal = in;}
    void setOutVal(vector<double>& out){yVal = out;}
    void initialize();
    void createTrainTest();
    void predict();
    void train(const string&);
    void maxLikeEstim();
    void bayesEstim(const double& );
};


#endif //MACHINE_LEARNING_NAVIEBAYES_H
