//
// Created by wyb on 18-12-13.
//

#ifndef MACHINE_LEARNING_PERCEPTRON_H
#define MACHINE_LEARNING_PERCEPTRON_H

#include <vector>
#include <array>
#include <utility>
#include "model_base.h"


class Perceptron: public Base{
private:
    std::vector<double> w;
    double b;
public:
    virtual void getData(const std::string& filename);
    virtual void run();
    void splitData(const float& );
    void createFeatureGt();//create feature for test,using trainData, testData
    void setDim(const unsigned long& iDim){indim = iDim;}
    double inference(const std::vector<double>&) ;
    void initialize(std::vector<double>& init);
    void train(const int& step,const float& lr);
    int predict(const std::vector<double>& inputData, const double& GT);
    double loss(const std::vector<double>& inputData, const double& groundTruth);
    std::pair<std::vector<double>, double> computeGradient(const std::vector<double>& inputData, const double& groundTruth);
    std::vector<std::vector<double>> getTestDataFeature(){return testDataF;}
    std::vector<double> getTestGT(){ return testDataGT;}
};



#endif //MACHINE_LEARNING_PERCEPTRON_H
