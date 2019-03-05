//
// Created by wyb on 19-2-27.
//

#ifndef MACHINE_LEARNING_GMM_H
#define MACHINE_LEARNING_GMM_H
#include <vector>
#include <string>

#include "model_base.h"

class GMM : public Base {
private:
    std::vector<double> alpha;
    std::vector<std::vector<std::vector<double>>> sigma;
    std::vector<std::vector<double>> mu;
    std::vector<std::vector<double>> gamma;
    std::vector<std::vector<double>> gaussVote;
public:
    virtual void getData(const std::string &filename);
    virtual void run();
    void createTrainTest();
    void EMAlgorithm(std::vector<double>& alphaOld,
                     std::vector<std::vector<std::vector<double>>>& sigmaOld,
                     std::vector<std::vector<double>>& muOld);
    void train(int steps, int k);
    double gaussian(std::vector<double>& muI, std::vector<std::vector<double>>& sigmaI,
                    vector<double>& observeValue);
    double getDet(const std::vector<std::vector<double>>& mat, int ignoreCol);
    std::vector<std::vector<double>> matInversion(std::vector<std::vector<double>>& mat);
    int predict(vector<double>& testF, double& testGT);
};

#endif //MACHINE_LEARNING_GMM_H
