//
// Created by wyb on 18-12-17.
//

#include "perceptron.h"

void Perceptron::getData(const std::string &filename) {
    //load data to a vector
    std::vector<double> temData;
    double onepoint;
    std::string line;
    inData.clear();
    std::ifstream infile(filename);
    std::cout<<"reading ..."<<std::endl;
    while(!infile.eof()){
        temData.clear();
        std::getline(infile, line);
        if(line.empty())
            continue;
        std::stringstream stringin(line);
        while(stringin >> onepoint){
            temData.push_back(onepoint);
        }
        indim = temData.size();
        indim -= 1;
        inData.push_back(temData);
    }
    std::cout<<"total data is "<<inData.size()<<std::endl;
}

void Perceptron::splitData(const float& trainTotalRatio){
    std::random_shuffle(inData.begin(), inData.end());
    unsigned long size = inData.size();
    unsigned long trainSize = size * trainTotalRatio;
    std::cout<<"total data is "<< size<<" ,train data has "<<trainSize<<std::endl;
    for(int i=0;i<size;++i){
        if (i<trainSize)
            trainData.push_back(inData[i]);
        else
            testData.push_back(inData[i]);

    }

}
void Perceptron::createFeatureGt() {
    //create feature for test,using trainData, testData
    for (const auto& data:trainData){
        std::vector<double> trainf;
        trainf.assign(data.begin(), data.end()-1);
        trainDataF.push_back(trainf);
        trainDataGT.push_back(*(data.end()-1));
    }
    for (const auto& data:testData){
        std::vector<double> testf;
        testf.assign(data.begin(), data.end()-1);
        testDataF.push_back(testf);
        testDataGT.push_back(*(data.end()-1));
    }
}

void Perceptron::initialize(std::vector<double>& init) {
    // must initialize parameter first, using vector to initialize
    if(init.size()!=indim+1) {
        std::cout<<"input dimension is should be "+std::to_string(indim+1)<<std::endl;
        throw init.size();
    }
    w.assign(init.begin(), init.end()-1);
    b = *(init.end()-1);
}



double Perceptron::inference(const std::vector<double>& inputData){
    //just compute wx+b , for compute loss and predict.
    if (inputData.size()!=indim){
        std::cout<<"input dimension is incorrect. "<<std::endl;
        throw inputData.size();
    }

    double sum_tem = 0.0;
    sum_tem = inputData * w;
    sum_tem += b;
    return sum_tem;
}



double Perceptron::loss(const std::vector<double>& inputData, const double& groundTruth){
    double loss = -1.0 * groundTruth * inference(inputData);
    std::cout<<"loss is "<< loss <<std::endl;
    return loss;
}



std::pair<std::vector<double>, double> Perceptron::computeGradient(const std::vector<double>& inputData, const double& groundTruth) {
    double lossVal = loss(inputData, groundTruth);
    std::vector<double> w;
    double b;
    if (lossVal > 0.0)
    {
        for(auto indata:inputData) {
            w.push_back(indata*groundTruth);
        }
        b = groundTruth;
    }
    else{
        for(auto indata:inputData) {
            w.push_back(0.0);
        }
        b = 0.0;
    }
    return std::pair<std::vector<double>, double>(w, b);//here, for understandable, we use pair to represent w and b.
    //you also could return a vector which contains w and b.
}


void Perceptron::train(const int & step, const float & lr) {
    int count = 0;
    createFeatureGt();
    for(int i=0; i<step; ++i){
        if (count==trainDataF.size()-1)
            count = 0;
        count++;
        std::vector<double> inputData = trainDataF[count];
        double groundTruth = trainDataGT[count];
        auto grad = computeGradient(inputData, groundTruth);
        auto grad_w = grad.first;
        double grad_b = grad.second;
        for (int j=0; j<indim;++j){
            w[j] += lr * (grad_w[j]);
        }
        b += lr * (grad_b);
    }
}


int Perceptron::predict(const std::vector<double>& inputData, const double& GT) {

    double out = inference(inputData);
    std::cout<<"The right class is "<<GT<<std::endl;
    if(out>=0.0){
        std::cout<<"The predict class is 1"<<std::endl;
        return 1;
    }
    else{
        std::cout<<"The right class is -1"<<std::endl;
        return -1;
    }


}

void Perceptron::run(){
    getData("../data/perceptrondata.txt");
    splitData(0.6);//below is split data , and store it in  trainData, testData
    std::vector<double> init = {1.0,1.0,1.0};
    initialize(init);
    train(20, 1.0);//20 is steps and 1.0 is learning rate
    std::vector<std::vector<double>>  testData = getTestDataFeature();
    std::vector<double> testGT = getTestGT();
    for(int i=0; i<testData.size(); ++i){
        std::cout<<i<<std::endl;
        predict(testData[i], testGT[i]);
    }
}