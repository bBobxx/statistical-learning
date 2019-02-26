//
// Created by wyb on 19-1-16.
//
#include "Logistic.h"

using std::string;
using std::vector;
using std::pair;

void Logistic::getData(const string &filename){
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
        inData.push_back(temData);
    }
    std::cout<<"total data is "<<inData.size()<<std::endl;
}


void Logistic::createTrainTest() {
    std::random_shuffle(inData.begin(), inData.end());
    unsigned long size = inData.size();
    unsigned long trainSize = size * 0.6;
    std::cout<<"total data is "<< size<<" ,train data has "<<trainSize<<std::endl;
    for(int i=0;i<size;++i){
        if (i<trainSize)
            trainData.push_back(inData[i]);
        else
            testData.push_back(inData[i]);

    }
    //create feature for test,using trainData, testData
    for (const auto& data:trainData){
        std::vector<double> trainf;
        trainf.assign(data.begin(), data.end()-1);
        trainf.push_back(1.0);
        trainDataF.push_back(trainf);
        trainDataGT.push_back(*(data.end()-1));
    }
    for (const auto& data:testData){
        std::vector<double> testf;
        testf.assign(data.begin(), data.end()-1);
        testf.push_back(1.0);
        testDataF.push_back(testf);
        testDataGT.push_back(*(data.end()-1));
    }
}


double Logistic::logistic(const vector<double>& data){
    double expval = exp(w * data);
    return expval/(1.0+expval);
}

void Logistic::initialize(const vector<double>& wInit){
    w = wInit;
}

vector<double> Logistic::computeGradient(const vector<double>& trainFeature, double trainGrT){
    return -1*trainFeature*(trainGrT-logistic(trainFeature));
}

void Logistic::train(const int& step, const double& lr) {
    int count = 0;
    for(int i=0; i<step; ++i) {
        if (count == trainDataF.size() - 1)
            count = 0;
        count++;
        vector<double> grad = computeGradient(trainDataF[count], trainDataGT[count]);
        double fl;
        if (trainDataGT[count]==0)
            fl = 1;
        else
            fl = -1;
        w = w + fl*lr*grad;
        auto val = trainDataF[count]*w;
        double loss = -1*trainDataGT[count]*val + log(1 + exp(val));
        cout<<"step "<<i<<", train loss is "<<loss<<" gt "<<trainDataGT[count]<<endl;
    }
}

double Logistic::predict(const vector<double>& inputData, const double& GT){
    cout<<"The right class is "<<GT<<endl;
    double out = logistic(inputData);
    if(out>=0.5){
        std::cout<<"The predict class is 1"<<std::endl;
        return 1;
    }
    else{
        std::cout<<"The predict class is 0"<<std::endl;
        return 0;
    }
}

void Logistic::run(){
    getData("../data/logistic.txt");
    createTrainTest();
    std::vector<double> init (indim, 0.5);
    initialize(init);
    train(20, 1.0);//20 is steps and 1.0 is learning rate
    for(int i=0; i<testDataF.size(); ++i){
        std::cout<<i<<std::endl;
        predict(testDataF[i], testDataGT[i]);
    }
}
