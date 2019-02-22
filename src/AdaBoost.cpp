//
// Created by wyb on 19-2-22.
//
#include "AdaBoost.h"

using std::string;


void AdaBoost::getData(const string &filename){
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
        indim = temData.size()-1;
        inData.push_back(temData);
    }
    std::cout<<"total data is "<<inData.size()<<std::endl;
}


void AdaBoost::createTrainTest() {
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
        featrWeight.push_back(1.0);
        trainDataF.push_back(trainf);
        trainDataGT.push_back(*(data.end()-1));
    }
    for (const auto& data:testData){
        std::vector<double> testf;
        testf.assign(data.begin(), data.end()-1);
        testDataF.push_back(testf);
        testDataGT.push_back(*(data.end()-1));
    }
    featrWeight = featrWeight / featrWeight.size();
}

int AdaBoost::computeWeights(Perceptron* classifier) {
    vector<double> trainGT;
    for(int i =0; i<trainDataGT.size();++i)
        trainGT.push_back(trainDataGT[i]*featrWeight[i]);
    classifier->setTrainD(trainDataF, trainGT);
    classifier->setDim(indim);
    classifier->train(100, 0.9);
    double erroeRate = 0;
    for(int i = 0; i<trainDataF.size();++i) {
        if (classifier->predict(trainDataF[i])!=int(trainDataGT[i]))
            erroeRate += featrWeight[i];
    }
    if(erroeRate==0){
        if(clsfWeight.size()==0)
            clsfWeight.push_back(1);
        return 0;
    }

    double clsW;
    clsW = 0.5*std::log((1-erroeRate)/erroeRate);
    clsfWeight.push_back(clsW);


    double zm=0;
    for(int i = 0; i<trainDataF.size();++i) {
        zm+=featrWeight[i]*std::exp(-clsW*trainDataGT[i]*classifier->predict(trainDataF[i]));
    }

    for(int i = 0; i<featrWeight.size();++i ){
        featrWeight[i] = featrWeight[i]/zm*std::exp(-clsW*trainDataGT[i]*classifier->predict(trainDataF[i]));
    }
    return 1;
}

int AdaBoost::predict(vector<double> &testF) {
    double out = 0;
    for(int i = 0; i<clsfWeight.size();++i) {
        out += clsfWeight[i] * classifiers[i]->predict(testF);
    }
    if (out > 0)
        return 1;
    else
        return -1;
}

void AdaBoost::run() {
    getData("../data/perceptrondata.txt");
    createTrainTest();

    int isContinue = 1;
    while(isContinue){
        Perceptron* cls = new Perceptron();
        isContinue = computeWeights(cls);
        if(isContinue || classifiers.size()==0)
            classifiers.push_back(cls);

    }
    for(int i=0; i<testDataF.size(); ++i){
        std::cout<<i<<std::endl;
        std::cout<<"The right class is "<<testDataGT[i]<<std::endl;
        int out = predict(testDataF[i]);
        std::cout<<"The predict class is "<<out<<std::endl;
    }
    int nfc = clsfWeight.size();
    nfc = nfc == 0 ? 1:nfc;
    cout<<"The number of classfiers is "<<nfc<<endl;
    for(int i = 0; i<classifiers.size();++i)
        delete classifiers[i];
}

