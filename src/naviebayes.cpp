//
// Created by wyb on 18-12-20.
//
#include "NavieBayes.h"

void NavieBayes::getData(const std::string &filename) {
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



void NavieBayes::createTrainTest() {
    std::random_shuffle(inData.begin(), inData.end());
    unsigned long size = inData.size();
    unsigned long trainSize = size * 1;
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


void NavieBayes::maxLikeEstim(){
    for(const auto& gt: trainDataGT){
        priProb[std::to_string(gt)] += 1;
    }
    for(unsigned long i=0;i<indim;++i){
        for(unsigned long j=0;j<trainDataF.size();++j)
        {
            auto cond = std::make_pair(std::to_string(trainDataF[j][i]), std::to_string(trainDataGT[j]));
            condProb[i][cond] += 1.0/priProb[std::to_string(trainDataGT[j])];
        }
    }
    for(auto& iter:priProb)
        iter.second /= double(trainDataF.size());
}



void NavieBayes::bayesEstim(const double& lmbda = 1.0){
    for(const auto& gt: trainDataGT){
        priProb[std::to_string(gt)] += 1.0;
    }
    for(unsigned long i=0;i<indim;++i){
        for(unsigned long j=0;j<trainDataF.size();++j)
        {
            auto cond = std::make_pair(std::to_string(trainDataF[j][i]), std::to_string(trainDataGT[j]));

            condProb[i][cond] += 1.0/(priProb[std::to_string(trainDataGT[j])]+lmbda*xVal[i].size());
        }
    }
    for(unsigned long i=0;i<indim;++i){
        for(auto& d:condProb[i]){
            d.second += lmbda/(priProb[d.first.second]+lmbda*xVal[i].size());
        }
    }

    for(auto& iter:priProb)
        iter.second = (iter.second+lmbda)/(double(trainDataF.size()+yVal.size()));
}


void NavieBayes::initialize() {
    for(unsigned long i=0;i<indim;++i){
        map<pair<string,string>, double> m;
        for(const auto& xval : xVal[i])
            for(const auto& yval:yVal){
                auto cond = std::make_pair(std::to_string(xval), std::to_string(yval));
                m[cond]=0;
            }

        condProb.push_back(m);
    }
    for(const auto& val:yVal)
        priProb[std::to_string(val)]=0;
}


void NavieBayes::train(const string& estim="mle"){
    //train actually is create priProb and condProb.
    if(xVal.empty() && yVal.empty()){
        cout<<"please set range of x and y first."<<endl;
        throw;
    }
    initialize();
    if (estim == "mle")
        maxLikeEstim();
    else {
        if(estim == "byse")
            bayesEstim();
        else{
            cout<<"estimation nust be mle or byse."<<endl;
            throw ;
        }
    }
    cout<<"train over."<<endl;
    for(auto& a:condProb)
        for(auto& c:a){
            cout<<"the conditional probability of "<<c.first.first<<"/"<<c.first.second<<" is: "<<c.second<<endl;
        }
    for(auto& a:priProb)
        cout<<"the priori probability of "<<a.first<<" is: "<<a.second<<endl;
}

void NavieBayes::predict() {
    for(unsigned long j=0;j<testDataF.size();++j){
        double y_t=0;
        double pre = 0;
        cout<<"the test data ture class is "<<testDataGT[j]<<endl;
        for(const auto& y: yVal){
            auto pr = priProb[std::to_string(y)];
            for(unsigned long i=0;i<indim;++i)
                pr *= condProb[i][std::make_pair(std::to_string(testDataF[j][i]), std::to_string(y))];
            cout<<"predict probability of "<<y<<" is "<<pr<<endl;
            if(pr>pre){
                pre = pr;
                y_t = y;
            }

        }
        cout<<"the test data predict class is "<<y_t<<endl;
    }
}
void NavieBayes::run() {
    getData("../data/naviebayes.txt");
    createTrainTest();
    vector<vector<double>> x {{1,2,3},{4,5,6}};//书中例题第二维的取值是字母，为了方便换成了4,5,6
    vector<double> y {-1,1};
    setInVal(x);
    setOutVal(y);
    train("byse");
    vector<double> testDF {2, 4};
    testDataF.push_back(testDF);
    testDataGT.push_back(-1);
    predict();
}
