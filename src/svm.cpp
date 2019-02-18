//
// Created by wyb on 19-2-18.
//
#include "SVM.h"

using std::string ;

void SVM::getData(const string &filename){
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


void SVM::createTrainTest() {
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


void SVM::SMO() {
    /*
     * this function reference the Platt J. Sequential minimal optimization: A fast algorithm for training support vector machines[J]. 1998.
     */
    int numChanged = 0;
    int examineAll = 1;
    while(numChanged > 0 || examineAll){
        numChanged = 0;
        if (examineAll){
            for (int i=0; i<trainDataF.size();++i)
                numChanged+=SMOExamineExample(i);
        }
        else{
            for (int i=0; i<trainDataF.size();++i){
                if(alpha[i]!=0&&alpha[i]!=C)
                    numChanged+=SMOExamineExample(i);
            }
        }
        if(examineAll==1)
            examineAll=0;
        else{
            if(numChanged==0)
                examineAll=1;
        }
    }

}
double SVM::kernel(vector<double> & x1, vector<double> & x2) {
    //here use linear kernel
    return x1 * x2;
}
double SVM::computeE(int& i) {
    double e = 0;
    for(int j =0 ; j < trainDataF.size(); ++j){
        e += alpha[j]*trainDataGT[j]*kernel(trainDataF[j], trainDataF[i]);
    }
    e += b;
    e -= trainDataGT[i];
    //e = w*trainDataF[i]+b-trainDataGT[i];
    return e;

}


pair<double, double> SVM::SMOComputeOB(int& i1, int& i2, double&L, double& H) {
    double y1 = trainDataGT[i1];
    double y2 = trainDataGT[i2];
    double s = y1 * y2;
    double f1 = y1 * (E[i1] + b) - alpha[i1] * kernel(trainDataF[i1], trainDataF[i1]) -
                s * alpha[i2] * kernel(trainDataF[i1], trainDataF[i2]);
    double f2 = y2 * (E[i2] + b) - s * alpha[i1] * kernel(trainDataF[i1], trainDataF[i2]) -
                alpha[i2] * kernel(trainDataF[i2], trainDataF[i2]);
    double L1 = alpha[i1] + s * (alpha[i2] - L);
    double H1 = alpha[i1] + s * (alpha[i2] - H);
    double obL = L1 * f1 + L * f2 + 0.5 * L1 * L1 * kernel(trainDataF[i1], trainDataF[i1]) +
                 0.5 * L * L * kernel(trainDataF[i2], trainDataF[i2]) +
                 s * L * L1 * kernel(trainDataF[i1], trainDataF[i2]);
    double obH = H1 * f1 + H * f2 + 0.5 * H1 * H1 * kernel(trainDataF[i1], trainDataF[i1]) +
                 0.5 * H * H * kernel(trainDataF[i2], trainDataF[i2]) +
                 s * H * H1 * kernel(trainDataF[i1], trainDataF[i2]);
    return std::make_pair(obL, obH);
}


int SVM::SMOTakeStep(int& i1, int& i2) {
    if (i1 == i2)
        return 0;
    double y1 = trainDataGT[i1];
    double y2 = trainDataGT[i2];
    double s = y1 * y2;
    double L, H;
    if (y1 != y2) {
        L = (alpha[i1] - alpha[i2]) > 0 ? alpha[i1] - alpha[i2] : 0;
        H = (alpha[i1] - alpha[i2] + C) < C ? alpha[i1] - alpha[i2] + C : C;
    } else {
        L = (alpha[i1] + alpha[i2] - C) > 0 ? alpha[i1] + alpha[i2] - C : 0;
        H = (alpha[i1] + alpha[i2]) < C ? alpha[i1] + alpha[i2] : C;
    }
    if (L == H)
        return 0;
    double k11 = kernel(trainDataF[i1], trainDataF[i1]);
    double k12 = kernel(trainDataF[i1], trainDataF[i2]);
    double k22 = kernel(trainDataF[i2], trainDataF[i2]);
    double eta = k11 + k22 - 2 * k12;
    double a2;
    if (eta > 0) {
        a2 = alpha[i2] + y2 * (E[i1] - E[i2]) / eta;
        if (a2 < L)
            a2 = L;
        else {
            if (a2 > H)
                a2 = H;
        }
    } else {
        pair<double, double> ob = SMOComputeOB(i1, i2, L, H);
        double Lobj = ob.first;
        double Hobj = ob.second;
        if (Lobj < Hobj - eps)
            a2 = L;
        else {
            if (Lobj > Hobj + eps)
                a2 = H;
            else
                a2 = alpha[i2];
        }
    }
    if (std::abs(a2 - alpha[i2]) < eps * (a2 + alpha[i2] + eps))
        return 0;
    double a1 = alpha[i1] + s * (alpha[i2] - a2);
    double b1;
    //please notice that the update equation is from <<统计学习方法>>p130, not the equation in paper
    b1= -E[i1] - y1 * (a1 - alpha[i1]) * kernel(trainDataF[i1], trainDataF[i1]) -
                y2 * (a2 - alpha[i2]) * kernel(trainDataF[i1], trainDataF[i2]) + b;
    double b2;
    b2 = -E[i2] - y1 * (a1 - alpha[i1]) * kernel(trainDataF[i1], trainDataF[i2]) -
                y2 * (a2 - alpha[i2]) * kernel(trainDataF[i2], trainDataF[i2]) + b;
    double bNew = (b1 + b2) / 2;
    b = bNew;
    w = w + y1 * (a1 - alpha[i1]) * trainDataF[i1] + y2 * (a2 - alpha[i2]) *
                                                     trainDataF[i2];
    //this is the linear SVM case, this equation are from the paper equation 22
    alpha[i1] = a1;
    alpha[i2] = a2;
//    vector<double> wtmp (indim);
//    for (int i=0; i<trainDataF.size();++i)
//    {
//        auto tmp = alpha[i]*trainDataF[i]*trainDataGT[i];
//        wtmp = wtmp+tmp;
//    }
//    w = wtmp;
    E[i1] = computeE(i1);
    E[i2] = computeE(i2);
    return 1;
}

int SVM::SMOExamineExample(int i2){
    double y2 = trainDataGT[i2];
    double alph2 = alpha[i2];
    double E2 = E[i2];
    double r2 = E2*y2;
    if((r2<-tol && alph2<C)||(r2>tol && alph2>0)){
        int alphNum = 0;
        for (auto& a:alpha){
            if (a != 0 && a != C)
                alphNum++;
        }
        if (alphNum>1){
            double dis = 0;
            int i1 ;
            for(int j=0;j<E.size();++j){
                if (std::abs(E[j]-E[i2])>dis){
                    i1 = j;
                    dis = std::abs(E[j]-E[i2]);
                }

            }

            if (SMOTakeStep(i1,i2))
                return 1;
        }
        for (int i = 0; i < alpha.size();++i){
            if (alpha[i] != 0 && alpha[i] != C){
                int i1 = i;
                if (SMOTakeStep(i1, i2))
                    return 1;
            }
        }
        for(int i = 0; i < trainDataF.size();++i){
            int i1 = i;
            if (SMOTakeStep(i1, i2))
                return 1;
        }

    }
    return 0;
}

void SVM::initialize() {
    b = 0;
    for(int i=0;i<trainDataF.size();++i){
        alpha.push_back(0.0);
    }
    for(int i=0;i<indim;++i){
        w.push_back(0.0);
    }
    for(int i=0;i<trainDataF.size();++i){
        double e = computeE(i);
        E.push_back(e);
    }


}

void SVM::train() {
    initialize();
    SMO();
}

double SVM::predict(const vector<double> &inputData, const double &GT) {
    double p = w*inputData+b;
    if(p>0)
        return 1.0;
    else
        return -1.0;
}

void SVM::run() {
    getData("../data/perceptrondata.txt");
    createTrainTest();
    train();
    cout<<"w and b is: "<<endl;
    for(auto&c : w)
        cout<<c<<" ";
    cout<<b<< endl;
    for(int i = 0; i<testDataF.size();++i){
        cout<<"the true class of this point is "<<testDataGT[i];
        double pre = predict(testDataF[i], testDataGT[i]);
        cout<<", the predict class of this point is "<<pre<<endl;

    }
}

