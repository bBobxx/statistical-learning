//
// Created by wyb on 19-3-5.
//

#include "GMM.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;


void GMM::getData(const std::string &filename) {
    //load data to a vector
    std::vector<double> temData;
    double onepoint;
    std::string line;
    inData.clear();
    std::ifstream infile(filename);
    std::cout << "reading ..." << std::endl;
    while(!infile.eof()){
        temData.clear();
        std::getline(infile, line);
        if(line.empty())
            continue;
        std::stringstream stringin (line);
        while (stringin >> onepoint) {
            temData.push_back(onepoint);
        }
        indim = temData.size();
        indim -= 1;
        inData.push_back(temData);
    }
    std::cout<<"total data is "<<inData.size()<<std::endl;
}



void GMM::createTrainTest() {
    std::random_shuffle(inData.begin(), inData.end());
    unsigned long size = inData.size();
    unsigned long trainSize = size * 0.7;
    std::cout << "total data is " << size << " ,train data has " << trainSize << std::endl;
    for (int i = 0;i < size; ++i) {
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
double GMM::getDet(const vector<vector<double>> &mat, int ignoreCol=-1) {
    // compute determinant of a matrix
    if (mat.empty())
        throw "mat must be a Square array";
    if (mat.size() == 1) {
        return mat[0][0];
    }
    if (mat.size() == 2) {
        if (mat[0].size() != 2 || mat[1].size() != 2)
            throw "mat must be a Square array";
        return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0];
    }
    double det = 0;
    for (int numCol = 0; numCol < mat.size(); ++numCol) {
        // below is to compute sub mat.
        vector<vector<double>> newMat;
        ignoreCol = numCol;
        for (int i = 0 ; i < mat.size(); ++i) {
            vector<double> matRow;
            for (int j = 0; j < mat.size(); ++j) {
                if (i == 0 || j == ignoreCol)
                    continue;
                matRow.push_back(mat[i][j]);
            }
            if (matRow.size()!=0)
                newMat.push_back(matRow);
        }
        int factor;
        factor = numCol%2 == 0 ? 1 : -1;
        det += factor*mat[0][numCol]*getDet(newMat, numCol);
    }
    return det;
}


vector<vector<double>> GMM::matInversion(vector<vector<double>> &mat) {
    // compute Inversion of a matrix
    double det = getDet(mat);
    if (std::abs(det)<1e-10)
        std::cerr<< "det of mat must not be 0" << endl;
    vector<vector<double>> invMat (mat.size(), vector<double>(mat.size(), 0));
    for (int i = 0 ; i < invMat.size(); ++i) {
        for (int j = 0; j < invMat.size(); ++j) {
            // below is to compute sub mat.
            vector<vector<double>> newMat;
            for (int x = 0; x < mat.size(); ++x) {
                vector<double> matRow;
                for (int y = 0; y < mat.size(); ++y) {
                    if (x == i || y == j)
                        continue;
                    matRow.push_back(mat[i][j]);
                }
                if (!matRow.empty())
                    newMat.push_back(matRow);
            }
            invMat[j][i] = getDet(newMat) / det; // note the i and j
        }
    }
    return invMat;

}
double GMM::gaussian(vector<double>& muI, vector<vector<double>>& sigmaI,
                     vector<double>& observeValue) {
    vector<double> xMinusMu = observeValue - muI;
    vector<double> rightMul;
    vector<vector<double>> matInvers = transpose(matInversion(sigmaI));
    // for compute convenience, i use the transpose mat for my operator *
    for (auto& vec : matInvers) {
        rightMul.push_back(xMinusMu*vec);
    }
    double finalMul = rightMul*xMinusMu;
    double det = getDet(sigmaI);
    double gaussianVal;
    gaussianVal = 1 / (std::pow(2 * 3.14, indim/2) * std::pow(det, 0.5)) * std::exp(-0.5 * finalMul);
}

void GMM::EMAlgorithm(vector<double> &alphaOld, vector<vector<vector<double>>> &sigmaOld,
        vector<vector<double>> &muOld) {
// compute gamma
    for (int i = 0; i < trainDataF.size(); ++i) {
        double probSum = 0;
        for (int l = 0; l < alpha.size(); ++l) {
            double gas = gaussian(muOld[l], sigmaOld[l], trainDataF[i]);
            probSum += alphaOld[l] * gas;
        }
        for (int k = 0; k < alpha.size(); ++k) {
            double gas = gaussian(muOld[k], sigmaOld[k], trainDataF[i]);
            gamma[i][k] = alphaOld[k] * gas / probSum;
        }
    }
// update mu, sigma, alpha
    for (int k = 0; k < alpha.size(); ++k) {
        vector<double> muNew;
        vector<vector<double>> sigmaNew;
        double alphaNew;
        vector<double> muNumerator;
        double sumGamma = 0.0;
        for (int i = 0; i < trainDataF.size(); ++i) {
            sumGamma += gamma[i][k];
            if (i==0) {
                muNumerator = gamma[i][k] * trainDataF[i];
            }
            else {
                muNumerator = muNumerator + gamma[i][k] * trainDataF[i];
            }
        }
        muNew = muNumerator / sumGamma;
        for (int i = 0; i < trainDataF.size(); ++i) {
            if (i==0) {
                auto temp1 = gamma[i][k]/ sumGamma * (trainDataF[i] - muNew);
                auto temp2 = trainDataF[i] - muNew;
                sigmaNew = vecMulVecToMat(temp1, temp2);
            }
            else {
                auto temp1 = gamma[i][k] / sumGamma * (trainDataF[i] - muNew);
                auto temp2 = trainDataF[i] - muNew;
                sigmaNew = sigmaNew + vecMulVecToMat(temp1, temp2);
            }
        }
        alphaNew = sumGamma / trainDataF.size();
        mu[k] = muNew;
        sigma[k] = sigmaNew;
        alpha[k] = alphaNew;
    }
}

void GMM::train(int steps, int k) {
    // Initialize the variable
    if (alpha.empty() && mu.empty() && sigma.empty() && gamma.empty()) {
        for (int i = 0; i < k; ++i) {
            alpha.push_back(1.0/k);
            for (int index = 0; index < trainDataGT.size(); ++index){
                if((int)trainDataGT[index] == i+1) {
                    mu.push_back(trainDataF[index]);
                    break;
                }
            }
            vector<vector<double>> sigm (indim, vector<double> (indim));
            for (int row = 0; row < indim; ++row) {
                for (int col = 0; col < indim; ++col){
                    if (row == col)
                        sigm[row][col] = 0.1;
                }
            }
            sigma.push_back(sigm);
        }
        for (int i = 0; i < trainDataF.size(); ++i) {
            vector<double> gammaTemp;
            for (int j = 0; j < k; ++j)
                gammaTemp.push_back(1.0/(trainDataF.size() * k));
            gamma.push_back(gammaTemp);
        }
    }
    for (int step = 0; step < steps ; ++step)
        EMAlgorithm(alpha, sigma, mu);
    vector<vector<double>> vote (alpha.size(), vector<double> (alpha.size()));
    for (int i = 0; i < trainDataF.size(); ++i) {
        double prob = 0.0;
        int index = -1;
        for (int l = 0; l < alpha.size(); ++l) {
            double probk = gaussian(mu[l], sigma[l], trainDataF[i]);
            if (probk > prob) {
                prob = probk;
                index = l;
            }
        }
        int cls = (int)trainDataGT[i]-1;
        vote[index][cls] += 1;
    }
    gaussVote = vote;
}


int GMM::predict(vector<double>& testF, double& testGT) {
    cout << "the true class is " << testGT << endl;
    double prob = 0.0;
    int index = -1;
    for (int k = 0; k < alpha.size(); ++k) {
        double probk = gaussian(mu[k], sigma[k], testF);
        if (probk > prob) {
            prob = probk;
            index = k;
        }
    }
    int pred = std::distance(gaussVote[index].begin(),
                             std::max_element(gaussVote[index].begin(), gaussVote[index].end()));
    cout << "the predict class is " << pred+1 << endl;
    return pred;
}

void GMM::run() {
    getData("../data/GMM.txt");
    createTrainTest();
    train(10, 3);
    for(int i = 0; i < testDataF.size(); ++i) {
        predict(testDataF[i], testDataGT[i]);
    }
}