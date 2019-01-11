//
// Created by wyb on 19-1-7.
//

#ifndef MACHINE_LEARNING_DECISIONTREE_H
#define MACHINE_LEARNING_DECISIONTREE_H

#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <queue>
#include <utility>
#include "model_base.h"

using std::vector;
using std::string;
using std::map;
using std::set;
using std::pair;
using std::cout;
using std::endl;
using std::priority_queue;


struct DtreeNode {
    vector<vector<double>> leafValue;
    int axis;//split axis
    double splitVal;//split value
    bool isLeaf;
    vector<vector<double>> leftTreeVal;
    vector<vector<double>> rightTreeVal;
    DtreeNode* left;
    DtreeNode* right;
    DtreeNode(): isLeaf(false), axis(0), splitVal(0.0), left(nullptr), right(nullptr){};

};


class DecisionTree: public Base{
private:
    DtreeNode* root = nullptr;
    vector<int> features;
public:
    virtual void getData(const std::string &filename);
    virtual void run();
    void createTrainTest();
    void initializeRoot();
    DtreeNode* buildTree(DtreeNode* node, vector<vector<double >>& valRange);
    pair<int, double> createSplitFeature(vector<vector<double >>& valRange);
    void showTree(DtreeNode* node);

};

void DecisionTree::getData(const string &filename){
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
    for (int i = 0; i < indim; ++i)
        features.push_back(i);//initialize features
    std::cout<<"total data is "<<inData.size()<<std::endl;
}


void DecisionTree::createTrainTest() {
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

void DecisionTree::initializeRoot(){
    root = new DtreeNode();
}
DtreeNode* DecisionTree::buildTree(DtreeNode* node, vector<vector<double >>& valRange) {
    if (!node)
        return nullptr;
    if (features.empty())
        return nullptr;
    pair<int, double> splitFeatureAndValue = createSplitFeature(valRange);
    node->axis = splitFeatureAndValue.first;
    node->splitVal = splitFeatureAndValue.second;
    set<double> cls_left;
    set<double> cls_right;
    for (const auto& data : valRange){
        if (data[node->axis] == node->splitVal) {
            node->leftTreeVal.push_back(data);
            cls_left.insert(data.back());
        }
        else {
            node->rightTreeVal.push_back(data);
            cls_right.insert(data.back());
        }
    }
    if (cls_left.size()<=1){//belong to the same class
        node->left = new DtreeNode();
        node->left->isLeaf = true;
        node->left->leafValue = node->leftTreeVal;
    }
    else if (!node->leftTreeVal.empty()) {
        node->left = new DtreeNode();
        node->left = buildTree(node->left, node->leftTreeVal);
    } else{
        return nullptr;
    }
    if (cls_right.size()<=1){//belong to the same class
        node->right = new DtreeNode();
        node->right->isLeaf = true;
        node->right->leafValue = node->rightTreeVal;
    }
    else if (!node->rightTreeVal.empty()){
        node->right = new DtreeNode();
        node->right = buildTree(node->right, node->rightTreeVal);
    } else{
        return nullptr;
    }
    if (!node->right&&!node->left){
        node->isLeaf=true;
        node->leafValue = valRange;
    }
    return node;
}


pair<int, double> DecisionTree::createSplitFeature(vector<vector<double >>& valRange){
    priority_queue<pair<double, pair<int, double>>, vector<pair<double, pair<int, double>>>, std::greater<pair<double, pair<int, double>>>> minheap;
    //pair<double, pair<int, double>> first value is Gini value, second pair (pair<int, double>) first value is split
    //axis, second value is split value
    vector<map<double, int>> dataDivByFeature(indim);//vector size is num of axis, map's key is the value of feature, map's value is
    //num belong to feature'value
    vector<set<double>> featureVal(indim);//store value for each axis
    vector<map<pair<double, double>, int>> datDivByFC(indim);//vector size is num of axis, map's key is the feature value and class value, map's value is
    //num belong to that feature value and class
    set<double> cls;//store num of class
    for(const auto& featureId:features) {
        if (featureId<0)
            continue;
        map<double, int> dataDivByF;
        map<pair<double, double>, int> dtDivFC;
        set<double> fVal;
        for (auto& data:valRange){//below data[featureId] is the value of one feature axis, data.back() is class value
            cls.insert(data.back());
            fVal.insert(data[featureId]);
            if (dataDivByF.count(data[featureId]))
                dataDivByF[data[featureId]] += 1;
            else
                dataDivByF[data[featureId]] = 0;
            if (dtDivFC.count(std::make_pair(data[featureId], data.back())))
                dtDivFC[std::make_pair(data[featureId], data.back())] += 1;
            else
                dtDivFC[std::make_pair(data[featureId], data.back())] = 0;
        }
        featureVal[featureId] = fVal;
        dataDivByFeature[featureId] = dataDivByF;
        datDivByFC[featureId] = dtDivFC;
    }
    for (auto& featureId: features){//for each feature axis
        if (featureId<0)
            continue;
        for (auto& feVal: featureVal[featureId]){//for each feature value
            cout<<featureId<<","<<feVal<<endl;
            double gini1 = 0 ;
            double gini2 = 0 ;

            double prob1 = dataDivByFeature[featureId][feVal]/double(valRange.size());
            double prob2 = 1 - prob1;
            for (auto& c : cls){//for each class
                double pro1 = double(datDivByFC[featureId][std::make_pair(feVal, c)])/dataDivByFeature[featureId][feVal];
                gini1 += pro1*(1-pro1);
                int numC = 0;
                for (auto& feVal2: featureVal[featureId])
                    numC += datDivByFC[featureId][std::make_pair(feVal2, c)];
                double pro2 = double(numC-datDivByFC[featureId][std::make_pair(feVal, c)])/(valRange.size()-dataDivByFeature[featureId][feVal]);
                gini2 += pro2*(1-pro2);
            }
            double gini = prob1*gini1+prob2*gini2;

            minheap.push(std::make_pair(gini, std::make_pair(featureId, feVal)));
        }
    }
    features[minheap.top().second.first]=-1;
    return minheap.top().second;
}

void DecisionTree::showTree(DtreeNode* node) {
    if(node == nullptr)
        return;
    cout<<"the leaf class is "<< node->isLeaf<<endl;
    cout<<" the splitaxis is "<<node->axis<<endl;
    cout<<" the splitval is "<<node->splitVal<<endl;
    showTree(node->left);
    showTree(node->right);
}

void DecisionTree::run(){
    getData("../data/decisiontree.txt");
    createTrainTest();
    initializeRoot();
    buildTree(root, trainData);
    showTree(root);
}
#endif //MACHINE_LEARNING_DECISIONTREE_H
