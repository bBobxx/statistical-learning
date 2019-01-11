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

#endif //MACHINE_LEARNING_DECISIONTREE_H
