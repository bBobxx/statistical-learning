//
// Created by wyb on 18-12-17.
//

#ifndef MACHINE_LEARNING_KNN_H
#define MACHINE_LEARNING_KNN_H

#include <vector>
#include <stack>
#include <utility>
#include <queue>
#include "model_base.h"
using std::vector;
using std::stack;
using std::pair;
using std::cout;
using std::endl;
using std::priority_queue;


struct KdtreeNode {
    vector<double> val;//store val for feature
    int cls;//store class
    unsigned long axis;//split axis
    double splitVal;//mid val for axis
    vector<vector<double>> leftTreeVal;
    vector<vector<double>> rightTreeVal;
    KdtreeNode* parent;
    KdtreeNode* left;
    KdtreeNode* right;
    KdtreeNode(): cls(0), axis(0), splitVal(0.0), parent(nullptr), left(nullptr), right(nullptr){};

};


class Knn: public Base{
private:
    stack<unsigned long> axisVec;
    KdtreeNode* root = new KdtreeNode();
    unsigned long K;
    priority_queue<pair<double, KdtreeNode*>> maxHeap;
public:
    virtual void getData(const std::string& filename);
    virtual void run();
    void createTrainTest(const float& trainTotalRatio);
    KdtreeNode* buildTree(KdtreeNode*root, vector<vector<double>>& data, stack<unsigned long>& axisStack);
    void setRoot();
    void createSplitAxis();
    KdtreeNode* getRoot(){return root;}
    void setK(unsigned long k){K = k;}
    void findKNearest(vector<double>& testD);
    double computeDis(const vector<double>& v1, const vector<double>& v2);
    void DeleteRoot(KdtreeNode *pRoot);
    void showTree(KdtreeNode* root);
    ~Knn();
};



#endif //MACHINE_LEARNING_KNN_H
