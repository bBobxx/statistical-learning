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



struct KdtreeNode {
    std::vector<double> val;//store val for feature
    int cls;//store class
    unsigned long axis;//split axis
    double splitVal;//mid val for axis
    std::vector<std::vector<double>> leftTreeVal;
    std::vector<std::vector<double>> rightTreeVal;
    KdtreeNode* parent;
    KdtreeNode* left;
    KdtreeNode* right;
    KdtreeNode(): cls(0), axis(0), splitVal(0.0), parent(nullptr), left(nullptr), right(nullptr){};

};


class Knn: public Base{
private:
    std::stack<unsigned long> axisVec;
    KdtreeNode* root = new KdtreeNode();
    unsigned long K;
    std::priority_queue<std::pair<double, KdtreeNode*>> maxHeap;
public:
    virtual void getData(const std::string& filename);
    virtual void run();
    void createTrainTest(const float& trainTotalRatio);
    KdtreeNode* buildTree(KdtreeNode*root, std::vector<std::vector<double>>& data, std::stack<unsigned long>& axisstack);
    void setRoot();
    void createSplitAxis();
    KdtreeNode* getRoot(){return root;}
    void setK(unsigned long k){K = k;}
    void findKNearest(std::vector<double>& testD);
    double computeDis(const std::vector<double>& v1, const std::vector<double>& v2);
    void DeleteRoot(KdtreeNode *pRoot);
    void showTree(KdtreeNode* root);
    ~Knn();
};



#endif //MACHINE_LEARNING_KNN_H
