//
// Created by wyb on 18-12-18.
//
#include "knn.h"
using std::string;
using std::vector;
using std::pair;
using std::priority_queue;
using std::stack;

void Knn::getData(const std::string &filename) {
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


void Knn::createTrainTest(const float& trainTotalRatio){
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


void Knn::createSplitAxis(){
    cout<<"createSplitAxis..."<<endl;
    //the last element of trainData is gt
    vector<pair<unsigned long, double>> varianceVec;
    auto sumv = trainData[0];
    for(unsigned long i=1;i<trainData.size();++i){
        sumv = sumv + trainData[i];
    }
    auto meanv = sumv/trainData.size();
    vector<decltype(trainData[0]-meanv)> subMean;
    for(const auto& c:trainData)
        subMean.push_back(c-meanv);
    for (unsigned long i = 0; i < trainData.size(); ++i) {
        for (unsigned long j = 0; j < indim; ++j) {
            subMean[i][j] *= subMean[i][j];
        }

    }
    auto varc = subMean[0];
    for(unsigned long i=1;i<subMean.size();++i){
        varc = varc + subMean[i];
    }
    auto var = varc/subMean.size();
    for(unsigned long i=0;i<var.size()-1;++i){//here not contain the axis of gt
        varianceVec.push_back(pair<unsigned long, double>(i, var[i]));
    }
    std::sort(varianceVec.begin(), varianceVec.end(), [](pair<unsigned long, double> &left, pair<unsigned long, double> &right) {
        return left.second < right.second;
    });
    for(const auto& variance:varianceVec){
        axisVec.push(variance.first);//the maximum variance is on the top
    }
    cout<<"createSplitAxis over"<<endl;
}



void Knn::setRoot() {
    if(axisVec.empty()){
        cout<<"please run createSplitAxis first."<<endl;
        throw axisVec.empty();
    }
    auto axisv = axisVec;
    auto axis = axisv.top();
    axisv.pop();
    std::sort(trainData.begin(), trainData.end(), [&axis](vector<double> &left, vector<double > &right) {
        return left[axis]<right[axis];
    });
    unsigned long mid = trainData.size()/2;
    for(unsigned long i = 0; i < trainData.size(); ++i){
        if(i!=mid){
            if (i<mid)
                root->leftTreeVal.push_back(trainData[i]);

            else
                root->rightTreeVal.push_back(trainData[i]);
        } else{
            root->val.assign(trainData[i].begin(),trainData[i].end()-1);
            root->splitVal = trainData[i][axis];
            root->axis = axis;
            root->cls = *(trainData[i].end()-1);
        }
    }
    cout<<"root node set over"<<endl;
}



KdtreeNode* Knn::buildTree(KdtreeNode*root, vector<vector<double>>& data, stack<unsigned long>& axisStack) {

    stack<unsigned long> aS;
    if(axisStack.empty())
        aS=axisVec;
    else
        aS=axisStack;
    auto node = new KdtreeNode();
    node->parent = root;

    auto axis2 = aS.top();
    aS.pop();

    std::sort(data.begin(), data.end(), [&axis2](vector<double> &left, vector<double > &right) {
        return left[axis2]<right[axis2];
    });

    unsigned long mid = data.size()/2;

    if(node->leftTreeVal.empty()&&node->rightTreeVal.empty()){
        for(unsigned long i = 0; i < data.size(); ++i){
            if(i!=mid){
                if (i<mid)
                    node->leftTreeVal.push_back(data[i]);
                else
                    node->rightTreeVal.push_back(data[i]);

            } else{
                node->val.assign(data[i].begin(),data[i].end()-1);
                node->splitVal = data[i][axis2];
                node->axis = axis2;
                node->cls = *(data[i].end()-1);
            }
        }
    }

    if(!node->leftTreeVal.empty()){
        node->left = buildTree(node, node->leftTreeVal, aS);
    }
    if(!node->rightTreeVal.empty()){
        node->right = buildTree(node, node->rightTreeVal, aS);
    }

    return node;
}


void Knn::showTree(KdtreeNode* root) {
    if(root == nullptr)
        return;
    cout<<"the feature is ";
    for(const auto& c:root->val)
        cout<<c<<" ";
    cout<<" the class is "<<root->cls<<endl;
    showTree(root->left);
    showTree(root->right);
}


void Knn::findKNearest(vector<double>& testD){
    cout<<"the test data is(the last is class) ";
    for(const auto& c:testD)
        cout<<c<<" ";
    cout<<"\nsearching "<<K<<" nearest val..."<<endl;
    stack<KdtreeNode*> path;

    auto curNode = root;
    while(curNode!= nullptr){
        path.push(curNode);
        if(testD[curNode->axis]<=curNode->splitVal)
            curNode = curNode->left;
        else
            curNode = curNode->right;
    }
    while(!path.empty()){
        auto curN = path.top();
        path.pop();
        vector<double> testDF(testD.begin(),testD.end()-1);
        double dis=0.0;
        dis = computeDis(testDF, curN->val);
        if(maxHeap.size()<K){
            maxHeap.push(pair<double, KdtreeNode*>(dis, curN));
        }
        else{
            if(dis<maxHeap.top().first){
                maxHeap.pop();
                maxHeap.push(pair<double, KdtreeNode*>(dis, curN));
            }
        }
        if(path.empty())
            continue;
        auto curNparent = path.top();
        KdtreeNode* curNchild;
        if(testDF[curNparent->axis]<=curNparent->splitVal)
            curNchild = curNparent->right;
        else
            curNchild = curNparent->left;
        if(curNchild == nullptr)
            continue;
        double childDis = computeDis(testDF, curNchild->val);
        if(childDis<maxHeap.top().first){
            maxHeap.pop();
            maxHeap.push(pair<double, KdtreeNode*>(childDis, curNchild));
            while(curNchild!= nullptr){//add subtree to path
                path.push(curNchild);
                if(testD[curNchild->axis]<=curNchild->splitVal)
                    curNchild = curNchild->left;
                else
                    curNchild = curNchild->right;
            }
        }
    }

}


double Knn::computeDis(const vector<double>& v1, const vector<double>& v2){
    auto v = v1 - v2;
    double di = v*v;
    return di;
}


void Knn::DeleteRoot(KdtreeNode *pRoot) //<根据根节点删除整棵树
{
    if (pRoot == nullptr) {
        return;
    }
    KdtreeNode *pLeft = pRoot->left;
    KdtreeNode *pRight = pRoot->right;
    delete pRoot;
    pRoot = nullptr;
    if (pLeft) {
        DeleteRoot(pLeft);
    }
    if (pRight) {
        DeleteRoot(pRight);
    }
    return;
}


Knn::~Knn(){
    DeleteRoot(root);
}


void Knn::run(){
    getData("../data/perceptrondata.txt");
    createTrainTest(0.6);
    createSplitAxis();
    setRoot();
    root->left = buildTree(root, root->leftTreeVal, axisVec);
    root->right = buildTree(root, root->rightTreeVal, axisVec);
    cout<<"show the tree in preorder traversal."<<endl;
    showTree(root);
    setK(2);
    for(auto& a:testData) {
        findKNearest(a);
        while (!maxHeap.empty()) {
            cout << "dis: " << maxHeap.top().first;
            cout << " val: ";
            for (auto &c :maxHeap.top().second->val)
                cout << c << " ";
            cout << endl;
            maxHeap.pop();
        }
    }
}

