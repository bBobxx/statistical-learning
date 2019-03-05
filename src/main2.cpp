//
// Created by wyb on 18-12-13.
//

#include <iostream>
#include <vector>
#include "perceptron.h"
#include "knn.h"
#include "NavieBayes.h"
#include "DecisionTree.h"
#include "Logistic.h"
#include "SVM.h"
#include "AdaBoost.h"
#include "GMM.h"
using std::vector;
using std::cout;
using std::endl;


int main() {
    //Base* obj = new Perceptron();
    //Base* obj = new Knn();
    //Base* obj = new NavieBayes();
    //Base* obj = new DecisionTree();
    //Base* obj = new Logistic();
    //Base* obj = new SVM();
    //Base* obj = new AdaBoost();
    Base* obj = new GMM();
    obj->run();
    delete obj;
    return 0;
}
