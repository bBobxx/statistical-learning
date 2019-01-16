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
using std::vector;
using std::cout;
using std::endl;


int main() {
    //Base* obj = new Perceptron();
    //Base* obj = new Knn();
    //Base* obj = new NavieBayes();
    //Base* obj = new DecisionTree();
    Base* obj = new Logistic();
    obj->run();
    delete obj;
    return 0;
}
