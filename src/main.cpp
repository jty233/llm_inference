#include <algorithm>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "model_parse.h"
#include "tensor.h"
#include "modules/linear.h"
#include "module.h"
#include "mnist_test.h"
using namespace std;
int main()
{
    ios::sync_with_stdio(false);
    cout.tie(0);
    mnist_test();

}