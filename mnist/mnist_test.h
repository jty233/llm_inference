#pragma once
#include <chrono>
#include <opencv2/opencv.hpp>
#include "mnist.h"
#include <iostream>
#include <cstring>
#include <ostream>
#include <string>
#include "time_calc.h"

inline void mnist_test() {
    startTimeCalc("mnist load");
    Mnist mnist("../mnist.safetensors");
    finishTimeCalc();
    Tensor<float> t;
    t.asShape({10,28 * 28});
    cv::Mat img;
    for (int num = 0; num < 10; num++) {
        img = cv::imread("../mnist/nums/" + std::to_string(num) + ".png", cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, {28 , 28});
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                t.at(num, i * 28 + j) = img.at<uchar>(i, j) / 255.;
            }
        }
    }
    
    startTimeCalc("mnist forward");
    auto res = mnist.forward(t);
    finishTimeCalc();
    std::cout << res;
}