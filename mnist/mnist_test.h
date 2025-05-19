#pragma once
#include <opencv2/opencv.hpp>
#include "mnist.h"
#include <string>
#include "mnist_mha.h"
#include "time_calc.h"

inline void mnist_test() {
    Mnist mnist("../mnist.safetensors");
    Tensor<float> t;
    int batch_size = 10;
    t.asShape({batch_size,28 * 28});
    cv::Mat img[10];
    for (int num = 0; num < 10; num++) {
        img[num] = cv::imread("../mnist/nums/" + std::to_string(num % 10) + ".png", cv::IMREAD_GRAYSCALE);
        cv::resize(img[num], img[num], {28 , 28});
    }
    for (int num = 0; num < batch_size; num++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                t.at(num, i * 28 + j) = img[num % 10].at<uchar>(i, j) / 255.;
            }
        }
    }
    
    startTimeCalc("mnist forward");
    auto res = mnist.forward(t);
    coutTimeCalc();
    std::cout << res;
}

inline void mnist_mha_test() {
    MnistMha mnist("../mnist_mha.safetensors");
    Tensor<float> t;
    int batch_size = 10000;
    t.asShape({batch_size,1, 28 * 28});
    cv::Mat img[10];
    for (int num = 0; num < 10; num++) {
        img[num] = cv::imread("../mnist/nums/" + std::to_string(num % 10) + ".png", cv::IMREAD_GRAYSCALE);
        cv::resize(img[num], img[num], {28 , 28});
    }
    for (int num = 0; num < batch_size; num++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                t.at(num, 0, i * 28 + j) = img[num % 10].at<uchar>(i, j) / 255.;
            }
        }
    }
    
    startTimeCalc("mnist_mha forward");
    auto res = mnist.forward(t);
    coutTimeCalc();
    // std::cout << res;
}