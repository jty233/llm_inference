#pragma once
#include <opencv2/opencv.hpp>
#include "mnist.h"
#include <iostream>
#include <cstring>
#include <string>

inline void mnist_test() {
    Mnist mnist("../mnist.safetensors");
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
    

    auto res = mnist.forward(t);
    std::cout << res << std::endl;
}