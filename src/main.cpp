#include <iostream>
#include "paddledet.h"

int main(void)
{
    PPDet det = PPDet("../config/ppdet.yaml");
    
    std::vector<cv::String> cv_all_img_names;
    cv::glob("../images", cv_all_img_names);

    for (int i = 0; i < cv_all_img_names.size(); ++i)
    {
        cv::Mat srcImg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
        std::vector<std::string> res;

        auto start = std::chrono::system_clock::now();
        det.Detect(srcImg);
        auto end = std::chrono::system_clock::now();
        std::cout << "------------------------------" << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        cv::imwrite("../result.jpg", srcImg);
    }
}