#pragma once

#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// paddle
#include "paddle_inference_api.h"
// yaml
#include "yaml-cpp/yaml.h"

#include "config_parser.h"
#include "picodet_postprocess.h"
#include "preprocess_op.h"
#include "utils.h"

using namespace paddle_infer;
using namespace PaddleDetection;

// PaddleDetection目标检测类
class PPDet
{
public:
    // 构造、析构函数
    explicit PPDet(std::string cfgfile);
    ~PPDet();
    // 主流程
    void Detect(cv::Mat& srcImg);

private:
    // 目标检测推理对象
    std::shared_ptr<Predictor> m_detector;
    // 预处理类对象
    Preprocessor m_preprocessor;
    // 输入
    ImageBlob m_inputs;
    // 参数解析对象，解析的是生成的infer_cfg.yml
    ConfigPaser m_config;
    // 结果
    std::vector<PaddleDetection::ObjectResult> m_result;
    // 通用参数
    std::string m_device;       // 推理设备，CPU/GPU/XPU
    bool m_use_mkldnn;          // 是否使用mkldnn库
    int m_cpu_num_threads;      // CPU预测时的线程数，在机器核数充足的情况下，该值越大，预测速度越快
    int m_gpu_id;               // GPU id
    std::string m_run_mode;     // 模型精度，paddle（无tensorrt下）或trt_fp32/trt_fp16/trt_int8（tensorrt）
    // 检测模型参数
    std::string m_model_path;   // 模型路径
    int m_trt_min_shape;        // TRT模型最小DynamicShapeI
    int m_trt_max_shape;        // TRT模型最大DynamicShapeI
    int m_trt_opt_shape;        // TRT模型输入大小
    bool m_trt_calib_mode;      // If the model is produced by TRT offline quantitative calibration,trt_calib_mode need to set True
    // 推理模型中获取的参数
    float m_threshold;          // 阈值
    bool m_use_dynamic_shape;
    int m_min_subgraph_size;

private:
    // 加载paddle推理模型
    void LoadModel();
    // 预处理
    void Preprocess(const cv::Mat &srcImg);
    // 后处理
    void Postprocess(const cv::Mat& srcImg, std::vector<PaddleDetection::ObjectResult>& result,
                    std::vector<int> bbox_num, std::vector<float> output_data_,
                    std::vector<int> output_mask_data_, float nms_threshold, bool is_rbox);
    // 画结果
    void DrawResult(cv::Mat& srcImg);
};
