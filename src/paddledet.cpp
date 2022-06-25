#include "paddledet.h"

// 构造函数
PPDet::PPDet(std::string cfgfile) 
{
    // 加载配置文件
    YAML::Node det_conf = YAML::LoadFile(cfgfile);
    // 通用参数
    m_device = det_conf["common"]["device"].as<std::string>();
    m_use_mkldnn = det_conf["common"]["use_mkldnn"].as<bool>();
    m_cpu_num_threads = det_conf["common"]["cpu_num_threads"].as<int>();
    m_gpu_id = det_conf["common"]["gpu_id"].as<int>();
    m_run_mode = det_conf["common"]["run_mode"].as<std::string>();
    // 检测模型参数
    m_model_path = det_conf["det"]["model"].as<std::string>();
    m_trt_min_shape = det_conf["det"]["trt_min_shape"].as<int>();
    m_trt_max_shape = det_conf["det"]["trt_max_shape"].as<int>();
    m_trt_opt_shape = det_conf["det"]["trt_opt_shape"].as<int>();
    m_trt_calib_mode = det_conf["det"]["trt_calib_mode"].as<bool>();
    // 加载推理模型的配置
    m_config.load_config(m_model_path);
    m_use_dynamic_shape = m_config.use_dynamic_shape_;
    m_min_subgraph_size = m_config.min_subgraph_size_;
    m_threshold = m_config.draw_threshold_;
    // 获取预处理操作类型
    m_preprocessor.Init(m_config.preprocess_info_);
    // 加载模型
    LoadModel();
}

// 析构函数
PPDet::~PPDet()
{
}


// 加载模型并创建推理对象
void PPDet::LoadModel() 
{
    int batch_size = 1;
    paddle_infer::Config config;
    std::string prog_file = m_model_path + OS_PATH_SEP + "model.pdmodel";
    std::string params_file = m_model_path + OS_PATH_SEP + "model.pdiparams";
    config.SetModel(prog_file, params_file);
    // GPU
    if (m_device == "GPU") 
    {
        config.EnableUseGpu(200, m_gpu_id);
        config.SwitchIrOptim(true);
        // tensorrt模型设置
        if (m_run_mode != "paddle") 
        {
            // 模型精度
            auto precision = paddle_infer::Config::Precision::kFloat32;
            if (m_run_mode == "trt_fp32")
                precision = paddle_infer::Config::Precision::kFloat32;
            else if (m_run_mode == "trt_fp16")
                precision = paddle_infer::Config::Precision::kHalf;
            else if (m_run_mode == "trt_int8")
                precision = paddle_infer::Config::Precision::kInt8;
            else
                std::cout << "run_mode should be 'paddle', 'trt_fp32', 'trt_fp16' or 'trt_int8'" << std::endl;
            // 设置tensorrt
            config.EnableTensorRtEngine(1 << 30, batch_size, m_min_subgraph_size, precision, false, m_trt_calib_mode);
            // 设置dynamic shape
            if (m_use_dynamic_shape) 
            {
                // set DynamicShsape for image tensor
                const std::vector<int> min_input_shape = {
                    1, 3, m_trt_min_shape, m_trt_min_shape};
                const std::vector<int> max_input_shape = {
                    1, 3, m_trt_max_shape, m_trt_max_shape};
                const std::vector<int> opt_input_shape = {
                    1, 3, m_trt_opt_shape, m_trt_opt_shape};
                const std::map<std::string, std::vector<int>> map_min_input_shape = {
                    {"image", min_input_shape}};
                const std::map<std::string, std::vector<int>> map_max_input_shape = {
                    {"image", max_input_shape}};
                const std::map<std::string, std::vector<int>> map_opt_input_shape = {
                    {"image", opt_input_shape}};

                config.SetTRTDynamicShapeInfo(map_min_input_shape, map_max_input_shape, map_opt_input_shape);
                std::cout << "TensorRT dynamic shape enabled" << std::endl;
            }
        }
    }
    // XPU
    else if (m_device == "XPU") 
    {
        config.EnableXpu(10 * 1024 * 1024);
    }
    // CPU
    else 
    {
        config.DisableGpu();
        if (m_use_mkldnn) 
        {
            config.EnableMKLDNN();
            // cache 10 different shapes for mkldnn to avoid memory leak
            config.SetMkldnnCacheCapacity(10);
        }
        config.SetCpuMathLibraryNumThreads(m_cpu_num_threads);
    }
    config.SwitchUseFeedFetchOps(false);
    config.SwitchIrOptim(true);
    config.DisableGlogInfo();
    // Memory optimization
    config.EnableMemoryOptim();
    m_detector = std::move(CreatePredictor(config));
}


// 预处理
void PPDet::Preprocess(const cv::Mat& srcImg)
{
  cv::Mat im = srcImg.clone();
  cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
  m_preprocessor.Run(&im, &m_inputs);
}


// 后处理
void PPDet::Postprocess(const cv::Mat& srcImg, std::vector<PaddleDetection::ObjectResult>& result,
    std::vector<int> bbox_num,
    std::vector<float> output_data_,
    std::vector<int> output_mask_data_,
    float nms_threshold,
    bool is_rbox = false) 
{
    result.clear();
    int total_num = std::accumulate(bbox_num.begin(), bbox_num.end(), 0);
    int out_mask_dim = -1;
    if (m_config.mask_)
    {
        out_mask_dim = output_mask_data_.size() / total_num;
    }

    int rh = 1;
    int rw = 1;
    if (m_config.arch_ == "Face") 
    {
        rh = srcImg.rows;
        rw = srcImg.cols;
    }
    for (int j = 0; j < total_num; ++j) 
    {
        if (is_rbox) 
        {
            // Class id
            int class_id = static_cast<int>(round(output_data_[0 + j * 10]));
            // Confidence score
            float score = output_data_[1 + j * 10];
            if(score < 0.3)
                continue;
            int x1 = (output_data_[2 + j * 10] * rw);
            int y1 = (output_data_[3 + j * 10] * rh);
            int x2 = (output_data_[4 + j * 10] * rw);
            int y2 = (output_data_[5 + j * 10] * rh);
            int x3 = (output_data_[6 + j * 10] * rw);
            int y3 = (output_data_[7 + j * 10] * rh);
            int x4 = (output_data_[8 + j * 10] * rw);
            int y4 = (output_data_[9 + j * 10] * rh);

            PaddleDetection::ObjectResult result_item;
            result_item.rect = {x1, y1, x2, y2, x3, y3, x4, y4};
            result_item.class_id = class_id;
            result_item.confidence = score;
            result.push_back(result_item);
        } 
        else
        {
            // Class id
            int class_id = static_cast<int>(round(output_data_[0 + j * 6]));
            // Confidence score
            float score = output_data_[1 + j * 6];
            if(score < 0.3)
                continue;
            int xmin = (output_data_[2 + j * 6] * rw);
            int ymin = (output_data_[3 + j * 6] * rh);
            int xmax = (output_data_[4 + j * 6] * rw);
            int ymax = (output_data_[5 + j * 6] * rh);
            int wd = xmax - xmin;
            int hd = ymax - ymin;

            PaddleDetection::ObjectResult result_item;
            result_item.rect = {xmin, ymin, xmax, ymax};
            result_item.class_id = class_id;
            result_item.confidence = score;

            if (m_config.mask_) 
            {
                std::vector<int> mask;
                for (int k = 0; k < out_mask_dim; ++k) 
                {
                    if (output_mask_data_[k + j * out_mask_dim] > -1)
                        mask.push_back(output_mask_data_[k + j * out_mask_dim]);
                }
                result_item.mask = mask;
            }
            result.push_back(result_item);
        }
    }

    // NMS
    PaddleDetection::nms(result, nms_threshold);
}

// 目标检测主流程
void PPDet::Detect(cv::Mat& srcImg)
{
    // 输入数据
    std::vector<float> in_data;           // 输入网络的数据
    std::vector<float> im_shape(2);       // 预处理后的图像的shape
    std::vector<float> scale_factor(2);   // 预处理后的图像相比原始图像的scale
    cv::Mat in_net_img;                   // 预处理后的图像
    // 输出数据
    std::vector<const float*> output_data_list_;
    std::vector<int> out_bbox_num_data_;
    std::vector<int> out_mask_data_;

    // 预处理
    Preprocess(srcImg);
    im_shape[0] = m_inputs.im_shape_[0];
    im_shape[1] = m_inputs.im_shape_[1];
    scale_factor[0] = m_inputs.scale_factor_[0];
    scale_factor[01] = m_inputs.scale_factor_[1];
    // 获取输入数据
    in_data.insert(in_data.end(), m_inputs.im_data_.begin(), m_inputs.im_data_.end());
    // 预处理后的图像
    in_net_img = m_inputs.in_net_im_;

    // 准备输入的tensor
    auto input_names = m_detector->GetInputNames();
    for (const auto& tensor_name : input_names) 
    {
        auto in_tensor = m_detector->GetInputHandle(tensor_name);
        if (tensor_name == "image") 
        {
            int rh = m_inputs.in_net_shape_[0];
            int rw = m_inputs.in_net_shape_[1];
            in_tensor->Reshape({1, 3, rh, rw});
            in_tensor->CopyFromCpu(in_data.data());
        } 
        else if (tensor_name == "im_shape") 
        {
            in_tensor->Reshape({1, 2});
            in_tensor->CopyFromCpu(im_shape.data());
        } 
        else if (tensor_name == "scale_factor") 
        {
            in_tensor->Reshape({1, 2});
            in_tensor->CopyFromCpu(scale_factor.data());
        }
    }

    // 进行一个模型推理预测的大动作
    m_detector->Run();

    // 获取输出tensor
    std::vector<std::vector<float>> out_tensor_list;
    std::vector<std::vector<int>> output_shape_list;
    auto output_names = m_detector->GetOutputNames();
    for (int j = 0; j < output_names.size(); j++)
    {
        auto output_tensor = m_detector->GetOutputHandle(output_names[j]);
        std::vector<int> output_shape = output_tensor->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
        output_shape_list.push_back(output_shape);
        if (m_config.mask_ && (j == 2))
        {
            out_mask_data_.resize(out_num);
            output_tensor->CopyToCpu(out_mask_data_.data());
        } 
        else if (output_tensor->type() == paddle_infer::DataType::INT32) 
        {
            out_bbox_num_data_.resize(out_num);
            output_tensor->CopyToCpu(out_bbox_num_data_.data());
        } 
        else
        {
            std::vector<float> out_data;
            out_data.resize(out_num);
            output_tensor->CopyToCpu(out_data.data());
            out_tensor_list.push_back(out_data);
        }
    }

    bool is_rbox = false;
    int reg_max = 7;
    int num_class = 80;
    // 后处理
    m_result.clear();
    // PicoDet
    if (m_config.arch_ == "PicoDet")
    {
        for (int i = 0; i < out_tensor_list.size(); i++) 
        {
            if (i == 0)
                num_class = output_shape_list[i][2];
            if (i == m_config.fpn_stride_.size())
                reg_max = output_shape_list[i][2] / 4 - 1;
            float* buffer = new float[out_tensor_list[i].size()];
            memcpy(buffer, &out_tensor_list[i][0], out_tensor_list[i].size() * sizeof(float));
            output_data_list_.push_back(buffer);
        }
        PaddleDetection::PicoDetPostProcess(
            &m_result,
            output_data_list_,
            m_config.fpn_stride_,
            m_inputs.im_shape_,
            m_inputs.scale_factor_,
            m_config.nms_info_["score_threshold"].as<float>(),
            m_config.nms_info_["nms_threshold"].as<float>(),
            num_class,
            reg_max);
    }
    // 其他网络模型
    else 
    {
        is_rbox = output_shape_list[0][output_shape_list[0].size() - 1] % 10 == 0;
        Postprocess(srcImg,
                    m_result,
                    out_bbox_num_data_,
                    out_tensor_list[0],
                    out_mask_data_,
                    m_config.nms_info_["nms_threshold"].as<float>(),
                    is_rbox);
    }
    DrawResult(srcImg);
}

// 画结果
void PPDet::DrawResult(cv::Mat& srcImg)
{
    for (size_t i = 0; i < m_result.size(); ++i)
	{
		if (m_result[i].rect.size() > 6) 
        {
            for (int k = 0; k < 4; k++) 
            {
                cv::Point pt1 = cv::Point(m_result[i].rect[(k * 2) % 8],
                                        m_result[i].rect[(k * 2 + 1) % 8]);
                cv::Point pt2 = cv::Point(m_result[i].rect[(k * 2 + 2) % 8],
                                        m_result[i].rect[(k * 2 + 3) % 8]);
                cv::line(srcImg, pt1, pt2, cv::Scalar(0x27, 0xC1, 0x36), 2);
            }

        }
        else
        {
            int w = m_result[i].rect[2] - m_result[i].rect[0];
            int h = m_result[i].rect[3] - m_result[i].rect[1];
            cv::Rect roi = cv::Rect(m_result[i].rect[0], m_result[i].rect[1], w, h);
            cv::rectangle(srcImg, roi, cv::Scalar(0x27, 0xC1, 0x36), 2);
        }
		cv::putText(srcImg, std::to_string((int)m_result[i].class_id), cv::Point(m_result[i].rect[0], m_result[i].rect[1] - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
	}
}