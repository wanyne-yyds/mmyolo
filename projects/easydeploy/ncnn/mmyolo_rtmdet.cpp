#include "ncnn/layer.h"
#include "ncnn/net.h"
#include "model/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob.bin.h"
#include "model/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob.param.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <iostream>

#define YOLOX_NMS_THRESH  0.45 // nms threshold
#define RTMDET_INPUT_W_SIZE 640
#define RTMDET_INPUT_H_SIZE 352
#define YOLOX_CONF_THRESH 0.25 // threshold of bounding box prob

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_yolox(const cv::Mat& bgr, std::vector<Object>& objects)
{

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int w = img_w;
    int h = img_h;

    float scale = std::min((float)RTMDET_INPUT_W_SIZE / img_w, (float)RTMDET_INPUT_H_SIZE / img_h);
    // 计算缩放后的图像大小
    int w = static_cast<int>(img_w * scale);
    int h = static_cast<int>(img_h * scale);

    unsigned char* model = (unsigned char*)(rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_bin);
	unsigned char* param = (unsigned char*)(rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_bin);
	std::vector<int> inpNodes = std::vector<int>{rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_images};
	std::vector<int> oupNodes = std::vector<int>
    {
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_bbox8,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_cls8,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_bbox16,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_cls16,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_bbox32,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_cls32,
    };
    std::vector<std::vector<float>>	anchors;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);
    // pad to YOLOX_TARGET_SIZE rectangle
    int wpad = RTMDET_INPUT_W_SIZE - new_width;
    int hpad = RTMDET_INPUT_H_SIZE - new_height;

    // cv::Mat shrink;
    // cv::Size dsize = cv::Size(new_width, new_height);
    // cv::resize(bgr, shrink, dsize, 0, 0, cv::INTER_AREA);
    // std::stringstream str0;
    // std::stringstream str1;
    // str0 << "./" << (0 + 1) * 10 << ".png";
    // imwrite(str0.str(), shrink);
    // cv::Mat border;
    // cv::copyMakeBorder(shrink, border, 0, hpad, 0, wpad, cv::BORDER_CONSTANT);
    // str1 << "./" << (1 + 1) * 10 << ".png";
    // imwrite(str1.str(), border);

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f);
    int nThread = 1;
    ncnn::Net yolox;
    yolox.load_param(param);
    yolox.load_model(model);
    ncnn::Extractor ex = yolox.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(nThread);

    ex.input(inpNodes[0], in_pad);
    std::vector<float> strides = std::vector<float>{ 8.f, 16.f, 32.f };
    std::vector<Object> proposals;
    // // std::vector<GridAndStride> grid_strides;
    // // generate_grids_and_stride(YOLOX_TARGET_SIZE, strides, grid_strides);
    // std::chrono::steady_clock::time_point btime = std::chrono::steady_clock::now();
    // for (int anchor_idx = 0; anchor_idx < strides.size(); anchor_idx++)
    // {
	// 	ncnn::Mat reg_blob, obj_blob, cls_blob;
	// 	ex.extract(oupNodes[anchor_idx * 3 + 0], reg_blob);
	// 	ex.extract(oupNodes[anchor_idx * 3 + 1], obj_blob);
	// 	ex.extract(oupNodes[anchor_idx * 3 + 2], cls_blob);

    //     const int stride = strides[anchor_idx];
    //     int num_grid_x = reg_blob.w;
    //     int num_grid_y = reg_blob.h;
    //     int nClasses = 11;
    //     for (int i = 0; i < num_grid_y; i++)
    //     {
    //         for (int j = 0; j < num_grid_x; j++)
    //         {
    //             float box_objectness = obj_blob.channel(0).row(i)[j];
    //             if (box_objectness < YOLOX_CONF_THRESH)
    //                 continue;
    //             for (int class_idx = 0; class_idx < nClasses; class_idx++)
    //             {
    //                 float box_cls_score = cls_blob.channel(class_idx).row(i)[j];
    //                 float box_prob = box_objectness * box_cls_score;
    //                 if (box_prob < YOLOX_CONF_THRESH)
    //                     continue;
    //                 // class loop
    //                 float x_center = (reg_blob.channel(0).row(i)[j] + j) * stride;
    //                 float y_center = (reg_blob.channel(1).row(i)[j] + i) * stride;
    //                 float w = exp(reg_blob.channel(2).row(i)[j]) * stride;
    //                 float h = exp(reg_blob.channel(3).row(i)[j]) * stride;
    //                 float x0 = x_center - w * 0.5f;
    //                 float y0 = y_center - h * 0.5f;
    //                 Object obj;
    //                 obj.rect.x = x0;
    //                 obj.rect.y = y0;
    //                 obj.rect.width = w;
    //                 obj.rect.height = h;
    //                 obj.label = class_idx;
    //                 obj.prob = box_prob;
    //                 proposals.push_back(obj);
    //             }
    //         }
    //     }
    // }
    // std::chrono::steady_clock::time_point etime = std::chrono::steady_clock::now();
    // std::cout << "runSession once: " << std::chrono::duration_cast<std::chrono::milliseconds>(etime - btime).count() << " ms" << std::endl;
    // // sort all proposals by score from highest to lowest
    // qsort_descent_inplace(proposals);
    // // apply nms with nms_threshold
    // std::vector<int> picked;
    // nms_sorted_bboxes(proposals, picked, YOLOX_NMS_THRESH);
    // int count = picked.size();
    // objects.resize(count);
    // for (int i = 0; i < count; i++)
    // {
    //     objects[i] = proposals[picked[i]];

    //     // adjust offset to original unpadded
    //     float x0 = (objects[i].rect.x) / scale;
    //     float y0 = (objects[i].rect.y) / scale;
    //     float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
    //     float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

    //     // clip
    //     x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    //     y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    //     x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    //     y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

    //     objects[i].rect.x = x0;
    //     objects[i].rect.y = y0;
    //     objects[i].rect.width = x1 - x0;
    //     objects[i].rect.height = y1 - y0;
    // }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    std::vector<Object> objects;
    detect_yolox(m, objects);
}