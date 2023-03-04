#include "ncnn/layer.h"
#include "ncnn/net.h"
#include "model/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob.bin.h"
#include "model/rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128-100e_coco_prob.param.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <iostream>

#define RTMDET_NMS_THRESH  0.45 // nms threshold
#define RTMDET_INPUT_W_SIZE 640
#define RTMDET_INPUT_H_SIZE 352
#define RTMDET_CONF_THRESH 0.45 // threshold of bounding box prob
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}
static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}
static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static int detect_yolox(const cv::Mat& bgr, std::vector<Object>& objects)
{

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int w = img_w;
    int h = img_h;

    float scale = std::min((float)RTMDET_INPUT_W_SIZE / img_w, (float)RTMDET_INPUT_H_SIZE / img_h);
    // 计算缩放后的图像大小
    w = static_cast<int>(img_w * scale);
    h = static_cast<int>(img_h * scale);

    unsigned char* model = (unsigned char*)(rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_bin);
    unsigned char* param = (unsigned char*)(rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_bin);
    std::vector<int> inpNodes = std::vector<int>{ rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_images };
    std::vector<int> oupNodes = std::vector<int>
    {
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_box8,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_cls8,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_box16,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_cls16,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_box32,
        rtmdet_tiny_syncbn_fast_mobilenetv2_10xb128_100e_coco_prob_param_id::BLOB_cls32,
    };
    std::vector<std::vector<float>>	anchors;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);
    // pad to RTMDET_TARGET_SIZE rectangle
    int wpad = RTMDET_INPUT_W_SIZE - w;
    int hpad = RTMDET_INPUT_H_SIZE - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 1/57.375f, 1 / 57.12f, 1 / 58.395f };
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    int nThread = 1;
    ncnn::Net rtmdet;
    rtmdet.load_param(param);
    rtmdet.load_model(model);
    ncnn::Extractor ex = rtmdet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(nThread);

    ex.input(inpNodes[0], in_pad);
    std::vector<float> strides = std::vector<float>{ 8.f, 16.f, 32.f };
    std::vector<Object> proposals;
    std::chrono::steady_clock::time_point btime = std::chrono::steady_clock::now();
    for (int anchor_idx = 0; anchor_idx < strides.size(); anchor_idx++)
    {
        ncnn::Mat reg_blob, cls_blob;
        ex.extract(oupNodes[anchor_idx * 2 + 0], reg_blob);
        ex.extract(oupNodes[anchor_idx * 2 + 1], cls_blob);

        const int stride = strides[anchor_idx];
        int num_grid_x = reg_blob.w;
        int num_grid_y = reg_blob.h;
        int nClasses = 7;
        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                // find label with max score
                int label = -1;
                float score = -FLT_MAX;
                for (int k = 0; k < nClasses; k++)
                {
                    float s = cls_blob.channel(k).row(i)[j];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }


                if (score >= RTMDET_CONF_THRESH)
                {

                    float x0 = (j - reg_blob.channel(0).row(i)[j]) * stride - (wpad / 2.);
                    float y0 = (i - reg_blob.channel(1).row(i)[j]) * stride - (hpad / 2.);
                    float x1 = (j + reg_blob.channel(2).row(i)[j]) * stride - (wpad / 2.);
                    float y1 = (i + reg_blob.channel(3).row(i)[j]) * stride - (hpad / 2.);

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = label;
                    obj.prob = score;
                    proposals.push_back(obj);
                }
            }
        }
    }
    std::chrono::steady_clock::time_point etime = std::chrono::steady_clock::now();
    std::cout << "runSession once: " << std::chrono::duration_cast<std::chrono::milliseconds>(etime - btime).count() << " ms" << std::endl;
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, RTMDET_NMS_THRESH);
    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}
static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
    "safety_belt",
    "not_safety_belt",
    "person",
    "wheel",
    "dark_phone",
    "bright_phone",
    "hand"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imwrite("./temp.jpg", image);
    // cv::imshow("image", image);
    // cv::waitKey(0);
}
int main(int argc, char** argv)
{
    if (argc != 2)
    {
       fprintf(stderr, "Usage: %s [imagepath]/n", argv[0]);
       return -1;
    }

    const char* imagepath = argv[1];
    cv::Mat m = cv::imread(imagepath);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed/n", imagepath);
        return -1;
    }
    std::vector<Object> objects;
    detect_yolox(m, objects);
    draw_objects(m, objects);
}