#ifndef __YOLOV3_INTERFACE_H_
#define __YOLOV3_INTERFACE_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>
#include <typeinfo>
#include <sys/time.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include "PluginFactory.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yolo_layer.h"
#include "image.h"
#include <chrono>

using namespace nvinfer1;
using namespace nvcaffeparser1;

static const int INPUT_H = 320;
static const int INPUT_W = 320;
static const int INPUT_C = 3;

static const int TIMING_ITERATIONS = 1;

static const int OUTPUT_SIZE0 = 18*10*10;
static const int OUTPUT_SIZE1 = 18*20*20;
static const int OUTPUT_SIZE2 = 18*40*40;


static Logger gLogger;

const char* INPUT_BLOB_NAME   = "data";

const char* OUTPUT_BLOB_NAME0 = "layer82-conv";
const char* OUTPUT_BLOB_NAME1 = "layer94-conv";
const char* OUTPUT_BLOB_NAME2 = "layer106-conv";

class TRTYOLOv3 {
public:
    void initialize(int classes, std::string deployFile, std::string modelFile);
    void detect(const std::string& img_name, const std::string& test_image_path, const std::string& output_folder);
    void free();
    int clsDetect;
private:
    void caffeToGIEModel(const std::string& deployFile, const std::string& modelFile, 
        unsigned int maxBatchSize, nvcaffeparser1::IPluginFactory* pluginFactory, IHostMemory *&gieModelStream);
    // nvcaffeparser1::IPluginFactory* pluginFactory;
    PluginFactory pluginFactory;
    IHostMemory *gieModelStream{nullptr};

    void deserializeEngine();
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext *context;

    void doInference(IExecutionContext& context, float* input, int batchSize,
        const char* imgFilename, const char* img_name, const char* out_folder);
};

extern "C" {
    void trtYolov3Init(int classes, std::string deployFile, std::string modelFile);
    void trtYolov3Detect(const std::string& img_name, const std::string& test_image_path, const std::string& output_folder);
    void trtYolov3Free();
}

#endif
