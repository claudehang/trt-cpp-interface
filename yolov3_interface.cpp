#include "./yolov3_interface.h"
#include "./image.h"
#include "./yolo_layer.h"

void TRTYOLOv3::initialize(int classes, std::string deployFile, std::string modelFile) {
    clsDetect = classes;
    cudaSetDevice(1);
    //gieModelStream{ nullptr };
    caffeToGIEModel(deployFile, modelFile, 1, &pluginFactory, gieModelStream);
    std::cout << "caffeToGIEModel is finished!!!!!!!!!!!!!" << std::endl;
    pluginFactory.destroyPlugin();
    std::cout << "Start deserializing CudaEngine model..." << std::endl;
}

void TRTYOLOv3::detect(const std::string& img_name, const std::string& test_image_path, const std::string& output_folder) {
    std::string imgFilename = test_image_path + img_name;
    image im = load_image_color(imgFilename.c_str(),0,0);
    doInference(*context, im.data, 1, imgFilename.c_str(), img_name.c_str(), output_folder.c_str());
    free_image(im);
    std::cout << "==========================\n\n\n\n" << std::endl;
}

void TRTYOLOv3::free() {
    context->destroy();
    engine->destroy();
    runtime->destroy();
    pluginFactory.destroyPlugin();
}

void TRTYOLOv3::caffeToGIEModel(const std::string& deployFile,      // name for caffe prototxt
                     const std::string& modelFile,                  // name for model 
                     unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
                     nvcaffeparser1::IPluginFactory* pluginFactory, // factory for plugin layers
                     IHostMemory *&gieModelStream)                  // output stream for the GIE model
{
    // create the builder
    std::cout << "start parsing model..." << std::endl;
    IBuilder* builder = createInferBuilder(gLogger);
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(pluginFactory);

    // seems that fp16 is not supported
    bool fp16 = builder->platformHasFastFp16();
    if (fp16)
        std::cout << "[Using FastFp16]" << std::endl;
    else
        std::cout << "[NOT Using FastFp16]" << std::endl;

    std::cout << "=== Starting pop out layer INFO ===" << std::endl;
    const IBlobNameToTensor* blobNameToTensor = parser->parse( deployFile.c_str(), modelFile.c_str(), *network, fp16 ? DataType::kHALF : DataType::kFLOAT);
    std::cout << "===  Ending pop out layer INFO  ===" << std::endl;

    // specify which tensors are outputs
    std::vector<std::string> outputs = { OUTPUT_BLOB_NAME0 ,OUTPUT_BLOB_NAME1,OUTPUT_BLOB_NAME2 };  // network outputs
    for (auto& s : outputs) {
        std::cout << "markOutput name is " << s << std::endl;
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    std::cout << "Start building the TensorRT engine and it should take a a LONG while..." << std::endl;
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setHalf2Mode(fp16);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
    
    std::cout << "End of parsing model and start really FAST inference!!!!!" << std::endl;
}

void TRTYOLOv3::deserializeEngine() {
    runtime = createInferRuntime(gLogger);
    std::cout << "start deserializeCudaEngine model..." << std::endl;
    engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
    std::cout << "end deserializeCudaEngine model..." << std::endl;
    context = engine->createExecutionContext();
    //context->setProfiler(&gProfiler);
}

void TRTYOLOv3::doInference(IExecutionContext& context, float* input, int batchSize,
    const char* imgFilename, const char* img_name, const char* out_folder)
{
    cv::Mat img = cv::imread(imgFilename);
    int w = img.cols;
    int h = img.rows;
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 4);
    void* buffers[4];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
        outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
        outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);

    // create GPU buffers and a stream
     cudaMalloc(&buffers[outputIndex0],  1 * OUTPUT_SIZE0 * sizeof(float)) ;
     cudaMalloc(&buffers[outputIndex1],  1 * OUTPUT_SIZE1 * sizeof(float)) ;
     cudaMalloc(&buffers[outputIndex2],  1 * OUTPUT_SIZE2 * sizeof(float)) ;

     cudaMalloc(&buffers[inputIndex],    1 * INPUT_C*INPUT_H * INPUT_W * sizeof(float)) ;

     cudaStream_t stream;
     cudaStreamCreate(&stream) ;

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    cudaMemcpyAsync(buffers[inputIndex], input, batchSize *INPUT_C* INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream) ;

    //auto beginDetectTime = std::chrono::high_resolution_clock::now();
    context.enqueue(batchSize, buffers, stream, nullptr);
    //auto endDetectTime = std::chrono::high_resolution_clock::now();
    //float eachDetectTime = std::chrono::duration<float, std::milli>(endDetectTime - beginDetectTime).count();
    //std::cout << "Time taken for net flow is " << eachDetectTime << " ms." << std::endl;
    //detectTime += eachDetectTime;


    //auto beginPostTime = std::chrono::high_resolution_clock::now(); 
    int nboxes = 0;
    detection *dets = get_detections(buffers,w,h,&nboxes,clsDetect);
    //auto endPostTime = std::chrono::high_resolution_clock::now();
    //float eachPostTime = std::chrono::duration<float, std::milli>(endPostTime - beginPostTime).count();
    //std::cout << "Time taken for post processing is " << eachPostTime << " ms." << std::endl;
    //postTime += eachPostTime;


    //show detection results
    //cv::Mat img = cv::imread(imgFilename);
    int i,j;

    for(i=0;i< nboxes;++i){
        int cls = -1;
        for(j=0;j<clsDetect;++j) {
            if(dets[i].prob[j] > 0.5){
                if(cls < 0){
                    cls = j;
                }
                printf("%d: %.0f%%\n",cls,dets[i].prob[j]*100);
            }
        }
        if(cls >= 0){
            box b = dets[i].bbox;
            printf("x = %f,y =  %f,w = %f,h =  %f\n",b.x,b.y,b.w,b.h);

            int left  = (b.x-b.w/2.)*w;
            int right = (b.x+b.w/2.)*w;
            int top   = (b.y-b.h/2.)*h;
            int bot   = (b.y+b.h/2.)*h;
            cv::rectangle(img,cv::Point(left,top),cv::Point(right,bot),cv::Scalar(0,0,255),3,8,0);
            printf("left = %d,right =  %d,top = %d,bot =  %d\n",left,right,top,bot);
        }
    }

    const size_t len = strlen(img_name) + strlen(out_folder);
    char* output_path = new char[len + 1];
    strcpy(output_path, out_folder);
    strcat(output_path, img_name);
    cv::imwrite(output_path, img);

    delete [] output_path;


    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex0]);
    cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);
}

TRTYOLOv3 model;

extern "C" void trtYolov3Init(int classes, std::string deployFile, std::string modelFile) {
    model.initialize(classes, deployFile, modelFile);
}

extern "C" void trtYolov3Detect(const std::string& img_name, const std::string& test_image_path, const std::string& output_folder) {
    model.detect(img_name, test_image_path, output_folder);
}

extern "C" void trtYolov3Free() {
    model.free();
}
