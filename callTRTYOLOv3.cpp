#include "./yolov3_interface.h"
#include "dlfcn.h"

#define LIB_TRTYOLOV3 "/data/home/claudehang/TensorRT-4.0.1.6/targets/x86_64-linux-gnu/samples/trt-yolo-320-interface/yolov3_interface.so"
//#define LIB_TRTYOLOV3 "/usr/local/lib/yolov3_interface.so"

extern "C"
{
    typedef void (*F_trtYolov3Init)(int classes, std::string deployFile, std::string modelFile);
    typedef void (*F_trtYolov3Detect)(const std::string& img_name, const std::string& test_image_path, const std::string& output_folder);
    typedef void (*F_trtYolov3Free)();
}

int main() {
    std::cout << "start loading interface" << std::endl;

    void *handle = dlopen(LIB_TRTYOLOV3,RTLD_LAZY);
    if(!handle)
    {
        printf("%s\n",dlerror());
        exit(EXIT_FAILURE);
    }

    char *error;
    dlerror();

    F_trtYolov3Init trtYolov3Init = (F_trtYolov3Init)dlsym(handle,"trtYolov3Init");
    F_trtYolov3Detect trtYolov3Detect = (F_trtYolov3Detect)dlsym(handle,"trtYolov3Detect");
    F_trtYolov3Free trtYolov3Free = (F_trtYolov3Free)dlsym(handle,"trtYolov3Free");
    
    if((error = dlerror()) != NULL)
    {
        printf("%s\n",error);
        exit(EXIT_FAILURE);
    }
    std::cout << "end of loading interface" << std::endl;

    // important parameters
    std::string deployFile = "/data/home/claudehang/TensorRT-4.0.1.6/bin/ca_man_yolov3.prototxt";
    std::string modelFile  = "/data/home/claudehang/TensorRT-4.0.1.6/bin/ca_man_yolov3.caffemodel";

    std::string test_image_path = "/data/home/claudehang/TensorRT-4.0.1.6/bin/lol-images/";
    std::string image_list_file = "/data/home/claudehang/TensorRT-4.0.1.6/bin/lol-images/image_list.txt";
    std::string output_folder   = "/data/home/claudehang/TensorRT-4.0.1.6/bin/lol-results/";

    int gpu_id = 1;
    int clsNum = 1;

    // read images in given directory
    std::ifstream img_list;
    img_list.open(image_list_file.data());
    if (!img_list)
    {
        std::cerr << image_list_file << " open error." << std::endl;
        exit(1);
    }
    std::string img_name;
    int count = 0;

    // initialize tensorrt model
    trtYolov3Init(clsNum, deployFile, modelFile);

    // do detection for each image
    while (getline(img_list, img_name)) {
        count++;
        //std::string imgFilename = test_image_path + img_name;
        std::cout << "YOLO on image < " << img_name << " >" << std::endl;
        trtYolov3Detect(img_name.c_str(), test_image_path.c_str(), output_folder.c_str());
    }
    trtYolov3Free();
    dlclose(handle);
}
