#ifndef __PLUGIN_FACTORY_H__  
#define __PLUGIN_FACTORY_H__  
  
#include <algorithm>  
#include <cassert>  
#include <iostream>  
#include <cstring>  
#include <sys/stat.h>  
#include <map>  
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "upsample_layer.h"
#include <unordered_map>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

//SSD Reshape layer : shape{0,-1,21}
template<int OutC>
class Reshape : public IPlugin
{
public:
    Reshape() {}
    Reshape(const void* buffer, size_t size)
    {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);
    }

    int getNbOutputs() const override
    {
        return 1;
    }
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
        return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }

    size_t getWorkspaceSize(int) const override
    {
        return 0;
    }

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(mCopySize);
    }

    void serialize(void* buffer) override
    {
        *reinterpret_cast<size_t*>(buffer) = mCopySize;
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
    {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;
};



class SofaMaxChannelLayer: public IPlugin
{
public:
    SofaMaxChannelLayer(int axis): _axis(axis),inner_num_(1),outer_num_(3462){}
    
    SofaMaxChannelLayer(int axis,const void* buffer,size_t size)
    {
 	  _axis = axis;
    }

    inline int getNbOutputs() const override { return 1; };
    
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
   
        return DimsCHW(1,3462, 2);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
 
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize(void* buffer) override
    {
     
    }

    void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
         
    }

protected:
    int _axis;
    int _size;
    float* scale = new float [3462]; //scale
    int inner_num_;
    int outer_num_;
    
    
  
};


//SSD Flatten layer
class LReluLayer : public IPlugin
{
public:
    LReluLayer(){}
    LReluLayer(float para):para_(para)
    {


              std::cout<<"LReluLayer0"<<std::endl;
    }

    LReluLayer(const void* buffer,size_t sz, float para):para_(para)
    {


        assert(sz == 4 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        float* p=(float*)(d+3);
	para_=p[0];
	channel_=d[0];
	w_=d[1];
	h_=d[2];

        //std::cout<<"LReluLayer1"<<para_ <<" " <<channel_<<" "<<w_ <<" "<<h_ <<std::endl;
        para_=0.1;

    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
         std::cout<<"getOutputDimensions  channel"<<inputs[0].d[0]<<"h:"<<inputs[0].d[1]<<"w:"<<inputs[0].d[2]<<std::endl;

         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];
         
        return DimsCHW(inputs[0].d[0], inputs[0].d[1] , inputs[0].d[2] );
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

 

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
	

     //   std::cout<<"LReluLayer1 enqueue : "<<batchSize<<"c:"<<channel_<<"w:"<<w_<<"h:"<<h_<<"para_"<<para_<<std::endl;
	ReluForward_gpu((const float*)inputs[0],(float*)outputs[0],batchSize,channel_,w_,h_,para_);

	 //ReLUForward1<<<CAFFE_GET_BLOCKS(batchSize*channel_*w_*h_), CAFFE_CUDA_NUM_THREADS>>>(batchSize*channel_*w_*h_,inputs[0],outputs[0],para_);

	//std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
	//CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
	//Forward_gpu (
	//  (float*)inputs[0],1,channel_, w_, h_, stride_, (float*)outputs[0] );


        //CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));

        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(int)*3+sizeof(float);
    }

    void serialize(void* buffer) override
    {
        
	 
	//
	//write(q+3, (float)para_);

        float* q = reinterpret_cast<float*>(buffer);
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = channel_; d[1] = w_; d[2] = h_;
        q[4]=para_;

	//serializeFromDevice(d, mKernelWeights);
	//serializeFromDevice(d, mBiasWeights);
	 
 

    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
      //  dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

    }

protected:
    float para_;
    int channel_;
    int w_;
    int h_;
};





//SSD Flatten layer
class UpsampleLayer : public IPlugin
{
public:
    UpsampleLayer(){}
    UpsampleLayer(size_t stride):stride_(stride)
    {
      std::cout<<"UpsampleLayer0"<<std::endl;


    }

    UpsampleLayer(const void* buffer,size_t sz, size_t stride):stride_(stride)
    {

        const int* d = reinterpret_cast<const int*>(buffer);
 
	channel_=d[0];
	w_=d[1];
	h_=d[2];


        std::cout<<"UpsampleLayer1"<<std::endl;

    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
         std::cout<<"channel"<<inputs[0].d[0]<<"h:"<<inputs[0].d[1]<<"w:"<<inputs[0].d[2]<<std::endl;

         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

        return DimsCHW(inputs[0].d[0], inputs[0].d[1]*stride_, inputs[0].d[2]*stride_);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }



    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {


    // std::cout<<"UpsampleLayer1 enqueue"<<std::endl;
        //std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
        //CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
     Forward_gpu((float*)inputs[0],1,channel_, w_, h_, stride_, (float*)outputs[0] );




        return 0;
    }


    size_t getSerializationSize() override
    {
        return 4*sizeof(int);
    }

    void serialize(void* buffer) override
    {
   
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = channel_; d[1] = w_; d[2] = h_;
        d[3]=stride_;
   
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
      //  dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

    }

protected:
    int stride_;
    int channel_;
    int w_;
    int h_;
};





//SSD Flatten layer
class FlattenLayer : public IPlugin
{
public:
    FlattenLayer(){}
    FlattenLayer(const void* buffer,size_t size)
    {
        assert(size == 3 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[0] * d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);
        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        return DimsCHW(_size, 1, 1);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
        CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 3 * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};



  
  
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory {  
public:
    std::unordered_map<std::string, int> UpsampleIDs = {
        std::make_pair("layer86-upsample", 0),
        std::make_pair("layer98-upsample", 1)};

    std::unordered_map<std::string, int> LReLUIDs = {
        std::make_pair("layer1-act",0),
        std::make_pair("layer2-act",1),  
        std::make_pair("layer3-act",2),  
        std::make_pair("layer4-act",3),  
        std::make_pair("layer6-act",4),  
        std::make_pair("layer7-act",5),
        std::make_pair("layer8-act",6),
        std::make_pair("layer10-act",7),  
        std::make_pair("layer11-act",8),  
        std::make_pair("layer13-act",9),  
        std::make_pair("layer14-act",10),  
        std::make_pair("layer15-act",11),  
        std::make_pair("layer17-act",12),  
        std::make_pair("layer18-act",13),  
        std::make_pair("layer20-act",14),  
        std::make_pair("layer21-act",15),  
        std::make_pair("layer23-act",16),  
        std::make_pair("layer24-act",17),  
        std::make_pair("layer26-act",18),  
        std::make_pair("layer27-act",19),  
        std::make_pair("layer29-act",20),


        std::make_pair("layer30-act",21),  
        std::make_pair("layer32-act",22),  
        std::make_pair("layer33-act",23),  
        std::make_pair("layer35-act",24),  
        std::make_pair("layer36-act",25),  
        std::make_pair("layer38-act",26),  
        std::make_pair("layer39-act",27),  
        std::make_pair("layer40-act",28),  
        std::make_pair("layer42-act",29),  
        std::make_pair("layer43-act",30),  
        std::make_pair("layer45-act",31),  
        std::make_pair("layer46-act",32),  
        std::make_pair("layer48-act",33),  
        std::make_pair("layer49-act",34),  
        std::make_pair("layer51-act",35),  
        std::make_pair("layer52-act",36),  
        std::make_pair("layer54-act",37),  
        std::make_pair("layer55-act",38),  
        std::make_pair("layer57-act",39),
        std::make_pair("layer58-act",40),


        std::make_pair("layer60-act",41),  
        std::make_pair("layer61-act",42),  
        std::make_pair("layer63-act",43),  
        std::make_pair("layer64-act",44),  
        std::make_pair("layer65-act",45),  
        std::make_pair("layer67-act",46),  
        std::make_pair("layer68-act",47),  
        std::make_pair("layer70-act",48),  
        std::make_pair("layer71-act",49),  
        std::make_pair("layer73-act",50),  
        std::make_pair("layer74-act",51),  
        std::make_pair("layer76-act",52),  
        std::make_pair("layer77-act",53),  
        std::make_pair("layer78-act",54),  
        std::make_pair("layer79-act",55),  
        std::make_pair("layer80-act",56),
        std::make_pair("layer81-act",57),
        std::make_pair("layer85-act",58), 
        std::make_pair("layer88-act",59),
        std::make_pair("layer89-act",60), 



        std::make_pair("layer90-act",61),  
        std::make_pair("layer91-act",62),  
        std::make_pair("layer92-act",63),  
        std::make_pair("layer93-act",64),  
        std::make_pair("layer97-act",65),  
        std::make_pair("layer100-act",66),
        std::make_pair("layer101-act",67),
        std::make_pair("layer102-act",68),
        std::make_pair("layer103-act",69),
        std::make_pair("layer104-act",70),
        std::make_pair("layer105-act",71)};
  
        // caffe parser plugin implementation  
        bool isPlugin(const char* layerName) override  
        {  
         printf("isPlugin %s\n",layerName);
        //  for (int layerNum = 1; layerNum < 150; layerNum++) {
        //     char matchLayer[40];
        //     if (strcmp(name, sprintf(matchLayer, "layer%d-act"))
        //         || strcmp(name, "layer85-upsample")
        //         || strcmp(name, "layer97-upsample")) { return true }
        //  }
        // return false;
        return ( !strcmp(layerName, "layer86-upsample")
            || !strcmp(layerName, "layer98-upsample")
            || !strcmp(layerName, "layer1-act")  
            || !strcmp(layerName, "layer2-act")  
            || !strcmp(layerName, "layer3-act")  
            || !strcmp(layerName, "layer4-act")  
            || !strcmp(layerName, "layer6-act")  
            || !strcmp(layerName, "layer7-act")
            || !strcmp(layerName, "layer8-act")
            || !strcmp(layerName, "layer10-act")  
            || !strcmp(layerName, "layer11-act")  
            || !strcmp(layerName, "layer13-act")  
            || !strcmp(layerName, "layer14-act")  
            || !strcmp(layerName, "layer15-act")  
            || !strcmp(layerName, "layer17-act")  
            || !strcmp(layerName, "layer18-act")  
            || !strcmp(layerName, "layer20-act")  
            || !strcmp(layerName, "layer21-act")  
            || !strcmp(layerName, "layer23-act")  
            || !strcmp(layerName, "layer24-act")  
            || !strcmp(layerName, "layer26-act")  
            || !strcmp(layerName, "layer27-act")  
            || !strcmp(layerName, "layer29-act")


            || !strcmp(layerName, "layer30-act")  
            || !strcmp(layerName, "layer32-act")  
            || !strcmp(layerName, "layer33-act")  
            || !strcmp(layerName, "layer35-act")  
            || !strcmp(layerName, "layer36-act")  
            || !strcmp(layerName, "layer38-act")  
            || !strcmp(layerName, "layer39-act")  
            || !strcmp(layerName, "layer40-act")  
            || !strcmp(layerName, "layer42-act")  
            || !strcmp(layerName, "layer43-act")  
            || !strcmp(layerName, "layer45-act")  
            || !strcmp(layerName, "layer46-act")  
            || !strcmp(layerName, "layer48-act")  
            || !strcmp(layerName, "layer49-act")  
            || !strcmp(layerName, "layer51-act")  
            || !strcmp(layerName, "layer52-act")  
            || !strcmp(layerName, "layer54-act")  
            || !strcmp(layerName, "layer55-act")  
            || !strcmp(layerName, "layer57-act")
            || !strcmp(layerName, "layer58-act")


            || !strcmp(layerName, "layer60-act")  
            || !strcmp(layerName, "layer61-act")  
            || !strcmp(layerName, "layer63-act")  
            || !strcmp(layerName, "layer64-act")  
            || !strcmp(layerName, "layer65-act")  
            || !strcmp(layerName, "layer67-act")  
            || !strcmp(layerName, "layer68-act")  
            || !strcmp(layerName, "layer70-act")  
            || !strcmp(layerName, "layer71-act")  
            || !strcmp(layerName, "layer73-act")  
            || !strcmp(layerName, "layer74-act")  
            || !strcmp(layerName, "layer76-act")  
            || !strcmp(layerName, "layer77-act")  
            || !strcmp(layerName, "layer78-act")  
            || !strcmp(layerName, "layer79-act")  
            || !strcmp(layerName, "layer80-act")
            || !strcmp(layerName, "layer81-act")
            || !strcmp(layerName, "layer85-act")  
            || !strcmp(layerName, "layer88-act")
            || !strcmp(layerName, "layer89-act") 



            || !strcmp(layerName, "layer90-act")  
            || !strcmp(layerName, "layer91-act")  
            || !strcmp(layerName, "layer92-act")  
            || !strcmp(layerName, "layer93-act")  
            || !strcmp(layerName, "layer97-act")  
            || !strcmp(layerName, "layer100-act")
            || !strcmp(layerName, "layer101-act")
            || !strcmp(layerName, "layer102-act")
            || !strcmp(layerName, "layer103-act")
            || !strcmp(layerName, "layer104-act")
            || !strcmp(layerName, "layer105-act")
        );  
  
        }  
  
        virtual IPlugin* createPlugin(const char* layerName, const Weights* weights, int nbWeights) override  
        {
            if(!strcmp(layerName, "layer86-upsample") || !strcmp(layerName, "layer98-upsample"))
            {
                const int i = UpsampleIDs[layerName];
                assert(mPluginUpsample[i] == nullptr);
                mPluginUpsample[i] = std::unique_ptr<UpsampleLayer>(new UpsampleLayer(2));
                return mPluginUpsample[i].get();
            }
            else if (  !strcmp(layerName, "layer1-act")  
            || !strcmp(layerName, "layer2-act")  
            || !strcmp(layerName, "layer3-act")  
            || !strcmp(layerName, "layer4-act")  
            || !strcmp(layerName, "layer6-act")  
            || !strcmp(layerName, "layer7-act")
            || !strcmp(layerName, "layer8-act")
            || !strcmp(layerName, "layer10-act")  
            || !strcmp(layerName, "layer11-act")  
            || !strcmp(layerName, "layer13-act")  
            || !strcmp(layerName, "layer14-act")  
            || !strcmp(layerName, "layer15-act")  
            || !strcmp(layerName, "layer17-act")  
            || !strcmp(layerName, "layer18-act")  
            || !strcmp(layerName, "layer20-act")  
            || !strcmp(layerName, "layer21-act")  
            || !strcmp(layerName, "layer23-act")  
            || !strcmp(layerName, "layer24-act")  
            || !strcmp(layerName, "layer26-act")  
            || !strcmp(layerName, "layer27-act")  
            || !strcmp(layerName, "layer29-act")


            || !strcmp(layerName, "layer30-act")  
            || !strcmp(layerName, "layer32-act")  
            || !strcmp(layerName, "layer33-act")  
            || !strcmp(layerName, "layer35-act")  
            || !strcmp(layerName, "layer36-act")  
            || !strcmp(layerName, "layer38-act")  
            || !strcmp(layerName, "layer39-act")  
            || !strcmp(layerName, "layer40-act")  
            || !strcmp(layerName, "layer42-act")  
            || !strcmp(layerName, "layer43-act")  
            || !strcmp(layerName, "layer45-act")  
            || !strcmp(layerName, "layer46-act")  
            || !strcmp(layerName, "layer48-act")  
            || !strcmp(layerName, "layer49-act")  
            || !strcmp(layerName, "layer51-act")  
            || !strcmp(layerName, "layer52-act")  
            || !strcmp(layerName, "layer54-act")  
            || !strcmp(layerName, "layer55-act")  
            || !strcmp(layerName, "layer57-act")
            || !strcmp(layerName, "layer58-act")


            || !strcmp(layerName, "layer60-act")  
            || !strcmp(layerName, "layer61-act")  
            || !strcmp(layerName, "layer63-act")  
            || !strcmp(layerName, "layer64-act")  
            || !strcmp(layerName, "layer65-act")  
            || !strcmp(layerName, "layer67-act")  
            || !strcmp(layerName, "layer68-act")  
            || !strcmp(layerName, "layer70-act")  
            || !strcmp(layerName, "layer71-act")  
            || !strcmp(layerName, "layer73-act")  
            || !strcmp(layerName, "layer74-act")  
            || !strcmp(layerName, "layer76-act")  
            || !strcmp(layerName, "layer77-act")  
            || !strcmp(layerName, "layer78-act")  
            || !strcmp(layerName, "layer79-act")  
            || !strcmp(layerName, "layer80-act")
            || !strcmp(layerName, "layer81-act")
            || !strcmp(layerName, "layer85-act")  
            || !strcmp(layerName, "layer88-act")
            || !strcmp(layerName, "layer89-act") 



            || !strcmp(layerName, "layer90-act")  
            || !strcmp(layerName, "layer91-act")  
            || !strcmp(layerName, "layer92-act")  
            || !strcmp(layerName, "layer93-act")  
            || !strcmp(layerName, "layer97-act")  
            || !strcmp(layerName, "layer100-act")
            || !strcmp(layerName, "layer101-act")
            || !strcmp(layerName, "layer102-act")
            || !strcmp(layerName, "layer103-act")
            || !strcmp(layerName, "layer104-act")
            || !strcmp(layerName, "layer105-act")
       ){
            const int i = LReLUIDs[layerName];
            assert(mPluginLReLU[i] == nullptr);
            mPluginLReLU[i] = std::unique_ptr<LReluLayer>(new LReluLayer(0.1));
            return mPluginLReLU[i].get();
        }else{  
             assert(0);  
             return nullptr;  
        }  
    }  
  
    // deserialization plugin implementation  
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override {                
        
        if(!strcmp(layerName, "layer86-upsample") || !strcmp(layerName, "layer98-upsample"))
        {
            const int i = UpsampleIDs[layerName];
            assert(mPluginUpsample[i] == nullptr);
            mPluginUpsample[i] = std::unique_ptr<UpsampleLayer>(new UpsampleLayer(serialData, serialLength,2));
            return mPluginUpsample[i].get();
        }
        else if ( !strcmp(layerName, "layer1-act")  
            || !strcmp(layerName, "layer2-act")  
            || !strcmp(layerName, "layer3-act")  
            || !strcmp(layerName, "layer4-act")  
            || !strcmp(layerName, "layer6-act")  
            || !strcmp(layerName, "layer7-act")
            || !strcmp(layerName, "layer8-act")
            || !strcmp(layerName, "layer10-act")  
            || !strcmp(layerName, "layer11-act")  
            || !strcmp(layerName, "layer13-act")  
            || !strcmp(layerName, "layer14-act")  
            || !strcmp(layerName, "layer15-act")  
            || !strcmp(layerName, "layer17-act")  
            || !strcmp(layerName, "layer18-act")  
            || !strcmp(layerName, "layer20-act")  
            || !strcmp(layerName, "layer21-act")  
            || !strcmp(layerName, "layer23-act")  
            || !strcmp(layerName, "layer24-act")  
            || !strcmp(layerName, "layer26-act")  
            || !strcmp(layerName, "layer27-act")  
            || !strcmp(layerName, "layer29-act")


            || !strcmp(layerName, "layer30-act")  
            || !strcmp(layerName, "layer32-act")  
            || !strcmp(layerName, "layer33-act")  
            || !strcmp(layerName, "layer35-act")  
            || !strcmp(layerName, "layer36-act")  
            || !strcmp(layerName, "layer38-act")  
            || !strcmp(layerName, "layer39-act")  
            || !strcmp(layerName, "layer40-act")  
            || !strcmp(layerName, "layer42-act")  
            || !strcmp(layerName, "layer43-act")  
            || !strcmp(layerName, "layer45-act")  
            || !strcmp(layerName, "layer46-act")  
            || !strcmp(layerName, "layer48-act")  
            || !strcmp(layerName, "layer49-act")  
            || !strcmp(layerName, "layer51-act")  
            || !strcmp(layerName, "layer52-act")  
            || !strcmp(layerName, "layer54-act")  
            || !strcmp(layerName, "layer55-act")  
            || !strcmp(layerName, "layer57-act")
            || !strcmp(layerName, "layer58-act")


            || !strcmp(layerName, "layer60-act")  
            || !strcmp(layerName, "layer61-act")  
            || !strcmp(layerName, "layer63-act")  
            || !strcmp(layerName, "layer64-act")  
            || !strcmp(layerName, "layer65-act")  
            || !strcmp(layerName, "layer67-act")  
            || !strcmp(layerName, "layer68-act")  
            || !strcmp(layerName, "layer70-act")  
            || !strcmp(layerName, "layer71-act")  
            || !strcmp(layerName, "layer73-act")  
            || !strcmp(layerName, "layer74-act")  
            || !strcmp(layerName, "layer76-act")  
            || !strcmp(layerName, "layer77-act")  
            || !strcmp(layerName, "layer78-act")  
            || !strcmp(layerName, "layer79-act")  
            || !strcmp(layerName, "layer80-act")
            || !strcmp(layerName, "layer81-act")
            || !strcmp(layerName, "layer85-act")  
            || !strcmp(layerName, "layer88-act")
            || !strcmp(layerName, "layer89-act") 



            || !strcmp(layerName, "layer90-act")  
            || !strcmp(layerName, "layer91-act")  
            || !strcmp(layerName, "layer92-act")  
            || !strcmp(layerName, "layer93-act")  
            || !strcmp(layerName, "layer97-act")  
            || !strcmp(layerName, "layer100-act")
            || !strcmp(layerName, "layer101-act")
            || !strcmp(layerName, "layer102-act")
            || !strcmp(layerName, "layer103-act")
            || !strcmp(layerName, "layer104-act")
            || !strcmp(layerName, "layer105-act")
       ){
            const int i = LReLUIDs[layerName];
            assert(mPluginLReLU[i] == nullptr);
            mPluginLReLU[i] = std::unique_ptr<LReluLayer>(new LReluLayer(serialData, serialLength,0.1));
            return mPluginLReLU[i].get();
        }else{  
            assert(0);  
            return nullptr;  
        }  
    }  
  
    void destroyPlugin()  
    {
        for (unsigned i = 0; i < LReLUIDs.size(); ++i) { mPluginLReLU[i].reset(); }
        for (unsigned j = 0; j < UpsampleIDs.size(); ++j) { mPluginUpsample[j].reset(); }
    }  
  
  
private:
    void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { ptr->destroy(); }};
    std::unique_ptr<LReluLayer> mPluginLReLU[72];
    std::unique_ptr<UpsampleLayer> mPluginUpsample[2]{nullptr, nullptr};
};
  
#endif  

