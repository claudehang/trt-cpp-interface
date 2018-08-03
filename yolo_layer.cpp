#include "yolo_layer.h"
#include "blas.h"
#include "cuda_yolo.h"
#include "activations.h"
#include "box.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <vector>
#include <fstream>


float biases[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};

layer make_yolo_layer(int batch,int w,int h,int n,int total,int classes)
{
    layer l = {0};
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes+ 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w*l.h*l.c;

    l.biases = (float*)calloc(total*2,sizeof(float));
    for(int i =0;i<total*2;++i){
        l.biases[i] = biases[i];
    }
    l.mask = (int*)calloc(n,sizeof(int));
    if(l.w == 10){
        int j = 6;
        for(int i =0;i<l.n;++i)
            l.mask[i] = j++;
    }
    if(l.w == 20){
        int j = 3;
        for(int i =0;i<l.n;++i)
            l.mask[i] = j++;
    }
    if(l.w == 40){
        int j = 0;
        for(int i =0;i<l.n;++i)
            l.mask[i] = j++;
    }
    l.outputs = l.inputs;
    l.output = (float*)calloc(batch*l.outputs,sizeof(float));
    l.output_gpu = cuda_make_array(l.output,batch*l.outputs);
    return l;
}

void free_yolo_layer(layer l)
{
    if(NULL != l.biases){
        free(l.biases);
        l.biases = NULL;
    }

    if(NULL != l.mask){
        free(l.mask);
        l.mask = NULL;
    }
    if(NULL != l.output){
        free(l.output);
        l.output = NULL;
    }

    if(NULL != l.output_gpu)
        cuda_free(l.output_gpu);
}

static int entry_index(layer l,int batch,int location,int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4 + l.classes + 1) + entry*l.w*l.h + loc;
 }

void forward_yolo_layer_gpu(const void* input,layer l)
{
    copy_gpu(l.batch*l.inputs,(float*)input,1,l.output_gpu,1);
    int b,n;
    for(b = 0;b < l.batch;++b){
        for(n =0;n< l.n;++n){
            int index = entry_index(l,b,n*l.w*l.h,0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h,LOGISTIC);
            index = entry_index(l,b,n*l.w*l.h,4);
            activate_array_gpu(l.output_gpu + index,(1 + l.classes)*l.w*l.h,LOGISTIC);
       }
    }
    cuda_pull_array(l.output_gpu,l.output,l.batch*l.outputs);
}



int yolo_num_detections(layer l,float thresh)
{
    int i,n;
    int count = 0;
    for(i=0;i<l.w*l.h;++i){
        for(n=0;n<l.n;++n){
            int obj_index = entry_index(l,0,n*l.w*l.h+i,4);
            if(l.output[obj_index] > thresh)
                ++count;
        }
    }
    return count;
}

int num_detections(std::vector<layer> layers_params,float thresh)
{
    int i;
    int s=0;
    for(i=0;i<layers_params.size();++i){
        layer l  = layers_params[i];
        s += yolo_num_detections(l,thresh);
    }
    return s;

}

detection* make_network_boxes(std::vector<layer> layers_params,float thresh,int* num)
{
    layer l = layers_params[0];
    int i;
    int nboxes = num_detections(layers_params,thresh);
    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes,sizeof(detection));
    for(i=0;i<nboxes;++i){
        dets[i].prob = (float*)calloc(l.classes,sizeof(float));
    }
    return dets;
}


void correct_yolo_boxes(detection* dets,int n,int w,int h,int netw,int neth,int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)){
        new_w = netw;
        new_h = (h * netw)/w;
    }
    else{
        new_h = neth;
        new_w = (w * neth)/h;
    }
    // std::cout << "new_w: " << new_w << ", new_h: " << new_h << std::endl;
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        //b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
	b.x = b.x;
        //b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
	b.y = b.y;
        //b.w *= (float)netw/new_w;
	b.w = b.w;
        //b.h *= (float)neth/new_h;
	b.h = b.h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}


box get_yolo_box(float* x,float* biases,int n,int index,int i,int j,int lw, int lh,int w, int h,int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n] / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n + 1] / h;
    return b;
}


int get_yolo_detections(layer l,int w, int h, int netw,int neth,float thresh,int *map,int relative,detection *dets)
{
    int i,j,n;
    float* predictions = l.output;
    int count = 0;
    for(i=0;i<l.w*l.h;++i){
        int row = i/l.w;
        int col = i%l.w;
        for(n = 0;n<l.n;++n){           
            int obj_index = entry_index(l,0,n*l.w*l.h + i,4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index = entry_index(l,0,n*l.w*l.h + i,0);

            dets[count].bbox = get_yolo_box(predictions,l.biases,l.mask[n],box_index,col,row,l.w,l.h,netw,neth,l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j=0;j<l.classes;++j){
                int class_index = entry_index(l,0,n*l.w*l.h+i,4+1+j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    // correct_yolo_boxes(dets,count,w,h,netw,neth,relative);
    return count;
}


void fill_network_boxes(std::vector<layer> layers_params,int w,int h,float thresh, float hier, int *map,int relative,detection *dets)
{
    int j;
    for(j=0;j<layers_params.size();++j){
        layer l = layers_params[j];
        int count = get_yolo_detections(l,w,h,320,320,thresh,map,relative,dets);
        dets += count;
    }
}


detection* get_network_boxes(std::vector<layer> layers_params,
                             int img_w,int img_h,float thresh,float hier,int* map,int relative,int *num)
{
    //make network boxes
    detection *dets = make_network_boxes(layers_params,thresh,num);

    //fill network boxes
    fill_network_boxes(layers_params,img_w,img_h,thresh,hier,map,relative,dets);
    return dets;
}

//get detection result
detection* get_detections(void** blobs,int img_w,int img_h,int *nboxes,int classNum)
{
    int classes = classNum;
    float thresh = 0.5;
    float hier_thresh = 0.5;
    float nms = 0.3;
    int blob_size_scale[3] = {10, 20, 40};

    std::vector<layer> layers_params;
    layers_params.clear();
    for(int i=0;i<3;++i){
        layer l_params = make_yolo_layer(1,blob_size_scale[i], blob_size_scale[i],3,9,classes);
        layers_params.push_back(l_params);
        forward_yolo_layer_gpu(blobs[i+1],l_params);
    }
    

    //get network boxes
    detection* dets = get_network_boxes(layers_params,img_w,img_h,thresh,hier_thresh,0,1,nboxes);

    //release layer memory
    for(int index =0;index < layers_params.size();++index){
        free_yolo_layer(layers_params[index]);
    }

    if(nms) do_nms_sort(dets,(*nboxes),classes,nms);
    return dets;       
}


//release detection memory
void free_detections(detection *dets,int nboxes)
{
    int i;
    for(i = 0;i<nboxes;++i){
        free(dets[i].prob);
    }
    free(dets);
}
