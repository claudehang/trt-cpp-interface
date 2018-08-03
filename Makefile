CC := g++
TARGET := main

CAFFE_LIB_PATH := /data/aicenter/third_party/dist/libs
#CAFFE_LIB_PATH := /data/home/ryanqqshi/mobilefacenet/caffe/build/lib

COMM_DEP := -I/usr/local/include -I./include -I/usr/local/cuda/include -L $(CAFFE_LIB_PATH) -lcaffe -Wl,-rpath=$(CAFFE_LIB_PATH)


SO := mtcnn_face_align.so
SO += arcface_face_identify.so
SO += face_compare.so

all: $(TARGET) $(SO)

mtcnn_face_align.so : detection_batch.cpp
	$(CC) -fPIC -shared -o$@ $^  $(COMM_DEP)

arcface_face_identify.so : identify_batch.cpp
	$(CC) -fPIC -shared -o$@ $^  $(COMM_DEP)

face_compare.so : compare.cpp
	$(CC) -fPIC -shared -o$@ $^

$(TARGET):  main.cpp
	$(CC) -o$@ $^ -rdynamic -ldl -lopencv_core -lopencv_highgui -lopencv_ml -lopencv_imgproc


clean:
	rm -rf $(SO) $(TARGET)
