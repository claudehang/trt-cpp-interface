#include "image.h"
#include <chrono>

#include <opencv2/opencv.hpp>

using namespace cv;

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

void ipl_into_image(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
}

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h*w*c, sizeof(float));
    return out;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}





image load_image_cv(const char *filename, int channels)
{
    IplImage* src = 0;
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }

    if( (src = cvLoadImage(filename, flag)) == 0 )
    {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    auto beginImgMemTime = std::chrono::high_resolution_clock::now();
    IplImage* src_mem = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
    cvCopy(src, src_mem, NULL);
    auto endImgMemTime = std::chrono::high_resolution_clock::now();
    float eachImgMemTime = std::chrono::duration<float, std::milli>(endImgMemTime - beginImgMemTime).count();
    std::cout << "Time taken for image read in memory is " << eachImgMemTime << " ms." << std::endl;
    //imgMemTime += eachImgMemTime;
    cvReleaseImage(&src);

    CvSize size;
    size.width  = 320;
    size.height = 320;
    auto beginResizeTime = std::chrono::high_resolution_clock::now();
    IplImage* dst = cvCreateImage(size, src_mem->depth, src_mem->nChannels);
    //auto beginResizeTime = std::chrono::high_resolution_clock::now();
    cvResize(src_mem, dst, CV_INTER_CUBIC);
    auto endResizeTime = std::chrono::high_resolution_clock::now();
    float eachResizeTime = std::chrono::duration<float, std::milli>(endResizeTime - beginResizeTime).count();
    std::cout << "Time taken for image resize is " << eachResizeTime << " ms." << std::endl;
    //resizeTime += eachResizeTime;

    image out = ipl_to_image(dst);
    //cvReleaseImage(&src);
    cvReleaseImage(&dst);
    cvReleaseImage(&src_mem);
    rgbgr_image(out);
    return out;
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

image load_image(const char* filename,int w,int h,int c)
{
    image out = load_image_cv(filename,c);

    if((h && w) && (h != out.h || w != out.w))
    {
        image resized = resize_image(out,w,h);
        free_image(out);
        out = resized;
    }
    return out;
}

image load_image_color(const char* filename,int w,int h)
{
    return load_image(filename,w,h,3);
}

void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}


image letterbox_image(image im, int w, int h)
{
    std::cout << "Inside letterbox_image()\n\n";
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    auto beginResizeTime = std::chrono::high_resolution_clock::now();

    image resized = resize_image(im, new_w, new_h);

    auto endResizeTime = std::chrono::high_resolution_clock::now();
    float eachResizeTime = std::chrono::duration<float, std::milli>(endResizeTime - beginResizeTime).count();
    std::cout << "Time taken for resize_image() is " << eachResizeTime << " ms." << std::endl;

    auto beginMakeTime = std::chrono::high_resolution_clock::now();

    image boxed = make_image(w, h, im.c);

    auto endMakeTime = std::chrono::high_resolution_clock::now();
    float eachMakeTime = std::chrono::duration<float, std::milli>(endMakeTime - beginMakeTime).count();
    std::cout << "Time taken for make_image() is " << eachMakeTime << " ms." << std::endl;

    auto beginFillTime = std::chrono::high_resolution_clock::now();

    fill_image(boxed, .5);

    auto endFillTime = std::chrono::high_resolution_clock::now();
    float eachFillTime = std::chrono::duration<float, std::milli>(endFillTime - beginFillTime).count();
    std::cout << "Time taken for fill_image() is " << eachFillTime << " ms." << std::endl;

    auto beginEmbedTime = std::chrono::high_resolution_clock::now();

    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);

    auto endEmbedTime = std::chrono::high_resolution_clock::now();
    float eachEmbedTime = std::chrono::duration<float, std::milli>(endEmbedTime - beginEmbedTime).count();
    std::cout << "Time taken for embed_image() is " << eachEmbedTime << " ms." << std::endl;

    free_image(resized);
    std::cout << "Outside letterbox_image()\n\n";
    return boxed;
}
