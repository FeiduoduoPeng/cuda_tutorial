#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<memory>
#include "cuda.h"
#include<iostream>
#include<memory>

using namespace std;

#define DIM 1000

class cuComplex{
public:
    double r,i;

    __device__ cuComplex(double a, double b): r(a),i(b){}
    __device__ cuComplex operator* (const cuComplex& a){
        return cuComplex(r*a.r - i*a.i, r*a.i + i*a.r);
    }
    __device__ cuComplex operator+ (const cuComplex& a){
        return cuComplex(r + a.r, i + a.i);
    }
    __device__ double magnitude2(void){
        return r*r+ i*i;
    }
};

__device__ int julia(int x, int y){
    double cx = 1.5*(double)(x-DIM/2)/(DIM/2);
    double cy = 1.5*(double)(y-DIM/2)/(DIM/2);

    cuComplex a(cx,cy);
    cuComplex c(-0.8, 0.156);

    for(int i=0; i<200; i++){
        a = a*a + c;
        if(a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

__global__ void kernel(unsigned char *ptr){
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = y*gridDim.x + x;

    int jV = julia(x, y);
    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 255*jV;
    ptr[offset*4 + 3] = 255;
}

int main(){
    cv::Mat_<cv::Vec3b> img(DIM, DIM);

    unsigned char ptr[4*DIM*DIM];
    unsigned char *dev_img;
    cudaMalloc((void**) &dev_img, 4*DIM*DIM*sizeof(unsigned char));

    dim3 grid(DIM,DIM);
    kernel<<<grid, 1>>>(dev_img);

    cudaMemcpy(
        ptr, 
        dev_img, 
        4*DIM*DIM*sizeof(unsigned char), 
        cudaMemcpyDeviceToHost);
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            for(int ch=0; ch<4; ch++){
                img.at<cv::Vec3b>(i,j)[ch] = ptr[ 4*(j*DIM+i) + ch];
            }
        }
    }
    cv::imshow("julia", img);
    cv::waitKey(0);

    return 0;
}