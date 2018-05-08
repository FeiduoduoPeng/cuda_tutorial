#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<memory>
#include "cuda.h"
#include<iostream>
#include<memory>

using namespace std;
#define DIM 1024
#define INF 2e10f
#define SPHERE 20
#define rnd(x) (x*rand()/RAND_MAX)


class Sphere{
public:
    double r, g, b;
    double x, y, z, radius;

    __device__ float hit(double ox, double oy, double *n){
        double dx = ox-x;
        double dy = oy-y;
        double xy_2 = dx*dx + dy*dy;
        double rad_2 = radius*radius;
        if( xy_2 < radius*radius){
            float dz = sqrtf(rad_2 - xy_2);
            *n = dz/radius;
            return dz+z;
        }
        return -INF;
    }
};

__constant__ Sphere s[SPHERE];
//__global__ void kernel(Sphere *s, unsigned char *ptrs){
__global__ void kernel(unsigned char *ptrs){
    int x =threadIdx.x + blockDim.x * blockIdx.x;
    int y =threadIdx.y + blockDim.y * blockIdx.y;
    int offset = x + y*blockDim.x * gridDim.x;

    double ox= x-DIM/2.0;
    double oy= y-DIM/2.0;
    double r=0, g=0, b=0; 
    double maxz = -INF;

    for(int i=0; i<SPHERE; i++){
        double n;
        double t = s[i].hit(ox, oy, &n);
        //if current hit is more closer to camera than last hit, use current data
        if( t>maxz){
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;//
        }
    }
    ptrs[4 * offset + 0] = int(r*255);
    ptrs[4 * offset + 1] = int(g*255);
    ptrs[4 * offset + 2] = int(b*255);
    ptrs[4 * offset + 3] = 255;
}

int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cv::Mat_<cv::Vec3b> img(DIM,DIM);
    unsigned char ptrs[4*DIM*DIM];
    unsigned char *dev_ptrs;
    Sphere temp_s[SPHERE];

    srand((unsigned) time(0));
    for(int i=0; i<SPHERE; i++){
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f)-500;
        temp_s[i].y = rnd(1000.0f)-500;
        temp_s[i].z = rnd(1000.0f)-500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }


    cudaMalloc((void**)&dev_ptrs, 4*DIM*DIM*sizeof(unsigned char));
    //cudaMalloc((void**)&s, SPHERE*sizeof(Sphere));
    //cudaMemcpy(s, temp_s, SPHERE*sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(s, temp_s, SPHERE * sizeof(Sphere));

    dim3 blocks(32, 32);
    dim3 threads(32, 32);
    kernel<<<blocks, threads>>>(dev_ptrs);
    
    cudaMemcpy(ptrs, dev_ptrs, 4*DIM*DIM*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float mytime;
    cudaEventElapsedTime(&mytime, start, stop);
    cout<<"performace:\n"<<mytime<<endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_ptrs);
    cudaFree(s);
    for(int y=0; y<img.rows; y++){
        for(int x=0; x<img.cols; x++){
            for(int ch=0; ch<3; ch++){
                img.at<cv::Vec3b>(x, y)[ch] = ptrs[ 4*(x+y*DIM) + ch];
            }
        }
    }
    cv::imshow("show", img);
    cv::waitKey(0);
    return 0;
}