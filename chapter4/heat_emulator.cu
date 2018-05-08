#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<memory>
#include "cuda.h"
#include<iostream>
#include<memory>

using namespace std;
#define DIM 1024
//系数应当小于0.25
#define SPEED 0.25

__global__ void mycopy(const float *src, float *dest){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    if( src[offset]!=0) dest[offset] = src[offset];
    //dest[offset] = src[offset];
}

__global__ void kernel(float *optrs, const float *iptrs, int tick){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int right = offset + 1;
    int left = offset - 1;
    if(x==DIM-1) right--;
    if(x==0) left++;

    int top = offset - DIM;
    int down = offset + DIM;
    if(y==DIM-1) down -= DIM;
    if(y==0) top += DIM;

    optrs[offset] = iptrs[offset] + SPEED*(iptrs[right] + iptrs[left] + iptrs[top] + iptrs[down] - 4*iptrs[offset] );
    //if(optrs[offset] > 255)
    //    optrs[offset] = 255 ;
    //if(optrs[offset] < 0)
    //    optrs[offset] = 0 ;
    
    //optrs[offset] = 255.0f;
}


int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cv::Mat_<cv::Vec3b> img(DIM,DIM);
    float img_ptrs[DIM*DIM]={0};
    float *out_ptr;
    float *const_ptr;
    float *in_ptr;
    dim3 blocks(32, 32);
    dim3 threads(32, 32);

    cudaMalloc((void**)&out_ptr, DIM*DIM*sizeof(float));
    cudaMalloc((void**)&const_ptr, DIM*DIM*sizeof(float));
    cudaMalloc((void**)&in_ptr, DIM*DIM*sizeof(float));
    
    for(int i=0; i<DIM; i++){
        for(int j=0; j<DIM; j++){
            if(i>110 && i<210 && j>110 && j<210)
                img_ptrs[i*DIM + j] = 255;
            if(i>210 && i<310 && j>210 && j<310)
                img_ptrs[i*DIM + j] = 255;
        }
    }

    cudaMemcpy( const_ptr, img_ptrs, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice );

    for(int i=0; i<DIM*DIM; i++)
        img_ptrs[i]=70;
    cudaMemcpy( in_ptr, img_ptrs, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice );

    for(int it=0; it<9000; it++){
        mycopy<<<blocks,threads>>>(const_ptr, in_ptr);

        kernel<<<blocks, threads>>>(out_ptr, in_ptr, it);
        swap(in_ptr, out_ptr);
        //cudaMemcpy(img_ptrs, in_ptr, DIM*DIM*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(img_ptrs, in_ptr, DIM*DIM*sizeof(float), cudaMemcpyDeviceToHost);

        cout<<it<<" pixel(215,205): "<<img_ptrs[215 + 205*DIM]<<endl;
        for(int i=0; i< img.rows; i++){
            for(int j=0; j<img.cols; j++){
                for(int ch=0; ch<3; ch++)
                    img.at<cv::Vec3b>(i,j)[ch]=img_ptrs[ j*DIM+i];
            }
        }
        if(it == 8999)
            cv::waitKey(0);
        cv::imshow("test", img);
        cv::waitKey(1);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float mytime;
    cudaEventElapsedTime(&mytime, start, stop);
    cout<<"performace:\n"<<mytime<<endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(in_ptr);
    cudaFree(out_ptr);
    cudaFree(const_ptr);

    return 0;
}