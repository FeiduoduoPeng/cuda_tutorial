#include<iostream>
using namespace std;

__global__ void kernel(void){

}

int main(){
    kernel<<<1,1>>>();
    cout<<"Hello world"<<endl;
    return 0;
}