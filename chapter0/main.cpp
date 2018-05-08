#include "vector_add.h"
#include<iostream>

using namespace std;

int main(){
    int a[N], b[N], c[N];

    for(int i=0; i<N; i++){
        a[i] = i;
        b[i] = 2;
    }
    kernel_wrapper(a, b, c);

    for(auto item:c){
        cout<<item<<endl;
    }
    return 0;
}