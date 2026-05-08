// check_gpu_sm.cu
#include <cuda_runtime.h>
#include <iostream>



int main() {
    int device_count = 0;

    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (device_count == 0) {
        std::cout << "CUDA GPU를 찾을 수 없습니다." << std::endl;
        return 0;
    }

    std::cout << "CUDA GPU count: " << device_count << std::endl;

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;

        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed on device "
                      << dev << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "\nDevice " << dev << ": " << prop.name << std::endl;
        std::cout << "Compute Capability: "
                  << prop.major << "." << prop.minor << std::endl;
        std::cout << "SM version: sm_"
                  << prop.major << prop.minor << std::endl;

        std::cout << "MultiProcessor Count: "
                  << prop.multiProcessorCount << std::endl;
        std::cout << "Max Threads Per Block: "
                  << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Warp Size: "
                  << prop.warpSize << std::endl;
    }

    return 0;
}