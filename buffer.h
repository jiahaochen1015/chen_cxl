#ifndef __BENCHMARK_BUFFER_H__
#define __BENCHMARK_BUFFER_H__

// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <memory>
#include <cstddef>
#include <cstdint>
#include "cuda.h"
#include "nvm_types.h"
#include "nvm_dma.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <new>
#include <cstdlib>
#include <iostream>
#include "util.h"
#include <numa.h>

#define ADDR (void *)(0x0UL)
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)
#define PROTECTION (PROT_READ | PROT_WRITE)

using error = std::runtime_error;
using std::string;

typedef std::shared_ptr<nvm_dma_t> DmaPtr;

typedef std::shared_ptr<void> BufferPtr;

DmaPtr createDma(const nvm_ctrl_t *ctrl, size_t size);

DmaPtr createDma(const nvm_ctrl_t *ctrl, size_t size, int cudaDevice);

DmaPtr createDma(const nvm_ctrl_t *ctrl, size_t size, uint32_t adapter, uint32_t id);

DmaPtr createDma(const nvm_ctrl_t *ctrl, size_t size, int cudaDevice, uint32_t adapter, uint32_t id);

BufferPtr createBuffer(size_t size);

BufferPtr createBuffer(size_t size, int cudaDevice);

// CHIA-HAO
bool useHugePage = false;

static void getDeviceMemory(int device, void *&bufferPtr, void *&devicePtr, size_t size, void *&origPtr)
{
    bufferPtr = nullptr;
    devicePtr = nullptr;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    size += 64 * 1024;
    // std::cout << "DMA Size: "<< size << std::endl;
    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }
    /*
    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    */

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }

    origPtr = bufferPtr;
    // devicePtr = (void*) (((uint64_t)attrs.devicePointer));
    devicePtr = (void *)((((uint64_t)attrs.devicePointer) + (64 * 1024)) & 0xffffffffff0000);
    bufferPtr = (void *)((((uint64_t)bufferPtr) + (64 * 1024)) & 0xffffffffff0000);
}

static void getDeviceMemory2(int device, void *&bufferPtr, size_t size, void *&origPtr)
{
    bufferPtr = nullptr;
    // devicePtr = nullptr;
    size += 128;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    /*
        cudaPointerAttributes attrs;
        err = cudaPointerGetAttributes(&attrs, bufferPtr);
        if (err != cudaSuccess)
        {
            cudaFree(bufferPtr);
            throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
        }

        devicePtr = (void*) (((uint64_t)attrs.devicePointer));
    */
    origPtr = bufferPtr;
    bufferPtr = (void *)((((uint64_t)bufferPtr) + (128)) & 0xffffffffffffe0);
    // std::cout << "getdeviceMemory: " << std::hex << bufferPtr <<  std::endl;
}

static void getDeviceMemory(int device, void *&bufferPtr, size_t size)
{
    void *notUsed = nullptr;
    getDeviceMemory(device, bufferPtr, notUsed, size, notUsed);
}

/*
static void getDeviceMemory2(int device, void*& bufferPtr, size_t size)
{
    void* notUsed = nullptr;
    getDeviceMemory2(device, bufferPtr, size);
}
*/

inline DmaPtr createDmaOnNode(const nvm_ctrl_t *ctrl, size_t size)
{
    nvm_dma_t *dma = nullptr;
    void *buffer = nullptr;
    int err = 0;

#define TARGET_NUMA_NODE 1
    bool is_numa_alloc = false;

    if (numa_available() >= 0)
    {
        printf("Allocating Host DMA buffer (size %lu) on NUMA node %d\n", size, TARGET_NUMA_NODE);
        buffer = numa_alloc_onnode(size, TARGET_NUMA_NODE);
        if (!buffer)
        {
            perror("numa_alloc_onnode");
            exit(1);
        }
        memset(buffer, 0, size); // "接触" 内存以在目标节点上提交
        is_numa_alloc = true;
    }
    else
    {
        // 如果 NUMA 不可用，回退到 posix_memalign (标准 4KB 页)
        fprintf(stderr, "WARN: numa_alloc_onnode not available. Falling back to posix_memalign for host DMA.\n");
        err = posix_memalign(&buffer, 65536, size);
        if (err)
        {
            throw error(string("Failed to allocate host memory (posix_memalign fallback): ") + std::to_string(err));
        }
    }

    if (err)
    {
        throw error(string("Failed to allocate host memory: ") + std::to_string(err));
    }

    int status = nvm_dma_map_host(&dma, ctrl, buffer, size);
    if (!nvm_ok(status))
    {
        if (is_numa_alloc)
        {
            numa_free(buffer, size);
        }
        else if (buffer != nullptr)
        {
            free(buffer);
        }
        throw error(string("Failed to map host memory: ") + nvm_strerror(status));
    }

    // 注意：这里的 deleter 专门处理 numa_free / free
    fprintf(stderr, "Creating Host DMA (OnNode): %p (size %lu)\n", buffer, size);
    fflush(stderr);
    return DmaPtr(dma, [buffer, is_numa_alloc, size](nvm_dma_t *dma)
                  {
                      nvm_dma_unmap(dma);
                      if (is_numa_alloc) {
                          fprintf(stderr, "Deleting Host DMA (numa_free): %p (size %lu)\n", buffer, size);
                          numa_free(buffer, size);
                      }
                      else {
                          fprintf(stderr, "Deleting Host DMA (free): %p (size %lu)\n", buffer, size);
                          free(buffer);
                      } });
}

// 创建一个dma对象，用于host与nvme之间进行数据传输。
inline DmaPtr createDma(const nvm_ctrl_t *ctrl, size_t size)
{
    nvm_dma_t *dma = nullptr;
    void *buffer = nullptr;

    /*
    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }
    */

    int err = 0;

    // if (useHugePage && !force_use_4k_page) {
    if (useHugePage)
    {
#if 0
        madvise(buffer, size, MADV_HUGEPAGE);
        err = posix_memalign(&buffer, 2*1024*1024, size);
#else
        buffer = mmap(ADDR, size, PROTECTION, FLAGS, -1, 0);
        if (buffer == MAP_FAILED)
        {
            perror("mmap");
            exit(1);
        }
#endif

        printf("buffer created %p\n", buffer);
    }
    else
    {
        // madvise(buffer, size, MADV_NOHUGEPAGE);
        err = posix_memalign(&buffer, 65536, size);
        // printf("allocated mem by posix_memalign\n");
    }
    // CHIA-HAO
    // cudaHostRegister(buffer, (size_t)8*1024*1024*1024, cudaHostRegisterDefault);
    // if (useHugePage) {
    //    madvise(buffer, size, MADV_HUGEPAGE);
    //}
    // mlock(buffer, size);

    if (err)
    {
        throw error(string("Failed to allocate host memory: ") + std::to_string(err));
    }
    int status = nvm_dma_map_host(&dma, ctrl, buffer, size);
    if (!nvm_ok(status))
    {
        // cudaFreeHost(buffer);
        //  free(buffer);

        if (useHugePage && buffer != MAP_FAILED)
        {
            munmap(buffer, size);
        }
        else if (!useHugePage && buffer != nullptr)
        {
            free(buffer);
        }
        throw error(string("Failed to map host memory: ") + nvm_strerror(status));
    }

    bool is_hugetlb = useHugePage;

    printf("Creating Host DMA: %p (size %lu)\n", buffer, size);
    return DmaPtr(dma, [buffer, is_hugetlb, size](nvm_dma_t *dma)
                  {
        nvm_dma_unmap(dma);
        //cudaFreeHost(buffer);

 if (is_hugetlb) {
            printf("Deleting Host DMA: %p (size %lu)\n", buffer, size);
            if (munmap(buffer, size)) {
                perror("munmap");
                exit(1);
            }
            else {
                printf("munamp succ: %p - size %lu\n", buffer, size);
            }
        }
        else {
            printf("free buffer %p\n", buffer);
            free(buffer);
        } });
}

// 和不带cudaDevice参数的createDma不同，该函数创建的是nvme和gpu之间的dma对象。
// page_cache_t::pdt.base_addr 是 GPU 显存，通过带有 cudaDevice 参数的 createDma 版本分配，并使用 nvm_dma_map_device 实现 GPUDirect Storage。
inline DmaPtr createDma(const nvm_ctrl_t *ctrl, size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createDma(ctrl, size);
    }

    nvm_dma_t *dma = nullptr;
    void *bufferPtr = nullptr;
    void *devicePtr = nullptr;
    void *origPtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size, origPtr);

    // std::cout << "Got Device mem\n";
    int status = nvm_dma_map_device(&dma, ctrl, bufferPtr, size);
    // std::cout << "Got dma_map_devce\n";
    if (!nvm_ok(status))
    {
        // std::cout << "Got dma_map_devce failed\n";
        // cudaFree(bufferPtr);
        throw error(string("Failed to map device memory: ") + nvm_strerror(status));
    }
    cudaError_t err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    dma->vaddr = bufferPtr;

    std::cout << "Createing DMA: " << origPtr << "\n";
    return DmaPtr(dma, [bufferPtr, origPtr](nvm_dma_t *dma)
                  {
        std::cout << "Deleting DMA: " << origPtr << "\n";
        if (dma)
            nvm_dma_unmap(dma);
        if (origPtr)
            cudaFree(origPtr);
        std::cout << "Deleting DMA: " << origPtr << " finished."<< "\n"; });
}

inline BufferPtr createBuffer(size_t size)
{
    void *buffer = nullptr;

    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }

    return BufferPtr(buffer, [](void *ptr)
                     { cudaFreeHost(ptr); });
}

inline BufferPtr createBuffer(size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createBuffer(size);
    }

    void *bufferPtr = nullptr;
    void *origPtr = nullptr;

    getDeviceMemory2(cudaDevice, bufferPtr, size, origPtr);
    // std::cout << "createbuffer: " << std::hex << bufferPtr <<  std::endl;

    return BufferPtr(bufferPtr, [origPtr](void *ptr)
                     {
                         if (ptr)
                         {
                             __ignore(ptr);
                             cudaFree(origPtr);
                         }
                         // std::cout << "Deleting Buffer\n";
                     });
}

/*

DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size)
{
    return createDma(ctrl, size);
}



DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice)
{
    return createDma(ctrl, size, cudaDevice);
}

*/
#endif
