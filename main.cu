/* References:
 *
 *      Coalesce
 *          Hong, Sungpack, et al.
 *          "Accelerating CUDA graph algorithms at maximum warp."
 *          Acm Sigplan Notices 46.8 (2011): 267-276.
 *
 */

#include <cuda.h>
#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <getopt.h>
// #include "helper_cuda.h"
#include <algorithm>
#include <vector>
#include <numeric>
#include <iterator>
#include <math.h>
#include <chrono>
#include <ctime>
#include <ratio>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>

#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <buffer.h>
#include "settings.h"
#include <ctrl.h>
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>

#include <iterator>
#include <functional>

#define UINT64MAX 0xFFFFFFFFFFFFFFFF

using error = std::runtime_error;
using std::string;
// const char* const ctrls_paths[] = {"/dev/libnvmpro0", "/dev/libnvmpro1", "/dev/libnvmpro2", "/dev/libnvmpro3", "/dev/libnvmpro4", "/dev/libnvmpro5", "/dev/libnvmpro6", "/dev/libnvmpro7"};
const char *const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};
// const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};
// const char* const ctrls_paths[] = {"/dev/libnvm_vmalloc0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};

#define WARP_SHIFT 5
#define WARP_SIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define BLOCK_NUM 1024ULL

#define MAXWARP 64

#define TYPE uint64_t

typedef uint64_t EdgeT;

typedef enum
{
    BASELINE = 0,
    OPTIMIZED = 1,
    BASELINE_PC = 2,
    OPTIMIZED_PC = 3,
} impl_type;

typedef enum
{
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    UVM_READONLY_NVLINK = 3,
    UVM_DIRECT_NVLINK = 4,
    BAFS_DIRECT = 6,
} mem_type;

__global__ //__launch_bounds__(64,32)
    void kernel_baseline(uint64_t n_elems, TYPE *A, TYPE *B, TYPE *sum)
{
    // uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elems)
    {
        sum[tid] = A[tid] + B[tid];
        // uint64_t val = A[tid] + B[tid];
        // atomicAdd(&sum[0], val);
        // printf("tid: %llu A:%llu B:%llu \n",tid,  A[tid], B[tid]);
    }
}

template <typename T>
__global__ __launch_bounds__(64, 32) void kernel_sequential_warp(T *A, T *B, uint64_t n_elems, uint64_t n_pages_per_warp, TYPE *sum, uint64_t n_warps, size_t page_size)
{

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    T val = 0;
    uint64_t idx = 0;

    if (tid == 0)
        printf("n_elems_per_page: %llu\n", n_elems_per_page);
    if (warp_id < n_warps)
    {
        size_t start_page = n_pages_per_warp * warp_id;
        ;
        for (size_t i = 0; i < n_pages_per_warp; i++)
        {
            size_t cur_page = start_page + i;
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += WARP_SIZE)
            {
                idx = start_idx + j;
                if (idx < n_elems)
                {
                    val = A[idx] + B[idx];
                    sum[idx] = val;
                    // atomicAdd(&sum[0], val);
                    //            printf("tid: %llu A:%llu B:%llu \n",idx,  A[tid], B[tid]);
                }
            }
        }
    }
}

template <typename T>
__global__ //__launch_bounds__(64,32)
    void kernel_sequential_warp_ptr_pc(array_d_t<T> *da, array_d_t<T> *db, uint64_t n_elems, uint64_t n_pages_per_warp, array_d_t<T> *dc, TYPE *sum, uint64_t n_warps, size_t page_size, uint64_t stride)
{

    bam_ptr<T> Aptr(da);
    bam_ptr<T> Bptr(db);
    bam_ptr<T> Cptr(dc);

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    T val = 0;
    uint64_t idx = 0;
    uint64_t nep = (n_warps + stride - 1) / stride;
    uint64_t warp_id = (old_warp_id / nep) + ((old_warp_id % nep) * stride);

    if (warp_id < n_warps)
    {
        size_t start_page = n_pages_per_warp * warp_id;
        ;
        for (size_t i = 0; i < n_pages_per_warp; i++)
        {
            size_t cur_page = start_page + i;
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += WARP_SIZE)
            {
                idx = start_idx + j;
                if (idx < n_elems)
                {
                    // val  = Aptr[idx] + Bptr[idx];
                    // Cptr[idx] = val;
                    // Cptr[idx] = Aptr[idx] + Bptr[idx];

                    TYPE a = Aptr[idx];
                    Aptr.fini();
                    TYPE b = Bptr[idx];
                    Bptr.fini();
                    // Cptr[idx] = a + b;
                    Cptr.write(idx, a + b);
                    Cptr.fini();

                    // sum[idx] = val;
                    // atomicAdd(&sum[0], val);
                    //            printf("tid: %llu A:%llu B:%llu \n",idx,  A[tid], B[tid]);
                }
            }
        }
        // sum[0] =val;
    }
}

template <typename T>
__global__ //__launch_bounds__(64,32)
    void kernel_sequential_warp_ptr_pc(array_d_t<T> *d_in, array_d_t<T> *d_out, uint64_t n_elems, uint64_t n_pages_per_warp, uint64_t n_warps, size_t page_size, uint64_t stride)
{

    bam_ptr<T> INptr(d_in);
    bam_ptr<T> OUTptr(d_out);

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    T val = 0;
    uint64_t idx = 0;
    uint64_t nep = (n_warps + stride - 1) / stride;
    uint64_t warp_id = (old_warp_id / nep) + ((old_warp_id % nep) * stride);

    if (warp_id < n_warps)
    {
        size_t start_page = n_pages_per_warp * warp_id;
        ;
        for (size_t i = 0; i < n_pages_per_warp; i++)
        {
            size_t cur_page = start_page + i;
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += WARP_SIZE)
            {
                idx = start_idx + j;
                if (idx < n_elems)
                {
                    d_out->seq_write(idx, d_in->seq_read(idx) + d_out->seq_read(idx));
                    // TYPE a = INptr[idx]; INptr.fini();
                    // TYPE b = OUTptr[idx]; OUTptr.fini();
                    // OUTptr.write(idx, a+b); OUTptr.fini();
                }
            }
        }
    }
}

// 预取算法1
template <typename T>
__global__ //__launch_bounds__(64,32)
    void kernel_sequential_warp_ptr_pc_prefetch(array_d_t<T> *d_in, array_d_t<T> *d_out, uint64_t n_elems, uint64_t n_pages_per_warp, uint64_t n_warps, size_t page_size, uint64_t stride, uint64_t prefetch_ahead)
{

    // --- 1. 索引计算 ---
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    const uint64_t total_pages = (n_elems + n_elems_per_page - 1) / n_elems_per_page;

    // Warp 调度重映射
    uint64_t nep = (n_warps + stride - 1) / stride;
    uint64_t warp_id = (old_warp_id / nep) + ((old_warp_id % nep) * stride);

    if (warp_id >= n_warps)
        return;

    // --- 2. 划定领地 (Territory) ---
    // 每个 Warp 负责 [my_start, my_end)
    size_t my_start = n_pages_per_warp * warp_id;
    size_t my_end = my_start + n_pages_per_warp;

    // 全局边界修正
    if (my_end > total_pages)
        my_end = total_pages;
    if (my_start >= total_pages)
        return;

    // --- 3. 启动阶段：填充流水线 (Priming) ---
    // 预取 [start, start + ahead - 1]
    // 这里的逻辑是：先把"前几个"页面发出去。
    // 如果 n_pages_per_warp=1, ahead=4，这里只会发第 0 页 (因为受 my_end 限制)，
    // 这正是我们想要的！(后面的页归下一个 Warp 管)
    if (lane == 0 && prefetch_ahead > 0)
    {
#pragma unroll 4
        for (uint64_t k = 0; k < prefetch_ahead; k++) // 注意是 < ahead
        {
            size_t pf_page = my_start + k;

            // [严格边界]: 撞墙即停，绝不碰别人的数据
            if (pf_page >= my_end)
                break;

            // 发射预取
            size_t pf_idx = pf_page * n_elems_per_page;
            if (pf_idx < n_elems)
            {
                d_in->prefetch_page_by_index(pf_idx);
                d_out->prefetch_page_by_index(pf_idx);
            }
        }
    }

    // --- 4. 主循环：计算 + 滚动预取 ---
    for (size_t i = 0; i < n_pages_per_warp; i++)
    {
        size_t cur_page = my_start + i;
        if (cur_page >= my_end)
            break;

        // [滚动预取]: 在处理第 i 页时，预取第 i + ahead 页
        // 这样刚好接上 Priming 留下的尾巴
        if (lane == 0 && prefetch_ahead > 0)
        {
            size_t pf_page = cur_page + prefetch_ahead;

            // [严格边界]: 同样，只预取自己领地内的
            if (pf_page < my_end)
            {
                size_t pf_idx = pf_page * n_elems_per_page;
                if (pf_idx < n_elems)
                {
                    d_in->prefetch_page_by_index(pf_idx);
                    d_out->prefetch_page_by_index(pf_idx);
                }
            }
        }

        // --- 5. 核心计算 (Vector Add) ---
        size_t start_idx = cur_page * n_elems_per_page + lane;

#pragma unroll
        for (size_t j = 0; j < n_elems_per_page; j += 32)
        {
            uint64_t idx = start_idx + j;
            if (idx < n_elems)
            {
                d_out->seq_write(idx, d_in->seq_read(idx) + d_out->seq_read(idx));
            }
        }
    }
}
// 预取算法2
template <typename T>
__global__ void kernel_sequential_warp_ptr_pc_prefetch_2(
    array_d_t<T> *d_in,
    array_d_t<T> *d_out,
    uint64_t n_elems,
    uint64_t n_pages_per_warp,
    uint64_t n_warps,
    size_t page_size,
    uint64_t stride,
    uint64_t prefetch_ahead)
{
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);

    // Warp 调度逻辑
    uint64_t nep = (n_warps + stride - 1) / stride;
    uint64_t warp_id = (old_warp_id / nep) + ((old_warp_id % nep) * stride);

    const uint64_t n_pages_total = (n_elems + n_elems_per_page - 1) / n_elems_per_page;

    if (warp_id < n_warps)
    {
        // 当前 Warp 负责的页面范围
        size_t start_page = n_pages_per_warp * warp_id;
        size_t end_page = min(start_page + n_pages_per_warp, n_pages_total);

        // =========================================================
        // 阶段 A: 启动坑填补 (Gap Filling) - 仅由 Warp 0 执行
        // =========================================================
        // 只有全局第 0 个 Warp 负责预取最前面的 "ahead" 个页面
        // 避免所有 Warp 都去预取这些页
        if (warp_id == 0 && lane == 0 && prefetch_ahead > 0)
        {
            // 限制一次填补的数量，防止 Warp 0 卡太久
            size_t gap_size = min((size_t)prefetch_ahead, n_pages_total);

#pragma unroll 4
            for (size_t k = 0; k < gap_size; k++)
            {
                // Warp 0 帮大家预取 Page 0 到 Page (Ahead-1)
                size_t pf_elem_idx = k * n_elems_per_page;
                d_in->prefetch_page_by_index(pf_elem_idx);
                d_out->prefetch_page_by_index(pf_elem_idx);
            }
        }

        // =========================================================
        // 阶段 B: 跨 Warp 接力预取 (Relay Prefetch)
        // =========================================================
        for (size_t cur_page = start_page; cur_page < end_page; cur_page++)
        {

            // 滚动预取：Warp i 负责预取 Page (i + AHEAD)
            // 这样 Page X 只会被 Warp (X - AHEAD) 预取一次，绝无重叠
            if (lane == 0 && prefetch_ahead > 0)
            {
                size_t target_page = cur_page + prefetch_ahead;

                // 只要未越界，就发射预取
                // 注意：这里不再检查 target_page 是否属于本 Warp (target_page < end_page)
                // 因为我们是在为"未来的 Warp"做预取
                if (target_page < n_pages_total)
                {
                    size_t pf_elem_idx = target_page * n_elems_per_page;
                    d_in->prefetch_page_by_index(pf_elem_idx);
                    d_out->prefetch_page_by_index(pf_elem_idx);
                }
            }

            // 2. 实际计算
            size_t start_idx = cur_page * n_elems_per_page + lane;
            for (size_t j = 0; j < n_elems_per_page; j += WARP_SIZE)
            {
                uint64_t idx = start_idx + j;
                if (idx < n_elems)
                {
                    d_out->seq_write(idx, d_in->seq_read(idx) + d_out->seq_read(idx));
                }
            }
        }
    }
}
__global__ void print_reuse_dist(page_cache_d_t *pc, int range, uint64_t num_pages, uint64_t *out)
{
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    // for (int i = tid; i < num_pages; tid += blockDim.x*blockIdx.x) {
    out[tid] = pc->ranges_reuse_hist[range][tid].estimated_remaining_reuse_dist;
    //}
}
int main(int argc, char *argv[])
{
    using namespace std::chrono;

    Settings settings;
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string &e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }

    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    std::ifstream filea, fileb;
    std::string a_file, b_file;
    std::string a_file_bin, b_file_bin;
    std::string filename;

    impl_type type;
    mem_type mem;
    uint32_t *pad;
    TYPE *a_h, *a_d;
    TYPE *b_h, *b_d;
    TYPE *c_h, *c_d;
    uint64_t n_elems, n_size;
    uint64_t typeT;
    uint64_t numblocks, numthreads;
    size_t freebyte, totalbyte;

    float milliseconds;

    uint64_t pc_page_size;
    uint64_t pc_pages;

    try
    {

        a_file = std::string(settings.input_a);
        b_file = std::string(settings.input_b);

        type = (impl_type)settings.type;
        mem = (mem_type)settings.memalloc;

        pc_page_size = settings.pageSize;
        pc_pages = ceil((float)settings.maxPageCacheSize / pc_page_size);

        numthreads = settings.numThreads;

        cuda_err_chk(cudaSetDevice(settings.cudaDevice));

        cudaEvent_t start, end, tstart, tend;
        cuda_err_chk(cudaEventCreate(&start));
        cuda_err_chk(cudaEventCreate(&end));
        cuda_err_chk(cudaEventCreate(&tstart));
        cuda_err_chk(cudaEventCreate(&tend));

        a_file_bin = a_file + ".dst";
        b_file_bin = b_file + ".dst";

        std::cout << "A: " << a_file_bin << " B: " << b_file_bin << std::endl;

        uint64_t n_elems = settings.n_elems;
        uint64_t n_elems_size = n_elems * sizeof(TYPE);
        printf("Total elements: %llu \n", n_elems);
        uint64_t tmp;

        // Read files
        filea.open(a_file_bin.c_str(), std::ios::in | std::ios::binary);
        if (!filea.is_open())
        {
            printf("A file open failed\n");
            // exit(1);
        };

        // filea.read((char*)(&tmp), 16);
        if (mem != BAFS_DIRECT)
            a_h = (TYPE *)malloc(n_elems_size);
        if ((mem != BAFS_DIRECT) && (mem != UVM_DIRECT))
        {
            // printf("before mem switch\n");
            // fflush(stdout);
            filea.read((char *)a_h, n_elems_size);
            filea.close();
        }

        // Read files
        fileb.open(b_file_bin.c_str(), std::ios::in | std::ios::binary);
        if (!fileb.is_open())
        {
            printf("A file open failed\n");
            // exit(1);
        };

        // fileb.read((char*)(&tmp), 16);
        if (mem != BAFS_DIRECT)
            b_h = (TYPE *)malloc(n_elems_size);
        if ((mem != BAFS_DIRECT) && (mem != UVM_DIRECT))
        {
            fileb.read((char *)b_h, n_elems_size);
            fileb.close();
        }

        /**/
        // uint64_t* tmp_buf_d;
        // cudaMalloc((void**)&tmp_buf_d, sizeof(uint64_t)*16384);
        // cudaMemcpyToSymbol(tmp_buf, &tmp_buf_d, sizeof(uint64_t*));

        switch (mem)
        {
        case GPUMEM:
        {
            cuda_err_chk(cudaMalloc((void **)&a_d, n_elems_size));
            cuda_err_chk(cudaMalloc((void **)&b_d, n_elems_size));
            high_resolution_clock::time_point mc1 = high_resolution_clock::now();
            cuda_err_chk(cudaMemcpy(a_d, a_h, n_elems_size, cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(b_d, b_h, n_elems_size, cudaMemcpyHostToDevice));
            high_resolution_clock::time_point mc2 = high_resolution_clock::now();
            duration<double> mc_time_span = duration_cast<duration<double>>(mc2 - mc1);
            std::cout << "Memcpy time for loading the inputs: " << mc_time_span.count() << std::endl;
            break;
        }
        case UVM_READONLY:
        {
            cuda_err_chk(cudaMallocManaged((void **)&a_d, n_elems_size));
            cuda_err_chk(cudaMallocManaged((void **)&b_d, n_elems_size));
            cuda_err_chk(cudaMemcpy(a_d, a_h, n_elems_size, cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(b_d, b_h, n_elems_size, cudaMemcpyHostToDevice));
            // TODO: we can move that read op here.
            // file.read((char*)edgeList_d, edge_size);
            cuda_err_chk(cudaMemAdvise(a_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
            cuda_err_chk(cudaMemAdvise(b_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
            cuda_err_chk(cudaMemGetInfo(&freebyte, &totalbyte));
            break;
        }
        case UVM_DIRECT:
        {
            filea.close();
            fileb.close();
            int fda = open(a_file_bin.c_str(), O_RDONLY | O_DIRECT);
            int fdb = open(b_file_bin.c_str(), O_RDONLY | O_DIRECT);
            FILE *fa_tmp = fdopen(fda, "rb");
            if ((fa_tmp == NULL) || (fda == -1))
            {
                printf("A file fd open failed\n");
                exit(1);
            }
            FILE *fb_tmp = fdopen(fdb, "rb");
            if ((fb_tmp == NULL) || (fdb == -1))
            {
                printf("A file fd open failed\n");
                exit(1);
            }

            uint64_t count_4k_aligned = ((n_elems + 2 + 4096 / sizeof(TYPE)) / (4096 / sizeof(TYPE))) * (4096 / sizeof(TYPE));
            // uint64_t count_4k_aligned = n_elems;
            uint64_t size_4k_aligned = count_4k_aligned * sizeof(TYPE);

            cuda_err_chk(cudaMallocManaged((void **)&a_d, size_4k_aligned));
            cuda_err_chk(cudaMallocManaged((void **)&b_d, size_4k_aligned));
            cuda_err_chk(cudaMemAdvise(a_d, size_4k_aligned, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
            cuda_err_chk(cudaMemAdvise(b_d, size_4k_aligned, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
            high_resolution_clock::time_point ft1 = high_resolution_clock::now();

            if (fread(a_d, sizeof(TYPE), count_4k_aligned, fa_tmp) < 0)
            {
                printf("A file fread failed: %llu \t %llu\n", count_4k_aligned, n_elems + 2);
                exit(1);
            }
            fclose(fa_tmp);
            close(fda);

            if (fread(b_d, sizeof(TYPE), count_4k_aligned, fb_tmp) < 0)
            {
                printf("B file fread failed\n");
                exit(1);
            }
            fclose(fb_tmp);
            close(fdb);

            a_d = a_d + 2;
            b_d = b_d + 2;

            high_resolution_clock::time_point ft2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(ft2 - ft1);
            std::cout << "file read time: " << time_span.count() << std::endl;

            /* //THIS DOES NOT WORK
            high_resolution_clock::time_point ft1 = high_resolution_clock::now();
            cuda_err_chk(cudaMallocManaged((void**)&a_d, n_elems_size));
            cuda_err_chk(cudaMallocManaged((void**)&b_d, n_elems_size));
            filea.read((char*)a_d, n_elems_size);
            fileb.read((char*)b_d, n_elems_size);
            cuda_err_chk(cudaMemAdvise(a_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
            cuda_err_chk(cudaMemAdvise(b_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
            //cuda_err_chk(cudaMemAdvise(a_d, n_elems_size, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
            //cuda_err_chk(cudaMemAdvise(b_d, n_elems_size, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
            high_resolution_clock::time_point ft2 = high_resolution_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(ft2 -ft1);
            std::cout<< "file read time: "<< time_span.count() <<std::endl;
            */

            break;
        }
        case BAFS_DIRECT:
        {
            break;
        }
        }

        uint64_t n_pages = ceil(((float)n_elems_size) / pc_page_size);

        // Allocate memory for GPU
        TYPE *sum_d;
        TYPE *sum_h;
        // sum_h = (TYPE*) malloc(n_elems*sizeof(TYPE));

        // cuda_err_chk(cudaMalloc((void**)&sum_d, n_elems*sizeof(TYPE)));

        printf("Allocation finished\n");
        fflush(stdout);

        uint64_t n_warps;

        switch (type)
        {

        case OPTIMIZED:
        case OPTIMIZED_PC:
        {   
            uint64_t pages_per_warp = settings.pagesPerWarp;
            uint64_t n_elems_per_page = pc_page_size / sizeof(TYPE);
            uint64_t total_pages = (n_elems + n_elems_per_page - 1) / n_elems_per_page;
            uint64_t n_warps = (total_pages + pages_per_warp - 1) / pages_per_warp;
            numblocks = (n_warps * WARP_SIZE + numthreads - 1) / numthreads;
            break;
        }
        default:
            fprintf(stderr, "Invalid type\n");
            exit(1);
            break;
        }

        // dim3 blockDim(BLOCK_NUM, (numblocks+BLOCK_NUM)/BLOCK_NUM);
        dim3 blockDim(numblocks);

        if ((type == BASELINE_PC) || (type == OPTIMIZED_PC))
        {
            printf("page size: %d, pc_entries: %llu\n", pc_page_size, pc_pages);
        }
        std::vector<Controller *> ctrls(settings.n_ctrls);
        if (mem == BAFS_DIRECT)
        {
            cuda_err_chk(cudaSetDevice(settings.cudaDevice));
            for (size_t i = 0; i < settings.n_ctrls; i++)
            {
                ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
                // size_t total_queues_needed = settings.numQueues + 1;
                // printf("Allocating Controller with %lu queues (GPU: %lu, Host: 1)\n", total_queues_needed, settings.numQueues);
                // ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, total_queues_needed);
            }
            printf("Controllers Created\n");
        }
        printf("Initialization done\n");
        fflush(stdout);

        page_cache_t *h_pc;
        range_t<TYPE> *h_Arange;
        range_t<TYPE> *h_Brange;
        range_t<TYPE> *h_Crange;
        std::vector<range_t<TYPE> *> vec_Arange(1);
        std::vector<range_t<TYPE> *> vec_Brange(1);
        std::vector<range_t<TYPE> *> vec_Crange(1);
        array_t<TYPE> *h_Aarray;
        array_t<TYPE> *h_Barray;
        array_t<TYPE> *h_Carray;

        // CHIA-HAO: multiple vecs addition
        std::vector<range_t<TYPE> *> h_ranges(settings.iter + 1);
        std::vector<std::vector<range_t<TYPE> *>> vec_ranges(settings.iter + 1, std::vector<range_t<TYPE> *>(1));

        range_t<TYPE> *h_out_range;
        std::vector<range_t<TYPE> *> vec_out_range(1);

        std::vector<array_t<TYPE> *> h_arrays(settings.iter + 1);
        array_t<TYPE> *h_out_array;

        ///////////////////////////////////////////
        // chenjiahao
        // uint64_t cfileoffset = 720*1024*1024*1024;
        uint64_t cfileoffset = 72 * 1024 * 1024 * 1024;
        if ((type == BASELINE_PC) || (type == OPTIMIZED_PC))
        {
            // TODO: fix for 2 arrays
            h_pc = new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);

            // CHIA-HAO: multple vecs addition
            for (uint64_t i = 0; i < settings.iter + 1; i++)
            {
                h_ranges[i] = new range_t<TYPE>((uint64_t)0, (uint64_t)n_elems, (uint64_t)n_pages * i, (uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice, REPLICATE);
                vec_ranges[i][0] = h_ranges[i];
                h_arrays[i] = new array_t<TYPE>(n_elems, (uint64_t)n_pages * pc_page_size * i, vec_ranges[i], settings.cudaDevice, i);
            }

            printf("Page cache initialized\n");
            fflush(stdout);
        }

// CHIA-HAO
#if USE_HOST_CACHE
        HostCache *host_cache = createHostCache(ctrls[0], settings.maxPageCacheSize, settings.batchSize,
                                                (uint32_t)settings.nNvmeQueues,
                                                (uint32_t)settings.hostQueueEntries,
                                                (uint32_t)settings.spinThreshold);
        fprintf(stderr, "[DEBUG] Main Thread: Returned from createHostCache. Registering LBAs...\n");
        fflush(stderr);
        uint64_t offset = 0;
        host_cache->registerRangesLBA(offset / 512);
        offset += n_pages * pc_page_size;
        host_cache->registerRangesLBA(offset / 512);
        offset += n_pages * pc_page_size;
        host_cache->registerRangesLBA(offset / 512);
        offset += n_pages * pc_page_size;
        host_cache->registerRangesLBA(offset / 512);
        offset += n_pages * pc_page_size;
        host_cache->registerRangesLBA(offset / 512);
        offset += n_pages * pc_page_size;
        host_cache->registerRangesLBA(offset / 512);
        offset += n_pages * pc_page_size;
#if USE_PREFETCH
        // P/C线程模式
        fprintf(stderr, "[DEBUG] Main Thread: LBAs registered. Calling LaunchPrefetchThreads().\n");
        fflush(stderr);
        LaunchPrefetchThreads();
#endif
#endif

        // cuda_err_chk(cudaMalloc((void**)&c_d, n_elems_size));

        cuda_err_chk(cudaEventRecord(start, 0));
        auto itrstart = std::chrono::system_clock::now();
        for (int titr = 0; titr < settings.iter; titr += 1)
        {

            // cuda_err_chk(cudaMemset(sum_d, 0, n_elems*sizeof(TYPE)));

            switch (type)
            {

            case OPTIMIZED:
                printf("launching optimized: blockDim.x :%llu numthreads:%llu\n", blockDim.x, numthreads);
                kernel_sequential_warp<TYPE><<<blockDim, numthreads>>>(a_d, b_d, n_elems, 1, sum_d, n_warps, settings.pageSize);
                break;

            case OPTIMIZED_PC:
                printf("launching optimized_pc: blockDim.x :%llu numthreads:%llu\n", blockDim.x, numthreads);
                fflush(stdout);
#if USE_HOST_CACHE
                // if (titr == settings.iter - 1)
                // {
                //     lastIteration(&stream_mngr->kernel_stream);
                // }
#if USE_PREFETCH
                printf("== Using MANUAL PREFETCH Kernel ==\n");
                {
                    kernel_sequential_warp_ptr_pc_prefetch<TYPE><<<blockDim, numthreads, 0, stream_mngr->kernel_stream>>>(h_arrays[titr + 1]->d_array_ptr, h_arrays[0]->d_array_ptr, n_elems, settings.pagesPerWarp, n_warps, settings.pageSize, settings.stride, settings.prefetchAhead);
                }
#else
                printf("== Using BASE Kernel (No Prefetch) ==\n");
                kernel_sequential_warp_ptr_pc<TYPE><<<blockDim, numthreads, 0, stream_mngr->kernel_stream>>>(h_arrays[titr + 1]->d_array_ptr, h_arrays[0]->d_array_ptr, n_elems, 1, n_warps, settings.pageSize, settings.stride);
#endif
#else
                kernel_sequential_warp_ptr_pc<TYPE><<<blockDim, numthreads>>>(h_arrays[titr + 1]->d_array_ptr, h_arrays[0]->d_array_ptr, n_elems, 1, n_warps, settings.pageSize, settings.stride);
// h_pc->flush_cache();
#endif

                break;
            default:
                fprintf(stderr, "Invalid type\n");
                exit(1);
                break;
            }

            cudaError_t err{cudaGetLastError()};
            if (err != cudaSuccess)
            {
                fprintf(stderr, "cuda error %s\n", cudaGetErrorString(err));
                exit(1);
            }

#if USE_HOST_CACHE
            cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
#else
            cuda_err_chk(cudaDeviceSynchronize());
#endif
        }
#if USE_HOST_CACHE
        printf("itertation done!\n");
        // h_pc->flush_cache_to_hc(&stream_mngr->kernel_stream);
        // 确保所有计算内核（包括预取）都在 last_iteration=false 时运行
        if (settings.iter > 0)
        {
            printf("Setting last iteration flag for final flush...\n");
            lastIteration(&stream_mngr->kernel_stream); //
            // 确保 GPU 内核设置标志的动作在 CPU 继续执行 flush 之前完成
            cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
        }
        h_pc->flush_cache();
        // flushHostCache();
        cuda_err_chk(cudaDeviceSynchronize());
        // cuda_err_chk(cudaStreamSynchronize(stream_mngr->kernel_stream));
        // flushHostCache();
#else
        printf("itertation done!\n");
        h_pc->flush_cache();
        cuda_err_chk(cudaDeviceSynchronize());
#endif
        cuda_err_chk(cudaEventRecord(end, 0));
        cuda_err_chk(cudaEventSynchronize(end));
        cuda_err_chk(cudaEventElapsedTime(&milliseconds, start, end));

        // printf("sum: %llu\n", sum_h[0]);

        auto itrend = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(itrend - itrstart);

        // if(mem == BAFS_DIRECT) {
        //          h_Aarray->print_reset_stats();
        //          h_Barray->print_reset_stats();
        //          h_Carray->print_reset_stats();
        //  printf("VA SSD: %d PageSize: %d itrTime: %f\n", settings.n_ctrls, settings.pageSize, (double)elapsed.count());
        // }

        if (mem == BAFS_DIRECT)
        {
            cuda_err_chk(cudaDeviceSynchronize());
        }
        printf("\nA:%s \t B:%s Impl: %d \t SSD: %d \t CL: %d \t Cache: %llu \t Stride: %llu \t TotalTime %f ms (high resolution clock %u ms)\n", a_file_bin.c_str(), b_file_bin.c_str(), type, settings.n_ctrls, settings.pageSize, settings.maxPageCacheSize, settings.stride, milliseconds, elapsed.count());
        fflush(stdout);
        //}

        /*
        uint64_t* out_h;
        uint64_t* out_d;
        out_h = (uint64_t*) malloc(sizeof(uint64_t)*n_pages);
        cudaMalloc(&out_d, sizeof(uint64_t)*n_pages);
        print_reuse_dist<<<n_pages/256, 256>>>(h_pc->d_pc_ptr, 0, n_pages, out_d);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "cuda error %s\n", cudaGetErrorString(err));
            exit(1);
        }
        cudaDeviceSynchronize();
        cudaMemcpy(out_h, out_d, sizeof(uint64_t)*n_pages, cudaMemcpyDeviceToHost);
        printf("reuse dist copy back done\n");
        for (uint32_t i = 0; i < n_pages; i++) {
            printf("[%u]: %lu\n", i, out_h[i]);
        }
        */
        // flushHostCache();

        for (uint64_t i = 0; i < settings.iter + 1; i++)
        {
            h_arrays[i]->print_reset_stats();
        }
#if PREFETCH_DEBUG
        if (h_pc)
        {
            h_pc->print_reset_stats();
        }
#endif
        revokeHostRuntime();

        if (mem != BAFS_DIRECT)
        {
            free(a_h);
            free(b_h);
        }

        if ((type == BASELINE_PC) || (type == OPTIMIZED_PC))
        {
            // TODO: Fix this
            delete h_pc;

            for (uint32_t i = 0; i < settings.iter; i++)
            {
                delete h_ranges[i];
                delete h_arrays[i];
            }
        }

        cuda_err_chk(cudaFree(sum_d));
        if (mem != BAFS_DIRECT)
        {
            if (mem == UVM_DIRECT)
            {
                a_d = a_d - 2;
                b_d = b_d - 2;
            }
            cuda_err_chk(cudaFree(a_d));
            cuda_err_chk(cudaFree(b_d));
        }

        for (size_t i = 0; i < settings.n_ctrls; i++)
            delete ctrls[i];

        // revokeHostRuntime();
    }
    catch (const error &e)
    {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    return 0;
}
