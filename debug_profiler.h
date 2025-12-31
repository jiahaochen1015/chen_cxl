#pragma once
#include <nvtx3/nvToolsExt.h>
#include <atomic>
#include <mutex>
#include <stdint.h>

// ==========================================
// 1. GMT 高对比度配色方案 (ARGB)
// ==========================================
namespace GMT_Color {
    // ------------------------------------------
    // [I/O] 硬件与驱动层 (橙/青)
    // ------------------------------------------
    static const uint32_t IO_FLYING     = 0xFFFF4500; // [橙红] SSD 硬件处理中 (Async Life-cycle)
    static const uint32_t IO_SUBMIT     = 0xFF00FFFF; // [青色] 提交命令 (软件开销)
    static const uint32_t IO_POLL       = 0xFFFFA07A; // [浅鲑红] 轮询等待 (CPU空转)

    // ------------------------------------------
    // [Prefetch] 预取线程行为 (蓝色系 - 冷色调)
    // ------------------------------------------
    static const uint32_t PRE_LOGIC     = 0xFF1E90FF; // [道奇蓝] 预取主逻辑
    static const uint32_t PRE_MAP_CHK   = 0xFF4682B4; // [钢蓝] T2 Map 查重
    static const uint32_t PRE_ALLOC     = 0xFF87CEFA; // [淡天蓝] 寻找空闲 Slot
    
    // ------------------------------------------
    // [Completion] 完成线程 (绿色系)
    // ------------------------------------------
    static const uint32_t CMPL_PROCESS  = 0xFF32CD32; // [酸橙绿] 完成后处理
    static const uint32_t STATE_UPDATE  = 0xFF98FB98; // [苍绿] 状态位更新

    // ------------------------------------------
    // [Demand/Fetch] 按需抓取 (高亮/紫色系 - 区分度高)
    // ------------------------------------------
    // 修复报错: 对应 Demand::Handle_Entry
    static const uint32_t FETCH_ROUTINE = 0xFF9400D3; // [暗紫] Demand 入口 / 处理总耗时
    
    // 修复报错: 对应 Fetch::Map_Lookup
    static const uint32_t FETCH_LOOKUP  = 0xFF8A2BE2; // [蓝紫] 查表 (与预取的蓝色区分)
    
    // 修复报错: 对应 Fetch::Wait_Prefetch (关键瓶颈!)
    static const uint32_t FETCH_WAIT    = 0xFFFF0000; // [纯红] 阻塞! 等待预取完成 (IO 慢了)
    
    // 修复报错: 对应 Fetch::Lock_Slot
    static const uint32_t FETCH_SLOT_LK = 0xFFFF1493; // [深粉] 抢 Slot 锁 (竞争点)
    
    // 修复报错: 对应 Fetch::H2D_Copy
    static const uint32_t FETCH_COPY    = 0xFF00FF00; // [鲜绿] PCIe H2D 拷贝 (数据进卡)

    // ------------------------------------------
    // [Evict] 驱逐逻辑 (深红/热粉 - 警告色)
    // ------------------------------------------
    static const uint32_t EVICT_SCAN    = 0xFF8B0000; // [深红] 驱逐扫描 (寻找 Victim)
    static const uint32_t EVICT_COPY    = 0xFFFF69B4; // [热粉] PCIe D2H 拷贝 (数据退回 Host)
    static const uint32_t EVICT_META    = 0xFF9932CC; // [暗兰紫] 驱逐元数据更新
    
    // ------------------------------------------
    // [Common] 通用瓶颈
    // ------------------------------------------
    static const uint32_t LOCK_SPIN     = 0xFFDC143C; // [猩红] 锁自旋 (严重竞争)
}

// ==========================================
// 2. 作用域性能追踪器 (RAII)
// ==========================================
class NvtxScope {
public:
    __forceinline__ NvtxScope(const char* name, uint32_t color) {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name;
        nvtxRangePushEx(&eventAttrib);
    }
    __forceinline__ ~NvtxScope() {
        nvtxRangePop();
    }
};

// 宏定义简化调用
#define PROFILE_SCOPE(name, color) NvtxScope nvtx_scope_##__LINE__(name, color)

// ==========================================
// 3. 高性能异步 IO 追踪器 (Lock-Free)
// ==========================================
#define MAX_TRACED_QUEUES 64
#define MAX_TRACED_CIDS   65536

class AsyncIoTracer {
public:
    static AsyncIoTracer& Get() {
        static AsyncIoTracer instance;
        return instance;
    }

    // 开始追踪 (Atomic Store)
    __forceinline__ void Start(const char* name, uint32_t qid, uint16_t cid) {
        if (qid >= MAX_TRACED_QUEUES) return;
        
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = GMT_Color::IO_FLYING;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name;

        // 使用 memory_order_release 确保此前的数据写入对 consumer 可见
        active_ranges[qid][cid].store(nvtxRangeStartEx(&eventAttrib), std::memory_order_release);
    }

    // 结束追踪 (Atomic Exchange)
    __forceinline__ void End(uint32_t qid, uint16_t cid) {
        if (qid >= MAX_TRACED_QUEUES) return;
        
        // 使用 exchange 原子地读取并置零，防止多线程重复关闭导致的错误
        nvtxRangeId_t rangeId = active_ranges[qid][cid].exchange(0, std::memory_order_acquire);
        
        if (rangeId != 0) {
            nvtxRangeEnd(rangeId);
        }
    }

private:
    // 使用 atomic 替代 volatile + mutex，实现真正的无锁追踪
    // 内存消耗: 64 * 65536 * 8 bytes ≈ 32MB (Host 内存充足)
    std::atomic<nvtxRangeId_t> active_ranges[MAX_TRACED_QUEUES][MAX_TRACED_CIDS];

    AsyncIoTracer() {
        // 初始化
        for(int i=0; i<MAX_TRACED_QUEUES; ++i)
            for(int j=0; j<MAX_TRACED_CIDS; ++j)
                active_ranges[i][j].store(0);
    }
};

#define TRACE_IO_START(name, cid, qid) AsyncIoTracer::Get().Start(name, qid, cid)
#define TRACE_IO_END(cid, qid)         AsyncIoTracer::Get().End(qid, cid)