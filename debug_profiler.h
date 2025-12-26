// debug_profiler.h
#pragma once
#include <nvtx3/nvToolsExt.h>
#include <map>
#include <mutex>

// ==========================================
// 1. GMT 专用配色方案 (ARGB)
// ==========================================
namespace GMT_Color {
    // 通用类
    static const uint32_t KERNEL_LAUNCH = 0xFF00FF00; // 鲜绿: GPU Kernel 启动
    static const uint32_t PREFETCH_OPS  = 0xFFFFFF00; // 黄色: Host 预取逻辑
    static const uint32_t IO_SUBMIT     = 0xFF00FFFF; // 青色: 提交 IO 命令
    static const uint32_t IO_FLYING     = 0xFFFF4500; // 橙红: IO 在 SSD 中飞行 (Async)

    // Demand Fetch 细分 (关键路径)
    static const uint32_t FETCH_LOOKUP  = 0xFFFF00FF; // 紫色: 查表 / 抢分片锁
    static const uint32_t FETCH_WAIT    = 0xFFFF0000; // 纯红: [瓶颈] 等待预取完成 (自旋)
    static const uint32_t FETCH_COPY    = 0xFF32CD32; // 酸橙绿: PCIe H2D 拷贝
    static const uint32_t FETCH_SLOT_LK = 0xFF8A2BE2; // 蓝紫: 抢 Slot 锁

    // Demand/Fault Handling 入口
    // 复用已有颜色，避免引入新的硬编码色值
    static const uint32_t FETCH_ROUTINE = PREFETCH_OPS;

    // Prefetch 细分 (独立配色，便于与其它阶段区分)
    static const uint32_t PREFETCH_PROCESS = 0xFF1E90FF; // 道奇蓝: Prefetch 处理逻辑
    static const uint32_t PREFETCH_LOCK    = 0xFFFFD700; // 金色: Prefetch 锁竞争/临界区

    // Evict 细分
    static const uint32_t EVICT_SEARCH  = 0xFFFFA500; // 橙色: 寻找空闲 Slot (哈希冲突/满)
    static const uint32_t EVICT_UPDATE  = 0xFF00BFFF; // 深天蓝: 更新 Map
    static const uint32_t EVICT_COPY    = 0xFF98FB98; // 苍绿: PCIe D2H 拷贝
}

// ==========================================
// 2. 作用域性能追踪器 (RAII)
// ==========================================
// 用法: { PROFILE_SCOPE("MyBlock", GMT_Color::FETCH_WAIT); ... }
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
// 3. 异步 IO 追踪器 (跨线程)
// ==========================================
// 用于追踪 submit -> completion 的硬件耗时
static std::map<uint16_t, nvtxRangeId_t> g_async_trace_map;
static std::mutex g_async_trace_mtx;

inline void TRACE_IO_START(const char* name, uint16_t cid) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = GMT_Color::IO_FLYING;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    
    // 启动范围并保存 ID
    std::lock_guard<std::mutex> lock(g_async_trace_mtx);
    g_async_trace_map[cid] = nvtxRangeStartEx(&eventAttrib);
}

inline void TRACE_IO_END(uint16_t cid) {
    std::lock_guard<std::mutex> lock(g_async_trace_mtx);
    auto it = g_async_trace_map.find(cid);
    if (it != g_async_trace_map.end()) {
        nvtxRangeEnd(it->second);
        g_async_trace_map.erase(it);
    }
}