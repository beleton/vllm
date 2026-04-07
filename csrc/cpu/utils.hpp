#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <map>
#include <mutex>
#include <tuple>
#include <unistd.h>
#include <vector>
#include <ATen/cpu/Utils.h>

#include "cpu/cpu_types.hpp"

namespace cpu_utils {
enum class ISA { AMX, VEC };

inline ISA get_isa(const std::string& isa) {
  if (isa == "amx") {
    return ISA::AMX;
  } else if (isa == "vec") {
    return ISA::VEC;
  } else {
    TORCH_CHECK(false, "Invalid isa type: " + isa);
  }
}

template <typename T>
struct VecTypeTrait {
  using vec_t = void;
};

template <>
struct VecTypeTrait<float> {
  using vec_t = vec_op::FP32Vec16;
};

#if !defined(__aarch64__) || defined(ARM_BF16_SUPPORT)
template <>
struct VecTypeTrait<c10::BFloat16> {
  using vec_t = vec_op::BF16Vec16;
};
#endif

#if !defined(__powerpc__)
template <>
struct VecTypeTrait<c10::Half> {
  using vec_t = vec_op::FP16Vec16;
};
#endif

struct Counter {
  std::atomic<int64_t> counter;
  char _padding[56];

  Counter() : counter(0) {}

  void reset_counter() { counter.store(0); }

  int64_t acquire_counter() { return counter++; }
};

struct ThreadLocalityGroupInfo {
  int32_t numa_node;
  int32_t socket_id;
  int32_t l3_cache_id;
  std::vector<int32_t> thread_ids;
  std::vector<int32_t> cpu_ids;
};

class ThreadLocalityManager {
 public:
  static ThreadLocalityManager* get_thread_locality_manager();

  void set_thread_cpu_ids(const std::vector<int>& omp_cpu_ids);

  const std::vector<ThreadLocalityGroupInfo>& get_groups() const;

  int32_t get_group_id_for_thread(int32_t omp_thread_id) const;

  int32_t get_local_offset_in_group(int32_t omp_thread_id) const;

 private:
  mutable std::mutex mutex_;
  std::vector<int32_t> thread_to_group_id_;
  std::vector<int32_t> thread_to_local_offset_;
  std::vector<ThreadLocalityGroupInfo> groups_;
};

std::string describe_cpu_locality_groups(const std::string& cpu_ids);

inline int64_t get_available_l2_size() {
  static int64_t size = []() {
    const uint32_t l2_cache_size = at::cpu::L2_cache_size();
    return l2_cache_size >> 1;  // use 50% of L2 cache
  }();
  return size;
}

template <int32_t alignment_v, typename T>
inline T round_up(T size) {
  T alignment = alignment_v;
  return (((size + alignment - 1) / alignment) * alignment);
}

template <int32_t alignment_v, typename T>
inline T round_down(T size) {
  T alignment = alignment_v;
  return (size / alignment) * alignment;
}

template <typename T>
inline void print_logits(const char* name, T* ptr, int32_t row, int32_t col,
                         int32_t stride) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(5) << name << ": [\n";
  auto* curr_logits_buffer = ptr;
  for (int32_t m = 0; m < row; ++m) {
    for (int32_t n = 0; n < col; ++n) {
      ss << curr_logits_buffer[n] << ", ";
    }
    ss << "\n";
    curr_logits_buffer += stride;
  }
  ss << "]\n";
  std::printf("%s", ss.str().c_str());
}

class ScratchPadManager {
 public:
  static constexpr size_t allocation_unit = 4 * 1024;  // 4KB

  static ScratchPadManager* get_scratchpad_manager();

  ScratchPadManager();

  template <typename T>
  T* get_data() {
    return reinterpret_cast<T*>(ptr_);
  }

  static size_t round(size_t size) {
    return ((size + allocation_unit - 1) / allocation_unit) * allocation_unit;
  }

  void realloc(size_t new_size);

 private:
  size_t size_;
  void* ptr_;
};
}  // namespace cpu_utils

#endif
