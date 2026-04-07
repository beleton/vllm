#ifndef VLLM_NUMA_DISABLED
  #include <numa.h>
  #include <sched.h>
#endif
#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
  #include <unistd.h>
  #include <sys/syscall.h>
  #define gettid() syscall(SYS_gettid)
#endif

#include "cpu/utils.hpp"

namespace {

std::vector<int> parse_cpu_ids(const std::string& cpu_ids) {
  std::vector<int> parsed_cpu_ids;
  std::stringstream ss(cpu_ids);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(
        std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    if (token.empty()) {
      continue;
    }
    size_t dash_pos = token.find('-');
    if (dash_pos == std::string::npos) {
      parsed_cpu_ids.emplace_back(std::stoi(token));
      continue;
    }
    int start = std::stoi(token.substr(0, dash_pos));
    int end = std::stoi(token.substr(dash_pos + 1));
    TORCH_CHECK(start <= end, "Invalid cpu range: ", token);
    for (int cpu_id = start; cpu_id <= end; ++cpu_id) {
      parsed_cpu_ids.emplace_back(cpu_id);
    }
  }
  return parsed_cpu_ids;
}

int32_t read_int_from_sysfs(const std::string& path, int32_t default_value = -1) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return default_value;
  }
  int32_t value = default_value;
  file >> value;
  return file.fail() ? default_value : value;
}

int32_t get_socket_id_for_cpu(int32_t cpu_id) {
  return read_int_from_sysfs("/sys/devices/system/cpu/cpu" +
                             std::to_string(cpu_id) +
                             "/topology/physical_package_id");
}

int32_t get_l3_cache_id_for_cpu(int32_t cpu_id) {
  return read_int_from_sysfs("/sys/devices/system/cpu/cpu" +
                             std::to_string(cpu_id) + "/cache/index3/id");
}

int32_t get_numa_node_for_cpu(int32_t cpu_id) {
#ifndef VLLM_NUMA_DISABLED
  if (numa_available() != -1) {
    return numa_node_of_cpu(cpu_id);
  }
#endif
  return -1;
}

std::vector<cpu_utils::ThreadLocalityGroupInfo> build_locality_groups(
    const std::vector<int>& cpu_ids, bool use_thread_ids) {
  using GroupKey = std::tuple<int32_t, int32_t, int32_t>;
  std::map<GroupKey, cpu_utils::ThreadLocalityGroupInfo> groups_by_key;

  for (size_t idx = 0; idx < cpu_ids.size(); ++idx) {
    const int32_t cpu_id = cpu_ids[idx];
    const int32_t numa_node = get_numa_node_for_cpu(cpu_id);
    const int32_t socket_id = get_socket_id_for_cpu(cpu_id);
    const int32_t l3_cache_id = get_l3_cache_id_for_cpu(cpu_id);
    GroupKey key{numa_node, socket_id, l3_cache_id};
    auto& group = groups_by_key[key];
    group.numa_node = numa_node;
    group.socket_id = socket_id;
    group.l3_cache_id = l3_cache_id;
    group.cpu_ids.emplace_back(cpu_id);
    if (use_thread_ids) {
      group.thread_ids.emplace_back(static_cast<int32_t>(idx));
    }
  }

  std::vector<cpu_utils::ThreadLocalityGroupInfo> groups;
  groups.reserve(groups_by_key.size());
  for (auto& [_, group] : groups_by_key) {
    groups.emplace_back(std::move(group));
  }
  return groups;
}

std::string locality_groups_to_json(
    const std::vector<cpu_utils::ThreadLocalityGroupInfo>& groups) {
  std::stringstream ss;
  ss << '[';
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const auto& group = groups[group_idx];
    if (group_idx > 0) {
      ss << ',';
    }
    ss << '{';
    ss << "\"numa_node\":" << group.numa_node << ',';
    ss << "\"socket_id\":" << group.socket_id << ',';
    ss << "\"l3_cache_id\":" << group.l3_cache_id << ',';
    ss << "\"cpu_ids\":[";
    for (size_t cpu_idx = 0; cpu_idx < group.cpu_ids.size(); ++cpu_idx) {
      if (cpu_idx > 0) {
        ss << ',';
      }
      ss << group.cpu_ids[cpu_idx];
    }
    ss << "],";
    ss << "\"num_cpus\":" << group.cpu_ids.size();
    ss << '}';
  }
  ss << ']';
  return ss.str();
}

}  // namespace

#ifdef VLLM_NUMA_DISABLED
std::string init_cpu_threads_env(const std::string& cpu_ids) {
  cpu_utils::ThreadLocalityManager::get_thread_locality_manager()
      ->set_thread_cpu_ids(parse_cpu_ids(cpu_ids));
  return std::string(
      "Warning: NUMA is not enabled in this build. `init_cpu_threads_env` has "
      "no effect to setup thread affinity.");
}

#endif

#ifndef VLLM_NUMA_DISABLED
std::string init_cpu_threads_env(const std::string& cpu_ids) {
  std::vector<int> omp_cpu_ids = parse_cpu_ids(cpu_ids);
  TORCH_CHECK(!omp_cpu_ids.empty(), "Failed to parse CPU string: " + cpu_ids);
  cpu_utils::ThreadLocalityManager::get_thread_locality_manager()
      ->set_thread_cpu_ids(omp_cpu_ids);

  // Memory node binding
  if (numa_available() != -1) {
    std::set<int> node_ids;
    for (const auto& cpu_id : omp_cpu_ids) {
      int node_id = numa_node_of_cpu(cpu_id);
      if (node_id != -1) {
        node_ids.insert(node_id);
      }
    }
    // Concatenate all node_ids into a single comma-separated string
    if (!node_ids.empty()) {
      std::string node_ids_str;
      for (const int node_id : node_ids) {
        if (!node_ids_str.empty()) {
          node_ids_str += ",";
        }
        node_ids_str += std::to_string(node_id);
      }

      bitmask* mask = numa_parse_nodestring(node_ids_str.c_str());
      bitmask* src_mask = numa_get_mems_allowed();

      int pid = getpid();

      if (mask && src_mask) {
        // move all existing pages to the specified numa node.
        *(src_mask->maskp) = *(src_mask->maskp) ^ *(mask->maskp);
        int page_num = numa_migrate_pages(pid, src_mask, mask);
        if (page_num == -1) {
          TORCH_WARN("numa_migrate_pages failed. errno: " +
                     std::to_string(errno));
        }

        // Restrict memory allocation to the selected NUMA node(s).
        // Enhances memory locality for the threads bound to those NUMA CPUs.
        if (node_ids.size() > 1) {
          errno = 0;
          numa_set_interleave_mask(mask);
          if (errno != 0) {
            TORCH_WARN("numa_set_interleave_mask failed. errno: " +
                       std::to_string(errno));
          } else {
            TORCH_WARN(
                "NUMA binding: Using INTERLEAVE policy for memory "
                "allocation across multiple NUMA nodes (nodes: " +
                node_ids_str +
                "). Memory allocations will be "
                "interleaved across the specified NUMA nodes.");
          }
        } else {
          errno = 0;
          numa_set_membind(mask);
          if (errno != 0) {
            TORCH_WARN("numa_set_membind failed. errno: " +
                       std::to_string(errno));
          } else {
            TORCH_WARN(
                "NUMA binding: Using MEMBIND policy for memory "
                "allocation on the NUMA nodes (" +
                node_ids_str +
                "). Memory allocations will be "
                "strictly bound to these NUMA nodes.");
          }
        }

        numa_set_strict(1);

        numa_free_nodemask(mask);
        numa_free_nodemask(src_mask);
      } else {
        TORCH_WARN(
            "numa_parse_nodestring or numa_get_run_node_mask failed. errno: " +
            std::to_string(errno));
      }
    }
  }

  // OMP threads binding
  omp_set_num_threads((int)omp_cpu_ids.size());
  torch::set_num_threads((int)omp_cpu_ids.size());
  TORCH_CHECK_EQ(omp_cpu_ids.size(), torch::get_num_threads());
  TORCH_CHECK_EQ(omp_cpu_ids.size(), omp_get_max_threads());

  std::vector<std::pair<int, int>> thread_core_mapping;
  thread_core_mapping.reserve(omp_cpu_ids.size());
  omp_lock_t writelock;
  omp_init_lock(&writelock);

  #pragma omp parallel for schedule(static, 1)
  for (size_t i = 0; i < omp_cpu_ids.size(); ++i) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(omp_cpu_ids[i], &mask);
    int ret = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
    if (ret == -1) {
      TORCH_CHECK(false,
                  "sched_setaffinity failed. errno: " + std::to_string(errno));
    }

    omp_set_lock(&writelock);
    thread_core_mapping.emplace_back(gettid(), omp_cpu_ids[i]);
    omp_unset_lock(&writelock);
  }

  omp_destroy_lock(&writelock);

  std::stringstream ss;
  ss << "OMP threads binding of Process " << getpid() << ":\n";
  std::sort(thread_core_mapping.begin(), thread_core_mapping.end(),
            [](auto&& a, auto&& b) { return a.second < b.second; });
  for (auto&& item : thread_core_mapping) {
    ss << "\t"
       << "OMP tid: " << item.first << ", core " << item.second << "\n";
  }

  return ss.str();
}
#endif  // VLLM_NUMA_DISABLED

namespace cpu_utils {
ThreadLocalityManager* ThreadLocalityManager::get_thread_locality_manager() {
  static ThreadLocalityManager manager;
  return &manager;
}

void ThreadLocalityManager::set_thread_cpu_ids(const std::vector<int>& omp_cpu_ids) {
  std::lock_guard<std::mutex> lock(mutex_);
  groups_ = build_locality_groups(omp_cpu_ids, true);
  thread_to_group_id_.assign(omp_cpu_ids.size(), 0);
  thread_to_local_offset_.assign(omp_cpu_ids.size(), 0);

  for (size_t group_id = 0; group_id < groups_.size(); ++group_id) {
    const auto& group = groups_[group_id];
    for (size_t local_offset = 0; local_offset < group.thread_ids.size();
         ++local_offset) {
      const int32_t thread_id = group.thread_ids[local_offset];
      if (thread_id >= 0 &&
          thread_id < static_cast<int32_t>(thread_to_group_id_.size())) {
        thread_to_group_id_[thread_id] = static_cast<int32_t>(group_id);
        thread_to_local_offset_[thread_id] = static_cast<int32_t>(local_offset);
      }
    }
  }
}

const std::vector<ThreadLocalityGroupInfo>& ThreadLocalityManager::get_groups()
    const {
  return groups_;
}

int32_t ThreadLocalityManager::get_group_id_for_thread(int32_t omp_thread_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (omp_thread_id < 0 ||
      omp_thread_id >= static_cast<int32_t>(thread_to_group_id_.size())) {
    return 0;
  }
  return thread_to_group_id_[omp_thread_id];
}

int32_t ThreadLocalityManager::get_local_offset_in_group(
    int32_t omp_thread_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (omp_thread_id < 0 ||
      omp_thread_id >= static_cast<int32_t>(thread_to_local_offset_.size())) {
    return omp_thread_id;
  }
  return thread_to_local_offset_[omp_thread_id];
}

std::string describe_cpu_locality_groups(const std::string& cpu_ids) {
  return locality_groups_to_json(build_locality_groups(parse_cpu_ids(cpu_ids), false));
}

ScratchPadManager::ScratchPadManager() : size_(0), ptr_(nullptr) {
  this->realloc(allocation_unit * 128);
}

void ScratchPadManager::realloc(size_t new_size) {
  new_size = round(new_size);
  if (new_size > size_) {
    if (ptr_ != nullptr) {
      std::free(ptr_);
    }
    ptr_ = std::aligned_alloc(64, new_size);
    size_ = new_size;
  }
}

ScratchPadManager* ScratchPadManager::get_scratchpad_manager() {
  static ScratchPadManager manager;
  return &manager;
}
}  // namespace cpu_utils
