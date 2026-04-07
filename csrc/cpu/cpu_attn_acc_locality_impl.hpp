#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <torch/library.h>

#include "cpu_attn_impl.hpp"
#include "cpu/utils.hpp"

at::Tensor get_scheduler_metadata(
    int64_t num_req, int64_t num_heads_q, int64_t num_heads_kv,
    int64_t head_dim, const at::Tensor& seq_lens, at::ScalarType dtype,
    const at::Tensor& query_start_loc, bool casual, int64_t window_size,
    const std::string& isa_hint, bool enable_kv_split);

namespace cpu_attention_acc_locality {

constexpr int32_t kAccLocalityMetadataMagic = 0x41434c33;  // "ACL3"
constexpr int32_t kMaxThreadNum = 1024;
constexpr int32_t kMaxKvHeadNum = 1024;

struct alignas(64) AttentionMetadata {
  int32_t magic;
  int32_t version;
  int32_t thread_num;
  int32_t subgroup_num;
  int32_t legacy_thread_num;
  int32_t legacy_effective_thread_num;
  int32_t actual_kv_head_num;
  int32_t reduction_item_num;
  int32_t attention_task_num;
  int32_t legacy_attention_task_num;
  int32_t reduction_task_num;
  int32_t max_subgroup_thread_num;
  int64_t legacy_metadata_offset;
  int64_t legacy_metadata_size;
  int32_t subgroup_thread_num[kMaxThreadNum];
  int32_t thread_to_group_id[kMaxThreadNum];
  int32_t thread_to_local_offset[kMaxThreadNum];
  int32_t kv_head_to_subgroup[kMaxKvHeadNum];

  AttentionMetadata()
      : magic(kAccLocalityMetadataMagic),
        version(1),
        thread_num(0),
        subgroup_num(0),
        legacy_thread_num(0),
        legacy_effective_thread_num(0),
        actual_kv_head_num(0),
        reduction_item_num(0),
        attention_task_num(0),
        legacy_attention_task_num(0),
        reduction_task_num(0),
        max_subgroup_thread_num(0),
        legacy_metadata_offset(0),
        legacy_metadata_size(0) {
    std::fill_n(subgroup_thread_num, kMaxThreadNum, 0);
    std::fill_n(thread_to_group_id, kMaxThreadNum, 0);
    std::fill_n(thread_to_local_offset, kMaxThreadNum, 0);
    std::fill_n(kv_head_to_subgroup, kMaxKvHeadNum, 0);
  }

  cpu_attention::AttentionMetadata* legacy_metadata() {
    return reinterpret_cast<cpu_attention::AttentionMetadata*>(
        reinterpret_cast<char*>(this) + legacy_metadata_offset);
  }

  const cpu_attention::AttentionMetadata* legacy_metadata() const {
    return reinterpret_cast<const cpu_attention::AttentionMetadata*>(
        reinterpret_cast<const char*>(this) + legacy_metadata_offset);
  }

  std::string to_json() const {
    std::stringstream ss;
    ss << '{';
    ss << "\"thread_num\":" << thread_num << ',';
    ss << "\"subgroup_num\":" << subgroup_num << ',';
    ss << "\"legacy_thread_num\":" << legacy_thread_num << ',';
    ss << "\"legacy_effective_thread_num\":" << legacy_effective_thread_num
       << ',';
    ss << "\"actual_kv_head_num\":" << actual_kv_head_num << ',';
    ss << "\"attention_task_num\":" << attention_task_num << ',';
    ss << "\"legacy_attention_task_num\":" << legacy_attention_task_num << ',';
    ss << "\"reduction_task_num\":" << reduction_task_num << ',';
    ss << "\"subgroup_thread_num\":[";
    for (int32_t i = 0; i < subgroup_num; ++i) {
      if (i > 0) {
        ss << ',';
      }
      ss << subgroup_thread_num[i];
    }
    ss << "],";
    ss << "\"kv_head_to_subgroup\":[";
    for (int32_t i = 0; i < actual_kv_head_num; ++i) {
      if (i > 0) {
        ss << ',';
      }
      ss << kv_head_to_subgroup[i];
    }
    ss << "]}";
    return ss.str();
  }
};

struct AttentionInput {
  AttentionMetadata* metadata;
  int32_t num_tokens;
  int32_t num_heads;
  int32_t num_kv_heads;
  int32_t block_size;
  void* query;
  int64_t query_num_tokens_stride;
  int64_t query_num_heads_stride;
  int64_t cache_num_blocks_stride;
  int64_t cache_num_kv_heads_stride;
  int64_t blt_num_tokens_stride;
  void* key_cache;
  void* value_cache;
  void* output;
  int32_t* query_start_loc;
  int32_t* seq_lens;
  int32_t* block_table;
  float* alibi_slopes;
  c10::BFloat16* s_aux;
  float scale;
  bool causal;
  int32_t sliding_window_left;
  int32_t sliding_window_right;
  float softcap;
};

inline int32_t get_actual_kv_head_num(int32_t num_heads_q, int32_t num_heads_kv,
                                      int32_t max_num_q_per_iter) {
  const int32_t q_heads_per_kv = num_heads_q / num_heads_kv;
  const bool use_gqa = (max_num_q_per_iter % q_heads_per_kv == 0);
  return use_gqa ? num_heads_kv : num_heads_q;
}

inline void init_default_locality_mapping(AttentionMetadata* metadata,
                                          int32_t thread_num) {
  metadata->subgroup_num = 1;
  metadata->max_subgroup_thread_num = thread_num;
  metadata->subgroup_thread_num[0] = thread_num;
  for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
    metadata->thread_to_group_id[thread_id] = 0;
    metadata->thread_to_local_offset[thread_id] = thread_id;
  }
}

inline void init_thread_locality_mapping(AttentionMetadata* metadata,
                                         int32_t thread_num) {
  const auto& groups = cpu_utils::ThreadLocalityManager::get_thread_locality_manager()
                           ->get_groups();
  bool valid = !groups.empty();
  int32_t subgroup_id = 0;
  int32_t total_assigned_thread_num = 0;
  int32_t max_subgroup_thread_num = 0;

  if (valid) {
    for (const auto& group : groups) {
      if (group.thread_ids.empty()) {
        continue;
      }
      TORCH_CHECK_LE(group.thread_ids.size(), static_cast<size_t>(thread_num));
      metadata->subgroup_thread_num[subgroup_id] =
          static_cast<int32_t>(group.thread_ids.size());
      max_subgroup_thread_num = std::max(
          max_subgroup_thread_num, metadata->subgroup_thread_num[subgroup_id]);
      for (size_t local_offset = 0; local_offset < group.thread_ids.size();
           ++local_offset) {
        const int32_t thread_id = group.thread_ids[local_offset];
        if (thread_id < 0 || thread_id >= thread_num) {
          valid = false;
          break;
        }
        metadata->thread_to_group_id[thread_id] = subgroup_id;
        metadata->thread_to_local_offset[thread_id] =
            static_cast<int32_t>(local_offset);
        ++total_assigned_thread_num;
      }
      if (!valid) {
        break;
      }
      ++subgroup_id;
    }
  }

  if (!valid || subgroup_id == 0 || total_assigned_thread_num != thread_num) {
    init_default_locality_mapping(metadata, thread_num);
    return;
  }

  metadata->subgroup_num = subgroup_id;
  metadata->max_subgroup_thread_num = max_subgroup_thread_num;
}

inline at::Tensor build_scheduler_metadata(
    int64_t num_req, int64_t num_heads_q, int64_t num_heads_kv,
    int64_t head_dim, const at::Tensor& seq_lens, at::ScalarType dtype,
    const at::Tensor& query_start_loc, bool casual, int64_t window_size,
    const std::string& isa_hint, bool enable_kv_split,
    int32_t max_num_q_per_iter) {
  TORCH_CHECK_LE(omp_get_max_threads(), kMaxThreadNum);
  const int32_t thread_num = omp_get_max_threads();
  const int32_t actual_kv_head_num = get_actual_kv_head_num(
      static_cast<int32_t>(num_heads_q), static_cast<int32_t>(num_heads_kv),
      max_num_q_per_iter);
  TORCH_CHECK_LE(actual_kv_head_num, kMaxKvHeadNum);

  at::Tensor legacy_metadata_tensor =
      ::get_scheduler_metadata(num_req, num_heads_q, num_heads_kv, head_dim,
                               seq_lens, dtype, query_start_loc, casual,
                               window_size, isa_hint, enable_kv_split);
  const int64_t legacy_metadata_size =
      static_cast<int64_t>(legacy_metadata_tensor.numel());
  const int64_t legacy_metadata_offset =
      cpu_utils::round_up<64>(static_cast<int64_t>(sizeof(AttentionMetadata)));
  const int64_t total_metadata_size =
      legacy_metadata_offset + legacy_metadata_size;

  auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
  at::Tensor metadata_tensor = torch::empty({total_metadata_size}, options);
  AttentionMetadata* metadata = new (metadata_tensor.data_ptr()) AttentionMetadata();
  metadata->thread_num = thread_num;
  metadata->actual_kv_head_num = actual_kv_head_num;
  metadata->legacy_metadata_offset = legacy_metadata_offset;
  metadata->legacy_metadata_size = legacy_metadata_size;
  init_thread_locality_mapping(metadata, thread_num);

  std::memcpy(reinterpret_cast<char*>(metadata) + legacy_metadata_offset,
              legacy_metadata_tensor.data_ptr(), legacy_metadata_size);

  cpu_attention::AttentionMetadata* legacy_metadata = metadata->legacy_metadata();
  legacy_metadata->workitem_groups_ptr =
      reinterpret_cast<cpu_attention::AttentionWorkItemGroup*>(
          reinterpret_cast<char*>(legacy_metadata) +
          sizeof(cpu_attention::AttentionMetadata));
  legacy_metadata->reduction_items_ptr =
      reinterpret_cast<cpu_attention::ReductionWorkItemGroup*>(
          reinterpret_cast<char*>(legacy_metadata) +
          sizeof(cpu_attention::AttentionMetadata) +
          legacy_metadata->workitem_group_num *
              sizeof(cpu_attention::AttentionWorkItemGroup));
  legacy_metadata->reset_counter();

  metadata->legacy_thread_num = legacy_metadata->thread_num;
  metadata->legacy_effective_thread_num = legacy_metadata->effective_thread_num;
  metadata->reduction_item_num = legacy_metadata->reduction_item_num;
  metadata->legacy_attention_task_num =
      actual_kv_head_num * legacy_metadata->effective_thread_num;

  for (int32_t kv_head_idx = 0; kv_head_idx < actual_kv_head_num;
       ++kv_head_idx) {
    const int32_t subgroup_id =
        metadata->subgroup_num > 0 ? (kv_head_idx % metadata->subgroup_num) : 0;
    metadata->kv_head_to_subgroup[kv_head_idx] = subgroup_id;
    metadata->attention_task_num +=
        std::min(metadata->subgroup_thread_num[subgroup_id],
                 legacy_metadata->effective_thread_num);
    metadata->reduction_task_num +=
        std::min(metadata->subgroup_thread_num[subgroup_id],
                 legacy_metadata->reduction_item_num);
  }

  return metadata_tensor;
}

inline std::string inspect_scheduler_metadata(const at::Tensor& scheduler_metadata) {
  const AttentionMetadata* metadata =
      reinterpret_cast<const AttentionMetadata*>(scheduler_metadata.data_ptr());
  TORCH_CHECK_EQ(metadata->magic, kAccLocalityMetadataMagic);
  return metadata->to_json();
}

inline bool should_log_runtime_summary() {
  const char* env = std::getenv("VLLM_CPU_ATTN_ACC_LOCALITY_DEBUG");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline std::string format_int_list(const std::vector<int32_t>& values) {
  std::stringstream ss;
  ss << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      ss << ',';
    }
    ss << values[i];
  }
  ss << ']';
  return ss.str();
}

inline std::string build_runtime_summary(const AttentionMetadata& metadata) {
  std::stringstream ss;
  ss << "CPU attention acc-locality runtime summary\n";
  ss << "  thread_num=" << metadata.thread_num
     << ", subgroup_num=" << metadata.subgroup_num
     << ", legacy_effective_thread_num=" << metadata.legacy_effective_thread_num
     << ", actual_kv_head_num=" << metadata.actual_kv_head_num
     << ", attention_task_num=" << metadata.attention_task_num
     << ", legacy_attention_task_num=" << metadata.legacy_attention_task_num
     << ", reduction_item_num=" << metadata.reduction_item_num << '\n';

  for (int32_t subgroup_id = 0; subgroup_id < metadata.subgroup_num; ++subgroup_id) {
    std::vector<int32_t> thread_ids;
    std::vector<int32_t> kv_heads;
    std::vector<int32_t> legacy_slots;
    const int32_t subgroup_thread_num = metadata.subgroup_thread_num[subgroup_id];

    for (int32_t thread_id = 0; thread_id < metadata.thread_num; ++thread_id) {
      if (metadata.thread_to_group_id[thread_id] == subgroup_id) {
        thread_ids.push_back(thread_id);
      }
    }
    std::sort(thread_ids.begin(), thread_ids.end(),
              [&](int32_t lhs, int32_t rhs) {
                return metadata.thread_to_local_offset[lhs] <
                       metadata.thread_to_local_offset[rhs];
              });

    for (int32_t kv_head_idx = 0; kv_head_idx < metadata.actual_kv_head_num;
         ++kv_head_idx) {
      if (metadata.kv_head_to_subgroup[kv_head_idx] == subgroup_id) {
        kv_heads.push_back(kv_head_idx);
      }
    }

    if (subgroup_thread_num > 0) {
      for (int32_t local_offset = 0; local_offset < subgroup_thread_num; ++local_offset) {
        for (int32_t legacy_slot = local_offset;
             legacy_slot < metadata.legacy_effective_thread_num;
             legacy_slot += subgroup_thread_num) {
          legacy_slots.push_back(legacy_slot);
        }
      }
    }

    ss << "  subgroup " << subgroup_id
       << ": threads=" << format_int_list(thread_ids)
       << ", kv_heads=" << format_int_list(kv_heads)
       << ", legacy_slots=" << format_int_list(legacy_slots) << '\n';

    for (int32_t thread_id : thread_ids) {
      std::vector<int32_t> thread_legacy_slots;
      const int32_t local_offset = metadata.thread_to_local_offset[thread_id];
      for (int32_t legacy_slot = local_offset;
           legacy_slot < metadata.legacy_effective_thread_num;
           legacy_slot += subgroup_thread_num) {
        thread_legacy_slots.push_back(legacy_slot);
      }
      ss << "    thread " << thread_id
         << ": local_offset=" << local_offset
         << ", legacy_slots=" << format_int_list(thread_legacy_slots) << '\n';
    }
  }

  return ss.str();
}

template <typename attention_impl_t>
class AttentionMainLoop
    : public cpu_attention::AttentionMainLoop<attention_impl_t> {
 public:
  using Base = cpu_attention::AttentionMainLoop<attention_impl_t>;
  using query_t = typename Base::query_t;
  using q_buffer_t = typename Base::q_buffer_t;
  using kv_cache_t = typename Base::kv_cache_t;
  using logits_buffer_t = typename Base::logits_buffer_t;
  using partial_output_buffer_t = typename Base::partial_output_buffer_t;

  template <typename tile_gemm_t>
  using Attention = typename Base::template Attention<tile_gemm_t>;

  static constexpr int64_t max_q_head_num_per_iter =
      attention_impl_t::MaxQHeadNumPerIteration;
  static constexpr int64_t blocksize_alignment =
      attention_impl_t::BlockSizeAlignment;
  static constexpr int64_t head_dim = attention_impl_t::HeadDim;

  void operator()(const AttentionInput* input) {
    const int32_t thread_num = omp_get_max_threads();
    AttentionMetadata& metadata = *input->metadata;
    TORCH_CHECK_EQ(metadata.thread_num, thread_num);
    cpu_attention::AttentionMetadata& legacy_metadata = *metadata.legacy_metadata();
    if (should_log_runtime_summary()) {
      const std::string summary = build_runtime_summary(metadata);
      std::printf("%s", summary.c_str());
      std::fflush(stdout);
    }
    std::atomic<int32_t> guard_counter(0);
    std::atomic<int32_t>* guard_counter_ptr = &guard_counter;

#pragma omp parallel for schedule(static, 1)
    for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
      if (legacy_metadata.workitem_group_num == 0) {
        continue;
      }

      attention_impl_t attn_impl;

      const int32_t q_head_num = input->num_heads;
      const int32_t kv_head_num = input->num_kv_heads;
      const int32_t q_heads_per_kv = q_head_num / kv_head_num;
      const bool use_gqa =
          (max_q_head_num_per_iter % q_heads_per_kv == 0) ? true : false;
      const int32_t actual_kv_head_num = use_gqa ? kv_head_num : q_head_num;
      const int32_t actual_q_heads_per_kv = use_gqa ? q_heads_per_kv : 1;
      TORCH_CHECK_EQ(actual_kv_head_num, metadata.actual_kv_head_num);
      TORCH_CHECK_LE(actual_q_heads_per_kv, max_q_head_num_per_iter);
      const int32_t max_q_token_num_per_iter =
          max_q_head_num_per_iter / actual_q_heads_per_kv;
      const int64_t q_token_num_stride = input->query_num_tokens_stride;
      const int64_t q_head_num_stride = input->query_num_heads_stride;
      const int64_t kv_cache_head_num_stride = input->cache_num_kv_heads_stride;
      const int64_t kv_cache_block_num_stride = input->cache_num_blocks_stride;
      const int32_t sliding_window_left = input->sliding_window_left;
      const int32_t sliding_window_right = input->sliding_window_right;
      const int32_t block_size = input->block_size;
      const float scale = input->scale;
      const float softcap_scale = input->softcap;
      const float* alibi_slopes = input->alibi_slopes;
      const c10::BFloat16* s_aux = input->s_aux;
      const bool casual = input->causal;
      int32_t* const block_table = input->block_table;
      const int64_t block_table_stride = input->blt_num_tokens_stride;

      void* scratchpad_ptr =
          cpu_utils::ScratchPadManager::get_scratchpad_manager()
              ->get_data<void>();
      cpu_attention::AttentionScratchPad buffer_manager(thread_id, legacy_metadata,
                                                       scratchpad_ptr);

      const int32_t total_reduction_split_num = legacy_metadata.reduction_split_num;
      if (total_reduction_split_num > 0) {
        for (int32_t head_idx = thread_id; head_idx < actual_kv_head_num;
             head_idx += thread_num) {
          buffer_manager.update(head_idx, total_reduction_split_num, head_dim, 0,
                                sizeof(partial_output_buffer_t));
          volatile bool* curr_flag_ptr = buffer_manager.get_reduce_flag_buffer();
          for (int32_t split_idx = 0; split_idx < total_reduction_split_num;
               ++split_idx) {
            curr_flag_ptr[split_idx] = false;
          }
        }
      }

      const int64_t available_cache_size = cpu_utils::get_available_l2_size();
      const int32_t default_tile_size =
          cpu_attention::AttentionScheduler::calcu_default_tile_size(
              available_cache_size, head_dim, sizeof(kv_cache_t),
              sizeof(q_buffer_t), sizeof(logits_buffer_t),
              sizeof(partial_output_buffer_t), max_q_head_num_per_iter,
              max_q_head_num_per_iter);
      const int32_t default_q_tile_token_num =
          default_tile_size / actual_q_heads_per_kv;

      cpu_attention::AttentionWorkItemGroup* const workitem_groups =
          legacy_metadata.workitem_groups_ptr;
      const int32_t* cu_workitem_num_per_thread =
          legacy_metadata.cu_workitem_num_per_thread;
      cpu_attention::ReductionWorkItemGroup* const reduction_items =
          legacy_metadata.reduction_items_ptr;
      const int32_t effective_thread_num = legacy_metadata.effective_thread_num;
      const int32_t reduction_item_num = legacy_metadata.reduction_item_num;
      const int32_t split_kv_q_token_num_threshold =
          legacy_metadata.split_kv_q_token_num_threshold;

      if (total_reduction_split_num > 0) {
        ++(*guard_counter_ptr);
        while (guard_counter_ptr->load() != thread_num) {
#ifdef FAST_SPINNING
          FAST_SPINNING
#else
          std::this_thread::yield();
#endif
        }
      }

      const int32_t subgroup_id = metadata.thread_to_group_id[thread_id];
      const int32_t local_offset = metadata.thread_to_local_offset[thread_id];
      if (subgroup_id < 0 || subgroup_id >= metadata.subgroup_num) {
        continue;
      }
      const int32_t subgroup_thread_num = metadata.subgroup_thread_num[subgroup_id];
      if (subgroup_thread_num <= 0) {
        continue;
      }

      auto run_attention_for_legacy_thread = [&](int32_t kv_head_idx,
                                                 int32_t legacy_thread_offset) {
        cpu_attention::AttentionWorkItemGroup* const curr_workitem_groups =
            workitem_groups + cu_workitem_num_per_thread[legacy_thread_offset];
        const int32_t curr_workitem_groups_num =
            cu_workitem_num_per_thread[legacy_thread_offset + 1] -
            cu_workitem_num_per_thread[legacy_thread_offset];
        const int32_t q_head_start_idx = kv_head_idx * actual_q_heads_per_kv;

        for (int32_t workitem_group_idx = 0;
             workitem_group_idx < curr_workitem_groups_num;
             ++workitem_group_idx) {
          cpu_attention::AttentionWorkItemGroup* const current_workitem_group =
              &curr_workitem_groups[workitem_group_idx];

          const int32_t current_group_idx = current_workitem_group->req_id;
          const int32_t kv_start_pos = current_workitem_group->kv_split_pos_start;
          const int32_t kv_end_pos = current_workitem_group->kv_split_pos_end;
          const int32_t curr_split_id = current_workitem_group->split_id;
          const int32_t q_token_id_start = current_workitem_group->q_token_id_start;
          const int32_t q_token_num = current_workitem_group->q_token_num;

          const int32_t q_end = input->query_start_loc[current_group_idx + 1];
          const int32_t q_start = input->query_start_loc[current_group_idx];
          const int32_t seq_len = input->seq_lens[current_group_idx];
          const int32_t q_start_pos =
              (casual ? seq_len - (q_end - q_start) : 0);
          bool use_sink = (s_aux != nullptr &&
                           current_workitem_group->local_split_id == 0);

          for (int32_t q_token_offset = 0; q_token_offset < q_token_num;
               q_token_offset += default_q_tile_token_num) {
            bool first_iter_flag[cpu_attention::AttentionScheduler::MaxQTileIterNum];
            for (int32_t i = 0;
                 i < cpu_attention::AttentionScheduler::MaxQTileIterNum; ++i) {
              first_iter_flag[i] = true;
            }

            const int32_t q_token_start_idx =
                q_start + q_token_offset + q_token_id_start;
            const int32_t actual_q_token_num = std::min(
                default_q_tile_token_num, q_token_num - q_token_offset);
            const int32_t q_head_tile_size =
                actual_q_token_num * actual_q_heads_per_kv;
            const int32_t rounded_q_head_tile_size =
                ((q_head_tile_size + max_q_head_num_per_iter - 1) /
                 max_q_head_num_per_iter) *
                max_q_head_num_per_iter;
            const int32_t kv_tile_size =
                cpu_attention::AttentionScheduler::calcu_tile_size_with_constant_q(
                    available_cache_size, head_dim, sizeof(kv_cache_t),
                    sizeof(q_buffer_t), sizeof(logits_buffer_t),
                    sizeof(partial_output_buffer_t), max_q_head_num_per_iter,
                    blocksize_alignment, rounded_q_head_tile_size,
                    rounded_q_head_tile_size <= max_q_head_num_per_iter);

            buffer_manager.update(
                head_dim, sizeof(q_buffer_t), sizeof(logits_buffer_t),
                sizeof(partial_output_buffer_t), max_q_head_num_per_iter,
                rounded_q_head_tile_size, kv_tile_size);
            q_buffer_t* q_buffer = buffer_manager.get_q_buffer<q_buffer_t>();
            float* logits_buffer = buffer_manager.get_logits_buffer();
            float* partial_q_buffer = buffer_manager.get_output_buffer();
            float* max_buffer = buffer_manager.get_max_buffer();
            float* sum_buffer = buffer_manager.get_sum_buffer();

            const int32_t q_tile_start_pos =
                q_start_pos + q_token_offset + q_token_id_start;
            const int32_t q_tile_end_pos =
                q_tile_start_pos + actual_q_token_num;
            const auto [kv_tile_start_pos, kv_tile_end_pos] =
                cpu_attention::AttentionScheduler::calcu_kv_tile_pos(
                    kv_start_pos, kv_end_pos, q_tile_start_pos, q_tile_end_pos,
                    sliding_window_left, sliding_window_right);
            const auto [rounded_kv_tile_start_pos, rounded_kv_tile_end_pos] =
                cpu_attention::AttentionScheduler::align_kv_tile_pos(
                    kv_tile_start_pos, kv_tile_end_pos, blocksize_alignment);

            const int32_t curr_kv_head_idx =
                use_gqa ? kv_head_idx : (kv_head_idx / q_heads_per_kv);

            kv_cache_t* curr_k_cache =
                reinterpret_cast<kv_cache_t*>(input->key_cache) +
                curr_kv_head_idx * kv_cache_head_num_stride;
            kv_cache_t* curr_v_cache =
                reinterpret_cast<kv_cache_t*>(input->value_cache) +
                curr_kv_head_idx * kv_cache_head_num_stride;
            query_t* const q_tile_ptr =
                reinterpret_cast<query_t*>(input->query) +
                q_token_start_idx * q_token_num_stride +
                q_head_start_idx * q_head_num_stride;
            size_t output_buffer_offset =
                q_token_start_idx * q_head_num * head_dim +
                q_head_start_idx * head_dim;
            int32_t* curr_block_table =
                block_table + current_group_idx * block_table_stride;
            const float* curr_alibi_slopes =
                (alibi_slopes != nullptr ? alibi_slopes + q_head_start_idx
                                         : nullptr);
            const c10::BFloat16* curr_s_aux =
                (s_aux != nullptr ? s_aux + q_head_start_idx : nullptr);

            attn_impl.copy_q_heads_tile(q_tile_ptr, q_buffer, actual_q_token_num,
                                        actual_q_heads_per_kv, q_token_num_stride,
                                        q_head_num_stride, scale);

            if (use_sink) {
              alignas(64) float s_aux_fp32[16];
#if defined(__aarch64__) && !defined(ARM_BF16_SUPPORT)
              for (int i = 0; i < 16; ++i) {
                s_aux_fp32[i] = static_cast<float>(curr_s_aux[i]);
              }
#else
              vec_op::BF16Vec16 vec_bf16(curr_s_aux);
              vec_op::FP32Vec16 vec_fp32(vec_bf16);
              vec_fp32.save(s_aux_fp32);
#endif

              float* curr_sum_buffer = sum_buffer;
              float* curr_max_buffer = max_buffer;
              for (int32_t token_idx = 0; token_idx < actual_q_token_num;
                   ++token_idx) {
                for (int32_t head_idx = 0; head_idx < actual_q_heads_per_kv;
                     ++head_idx) {
                  curr_sum_buffer[head_idx] = 1.0f;
                  curr_max_buffer[head_idx] = s_aux_fp32[head_idx];
                }
                curr_sum_buffer += actual_q_heads_per_kv;
                curr_max_buffer += actual_q_heads_per_kv;
              }
            } else {
              float* curr_sum_buffer = sum_buffer;
              float* curr_max_buffer = max_buffer;
              for (int32_t token_idx = 0; token_idx < actual_q_token_num;
                   ++token_idx) {
                for (int32_t head_idx = 0; head_idx < actual_q_heads_per_kv;
                     ++head_idx) {
                  curr_sum_buffer[head_idx] = 0.0f;
                  curr_max_buffer[head_idx] =
                      std::numeric_limits<float>::lowest();
                }
                curr_sum_buffer += actual_q_heads_per_kv;
                curr_max_buffer += actual_q_heads_per_kv;
              }
            }

            for (int32_t kv_tile_pos = rounded_kv_tile_start_pos;
                 kv_tile_pos < rounded_kv_tile_end_pos;
                 kv_tile_pos += kv_tile_size) {
              const int32_t kv_tile_pos_left = kv_tile_pos;
              const int32_t kv_tile_pos_right =
                  std::min(kv_tile_pos_left + kv_tile_size,
                           rounded_kv_tile_end_pos);
              for (int32_t q_head_tile_token_offset = 0;
                   q_head_tile_token_offset < actual_q_token_num;
                   q_head_tile_token_offset += max_q_token_num_per_iter) {
                const int32_t q_tile_pos_left =
                    q_tile_start_pos + q_head_tile_token_offset;
                const int32_t q_tile_token_num =
                    std::min(max_q_token_num_per_iter,
                             actual_q_token_num - q_head_tile_token_offset);
                const int32_t q_tile_head_offset =
                    q_head_tile_token_offset * actual_q_heads_per_kv;
                const int32_t q_tile_head_num =
                    q_tile_token_num * actual_q_heads_per_kv;
                const int32_t q_tile_pos_right =
                    q_tile_pos_left + q_tile_token_num;
                const auto [actual_kv_tile_pos_left, actual_kv_tile_pos_right] =
                    cpu_attention::AttentionScheduler::calcu_kv_tile_pos(
                        kv_tile_pos_left, kv_tile_pos_right, q_tile_pos_left,
                        q_tile_pos_right, sliding_window_left,
                        sliding_window_right);
                const int32_t q_iter_idx =
                    q_head_tile_token_offset / max_q_token_num_per_iter;

                if (actual_kv_tile_pos_right <= actual_kv_tile_pos_left) {
                  continue;
                }

                const auto [aligned_actual_kv_tile_pos_left,
                            aligned_actual_kv_tile_pos_right] =
                    cpu_attention::AttentionScheduler::align_kv_tile_pos(
                        actual_kv_tile_pos_left, actual_kv_tile_pos_right,
                        blocksize_alignment);
                const int32_t actual_kv_token_num =
                    aligned_actual_kv_tile_pos_right -
                    aligned_actual_kv_tile_pos_left;

                q_buffer_t* curr_q_heads_buffer =
                    q_buffer + q_tile_head_offset * head_dim;
                float* curr_partial_q_buffer =
                    partial_q_buffer + q_tile_head_offset * head_dim;
                float* curr_max_buffer = max_buffer + q_tile_head_offset;
                float* curr_sum_buffer = sum_buffer + q_tile_head_offset;
                constexpr bool debug_info = false;

                attn_impl.template execute_attention<Attention>(
                    curr_q_heads_buffer, curr_k_cache, curr_v_cache,
                    logits_buffer, curr_partial_q_buffer, curr_max_buffer,
                    curr_sum_buffer, curr_block_table,
                    aligned_actual_kv_tile_pos_left,
                    aligned_actual_kv_tile_pos_right, actual_kv_token_num,
                    kv_cache_block_num_stride, q_tile_head_num, q_tile_token_num,
                    q_tile_pos_left, actual_q_heads_per_kv, block_size,
                    sliding_window_left, sliding_window_right, scale,
                    softcap_scale, curr_alibi_slopes,
                    first_iter_flag[q_iter_idx], use_sink, debug_info);
                first_iter_flag[q_iter_idx] = false;
              }
            }

            if (curr_split_id == -1) {
              this->final_output(partial_q_buffer,
                                 reinterpret_cast<query_t*>(input->output) +
                                     output_buffer_offset,
                                 sum_buffer, actual_q_heads_per_kv,
                                 actual_q_token_num, q_head_num);
            } else {
              const int32_t stride =
                  actual_q_heads_per_kv * split_kv_q_token_num_threshold;
              buffer_manager.update(kv_head_idx, total_reduction_split_num,
                                    head_dim, stride, sizeof(float));
              volatile bool* split_flag_buffer =
                  buffer_manager.get_reduce_flag_buffer() + curr_split_id;
              float* split_output_buffer =
                  buffer_manager.get_reduce_output_buffer() +
                  curr_split_id * stride * head_dim;
              float* split_max_buffer =
                  buffer_manager.get_reduce_max_buffer() + curr_split_id * stride;
              float* split_sum_buffer =
                  buffer_manager.get_reduce_sum_buffer() + curr_split_id * stride;

              this->partial_output(partial_q_buffer, max_buffer, sum_buffer,
                                   q_head_tile_size, split_output_buffer,
                                   split_max_buffer, split_sum_buffer,
                                   split_flag_buffer);
            }
          }
        }
      };

      auto run_reduction_for_item = [&](int32_t kv_head_idx, int32_t item_offset) {
        cpu_attention::ReductionWorkItemGroup* const curr_workitem_group =
            reduction_items + item_offset;
        const int32_t curr_output_token_idx =
            curr_workitem_group->q_token_id_start;
        const int32_t curr_output_token_num =
            curr_workitem_group->q_token_id_num;
        const int32_t curr_split_id = curr_workitem_group->split_start_id;
        const int32_t curr_split_num = curr_workitem_group->split_num;
        const int32_t current_group_idx = curr_workitem_group->req_id;
        const int32_t curr_output_head_num =
            curr_output_token_num * actual_q_heads_per_kv;

        const int32_t q_start = input->query_start_loc[current_group_idx];
        const int32_t q_token_start_idx = q_start + curr_output_token_idx;
        const int32_t q_head_start_idx = kv_head_idx * actual_q_heads_per_kv;
        size_t output_buffer_offset =
            q_token_start_idx * q_head_num * head_dim + q_head_start_idx * head_dim;

        const int32_t stride =
            actual_q_heads_per_kv * split_kv_q_token_num_threshold;
        buffer_manager.update(kv_head_idx, total_reduction_split_num, head_dim,
                              stride, sizeof(float));
        volatile bool* split_flag_buffer =
            buffer_manager.get_reduce_flag_buffer() + curr_split_id;
        float* split_output_buffer =
            buffer_manager.get_reduce_output_buffer() +
            curr_split_id * stride * head_dim;
        float* split_max_buffer =
            buffer_manager.get_reduce_max_buffer() + curr_split_id * stride;
        float* split_sum_buffer =
            buffer_manager.get_reduce_sum_buffer() + curr_split_id * stride;

        this->reduce_splits(split_output_buffer, split_max_buffer, split_sum_buffer,
                            split_flag_buffer, stride, curr_output_head_num,
                            curr_split_num);
        this->final_output(
            split_output_buffer,
            reinterpret_cast<query_t*>(input->output) + output_buffer_offset,
            split_sum_buffer, actual_q_heads_per_kv, curr_output_token_num,
            q_head_num);
      };

      for (int32_t kv_head_idx = 0; kv_head_idx < actual_kv_head_num;
           ++kv_head_idx) {
        if (metadata.kv_head_to_subgroup[kv_head_idx] != subgroup_id) {
          continue;
        }
        for (int32_t legacy_thread_offset = local_offset;
             legacy_thread_offset < effective_thread_num;
             legacy_thread_offset += subgroup_thread_num) {
          run_attention_for_legacy_thread(kv_head_idx, legacy_thread_offset);
        }
      }

      for (int32_t kv_head_idx = 0; kv_head_idx < actual_kv_head_num;
           ++kv_head_idx) {
        if (metadata.kv_head_to_subgroup[kv_head_idx] != subgroup_id) {
          continue;
        }
        for (int32_t item_offset = local_offset; item_offset < reduction_item_num;
             item_offset += subgroup_thread_num) {
          run_reduction_for_item(kv_head_idx, item_offset);
        }
      }
    }
  }
};

}  // namespace cpu_attention_acc_locality
