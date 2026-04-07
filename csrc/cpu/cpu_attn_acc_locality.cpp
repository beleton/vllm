#include "cpu_attn_acc_locality_impl.hpp"

#include "cpu_attn_vec.hpp"
#include "cpu_attn_vec16.hpp"

#ifdef CPU_CAPABILITY_AMXBF16
  #include "cpu_attn_amx.hpp"
  #define AMX_DISPATCH(...)                                                   \
    case cpu_attention::ISA::AMX: {                                           \
      using attn_impl = cpu_attention::AttentionImpl<cpu_attention::ISA::AMX, \
                                                     scalar_t, head_dim>;     \
      return __VA_ARGS__();                                                   \
    }
#else
  #define AMX_DISPATCH(...) case cpu_attention::ISA::AMX:
#endif

#ifdef __aarch64__
  #include "cpu_attn_neon.hpp"
  #define NEON_DISPATCH(...)                                                   \
    case cpu_attention::ISA::NEON: {                                           \
      using attn_impl = cpu_attention::AttentionImpl<cpu_attention::ISA::NEON, \
                                                     scalar_t, head_dim>;      \
      return __VA_ARGS__();                                                    \
    }
#else
  #define NEON_DISPATCH(...) case cpu_attention::ISA::NEON:
#endif

#define CPU_ATTN_DISPATCH_CASE(HEAD_DIM, ...) \
  case HEAD_DIM: {                            \
    constexpr size_t head_dim = HEAD_DIM;     \
    return __VA_ARGS__();                     \
  }

#define CPU_ATTN_DISPATCH_CASE_HEADDIM(HEAD_DIM, ...)           \
  [&] {                                                         \
    switch (HEAD_DIM) {                                         \
      CPU_ATTN_DISPATCH_CASE(32, __VA_ARGS__)                   \
      CPU_ATTN_DISPATCH_CASE(64, __VA_ARGS__)                   \
      CPU_ATTN_DISPATCH_CASE(80, __VA_ARGS__)                   \
      CPU_ATTN_DISPATCH_CASE(96, __VA_ARGS__)                   \
      CPU_ATTN_DISPATCH_CASE(112, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(128, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(160, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(192, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(224, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(256, __VA_ARGS__)                  \
      default: {                                                \
        TORCH_CHECK(false, "Invalid CPU attention head_dim: " + \
                               std::to_string(HEAD_DIM));       \
      }                                                         \
    }                                                           \
  }()

#define CPU_ATTN_DISPATCH_IMPL(ISA_TYPE, ...)                                 \
  [&] {                                                                       \
    switch (ISA_TYPE) {                                                       \
      AMX_DISPATCH(__VA_ARGS__)                                               \
      NEON_DISPATCH(__VA_ARGS__)                                              \
      case cpu_attention::ISA::VEC: {                                         \
        using attn_impl =                                                     \
            cpu_attention::AttentionImpl<cpu_attention::ISA::VEC, scalar_t,   \
                                         head_dim>;                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case cpu_attention::ISA::VEC16: {                                       \
        using attn_impl =                                                     \
            cpu_attention::AttentionImpl<cpu_attention::ISA::VEC16, scalar_t, \
                                         head_dim>;                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default: {                                                              \
        TORCH_CHECK(false, "Invalid CPU attention ISA type.");                \
      }                                                                       \
    }                                                                         \
  }()

namespace {

cpu_attention::ISA parse_cpu_attention_isa(const std::string& isa_hint) {
  if (isa_hint == "amx") {
    return cpu_attention::ISA::AMX;
  } else if (isa_hint == "vec") {
    return cpu_attention::ISA::VEC;
  } else if (isa_hint == "vec16") {
    return cpu_attention::ISA::VEC16;
  } else if (isa_hint == "neon") {
    return cpu_attention::ISA::NEON;
  }
  TORCH_CHECK(false, "Unsupported CPU attention ISA hint: " + isa_hint);
}

}  // namespace

at::Tensor get_scheduler_metadata_acc_locality(
    int64_t num_req, int64_t num_heads_q, int64_t num_heads_kv,
    int64_t head_dim, const at::Tensor& seq_lens, at::ScalarType dtype,
    const at::Tensor& query_start_loc, bool casual, int64_t window_size,
    const std::string& isa_hint, bool enable_kv_split) {
  const cpu_attention::ISA isa = parse_cpu_attention_isa(isa_hint);
  int32_t max_num_q_per_iter = 0;

  VLLM_DISPATCH_FLOATING_TYPES(dtype, "get_scheduler_metadata_acc_locality", [&]() {
    CPU_ATTN_DISPATCH_CASE_HEADDIM(head_dim, [&] {
      CPU_ATTN_DISPATCH_IMPL(isa, [&]() {
        max_num_q_per_iter = attn_impl::MaxQHeadNumPerIteration;
      });
    });
  });

  return cpu_attention_acc_locality::build_scheduler_metadata(
      num_req, num_heads_q, num_heads_kv, head_dim, seq_lens, dtype,
      query_start_loc, casual, window_size, isa_hint, enable_kv_split,
      max_num_q_per_iter);
}

std::string inspect_cpu_attn_acc_locality_metadata(
    const at::Tensor& scheduler_metadata) {
  return cpu_attention_acc_locality::inspect_scheduler_metadata(
      scheduler_metadata);
}

void cpu_attention_with_kv_cache_acc_locality(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, at::Tensor& output,
    const at::Tensor& query_start_loc, const at::Tensor& seq_lens,
    double scale, bool causal,
    const std::optional<at::Tensor>& alibi_slopes,
    int64_t sliding_window_left, int64_t sliding_window_right,
    const at::Tensor& block_table, double softcap,
    const at::Tensor& scheduler_metadata, const std::optional<at::Tensor>& s_aux) {
  TORCH_CHECK_EQ(query.dim(), 3);
  TORCH_CHECK_EQ(query.stride(2), 1);
  TORCH_CHECK_EQ(key_cache.dim(), 4);
  TORCH_CHECK_EQ(value_cache.dim(), 4);

  cpu_attention_acc_locality::AttentionInput input;
  input.metadata =
      reinterpret_cast<cpu_attention_acc_locality::AttentionMetadata*>(
          scheduler_metadata.data_ptr());
  TORCH_CHECK_EQ(input.metadata->magic,
                 cpu_attention_acc_locality::kAccLocalityMetadataMagic);
  input.num_tokens = query.size(0);
  input.num_heads = query.size(1);
  input.num_kv_heads = key_cache.size(1);
  input.block_size = key_cache.size(2);
  input.query = query.data_ptr();
  input.query_num_tokens_stride = query.stride(0);
  input.query_num_heads_stride = query.stride(1);
  input.cache_num_blocks_stride = key_cache.stride(0);
  input.cache_num_kv_heads_stride = key_cache.stride(1);
  input.blt_num_tokens_stride = block_table.stride(0);
  input.key_cache = key_cache.data_ptr();
  input.value_cache = value_cache.data_ptr();
  input.output = output.data_ptr();
  input.query_start_loc = query_start_loc.data_ptr<int32_t>();
  input.seq_lens = seq_lens.data_ptr<int32_t>();
  input.block_table = block_table.data_ptr<int32_t>();
  input.alibi_slopes =
      alibi_slopes.has_value() ? alibi_slopes->data_ptr<float>() : nullptr;
  input.s_aux = s_aux.has_value() ? s_aux->data_ptr<c10::BFloat16>() : nullptr;
  input.scale = scale;
  input.causal = causal;
  input.sliding_window_left = sliding_window_left;
  input.sliding_window_right = sliding_window_right;
  if (input.causal) {
    input.sliding_window_right = 0;
  }
  input.softcap = static_cast<float>(softcap);

  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "cpu_attention_with_kv_cache_acc_locality", [&]() {
        CPU_ATTN_DISPATCH_CASE_HEADDIM(query.size(2), [&] {
          CPU_ATTN_DISPATCH_IMPL(input.metadata->legacy_metadata()->isa, [&]() {
            TORCH_CHECK_EQ(input.block_size % attn_impl::BlockSizeAlignment, 0);
            cpu_attention_acc_locality::AttentionMainLoop<attn_impl> mainloop;
            mainloop(&input);
          });
        });
      });
}
