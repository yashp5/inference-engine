#include "model.h"
#include "codec.h"

#include <cmath>
#include <iterator>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <fmt/format.h>
#include <iostream>
#include <limits>
#include <string>
#include <sys/types.h>

using json = nlohmann::json;

void Config::from_yalm(YALMData &yalm, int context) {
  dim = std::stoi(yalm.metadata.at("dim").get<std::string>());
  hidden_dim = std::stoi(yalm.metadata.at("hidden_dim").get<std::string>());
  head_dim = std::stoi(yalm.metadata.at("head_dim").get<std::string>());
  n_layers = std::stoi(yalm.metadata.at("n_layers").get<std::string>());
  n_heads = std::stoi(yalm.metadata.at("n_heads").get<std::string>());
  n_kv_heads = std::stoi(yalm.metadata.at("n_kv_heads").get<std::string>());
  vocab_size = std::stoi(yalm.metadata.at("vocab_size").get<std::string>());
  // mixture of experts
  n_experts = yalm.metadata.contains("n_experts")
                  ? std::stoi(yalm.metadata.at("n_experts").get<std::string>())
                  : 0;
  n_experts_active =
      yalm.metadata.contains("n_experts_active")
          ? std::stoi(yalm.metadata.at("n_experts_active").get<std::string>())
          : 0;

  // for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral
  // since window size isn't correctly specified
  max_seq_len = std::min(
      std::stoi(yalm.metadata.at("max_seq_len").get<std::string>()), 4096);
  if (context) {
    max_seq_len = context;
  }

  rope_theta = std::stof(yalm.metadata.at("rope_theta").get<std::string>());
  rotary_dim = std::stoi(yalm.metadata.at("rotary_dim").get<std::string>());

  norm_eps = std::stof(yalm.metadata.value("norm_eps", "1e-5"));

  std::string act_str = yalm.metadata.value("act_type", "gelu");
  if (act_str == "gelu") {
    act = ActivationType::GELU;
  } else if (act_str == "silu") {
    act = ActivationType::SILU;
  } else {
    std::cerr << "unsupported act_type, defaulting to GELU" << std::endl;
    act = ActivationType::GELU;
  }

  std::string norm_type_str = yalm.metadata.value("norm_type", "rmsnorm");
  if (norm_type_str == "rmsnorm") {
    norm_type = LayerNormType::RMSNorm;
  } else {
    std::cerr << "unsupported norm_type, defaulting to rmsnorm" << std::endl;
    norm_type = LayerNormType::RMSNorm;
  }

  qkv_clip = yalm.metadata.contains("qkv_clip")
                 ? std::stof(yalm.metadata.at("qkv_clip").get<std::string>())
                 : FLT_MAX;

  std::string dtype = yalm.metadata.at("dtype").get<std::string>();
  // Todo: support fp8
  if (dtype == "fp32") {
    weight_dtype = DType::F32;
  } else if (dtype == "fp16") {
    weight_dtype = DType::F16;
  } else {
    std::cerr << "FATAL: unsupported dtype" << std::endl;
    assert(false);
  }
}

size_t Config::active_bytes(size_t pos) const {
  size_t weight_size = dtype_size(weight_dtype);

  size_t bytes_per_block = 0;
  bytes_per_block += 2 * dim * sizeof(float); // rms_att_weight, rms_ffn_weight
  bytes_per_block += n_heads * head_dim * dim * weight_size;        // wq
  bytes_per_block += 2 * n_kv_heads * head_dim * dim * weight_size; // wk, wv
  bytes_per_block += n_heads * head_dim * dim * weight_size;        // wo
  if (n_experts > 0) {
    bytes_per_block += n_experts * dim * weight_size;
    bytes_per_block +=
        n_experts_active * 3 * dim * hidden_dim * weight_size; // w1, w2, w3
  } else {
    bytes_per_block += 3 * dim * hidden_dim * weight_size; // w1, w2, w3
  }
  size_t kv_len = std::min(static_cast<size_t>(max_seq_len), pos + 1);
  size_t kv_entry_size = sizeof(f16_t);
  bytes_per_block += 2 * kv_len * n_kv_heads * head_dim *
                     kv_entry_size; // key_cache, value_cache,

  size_t bytes = 0;
  bytes += dim * weight_size;                // 1 row of token_embedding_table
  bytes += n_layers * bytes_per_block;       // blocks
  bytes += dim * sizeof(float);              // rms_final_weight
  bytes += vocab_size * dim * sizeof(float); // wcls

  return bytes;
}

void *check_tensor(const Tensor *tensor, DType weight_dtype,
                   std::array<int, 4> shape) {
  if (tensor == nullptr) {
    std::cerr << "FATAL: missing tensor" << std::endl;
    assert(false);
    return nullptr;
  }
  if (tensor->dtype != weight_dtype || tensor->shape != shape) {
    std::cerr << "FATAL: tensor mismatch for " << tensor->name << std::endl;
    std::cerr << fmt::format("expected: dtype={}, shape=[{},{},{},{}]",
                             dtype_to_string(weight_dtype), shape[0], shape[1],
                             shape[2], shape[3])
              << std::endl;
    std::cerr << fmt::format("got: dtype={}, shape=[{},{},{},{}]",
                             dtype_to_string(tensor->dtype), tensor->shape[0],
                             tensor->shape[1], tensor->shape[2],
                             tensor->shape[3])
              << std::endl;
    assert(false);
  }
  return tensor->data;
}

const Tensor *get_tensor(const YALMData &yalm, const std::string &key) {
  auto it = yalm.tensors.find(key);
  if (it == yalm.tensors.end()) {
    std::cerr << "FATAL: missing tensor " << key << std::endl;
    assert(false);
    return nullptr;
  }
  const Tensor &tensor = it->second;
  return &tensor;
}

Block::Block(int layer_i, const std::shared_ptr<Config> config,
             const Tensor *rms_att_weight, const Tensor *rms_ffn_weight,
             const Tensor *wq, const Tensor *wk, const Tensor *wv,
             const Tensor *wo, const Tensor *w1, const Tensor *w2,
             const Tensor *w3, const Tensor *moegate) {
  _layer_i = layer_i;
  _config = config;
  switch (config->weight_dtype) {
  case DType::F32:
  case DType::F16: {
    break;
  }
  default: {
    std::cerr << "FATAL: unsupported weight dtype "
              << dtype_to_string(config->weight_dtype) << std::endl;
    assert(false);
    break;
  }
  }

  _rms_att_weight = static_cast<float *>(
      check_tensor(rms_att_weight, DType::F32, {config->dim, 0, 0, 0}));
  _rms_ffn_weight = static_cast<float *>(
      check_tensor(rms_ffn_weight, DType::F32, {config->dim, 0, 0, 0}));

  _wq = check_tensor(wq, config->weight_dtype,
                     {config->n_heads * config->head_dim, config->dim, 0, 0});
  _wk =
      check_tensor(wk, config->weight_dtype,
                   {config->n_kv_heads * config->head_dim, config->dim, 0, 0});
  _wv =
      check_tensor(wv, config->weight_dtype,
                   {config->n_kv_heads * config->head_dim, config->dim, 0, 0});
  _wo = check_tensor(wo, config->weight_dtype,
                     {config->dim, config->n_heads * config->head_dim, 0, 0});

  if (config->n_experts > 0) {
    _moegate = check_tensor(moegate, config->weight_dtype,
                            {config->n_experts, config->dim, 0, 0});
    _w1 = check_tensor(w1, config->weight_dtype,
                       {config->n_experts, config->hidden_dim, config->dim, 0});
    _w2 = check_tensor(w2, config->weight_dtype,
                       {config->n_experts, config->dim, config->hidden_dim, 0});
    _w3 = check_tensor(w3, config->weight_dtype,
                       {config->n_experts, config->hidden_dim, config->dim, 0});
  } else {
    _w1 = check_tensor(w1, config->weight_dtype,
                       {config->hidden_dim, config->dim, 0, 0});
    _w2 = check_tensor(w2, config->weight_dtype,
                       {config->dim, config->hidden_dim, 0, 0});
    _w3 = check_tensor(w3, config->weight_dtype,
                       {config->hidden_dim, config->dim, 0, 0});
  }

  _key_cache =
      new f16_t[config->max_seq_len * config->n_kv_heads * config->head_dim]();
  _value_cache =
      new f16_t[config->max_seq_len * config->n_kv_heads * config->head_dim]();
}

Block::~Block() {
  if (_device == Device::CPU) {
    delete[] _key_cache;
    delete[] _value_cache;
  } else {
    free_cuda(_key_cache);
    free_cuda(_value_cache);
  }
}

void Block::cuda() {
  if (_device != Device::CPU) {
    return;
  }
  _device = Device::CUDA;
  size_t weight_size = dtype_size(_config->weight_dtype);
  // norms
  _rms_att_weight = static_cast<float *>(
      upload_cuda(_rms_att_weight, _config->dim * sizeof(float)));
  _rms_ffn_weight = static_cast<float *>(
      upload_cuda(_rms_ffn_weight, _config->dim * sizeof(float)));

  // self-attention
  _wq = upload_cuda(_wq, _config->n_heads * _config->head_dim * _config->dim *
                             weight_size);
  _wk = upload_cuda(_wk, _config->n_kv_heads * _config->head_dim *
                             _config->dim * weight_size);
  _wv = upload_cuda(_wv, _config->n_kv_heads * _config->head_dim *
                             _config->dim * weight_size);
  _wo = upload_cuda(_wo, _config->dim * _config->n_heads * _config->head_dim *
                             weight_size);

  // ffn
  _w1 = upload_cuda(_w1, _config->hidden_dim * _config->dim * weight_size);
  _w2 = upload_cuda(_w2, _config->dim * _config->hidden_dim * weight_size);
  _w3 = upload_cuda(_w3, _config->hidden_dim * _config->dim * weight_size);

  // kv cache
  _key_cache = static_cast<f16_t *>(
      upload_cuda(_key_cache, _config->max_seq_len * _config->n_kv_heads *
                                  _config->head_dim * sizeof(f16_t)));
  _value_cache = static_cast<f16_t *>(
      upload_cuda(_value_cache, _config->max_seq_len * _config->n_kv_heads *
                                    _config->head_dim * sizeof(f16_t)));
}

void Block::block(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,        // number of sink tokens currently in the KV cache
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  if (_device == Device::CUDA) {
    switch (_config->weight_dtype) {
      case DType::F32: {
        _block_cuda<float>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case DType::F16: {
        _block_cuda<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for cuda");
      }
    }
  } else {
    switch (_config->weight_dtype) {
      case DType::F32: {
        _block_cpu<float>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case DType::F16: {
#if defined(__AVX2__) && defined(__F16C__)
        _block_cpu<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
#else
        assert(false && "float16 not supported on this platform");
#endif
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for cpu");
      }
    }
  }

}

InferenceState::InferenceState(const std::shared_ptr<Config> config):
_config(config){
    assert(config);
    _x = new float[config->dim]();
    _xb = new float[config->dim]();
    _xb2 = new float[config->dim]();
    _hb = new float[config->hidden_dim]();
    _hb2 = new float[config->hidden_dim]();
    _q = new float[config->n_heads * config->head_dim]();
    _k = new float[config->n_kv_heads * config->head_dim]();
    _v = new float[config->n_kv_heads * config->head_dim]();
    _att = new float[config->n_heads * config->max_seq_len]();
    _logits = new float[config->vocab_size]();
    if (config->n_experts > 0){
        _moe_weights = new float[config->n_experts]();
        _active_experts = new int[config->n_experts_active]();
        _active_experts_weights = new float[config->n_experts_active]();
    }
}
