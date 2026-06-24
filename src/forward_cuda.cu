// MooGPT — Phase 6b: GPU forward-inference path (CUDA).
//
// Standalone CUDA program that mirrors GPT::forward (model.cpp) on the device and dumps
// logits in the exact same binary format as `moogpt check`, so reference/check.py verifies
// it against the PyTorch oracle. Usage matches `check`:
//
//   moogpt_cuda <weights.bin> <input.bin> <out_logits.bin>
//
// Design choices (deliberately matching the project's discipline):
//   * Hand-written kernels, no cuBLAS — same "from scratch" spirit as the CPU code, and it
//     lets us keep every reduction in the SAME ascending order the CPU loops use, so the
//     result matches the verified CPU/PyTorch path to ~1e-6 (well inside the 1e-4 gate)
//     rather than drifting on a fused-BLAS summation order.
//   * One thread per output element (linear) or per row/(head,query) (layernorm/attention).
//     Not the fastest possible kernels — correctness and matching the oracle come first, per
//     the project's correctness -> clarity -> speed ordering. Tuning is future work.
//   * Self-contained: it re-parses the MGPT weight file itself (the format is the contract,
//     same as check.py duplicating the layout) so it needs none of the host C++ classes and
//     compiles as a single translation unit with nvcc.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                   cudaGetErrorString(err__), __FILE__, __LINE__);           \
      std::exit(1);                                                            \
    }                                                                         \
  } while (0)

void die(const std::string& msg) {
  std::fprintf(stderr, "forward_cuda: %s\n", msg.c_str());
  std::exit(1);
}

// ---- host-side weight file parsing (MGPT format, must match model.cpp::load) -----------

struct Config {
  int n_layer, n_head, n_embd, block_size, vocab_size;
  int head_dim() const { return n_embd / n_head; }
};

int32_t read_i32(std::ifstream& f) {
  int32_t v = 0;
  f.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!f) die("unexpected EOF in header");
  return v;
}

// Reads `n` float32 into a host vector.
std::vector<float> read_floats(std::ifstream& f, std::size_t n) {
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()),
         static_cast<std::streamsize>(n * sizeof(float)));
  if (!f) die("unexpected EOF reading tensor");
  return v;
}

// A device-resident weight tensor.
struct DTensor {
  float* d = nullptr;
  std::size_t n = 0;
};

DTensor upload(const std::vector<float>& host) {
  DTensor t;
  t.n = host.size();
  CUDA_CHECK(cudaMalloc(&t.d, t.n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(t.d, host.data(), t.n * sizeof(float), cudaMemcpyHostToDevice));
  return t;
}

struct BlockW {
  DTensor ln1_w, ln1_b;
  DTensor c_attn_w, c_attn_b;   // (3C,C),(3C)
  DTensor c_proj_w, c_proj_b;   // (C,C),(C)
  DTensor ln2_w, ln2_b;
  DTensor c_fc_w, c_fc_b;       // (4C,C),(4C)
  DTensor mlp_proj_w, mlp_proj_b;  // (C,4C),(C)
};

struct Weights {
  Config cfg;
  DTensor wte, wpe;             // (V,C),(B,C)
  std::vector<BlockW> blocks;
  DTensor ln_f_w, ln_f_b;       // (C),(C)
  DTensor lm_head_w;            // (V,C)
};

Weights load_weights(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open weight file: " + path);
  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, "MGPT", 4) != 0) die("bad magic (expected 'MGPT')");
  if (read_i32(f) != 1) die("unsupported version");

  Weights w;
  w.cfg.n_layer = read_i32(f);
  w.cfg.n_head = read_i32(f);
  w.cfg.n_embd = read_i32(f);
  w.cfg.block_size = read_i32(f);
  w.cfg.vocab_size = read_i32(f);
  const int C = w.cfg.n_embd, V = w.cfg.vocab_size, B = w.cfg.block_size;

  auto rd = [&](std::size_t n) { return upload(read_floats(f, n)); };

  // Canonical order — identical to model.cpp::load.
  w.wte = rd((std::size_t)V * C);
  w.wpe = rd((std::size_t)B * C);
  w.blocks.resize(w.cfg.n_layer);
  for (auto& b : w.blocks) {
    b.ln1_w = rd(C);
    b.ln1_b = rd(C);
    b.c_attn_w = rd((std::size_t)3 * C * C);
    b.c_attn_b = rd((std::size_t)3 * C);
    b.c_proj_w = rd((std::size_t)C * C);
    b.c_proj_b = rd(C);
    b.ln2_w = rd(C);
    b.ln2_b = rd(C);
    b.c_fc_w = rd((std::size_t)4 * C * C);
    b.c_fc_b = rd((std::size_t)4 * C);
    b.mlp_proj_w = rd((std::size_t)C * 4 * C);
    b.mlp_proj_b = rd(C);
  }
  w.ln_f_w = rd(C);
  w.ln_f_b = rd(C);
  w.lm_head_w = rd((std::size_t)V * C);

  f.peek();
  if (!f.eof()) die("trailing bytes in weight file (config/order mismatch?)");
  return w;
}

std::vector<int> read_input_ids(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open input file: " + path);
  int32_t T = 0;
  f.read(reinterpret_cast<char*>(&T), sizeof(T));
  std::vector<int> ids(T);
  for (int i = 0; i < T; ++i) ids[i] = read_i32(f);
  return ids;
}

}  // namespace

// ---- kernels ---------------------------------------------------------------------------
// Each kernel mirrors the corresponding CPU op in ops.cpp / attention.cpp, keeping the
// inner reduction in ascending order so the arithmetic matches the verified CPU path.

__constant__ float kInvSqrt2 = 0.70710678118654752440f;  // 1/sqrt(2)

// x[t,c] = wte[id[t], c] + wpe[t, c]
__global__ void embed_kernel(const int* ids, const float* wte, const float* wpe,
                             float* x, int T, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * C) return;
  int t = idx / C, c = idx % C;
  x[idx] = wte[ids[t] * C + c] + wpe[t * C + c];
}

// One thread per row t: biased-variance LayerNorm (eps 1e-5), matching ops.cpp.
__global__ void layernorm_kernel(const float* x, const float* g, const float* b,
                                 float* out, int T, int C) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* row = x + t * C;
  float* orow = out + t * C;
  float mean = 0.0f;
  for (int c = 0; c < C; ++c) mean += row[c];
  mean /= C;
  float var = 0.0f;
  for (int c = 0; c < C; ++c) {
    float d = row[c] - mean;
    var += d * d;
  }
  var /= C;
  float inv_std = 1.0f / sqrtf(var + 1e-5f);
  for (int c = 0; c < C; ++c) orow[c] = (row[c] - mean) * inv_std * g[c] + b[c];
}

// y[t,o] = bias[o] + sum_i x[t,i] * W[o,i]   (W is (out,in); bias optional).
__global__ void linear_kernel(const float* x, const float* W, const float* bias,
                              float* y, int T, int in, int out, int has_bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * out) return;
  int t = idx / out, o = idx % out;
  const float* xr = x + t * in;
  const float* wr = W + o * in;
  float acc = has_bias ? bias[o] : 0.0f;
  for (int i = 0; i < in; ++i) acc += xr[i] * wr[i];
  y[idx] = acc;
}

// out[i] = 0.5 x (1 + erf(x/sqrt2))  (exact-erf GELU, matching ops.cpp).
__global__ void gelu_kernel(const float* x, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = x[idx];
  out[idx] = 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

// out = a + b  (residual add).
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = a[idx] + b[idx];
}

// Causal multi-head attention, one thread per (head, query-row) — the same per-i scheme as
// attention.cpp. qkv is (T,3C) with cols [0,C)=Q,[C,2C)=K,[2C,3C)=V. `probs` is per-(h,i)
// scratch of length T (row of the (T,T) weight matrix). Writes head h's column slice of
// out_heads (T,C).
__global__ void attention_kernel(const float* qkv, float* probs, float* out_heads,
                                 int T, int C, int H, int d, float scale) {
  int hi = blockIdx.x * blockDim.x + threadIdx.x;
  if (hi >= H * T) return;
  int h = hi / T, i = hi % T;
  int qoff = h * d, koff = C + h * d, voff = 2 * C + h * d;
  const float* qi = qkv + i * (3 * C) + qoff;
  float* prow = probs + (std::size_t)hi * T;

  // scores[j] = (q_i . k_j) * scale for j <= i, then softmax in place over j <= i.
  float m = -INFINITY;
  for (int j = 0; j <= i; ++j) {
    const float* kj = qkv + j * (3 * C) + koff;
    float dot = 0.0f;
    for (int e = 0; e < d; ++e) dot += qi[e] * kj[e];
    float s = dot * scale;
    prow[j] = s;
    if (s > m) m = s;
  }
  float sum = 0.0f;
  for (int j = 0; j <= i; ++j) {
    float e = expf(prow[j] - m);
    prow[j] = e;
    sum += e;
  }
  float inv = 1.0f / sum;
  for (int j = 0; j <= i; ++j) prow[j] *= inv;

  // out_i = sum_{j<=i} p[i,j] * v_j  (head h's columns of out_heads row i).
  float* orow = out_heads + i * C + h * d;
  for (int e = 0; e < d; ++e) orow[e] = 0.0f;
  for (int j = 0; j <= i; ++j) {
    float p = prow[j];
    const float* vj = qkv + j * (3 * C) + voff;
    for (int e = 0; e < d; ++e) orow[e] += p * vj[e];
  }
}

namespace {

inline int grid(int n, int block) { return (n + block - 1) / block; }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <weights.bin> <input.bin> <out_logits.bin>\n", argv[0]);
    return 2;
  }
  Weights w = load_weights(argv[1]);
  std::vector<int> ids = read_input_ids(argv[2]);
  const Config& cfg = w.cfg;
  const int T = static_cast<int>(ids.size());
  const int C = cfg.n_embd, H = cfg.n_head, d = cfg.head_dim(), V = cfg.vocab_size;
  if (T == 0) die("empty input");
  if (T > cfg.block_size) die("sequence longer than block_size");
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  // Report the device so the run is self-documenting.
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("forward_cuda: device=%s sm_%d%d  T=%d C=%d H=%d V=%d layers=%d\n",
              prop.name, prop.major, prop.minor, T, C, H, V, cfg.n_layer);

  // Device input ids.
  int* d_ids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ids, T * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), T * sizeof(int), cudaMemcpyHostToDevice));

  // Scratch buffers (sized to the largest use).
  auto dalloc = [](std::size_t n) {
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    return p;
  };
  float* x = dalloc((std::size_t)T * C);        // residual stream
  float* ln = dalloc((std::size_t)T * C);       // layernorm output
  float* qkv = dalloc((std::size_t)T * 3 * C);  // fused QKV
  float* out_heads = dalloc((std::size_t)T * C);
  float* attn_proj = dalloc((std::size_t)T * C);
  float* x1 = dalloc((std::size_t)T * C);
  float* fc = dalloc((std::size_t)T * 4 * C);
  float* act = dalloc((std::size_t)T * 4 * C);
  float* mlp_out = dalloc((std::size_t)T * C);
  float* probs = dalloc((std::size_t)H * T * T);
  float* logits = dalloc((std::size_t)T * V);

  const int BS = 256;

  // x = wte[ids] + wpe
  embed_kernel<<<grid(T * C, BS), BS>>>(d_ids, w.wte.d, w.wpe.d, x, T, C);

  for (int l = 0; l < cfg.n_layer; ++l) {
    BlockW& b = w.blocks[l];
    // --- attention sublayer (pre-norm + residual) ---
    layernorm_kernel<<<grid(T, BS), BS>>>(x, b.ln1_w.d, b.ln1_b.d, ln, T, C);
    linear_kernel<<<grid(T * 3 * C, BS), BS>>>(ln, b.c_attn_w.d, b.c_attn_b.d, qkv, T, C,
                                               3 * C, 1);
    attention_kernel<<<grid(H * T, BS), BS>>>(qkv, probs, out_heads, T, C, H, d, scale);
    linear_kernel<<<grid(T * C, BS), BS>>>(out_heads, b.c_proj_w.d, b.c_proj_b.d, attn_proj,
                                           T, C, C, 1);
    add_kernel<<<grid(T * C, BS), BS>>>(x, attn_proj, x1, T * C);  // x1 = x + attn

    // --- mlp sublayer (pre-norm + residual) ---
    layernorm_kernel<<<grid(T, BS), BS>>>(x1, b.ln2_w.d, b.ln2_b.d, ln, T, C);
    linear_kernel<<<grid(T * 4 * C, BS), BS>>>(ln, b.c_fc_w.d, b.c_fc_b.d, fc, T, C, 4 * C, 1);
    gelu_kernel<<<grid(T * 4 * C, BS), BS>>>(fc, act, T * 4 * C);
    linear_kernel<<<grid(T * C, BS), BS>>>(act, b.mlp_proj_w.d, b.mlp_proj_b.d, mlp_out, T,
                                           4 * C, C, 1);
    add_kernel<<<grid(T * C, BS), BS>>>(x1, mlp_out, x, T * C);  // x = x1 + mlp
  }

  // final layernorm + lm_head (no bias)
  layernorm_kernel<<<grid(T, BS), BS>>>(x, w.ln_f_w.d, w.ln_f_b.d, ln, T, C);
  linear_kernel<<<grid(T * V, BS), BS>>>(ln, w.lm_head_w.d, nullptr, logits, T, C, V, 0);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy logits back and write in the same format as `moogpt check`.
  std::vector<float> host_logits((std::size_t)T * V);
  CUDA_CHECK(cudaMemcpy(host_logits.data(), logits, host_logits.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::ofstream f(argv[3], std::ios::binary);
  if (!f) die("cannot open output file");
  int32_t Ti = T, Vi = V;
  f.write(reinterpret_cast<const char*>(&Ti), sizeof(Ti));
  f.write(reinterpret_cast<const char*>(&Vi), sizeof(Vi));
  f.write(reinterpret_cast<const char*>(host_logits.data()),
          static_cast<std::streamsize>(host_logits.size() * sizeof(float)));
  std::printf("forward_cuda: wrote logits (%d, %d) to %s\n", T, V, argv[3]);
  return 0;
}
