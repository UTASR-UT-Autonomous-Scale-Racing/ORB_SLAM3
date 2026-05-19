/**
 * CUDA stages of the ORB extractor, bit-exact with the OpenCV calls the CPU
 * path makes: cv::FAST (9/16, non-max suppression, per sub-image borders),
 * cv::GaussianBlur 7x7 sigma 2 on CV_8U (OpenCV's fixed-point path) and the
 * rBRIEF comparisons of computeOrbDescriptor.
 */

#include "ORBextractorCuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ORB_SLAM3
{
namespace cuda
{
namespace
{

constexpr int kMaxLevels = 16;
constexpr int kBlurTaps = 7;
constexpr int kThreads = 256;

__constant__ signed char c_pattern[512 * 2];
__constant__ unsigned short c_blur[kBlurTaps];

void Check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
#define CK(call) Check((call), #call)

struct Levels
{
    int n;
    int w[kMaxLevels];
    int h[kMaxLevels];
    int off[kMaxLevels];   // pixel offset of the level in the packed buffers
};

struct CellDev
{
    int level;
    int x0, y0, x1, y1;    // pixels FAST tests: the sub-image minus its 3-pixel border
    int slot;              // first output slot of the cell
};

__device__ __forceinline__ int Reflect101(int p, int n)
{
    return p < 0 ? -p : (p >= n ? 2 * n - 2 - p : p);
}

// max(A, B) of cv::cornerScore<16>: A (B) is the best 9-pixel arc that is
// darker (brighter) than the centre, as the smallest difference on the arc.
// A pixel is a FAST corner at threshold t iff the result is > t, and its
// score is result - 1.
__global__ void FastScore(const unsigned char* __restrict__ img, unsigned char* __restrict__ score, Levels L, int minT)
{
    const int lv = blockIdx.z;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = L.w[lv], h = L.h[lv];
    if (lv >= L.n || x >= w || y >= h)
        return;
    unsigned char* out = score + L.off[lv] + y * w + x;
    if (x < 3 || y < 3 || x >= w - 3 || y >= h - 3)
    {
        *out = 0;
        return;
    }
    const unsigned char* p = img + L.off[lv] + y * w + x;
    const int v = p[0];
    // Every 9-pixel arc covers two neighbouring compass pixels (0, 4, 8, 12),
    // so a pixel whose neighbouring compass pairs never both pass minT can't
    // be a corner at any threshold used (score stored as 0).
    {
        const int c0 = p[3 * w], c1 = p[3], c2 = p[-3 * w], c3 = p[-3];
        const int dk = v - minT, br = v + minT;
        const int dark = (c0 < dk) | ((c1 < dk) << 1) | ((c2 < dk) << 2) | ((c3 < dk) << 3);
        const int brig = (c0 > br) | ((c1 > br) << 1) | ((c2 > br) << 2) | ((c3 > br) << 3);
        const int rd = (dark | (dark << 4)) >> 1, rb = (brig | (brig << 4)) >> 1;
        if (!((dark & rd) & 15) && !((brig & rb) & 15))
        {
            *out = 0;
            return;
        }
    }
    const int ox[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    const int oy[16] = {3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3};
    int d[16];
#pragma unroll
    for (int k = 0; k < 16; ++k)
        d[k] = v - p[oy[k] * w + ox[k]];
    int a = -256, b = -256;
#pragma unroll
    for (int s = 0; s < 16; ++s)
    {
        int mn = 256, mx = -256;
#pragma unroll
        for (int j = 0; j < 9; ++j)
        {
            const int dv = d[(s + j) & 15];
            mn = min(mn, dv);
            mx = max(mx, dv);
        }
        a = max(a, mn);
        b = max(b, -mx);
    }
    *out = (unsigned char)min(max(max(a, b), 0), 255);
}

__global__ void BlurRows(const unsigned char* __restrict__ img, unsigned short* __restrict__ rows, Levels L)
{
    const int lv = blockIdx.z;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (lv >= L.n || x >= L.w[lv] || y >= L.h[lv])
        return;
    const int w = L.w[lv];
    const unsigned char* row = img + L.off[lv] + y * w;
    unsigned int sum = 0;
#pragma unroll
    for (int t = 0; t < kBlurTaps; ++t)
        sum += (unsigned int)c_blur[t] * row[Reflect101(x + t - kBlurTaps / 2, w)];
    rows[L.off[lv] + y * w + x] = (unsigned short)sum;          // <= 255 * 256
}

__global__ void BlurCols(const unsigned short* __restrict__ rows, unsigned char* __restrict__ out, Levels L)
{
    const int lv = blockIdx.z;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (lv >= L.n || x >= L.w[lv] || y >= L.h[lv])
        return;
    const int w = L.w[lv], h = L.h[lv];
    const unsigned short* base = rows + L.off[lv] + x;
    unsigned int sum = 0;
#pragma unroll
    for (int t = 0; t < kBlurTaps; ++t)
        sum += (unsigned int)c_blur[t] * base[Reflect101(y + t - kBlurTaps / 2, h) * w];
    out[L.off[lv] + y * w + x] = (unsigned char)min((sum + 32768u) >> 16, 255u);
}

// Score of the pixel as a FAST keypoint of this cell at threshold t, or -1.
__device__ __forceinline__ int CellKeep(const unsigned char* s, int w, const CellDev& c, int x, int y, int t)
{
    const int sv = s[y * w + x];
    if (sv <= t)
        return -1;
    const int sc = sv - 1;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0)
                continue;
            const int nx = x + dx, ny = y + dy;
            int nb = 0;
            if (nx >= c.x0 && nx < c.x1 && ny >= c.y0 && ny < c.y1)
            {
                const int sn = s[ny * w + nx];
                nb = sn > t ? sn - 1 : 0;
            }
            if (!(sc > nb))
                return -1;
        }
    return sc;
}

// One block per cell: FAST at iniT, at minT if the cell had none, written in
// row-major order (the order cv::FAST returns them).
__global__ void DetectCells(const unsigned char* __restrict__ score, Levels L, const CellDev* __restrict__ cells,
                            int iniT, int minT, int minBorder,
                            unsigned int* __restrict__ outXY, unsigned char* __restrict__ outScore,
                            int* __restrict__ cellCount)
{
    const CellDev c = cells[blockIdx.x];
    const int w = L.w[c.level];
    const unsigned char* s = score + L.off[c.level];
    const int vw = c.x1 - c.x0, vh = c.y1 - c.y0;
    if (vw <= 0 || vh <= 0)
    {
        if (threadIdx.x == 0)
            cellCount[blockIdx.x] = 0;
        return;
    }
    const int n = vw * vh;

    int any = 0;
    for (int k = threadIdx.x; k < n && !any; k += blockDim.x)
        any = CellKeep(s, w, c, c.x0 + k % vw, c.y0 + k / vw, iniT) >= 0;
    const int t = __syncthreads_or(any) ? iniT : minT;

    __shared__ int warpCount[kThreads / 32];
    __shared__ int base;
    if (threadIdx.x == 0)
        base = 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int k0 = 0; k0 < n; k0 += blockDim.x)
    {
        const int k = k0 + threadIdx.x;
        int sc = -1, x = 0, y = 0;
        if (k < n)
        {
            x = c.x0 + k % vw;
            y = c.y0 + k / vw;
            sc = CellKeep(s, w, c, x, y, t);
        }
        const unsigned int ballot = __ballot_sync(0xffffffffu, sc >= 0);
        if (lane == 0)
            warpCount[warp] = __popc(ballot);
        __syncthreads();
        int pos = base + __popc(ballot & ((1u << lane) - 1u));
        for (int i = 0; i < warp; ++i)
            pos += warpCount[i];
        if (sc >= 0)
        {
            outXY[c.slot + pos] = ((unsigned int)(y - minBorder) << 16) | (unsigned int)(x - minBorder);
            outScore[c.slot + pos] = (unsigned char)sc;
        }
        __syncthreads();
        if (threadIdx.x == 0)
            for (int i = 0; i < kThreads / 32; ++i)
                base += warpCount[i];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        cellCount[blockIdx.x] = base;
}

// One block per level: concatenate the cells' keypoints in cell order.
__global__ void CompactLevels(const int* __restrict__ cellCount, const CellDev* __restrict__ cells,
                              const int* __restrict__ levelCells,
                              const unsigned int* __restrict__ xy, const unsigned char* __restrict__ sc,
                              unsigned int* __restrict__ outXY, unsigned char* __restrict__ outScore,
                              int* __restrict__ levelCount)
{
    extern __shared__ int start[];          // exclusive prefix of the cell counts, + total
    const int c0 = levelCells[blockIdx.x], c1 = levelCells[blockIdx.x + 1];
    const int nc = c1 - c0;
    if (threadIdx.x == 0)
    {
        int run = 0;
        for (int i = 0; i < nc; ++i)
        {
            start[i] = run;
            run += cellCount[c0 + i];
        }
        start[nc] = run;
        levelCount[blockIdx.x] = run;
    }
    __syncthreads();
    if (nc == 0)
        return;
    const int total = start[nc], dst = cells[c0].slot;
    for (int e = threadIdx.x; e < total; e += blockDim.x)
    {
        int lo = 0, hi = nc - 1;                // cell holding entry e
        while (lo < hi)
        {
            const int mid = (lo + hi + 1) >> 1;
            if (start[mid] <= e) lo = mid; else hi = mid - 1;
        }
        const int src = cells[c0 + lo].slot + (e - start[lo]);
        outXY[dst + e] = xy[src];
        outScore[dst + e] = sc[src];
    }
}

// cvRound(x*b + y*a) and cvRound(x*a - y*b) of computeOrbDescriptor, rounded
// the way the host build computes them: GCC fuses them into FMAs when the
// host has FMA (x86 -march=native with FMA, aarch64), so the descriptors
// match bit for bit. cvRound = round half to even.
__device__ __forceinline__ int SampleRow(float x, float y, float a, float b)
{
#ifdef ORB_SLAM3_HOST_FMA
    return __float2int_rn(__fmaf_rn(x, b, __fmul_rn(y, a)));
#else
    return __float2int_rn(__fadd_rn(__fmul_rn(x, b), __fmul_rn(y, a)));
#endif
}

__device__ __forceinline__ int SampleCol(float x, float y, float a, float b)
{
#ifdef ORB_SLAM3_HOST_FMA
    return __float2int_rn(__fmaf_rn(x, a, -__fmul_rn(y, b)));
#else
    return __float2int_rn(__fsub_rn(__fmul_rn(x, a), __fmul_rn(y, b)));
#endif
}

// 32 threads per keypoint, one descriptor byte each (computeOrbDescriptor).
__global__ void DescribeKeypoints(const unsigned char* __restrict__ blurred, Levels L, const int4* __restrict__ kps,
                         const float2* __restrict__ cs, int n, unsigned char* __restrict__ out)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = gid >> 5, byte = gid & 31;
    if (i >= n)
        return;
    const int4 k = kps[i];
    const int w = L.w[k.z];
    const unsigned char* center = blurred + L.off[k.z] + k.y * w + k.x;
    const float a = cs[i].x, b = cs[i].y;
    const signed char* p = c_pattern + byte * 32;
    int val = 0;
#pragma unroll
    for (int t = 0; t < 8; ++t)
    {
        const float x0 = p[4 * t], y0 = p[4 * t + 1], x1 = p[4 * t + 2], y1 = p[4 * t + 3];
        const int r0 = SampleRow(x0, y0, a, b), q0 = SampleCol(x0, y0, a, b);
        const int r1 = SampleRow(x1, y1, a, b), q1 = SampleCol(x1, y1, a, b);
        val |= (center[r0 * w + q0] < center[r1 * w + q1]) << t;
    }
    out[i * 32 + byte] = (unsigned char)val;
}

// OpenCV's fixed-point Gaussian kernel for CV_8U (8 fractional bits, rounding
// error carried from the outside in, centre tap = 1 - rest).
void FixedPointGaussian(int n, double sigma, unsigned short* k)
{
    const double scale2 = -0.125 / (sigma * sigma);
    const int half = (n - 1) / 2;
    double vals[16], sum = 0.0;
    for (int i = 0, x = 1 - n; i < half; ++i, x += 2)
    {
        vals[i] = std::exp(double(x * x) * scale2);
        sum += vals[i];
    }
    sum = sum * 2.0 + 1.0;
    const double mul = 1.0 / sum;
    double err = 0.0;
    long total = 0;
    for (int i = 0; i < half; ++i)
    {
        const double adj = vals[i] * mul * 256.0 + err;
        const long v = std::lrint(adj);
        err = adj - double(v);
        k[i] = k[n - 1 - i] = (unsigned short)v;
        total += v;
    }
    k[half] = (unsigned short)(256 - 2 * total);
}

template <typename T>
struct HostBuf
{
    T* p = nullptr;
    size_t cap = 0;
    void Ensure(size_t n)
    {
        if (n <= cap)
            return;
        if (p)
            cudaFreeHost(p);
        CK(cudaMallocHost(&p, n * sizeof(T)));
        cap = n;
    }
    ~HostBuf() { if (p) cudaFreeHost(p); }
};

template <typename T>
struct DevBuf
{
    T* p = nullptr;
    size_t cap = 0;
    void Ensure(size_t n)
    {
        if (n <= cap)
            return;
        if (p)
            cudaFree(p);
        CK(cudaMalloc(&p, n * sizeof(T)));
        cap = n;
    }
    ~DevBuf() { if (p) cudaFree(p); }
};

std::once_flag g_constants;

} // namespace

bool Enabled()
{
    static const bool enabled = [] {
        const char* env = std::getenv("ORB_SLAM3_CUDA");
        if (env && std::string(env) == "0")
            return false;
        int n = 0;
        return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }();
    return enabled;
}

struct OrbCuda::Impl
{
    cudaStream_t stream = nullptr;
    Levels levels{};
    int totalPixels = 0;
    DevBuf<unsigned char> img, score, blurred;
    DevBuf<unsigned short> rows;
    DevBuf<CellDev> cells;
    DevBuf<int> cellCount, levelCells, levelCount;
    DevBuf<unsigned int> xy, cxy;
    DevBuf<unsigned char> sc, csc;
    DevBuf<int4> kps;
    DevBuf<float2> cs;
    DevBuf<unsigned char> desc;
    HostBuf<unsigned char> hImg, hSc, hDesc;
    HostBuf<CellDev> hCells;
    HostBuf<int> hLevelCells, hLevelCount;
    HostBuf<unsigned int> hXY;
    HostBuf<int4> hKps;
    HostBuf<float2> hCs;
    std::vector<int> levelSlot;
};

OrbCuda::OrbCuda(const std::vector<cv::Point>& pattern) : d(new Impl)
{
    if (pattern.size() != 512)
        throw std::runtime_error("ORB pattern must have 512 points");
    std::call_once(g_constants, [&] {
        signed char pat[1024];
        for (int i = 0; i < 512; ++i)
        {
            pat[2 * i] = (signed char)pattern[i].x;
            pat[2 * i + 1] = (signed char)pattern[i].y;
        }
        unsigned short k[kBlurTaps];
        FixedPointGaussian(kBlurTaps, 2.0, k);
        CK(cudaMemcpyToSymbol(c_pattern, pat, sizeof(pat)));
        CK(cudaMemcpyToSymbol(c_blur, k, sizeof(k)));
    });
    CK(cudaStreamCreateWithFlags(&d->stream, cudaStreamNonBlocking));
}

OrbCuda::~OrbCuda()
{
    if (d->stream)
        cudaStreamDestroy(d->stream);
}

void OrbCuda::Detect(const std::vector<cv::Mat>& images, const std::vector<std::vector<Cell> >& cellsPerLevel,
                     int minBorder, int iniThFAST, int minThFAST,
                     std::vector<std::vector<cv::KeyPoint> >& out)
{
    Impl& m = *d;
    const int nlevels = (int)images.size();
    if (nlevels > kMaxLevels)
        throw std::runtime_error("too many pyramid levels for the CUDA extractor");

    // pack the levels into pinned memory and upload
    Levels& L = m.levels;
    L.n = nlevels;
    int total = 0, maxW = 0, maxH = 0;
    for (int l = 0; l < nlevels; ++l)
    {
        L.w[l] = images[l].cols;
        L.h[l] = images[l].rows;
        L.off[l] = total;
        total += L.w[l] * L.h[l];
        maxW = std::max(maxW, L.w[l]);
        maxH = std::max(maxH, L.h[l]);
    }
    m.totalPixels = total;
    m.hImg.Ensure(total);
    for (int l = 0; l < nlevels; ++l)
        for (int y = 0; y < L.h[l]; ++y)
            std::memcpy(m.hImg.p + L.off[l] + y * L.w[l], images[l].ptr<unsigned char>(y), L.w[l]);
    m.img.Ensure(total);
    m.score.Ensure(total);
    m.blurred.Ensure(total);
    m.rows.Ensure(total);
    CK(cudaMemcpyAsync(m.img.p, m.hImg.p, total, cudaMemcpyHostToDevice, m.stream));

    const dim3 block(32, 8);
    const dim3 grid((maxW + block.x - 1) / block.x, (maxH + block.y - 1) / block.y, nlevels);
    FastScore<<<grid, block, 0, m.stream>>>(m.img.p, m.score.p, L, std::min(iniThFAST, minThFAST));
    BlurRows<<<grid, block, 0, m.stream>>>(m.img.p, m.rows.p, L);
    BlurCols<<<grid, block, 0, m.stream>>>(m.rows.p, m.blurred.p, L);

    // cells, with an output capacity each can't exceed: surviving corners are
    // never 8-neighbours, so at most ceil(w/2)*ceil(h/2) per cell
    int ncells = 0;
    for (const auto& c : cellsPerLevel)
        ncells += (int)c.size();
    m.hCells.Ensure(std::max(ncells, 1));
    m.hLevelCells.Ensure(nlevels + 1);
    m.levelSlot.assign(nlevels, 0);
    int slot = 0, ci = 0;
    for (int l = 0; l < nlevels; ++l)
    {
        m.hLevelCells.p[l] = ci;
        m.levelSlot[l] = slot;
        for (const Cell& c : cellsPerLevel[l])
        {
            CellDev cd{l, c.iniX + 3, c.iniY + 3, c.maxX - 3, c.maxY - 3, slot};
            const int vw = std::max(cd.x1 - cd.x0, 0), vh = std::max(cd.y1 - cd.y0, 0);
            slot += ((vw + 1) / 2) * ((vh + 1) / 2);
            m.hCells.p[ci++] = cd;
        }
    }
    m.hLevelCells.p[nlevels] = ci;
    const int slots = std::max(slot, 1);
    m.cells.Ensure(std::max(ncells, 1));
    m.cellCount.Ensure(std::max(ncells, 1));
    m.levelCells.Ensure(nlevels + 1);
    m.levelCount.Ensure(nlevels);
    m.xy.Ensure(slots);
    m.sc.Ensure(slots);
    m.cxy.Ensure(slots);
    m.csc.Ensure(slots);
    CK(cudaMemcpyAsync(m.cells.p, m.hCells.p, ncells * sizeof(CellDev), cudaMemcpyHostToDevice, m.stream));
    CK(cudaMemcpyAsync(m.levelCells.p, m.hLevelCells.p, (nlevels + 1) * sizeof(int), cudaMemcpyHostToDevice, m.stream));

    if (ncells > 0)
        DetectCells<<<ncells, kThreads, 0, m.stream>>>(m.score.p, L, m.cells.p, iniThFAST, minThFAST, minBorder,
                                                       m.xy.p, m.sc.p, m.cellCount.p);
    int maxCells = 0;
    for (const auto& c : cellsPerLevel)
        maxCells = std::max(maxCells, (int)c.size());
    CompactLevels<<<nlevels, kThreads, (maxCells + 1) * sizeof(int), m.stream>>>(
        m.cellCount.p, m.cells.p, m.levelCells.p, m.xy.p, m.sc.p, m.cxy.p, m.csc.p, m.levelCount.p);
    m.hLevelCount.Ensure(nlevels);
    CK(cudaMemcpyAsync(m.hLevelCount.p, m.levelCount.p, nlevels * sizeof(int), cudaMemcpyDeviceToHost, m.stream));
    CK(cudaStreamSynchronize(m.stream));
    CK(cudaGetLastError());

    m.hXY.Ensure(std::max(slots, 1));
    m.hSc.Ensure(std::max(slots, 1));
    for (int l = 0; l < nlevels; ++l)
    {
        const int n = m.hLevelCount.p[l], s = m.levelSlot[l];
        if (n == 0)
            continue;
        CK(cudaMemcpyAsync(m.hXY.p + s, m.cxy.p + s, n * sizeof(unsigned int), cudaMemcpyDeviceToHost, m.stream));
        CK(cudaMemcpyAsync(m.hSc.p + s, m.csc.p + s, n, cudaMemcpyDeviceToHost, m.stream));
    }
    CK(cudaStreamSynchronize(m.stream));

    out.assign(nlevels, std::vector<cv::KeyPoint>());
    for (int l = 0; l < nlevels; ++l)
    {
        const int n = m.hLevelCount.p[l], s = m.levelSlot[l];
        std::vector<cv::KeyPoint>& kps = out[l];
        kps.reserve(n);
        for (int k = 0; k < n; ++k)
        {
            const unsigned int v = m.hXY.p[s + k];
            kps.emplace_back((float)(v & 0xffffu), (float)(v >> 16), 7.f, -1.f, (float)m.hSc.p[s + k]);
        }
    }
}

void OrbCuda::Describe(const std::vector<std::vector<cv::KeyPoint> >& keypoints,
                       const std::vector<std::vector<float> >& cosA,
                       const std::vector<std::vector<float> >& sinA,
                       std::vector<cv::Mat>& descriptors)
{
    Impl& m = *d;
    const int nlevels = (int)keypoints.size();
    int n = 0;
    for (const auto& k : keypoints)
        n += (int)k.size();
    descriptors.assign(nlevels, cv::Mat());
    if (n == 0)
        return;
    m.hKps.Ensure(n);
    m.hCs.Ensure(n);
    int i = 0;
    for (int l = 0; l < nlevels; ++l)
        for (size_t k = 0; k < keypoints[l].size(); ++k, ++i)
        {
            const cv::KeyPoint& kp = keypoints[l][k];
            m.hKps.p[i] = make_int4(cvRound(kp.pt.x), cvRound(kp.pt.y), l, 0);
            m.hCs.p[i] = make_float2(cosA[l][k], sinA[l][k]);
        }
    m.kps.Ensure(n);
    m.cs.Ensure(n);
    m.desc.Ensure((size_t)n * 32);
    m.hDesc.Ensure((size_t)n * 32);
    CK(cudaMemcpyAsync(m.kps.p, m.hKps.p, n * sizeof(int4), cudaMemcpyHostToDevice, m.stream));
    CK(cudaMemcpyAsync(m.cs.p, m.hCs.p, n * sizeof(float2), cudaMemcpyHostToDevice, m.stream));
    const int threads = 128;
    DescribeKeypoints<<<(n * 32 + threads - 1) / threads, threads, 0, m.stream>>>(m.blurred.p, m.levels, m.kps.p, m.cs.p, n,
                                                                        m.desc.p);
    CK(cudaMemcpyAsync(m.hDesc.p, m.desc.p, (size_t)n * 32, cudaMemcpyDeviceToHost, m.stream));
    CK(cudaStreamSynchronize(m.stream));
    CK(cudaGetLastError());

    i = 0;
    for (int l = 0; l < nlevels; ++l)
    {
        const int nl = (int)keypoints[l].size();
        descriptors[l] = cv::Mat(nl, 32, CV_8U);
        if (nl)
            std::memcpy(descriptors[l].data, m.hDesc.p + (size_t)i * 32, (size_t)nl * 32);
        i += nl;
    }
}

} // namespace cuda
} // namespace ORB_SLAM3
