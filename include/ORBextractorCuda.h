/**
 * CUDA stages of the ORB extractor (FAST per cell with the low-threshold
 * retry, 7x7 Gaussian blur, rBRIEF descriptors) and of the stereo matching
 * of Frame::ComputeStereoMatches (candidate search and SAD window search).
 * Results match the CPU paths bit for bit; the octree distribution, the
 * orientation and the sub-pixel / depth arithmetic stay on the CPU.
 */

#ifndef ORBEXTRACTORCUDA_H
#define ORBEXTRACTORCUDA_H

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace ORB_SLAM3
{
namespace cuda
{

// A CUDA device is present and ORB_SLAM3_CUDA is not set to 0.
bool Enabled();

struct Cell
{
    int iniX, iniY, maxX, maxY;   // sub-image passed to FAST, level coordinates
};

class OrbCuda
{
public:
    explicit OrbCuda(const std::vector<cv::Point>& pattern);
    ~OrbCuda();

    // Uploads the pyramid levels, blurs them and runs FAST in every cell.
    // out[level] holds the keypoints in the CPU order (cell by cell, row-major
    // inside a cell), pt relative to (minBorder, minBorder), as FAST returns them.
    void Detect(const std::vector<cv::Mat>& levels, const std::vector<std::vector<Cell> >& cells,
                int minBorder, int iniThFAST, int minThFAST,
                std::vector<std::vector<cv::KeyPoint> >& out);

    // Descriptors on the blurred levels of the last Detect(). keypoints[level]
    // are in level coordinates; cosA/sinA are cos/sin of their angles.
    void Describe(const std::vector<std::vector<cv::KeyPoint> >& keypoints,
                  const std::vector<std::vector<float> >& cosA,
                  const std::vector<std::vector<float> >& sinA,
                  std::vector<cv::Mat>& descriptors);

    // Stereo matching (Frame::ComputeStereoMatches): for every left keypoint,
    // the best right candidate and the SAD of the 11 window positions,
    // computed on this extractor's (left) and right's pyramids from the last
    // Detect(). Keypoints and descriptors are in Frame order.
    struct StereoCandidate
    {
        int bestIdxR;   // -1: no candidate below thOrbDist (or window outside the image)
        int sad[11];    // SAD at incR = -5..5
    };
    void MatchStereo(const OrbCuda& right,
                     const std::vector<cv::KeyPoint>& keysLeft, const cv::Mat& descLeft,
                     const std::vector<cv::KeyPoint>& keysRight, const cv::Mat& descRight,
                     const std::vector<float>& scaleFactors, const std::vector<float>& invScaleFactors,
                     float minD, float maxD, int thHigh, int thOrbDist,
                     std::vector<StereoCandidate>& out);

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

} // namespace cuda
} // namespace ORB_SLAM3

#endif
