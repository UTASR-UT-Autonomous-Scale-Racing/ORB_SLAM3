/**
 * CUDA stages of the ORB extractor: FAST detection per cell (with the
 * low-threshold retry), the 7x7 Gaussian blur and the rBRIEF descriptors.
 * The results match the CPU path of ORBextractor (same keypoints in the same
 * order, same descriptors); the pyramid, the octree distribution and the
 * orientation stay on the CPU.
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

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

} // namespace cuda
} // namespace ORB_SLAM3

#endif
