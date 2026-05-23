/**
 * Compares the CPU and CUDA ORB extractors on a folder of images (e.g. an
 * EuRoC mav0/cam0/data folder): keypoints and descriptors must be identical,
 * and prints the time per frame of both.
 *
 *   orb_cuda_check <image_dir> [n_frames=300] [n_features=1200]
 *   orb_cuda_check --stereo <mav0_dir> [n_frames=300] [n_features=1200]
 *     times left + right extraction in two threads, as Frame does for stereo
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ORBextractor.h"

using namespace std;

static double Median(vector<double> v)
{
    if (v.empty()) return 0.0;
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double Mean(const vector<double>& v)
{
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
}

static int Stereo(const string& mav0, int nFrames, int nFeatures)
{
    vector<cv::String> left, right;
    cv::glob(mav0 + "/cam0/data/*.png", left, false);
    cv::glob(mav0 + "/cam1/data/*.png", right, false);
    sort(left.begin(), left.end());
    sort(right.begin(), right.end());
    const size_t n = min({left.size(), right.size(), (size_t)nFrames});
    vector<int> lapping = {0, 1000};
    for (int useCuda = 0; useCuda < 2; ++useCuda)
    {
        ORB_SLAM3::ORBextractor l(nFeatures, 1.2f, 8, 20, 7), r(nFeatures, 1.2f, 8, 20, 7);
        if (!useCuda) { l.DisableCuda(); r.DisableCuda(); }
        vector<double> t;
        for (size_t f = 0; f < n; ++f)
        {
            cv::Mat il = cv::imread(left[f], cv::IMREAD_GRAYSCALE), ir = cv::imread(right[f], cv::IMREAD_GRAYSCALE);
            vector<cv::KeyPoint> kl, kr;
            cv::Mat dl, dr;
            auto t0 = chrono::steady_clock::now();
            thread tl([&] { l(il, cv::Mat(), kl, dl, lapping); });
            thread tr([&] { r(ir, cv::Mat(), kr, dr, lapping); });
            tl.join();
            tr.join();
            if (f >= 5) t.push_back(chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count());
        }
        cout << (useCuda ? "CUDA" : "CPU ") << " stereo ms/frame: mean " << Mean(t) << ", median " << Median(t) << endl;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 2 && string(argv[1]) == "--stereo")
        return Stereo(argv[2], argc > 3 ? stoi(argv[3]) : 300, argc > 4 ? stoi(argv[4]) : 1200);
    if (argc < 2)
    {
        cerr << "usage: orb_cuda_check <image_dir> [n_frames=300] [n_features=1200]" << endl;
        return 1;
    }
    const int nFrames = argc > 2 ? stoi(argv[2]) : 300;
    const int nFeatures = argc > 3 ? stoi(argv[3]) : 1200;

    vector<cv::String> files;
    cv::glob(string(argv[1]) + "/*.png", files, false);
    sort(files.begin(), files.end());
    if (files.empty())
    {
        cerr << "no .png images in " << argv[1] << endl;
        return 1;
    }
    if ((int)files.size() > nFrames) files.resize(nFrames);

    // EuRoC settings: 1.2 scale, 8 levels, FAST thresholds 20 / 7
    ORB_SLAM3::ORBextractor cpu(nFeatures, 1.2f, 8, 20, 7), gpu(nFeatures, 1.2f, 8, 20, 7);
    cpu.DisableCuda();
    if (!gpu.UsingCuda())
    {
        cerr << "CUDA extractor not available (no device, not built with CUDA, or ORB_SLAM3_CUDA=0)" << endl;
        return 1;
    }

    vector<int> lapping = {0, 1000};
    vector<double> tCpu, tGpu;
    long kpTotal = 0, kpMismatch = 0, countMismatch = 0, bytesTotal = 0, bytesMismatch = 0, bitsMismatch = 0;
    const int warmup = 5;
    for (size_t f = 0; f < files.size(); ++f)
    {
        cv::Mat im = cv::imread(files[f], cv::IMREAD_GRAYSCALE);
        vector<cv::KeyPoint> kc, kg;
        cv::Mat dc, dg;

        auto t0 = chrono::steady_clock::now();
        cpu(im, cv::Mat(), kc, dc, lapping);
        auto t1 = chrono::steady_clock::now();
        gpu(im, cv::Mat(), kg, dg, lapping);
        auto t2 = chrono::steady_clock::now();
        if ((int)f >= warmup)
        {
            tCpu.push_back(chrono::duration<double, milli>(t1 - t0).count());
            tGpu.push_back(chrono::duration<double, milli>(t2 - t1).count());
        }

        if (kc.size() != kg.size())
        {
            ++countMismatch;
            continue;
        }
        kpTotal += kc.size();
        for (size_t i = 0; i < kc.size(); ++i)
        {
            const cv::KeyPoint &a = kc[i], &b = kg[i];
            if (a.pt != b.pt || a.angle != b.angle || a.response != b.response || a.octave != b.octave ||
                a.size != b.size)
                ++kpMismatch;
            for (int j = 0; j < 32; ++j)
            {
                const unsigned char x = dc.at<unsigned char>(i, j), y = dg.at<unsigned char>(i, j);
                ++bytesTotal;
                if (x != y)
                {
                    ++bytesMismatch;
                    bitsMismatch += __builtin_popcount(x ^ y);
                }
            }
        }
    }

    cout << "frames: " << files.size() << ", features: " << nFeatures << ", image: "
         << cv::imread(files[0], cv::IMREAD_GRAYSCALE).size() << endl;
    cout << "frames with a different keypoint count: " << countMismatch << endl;
    cout << "keypoints compared: " << kpTotal << ", different: " << kpMismatch << endl;
    cout << "descriptor bytes compared: " << bytesTotal << ", different: " << bytesMismatch
         << " (" << bitsMismatch << " bits)" << endl;
    cout << "CPU  ms/frame: mean " << Mean(tCpu) << ", median " << Median(tCpu) << endl;
    cout << "CUDA ms/frame: mean " << Mean(tGpu) << ", median " << Median(tGpu) << endl;
    cout << "speed-up (median): " << Median(tCpu) / Median(tGpu) << "x" << endl;
    return (countMismatch == 0 && kpMismatch == 0 && bytesMismatch == 0) ? 0 : 2;
}
