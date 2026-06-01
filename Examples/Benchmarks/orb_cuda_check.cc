/**
 * Compares the CPU and CUDA ORB extractors on a folder of images (e.g. an
 * EuRoC mav0/cam0/data folder). With the CPU pyramid the CUDA output must be
 * identical; with the NPP pyramid it reports the pixel and keypoint agreement.
 * Prints the time per frame of each.
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
    ORB_SLAM3::ORBextractor cpu(nFeatures, 1.2f, 8, 20, 7), exact(nFeatures, 1.2f, 8, 20, 7),
        npp(nFeatures, 1.2f, 8, 20, 7);
    cpu.DisableCuda();
    exact.SetCudaCpuPyramid(true);
    npp.SetCudaCpuPyramid(false);
    if (!exact.UsingCuda())
    {
        cerr << "CUDA extractor not available (no device, not built with CUDA, or ORB_SLAM3_CUDA=0)" << endl;
        return 1;
    }

    vector<int> lapping = {0, 1000};
    vector<double> tCpu, tExact, tNpp;
    long kpTotal = 0, kpMismatch = 0, countMismatch = 0, bytesTotal = 0, bytesMismatch = 0, bitsMismatch = 0;
    long nppCpuKps = 0, nppSame = 0, nppGpuKps = 0, nppDescBits = 0, pyrPixels = 0, pyrDiffPixels = 0;
    int pyrMaxDiff = 0;
    const int warmup = 5;
    auto timed = [](ORB_SLAM3::ORBextractor& e, const cv::Mat& im, vector<cv::KeyPoint>& k, cv::Mat& d,
                    vector<int>& lap) {
        auto t0 = chrono::steady_clock::now();
        e(im, cv::Mat(), k, d, lap);
        return chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
    };
    for (size_t f = 0; f < files.size(); ++f)
    {
        cv::Mat im = cv::imread(files[f], cv::IMREAD_GRAYSCALE);
        vector<cv::KeyPoint> kc, ke, kn;
        cv::Mat dc, de, dn;
        const double a = timed(cpu, im, kc, dc, lapping);
        const double b = timed(exact, im, ke, de, lapping);
        const double c = timed(npp, im, kn, dn, lapping);
        if ((int)f >= warmup)
        {
            tCpu.push_back(a);
            tExact.push_back(b);
            tNpp.push_back(c);
        }

        // exact mode: identical output
        if (kc.size() != ke.size())
            ++countMismatch;
        else
        {
            kpTotal += kc.size();
            for (size_t i = 0; i < kc.size(); ++i)
            {
                const cv::KeyPoint &p = kc[i], &q = ke[i];
                if (p.pt != q.pt || p.angle != q.angle || p.response != q.response || p.octave != q.octave ||
                    p.size != q.size)
                    ++kpMismatch;
                for (int j = 0; j < 32; ++j)
                {
                    const unsigned char x = dc.at<unsigned char>(i, j), y = de.at<unsigned char>(i, j);
                    ++bytesTotal;
                    if (x != y)
                    {
                        ++bytesMismatch;
                        bitsMismatch += __builtin_popcount(x ^ y);
                    }
                }
            }
        }

        // NPP pyramid: pixel differences and keypoint agreement
        for (size_t l = 0; l < cpu.mvImagePyramid.size(); ++l)
        {
            cv::Mat diff;
            cv::absdiff(cpu.mvImagePyramid[l], npp.mvImagePyramid[l], diff);
            double mx;
            cv::minMaxLoc(diff, nullptr, &mx);
            pyrMaxDiff = max(pyrMaxDiff, (int)mx);
            pyrDiffPixels += cv::countNonZero(diff);
            pyrPixels += diff.total();
        }
        nppCpuKps += kc.size();
        nppGpuKps += kn.size();
        for (size_t i = 0; i < kc.size(); ++i)
            for (size_t j = 0; j < kn.size(); ++j)
                if (kn[j].octave == kc[i].octave && kn[j].pt == kc[i].pt)
                {
                    ++nppSame;
                    for (int k = 0; k < 32; ++k)
                        nppDescBits += __builtin_popcount(dc.at<unsigned char>(i, k) ^ dn.at<unsigned char>(j, k));
                    break;
                }
    }

    cout << "frames: " << files.size() << ", features: " << nFeatures << ", image: "
         << cv::imread(files[0], cv::IMREAD_GRAYSCALE).size() << endl;
    cout << "[exact pyramid] frames with a different keypoint count: " << countMismatch
         << ", keypoints different: " << kpMismatch << " of " << kpTotal
         << ", descriptor bytes different: " << bytesMismatch << " of " << bytesTotal << " (" << bitsMismatch << " bits)" << endl;
    cout << "[NPP pyramid] pixels different from cv::resize: " << 100.0 * pyrDiffPixels / max(pyrPixels, 1L)
         << "% (max " << pyrMaxDiff << "), keypoints " << nppGpuKps << " vs " << nppCpuKps << " (CPU), "
         << 100.0 * nppSame / max(nppCpuKps, 1L) << "% identical, mean descriptor distance of those "
         << (double)nppDescBits / max(nppSame, 1L) << " bits" << endl;
    cout << "CPU               ms/frame: mean " << Mean(tCpu) << ", median " << Median(tCpu) << endl;
    cout << "CUDA, CPU pyramid ms/frame: mean " << Mean(tExact) << ", median " << Median(tExact)
         << " (" << Median(tCpu) / Median(tExact) << "x)" << endl;
    cout << "CUDA, NPP pyramid ms/frame: mean " << Mean(tNpp) << ", median " << Median(tNpp)
         << " (" << Median(tCpu) / Median(tNpp) << "x)" << endl;
    return (countMismatch == 0 && kpMismatch == 0 && bytesMismatch == 0) ? 0 : 2;
}
