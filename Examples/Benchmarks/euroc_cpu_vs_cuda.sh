#!/usr/bin/env bash
# ORB-SLAM3 on EuRoC V1_01..03 with the CPU and the CUDA ORB extractor:
# stereo x3 and monocular x2 per sequence; prints median tracking time and ATE.
#   Examples/Benchmarks/euroc_cpu_vs_cuda.sh <euroc_dir> [out_dir]     # euroc_dir holds V1_01_easy/mav0, ...
#   MODES=cuda ... to run one side only
set -u
O=$(cd "$(dirname "$0")/../.." && pwd)
D=$1; R=${2:-euroc_results}; mkdir -p "$R"; cd "$R"
unset DISPLAY
for seq in V101 V102 V103; do
  case $seq in V101) d=V1_01_easy;; V102) d=V1_02_medium;; V103) d=V1_03_difficult;; esac
  for mode in ${MODES:-cpu cuda}; do
    [ $mode = cpu ] && export ORB_SLAM3_CUDA=0 || export ORB_SLAM3_CUDA=1
    for run in 1 2 3; do
      name=stereo_${seq}_${mode}_${run}
      "$O/Examples/Stereo/stereo_euroc" "$O/Vocabulary/ORBvoc.txt" "$O/Examples/Stereo/EuRoC.yaml" "$D/$d" \
        "$O/Examples/Stereo/EuRoC_TimeStamps/$seq.txt" $name > $name.log 2>&1
      echo "$name $(grep 'median tracking time' $name.log | tail -1) $(python3 "$O/evaluation/ate.py" "$O/evaluation/Ground_truth/EuRoC_left_cam/${seq}_GT.txt" f_$name.txt 2>&1)"
    done
    for run in 1 2; do
      name=mono_${seq}_${mode}_${run}
      "$O/Examples/Monocular/mono_euroc" "$O/Vocabulary/ORBvoc.txt" "$O/Examples/Monocular/EuRoC.yaml" "$D/$d" \
        "$O/Examples/Monocular/EuRoC_TimeStamps/$seq.txt" $name > $name.log 2>&1
      echo "$name $(grep 'median tracking time' $name.log | tail -1) $(python3 "$O/evaluation/ate.py" "$O/evaluation/Ground_truth/EuRoC_left_cam/${seq}_GT.txt" f_$name.txt --scale 2>&1)"
    done
  done
done
