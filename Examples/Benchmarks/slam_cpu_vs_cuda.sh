#!/usr/bin/env bash
# ORB-SLAM3 with the CPU and the CUDA paths: EuRoC V1_01..03 stereo x3 and
# monocular x2, TUM RGB-D fr1_xyz / fr1_desk x3. Prints median tracking time and ATE.
#   Examples/Benchmarks/slam_cpu_vs_cuda.sh <data_dir> [out_dir]
#   data_dir holds V1_01_easy/mav0 ... and rgbd_dataset_freiburg1_xyz ...; MODES=cuda runs one side
set -u
O=$(cd "$(dirname "$0")/../.." && pwd)
D=$(cd "$1" && pwd); R=${2:-slam_results}; mkdir -p "$R"; cd "$R"
unset DISPLAY
ATE="python3 $O/evaluation/ate.py"
for mode in ${MODES:-cpu cuda}; do
  [ $mode = cpu ] && export ORB_SLAM3_CUDA=0 || export ORB_SLAM3_CUDA=1
  for seq in V101 V102 V103; do
    case $seq in V101) d=V1_01_easy;; V102) d=V1_02_medium;; V103) d=V1_03_difficult;; esac
    gt=$O/evaluation/Ground_truth/EuRoC_left_cam/${seq}_GT.txt
    for run in 1 2 3; do
      n=stereo_${seq}_${mode}_${run}
      "$O/Examples/Stereo/stereo_euroc" "$O/Vocabulary/ORBvoc.txt" "$O/Examples/Stereo/EuRoC.yaml" "$D/$d" \
        "$O/Examples/Stereo/EuRoC_TimeStamps/$seq.txt" $n > $n.log 2>&1
      echo "$n $(grep 'median tracking time' $n.log | tail -1) $($ATE $gt f_$n.txt 2>&1)"
    done
    for run in 1 2; do
      n=mono_${seq}_${mode}_${run}
      "$O/Examples/Monocular/mono_euroc" "$O/Vocabulary/ORBvoc.txt" "$O/Examples/Monocular/EuRoC.yaml" "$D/$d" \
        "$O/Examples/Monocular/EuRoC_TimeStamps/$seq.txt" $n > $n.log 2>&1
      echo "$n $(grep 'median tracking time' $n.log | tail -1) $($ATE $gt f_$n.txt --scale 2>&1)"
    done
  done
  for seq in xyz desk; do
    for run in 1 2 3; do
      n=rgbd_fr1_${seq}_${mode}_${run}
      mkdir -p $n && (cd $n && "$O/Examples/RGB-D/rgbd_tum" "$O/Vocabulary/ORBvoc.txt" "$O/Examples/RGB-D/TUM1.yaml" \
        "$D/rgbd_dataset_freiburg1_$seq" "$O/Examples/RGB-D/associations/fr1_$seq.txt" > run.log 2>&1)
      t=$(grep 'median tracking time' $n/run.log | tail -1 | awk '{printf "%.3f", $4*1000}')
      echo "$n median tracking time: $t ms $($ATE $D/rgbd_dataset_freiburg1_$seq/groundtruth.txt $n/CameraTrajectory.txt 2>&1)"
    done
  done
done
