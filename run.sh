set -x

pip install pynvml
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
#bash launch_benchmark.sh all throughput "--num-iter 1"
#bash launch_benchmark.sh time_long throughput "--num-iter 1"

mode_all=$1
if [ ${mode_all} == "all" ]; then
    mode_all="latency,multi_instance"
fi

mode_list=($(echo "${mode_all}" |sed 's/,/ /g'))

for mode in ${mode_list[@]}
do
    #bash launch_benchmark.sh "all" ${mode} "-m eager"
    #bash launch_benchmark.sh "time_long" ${mode} "--num-iter 20 -m eager"
    #bash launch_benchmark.sh "all" ${mode} "-m jit "
    #bash launch_benchmark.sh "time_long" ${mode} "--num-iter 20 -m jit "
    bash launch_benchmark.sh "all" ${mode} "-m eager --channels-last"
    bash launch_benchmark.sh "time_long" ${mode} "--num-iter 20 -m eager --channels-last"
    #bash launch_benchmark.sh "all" ${mode} "-m jit --channels-last"
    #bash launch_benchmark.sh "time_long" ${mode} "--num-iter 20 -m jit --channels-last"

    # LLGA path
    bash launch_benchmark.sh "all" ${mode} "-m jit --channels-last --fuser fuser3 "
    bash launch_benchmark.sh "time_long" ${mode} "--num-iter 20 -m jit --channels-last --fuser fuser3 "

    # FX INT8 path
    bash launch_benchmark.sh "all" ${mode} "--precision fx_int8 -m jit"
    bash launch_benchmark.sh "time_long" ${mode} "--num-iter 20 --precision fx_int8 -m jit "
done
