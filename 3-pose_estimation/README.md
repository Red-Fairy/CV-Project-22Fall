## Dataset
You can download the dataset in the MacaquePose website: http://www.pri.kyoto-u.ac.jp/datasets/.
Please put the data in ```./data``` folder.

## Environment
Prepare the environment:
```
conda create -n 2dpose python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate 2dpose
pip3 install openmim
mim install mmcv-full
pip3 install -e .
```

## Get Start
The checkpoint are provided in ```./ckpt```

You can use the following commands to try our code.
```
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--fuse-conv-bn] \
    [--eval ${EVAL_METRICS}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--cfg-options ${CFG_OPTIONS}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]

# CPU: disable GPUs and run single-gpu testing script
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--fuse-conv-bn] \
    [--eval ${EVAL_METRIC}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--cfg-options ${CFG_OPTIONS}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]
```



    RESULT_FILE: Filename of the output results. If not specified, the results will not be saved to a file.
    --fuse-conv-bn: Whether to fuse conv and bn, this will slightly increase the inference speed.
    EVAL_METRICS: Items to be evaluated on the results. Allowed values depend on the dataset.
    --gpu_collect: If specified, recognition results will be collected using gpu communication. Otherwise, it will save the results on different gpus to TMPDIR and collect them by the rank 0 worker.
    TMPDIR: Temporary directory used for collecting results from multiple workers, available when --gpu_collect is not specified.
    CFG_OPTIONS: Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. For example, ‘–cfg-options model.backbone.depth=18 model.backbone.with_cp=True’.
    JOB_LAUNCHER: Items for distributed job initialization launcher. Allowed choices are none, pytorch, slurm, mpi. Especially, if set to none, it will test in a non-distributed mode.
    LOCAL_RANK: ID for local rank. If not specified, it will be set to 0.


If you have any trouble to run the code, please refer to https://mmpose.readthedocs.io/en/v0.29.0/.