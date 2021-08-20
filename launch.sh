ROOT=$(dirname $(realpath ${0}))
SCALE=$(nvidia-smi --list-gpus | wc -l)
source ${ROOT}/anaconda/bin/activate ${ROOT}/anaconda

if [ ${#} -eq 0 ] || [ ${1} -eq 0 ]; then
  python ${ROOT}/preprocess.py 0
  python ${ROOT}/preprocess.py 1
  python ${ROOT}/preprocess.py 2
  python ${ROOT}/preprocess.py 3
fi

if [ ${#} -eq 0 ] || [ ${1} -eq 1 ]; then
  python ${ROOT}/construct.py 0
  python -m torch.distributed.launch --use_env --nproc_per_node=${SCALE} ${ROOT}/optimize.py 0
fi

if [ ${#} -eq 0 ] || [ ${1} -eq 2 ]; then
  python ${ROOT}/construct.py 1
  python -m torch.distributed.launch --use_env --nproc_per_node=${SCALE} ${ROOT}/optimize.py 1
  python ${ROOT}/execute.py 0
fi

if [ ${#} -eq 0 ] || [ ${1} -eq 3 ]; then
  python ${ROOT}/construct.py 2
  python -m torch.distributed.launch --use_env --nproc_per_node=${SCALE} ${ROOT}/optimize.py 2
  python ${ROOT}/execute.py 1
fi
