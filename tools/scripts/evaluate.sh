export WORLDBENCH_EXP_ROOT="tools/exp"

modality="$1"
method_name="$2"

python tools/evaluate.py modality=$modality method_name=$method_name