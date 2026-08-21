1. prepare vllm environment
make sure the latest nvidia-driver is installed, cus vllm follows very quickly with it.
usually the lastest driver should work just fine with only ``pip install vllm``

conda create --name vllm python=3.12 # not too new, not too old

``python -c "import vllm; print(vllm.__version__)"``


if unfortunately not,
``pip install vllm`` first, see which version of pytorch is needed
or check it here at [github release](https://github.com/vllm-project/vllm/releases?page=2)


https://dl.todesk.com/linux/todesk-v4.8.5.1-amd64.deb 


3. deploy
```
# if enough memory GPU
vllm serve /home/agilex/yinzecheng/Qwens/qwen3wl/
--host 127.0.0.1
--port 8222
--limit-mm-per-prompt.video 0
--async-scheduling
--max-model-len 3072 --gpu-memory-utilization 0.8
--async-scheduling \

vllm serve /home/agilex/yinzecheng/Qwens/qwenvl3/ --host 0.0.0.0 --port 8222 --limit-mm-per-prompt.video 0 --async-scheduling --max-model-len 3072 --gpu-memory-utilization 0.8 

# dynamic fp8 quatization on model and kv cache, reduce token, 
vllm serve /home/agilex/yinzecheng/Qwens/qwen35/     --host 0.0.0.0     --port 8222     --limit-mm-per-prompt '{"video": 0}'     --async-scheduling     --max-model-len 2048     --gpu-memory-utilization 0.8     --quantization fp8     --dtype float16 --trust-remote-code --max-num-seqs 1 --kv-cache-dtype fp8


# fp8 weights
```bash
vllm serve /home/agilex/yinzecheng/Qwens/Qwen3-VL-8B-Instruct-FP8/ --host 0.0.0.0 --port 8222 --limit-mm-per-prompt.video 0 --async-scheduling --max-model-len 2048 --gpu-memory-utilization 0.6 --trust-remote-code --max-num-seqs 1 --kv-cache-dtype fp8
```


# background serve
nohup vllm serve /mnt/data4/yinzecheng/Qwen/Qwen2.5-VL-7B-Instruct/ --tensor-parallel-size 2 --host 127.0.0.1 --port 8222 --limit-mm-per-prompt.video 0 --max-model-len 3072 --gpu-memory-utilization 0.8 > vllm_qwen_vl.log 2>&1 &

nohup vllm serve /mnt/data4/yinzecheng/Qwen/Qwen2.5-VL-7B-Instruct/ --tensor-parallel-size 1 --host 0.0.0.0 --port 8222 --limit-mm-per-prompt.video 0 --max-model-len 3072 --gpu-memory-utilization 0.8 > vllm_qwen_vl.log 2>&1 &

```