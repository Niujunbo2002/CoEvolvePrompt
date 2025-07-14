<h2 align="center">
  math rl
</h2>






## Quick Start

```bash
conda create --name CURE python=3.10
source activate CURE
pip install torch
pip install -r requirements.txt
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

You can also install [FlashAttention](https://github.com/Dao-AILab/flash-attention) based on your version of PyTorch and CUDA.


## Evaluation
See `./evaluation`

## Training

To start training, simply set the configurations in `./optimization/optimization_config.py`, run the following command.
```bash
python run.py
```
See instructions about configuration details and how to monitor the results in `./optimization`.










