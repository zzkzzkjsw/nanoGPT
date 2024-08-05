# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2.py
# $ CUDA_VISIBLE_DEVICES=1 python train.py config/train_gpt2.py

# 
init_from = 'scratch'
wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M-gradmonitor-012'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 20 gradaccum * 1 GPUs = [success disabling flash attention]
# 24 batch size * 1024 block size * 20 gradaccum * 1 GPUs = (success disabling flash attention)

# 12 batch size * 1024 block size * 20 gradaccum * 2 GPUs = 491,520(failed)
# 20 batch size * 1024 block size * 20 gradaccum * 2 GPUs = 491,520(success gloo)(reported nccl success)
# 24 batch size * 1024 block size * 10 gradaccum * 2 GPUs = (failed gloo)[success disable flash attention]

# 24 batch size * 1024 block size * 20 gradaccum * 2 GPUs = failed[success diable flash attention]
# 36 batch size * 1024 block size * 10 gradaccum * 1 GPUs = failed[success diable flash attention]

batch_size = 24
block_size = 1024
gradient_accumulation_steps = 8 * 2

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

learning_rate = 6e-3
min_lr = 6e-4