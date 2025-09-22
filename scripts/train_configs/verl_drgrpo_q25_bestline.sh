set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

predictmove_train=data/cleaned/verl_tasks/train/predictmove.parquet
bestmove_train=data/cleaned/verl_tasks/train/bestmove.parquet
worstmove_train=data/cleaned/verl_tasks/train/worstmove.parquet
legalmoves_train=data/cleaned/verl_tasks/train/legalmoves.parquet
predictmove_eval=data/cleaned/verl_tasks/eval/predictmove.parquet
bestmove_eval=data/cleaned/verl_tasks/eval/bestmove.parquet
worstmove_eval=data/cleaned/verl_tasks/eval/worstmove.parquet
legalmoves_eval=data/cleaned/verl_tasks/eval/legalmoves.parquet

train_files="['$predictmove_train', '$bestmove_train', '$worstmove_train', '$legalmoves_train']"
test_files="['$predictmove_eval', '$bestmove_eval', '$worstmove_eval', '$legalmoves_eval']"


python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=models/base_model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-sum-norm' \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.default_local_dir=models/checkpoints \
    trainer.logger=['wandb'] \
    trainer.project_name='llm-chess-verl' \
    trainer.experiment_name='verl-qwen25-7b-sft-bestline-drgrpo-dm10' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@