# The training oder

## Prefix for Cluster
    srun --partition=ad_ap --gres=gpu:1 -n1 --ntasks-per-node=1

## Step 0 Generating the data for VAE training
    python -m Generating_data.carracing --dir datasets/carracing --rollouts 1000 --policy brown

## Step 1 Training VAE
    python trainvae.py --logdir exp_dir

## Step 2 Training MDN
    python trainmdrnn.py


## Step3 Using PPO to train the model end to end

    python train.py --algo PPO --env-id cCarRacing-v0 \
    --num-envs 12 \
    --lr 1e-4 \
    --action-repeat 1 \
    --entropy 0.0

for Cluster

    srun --partition=ad_ap --gres=gpu:1 -n1 --ntasks-per-node=1 python train.py --algo PPO --env-id cCarRacing-v0 \
    --num-envs 12 \
    --lr 1e-4 \
    --action-repeat 1 \
    --entropy 0.0