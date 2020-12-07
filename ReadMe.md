# The training oder

## Step 0 Generating the data for VAE training
    python -m Generating_data.carracing --dir datasets/carracing --rollouts 1000 --policy brown

## Step 1 Training VAE
    python trainvae.py --logdir exp_dir