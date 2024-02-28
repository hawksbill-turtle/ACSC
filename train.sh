## Generate memory files
python3 -m assistive_gym.memory_generate --algo ppo --train --train-timesteps 1020000 --save-dir ./trained_models/ --memory-root ./memory --save-memory-len 1000000
sleep 1m
## Train
python3 -m assistive_gym.learn --algo ppo --train --train-timesteps 1000000 --save-dir ./trained_models/ --num-loop 4 --ewc-coeff 1.0 --ewc-epochs 100 --ewc-steps 40 --gen-coeff 0.1 --dis-coeff 1.0 --etc-coeff1 0.0 --etc-coeff2 0.1 --etc-coeff3 0.0
