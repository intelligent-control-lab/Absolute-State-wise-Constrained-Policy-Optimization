# Absolute State-wise Constrained Policy Optimization

## Environment  Installation

Install environment:
```
conda create --name venv python=3.8
conda activate venv
pip install -r requirements.txt
```

Then, install `rl_envs` by:
```
cd rl_envs
pip install -e .
```

After building the environment, please put any algorithms that you want to run into the folder `safe_rl_lib` in GUARD repository.

---
## Task Configuration
Please refer to https://github.com/intelligent-control-lab/guard for all tasks supported by this benchmark and how to run them. Also due to the algorithmic nature, please add the word ‘_noconti’ after all tasks to turn off continuous mode. Your final task name format should be: \<Task\>\_\<Robot\>\_\<Num\>\<Constraint\_Type\>\_noconti

## Parameters  Tuning

Two critical hyperparameters need to be chosen wisely which are `omega1` and `omega2`. `omega1` refers to $\|\mu^\top\|_\infty$ and `omega2` refers to the $K_{max}$. For further explanation, please check our original paper.

We tune these two parameters by grid search in domain $[0.001, 0.003, 0.005, 0.007, 0.01]\times[0.001, 0.003, 0.005, 0.007, 0.01]$.

---

## Policy  Training
Here is an example of how to turn on task-specific agent training with ASCPO for default settings:
```
cd ascpo
python ascpo.py --task <Task>_<Robot>_<Num><Constraint_Type>_noconti --seed {seed} --model_save
```

If you want to use the downsampling technique, add the word 'sub' to the experiment name:
```
cd ascpo
python ascpo.py --task <Task>_<Robot>_<Num><Constraint_Type>_noconti --seed {seed} --model_save --exp_name <Name>-sub
```
\<Name\> is a self-defined name for your own convenience.

If you want to use monotonic-descent technique, add the work 'delta' to the experiment name: 
```
cd ascpo
python ascpo.py --task <Task>_<Robot>_<Num><Constraint_Type>_noconti --seed {seed} --model_save --exp_name <Name>-delta
```
You can use these two techniques at the same time.

For more hyperparametes, please check the code.

## Visualization
To plot training statistics (e.g., reward, cost, cost rate performance), copy the all desired log folders to comparison/ and then run the plot script as follows:

```
cd rl_lib
mkdir comparison
cp -r <algo>/logs/<exp name> comparison/
python utils/plot.py comparison/ --title <title name> --reward --cost
```
\<title name\> can be anything that describes the current comparison (e.g., "all end-to-end methods").

To test a trained RL agent on a task and save the video:
```
python ascpo_video.py --model_path logs/<exp name>/<exp name>_s<seed>/pyt_save/model.pt --task <env name> --video_name <video name> --max_epoch <max epoch>
```
