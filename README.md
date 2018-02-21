# gym_vrep

OpenAI gym-like reinforcement environment created by V-REP

# Dependency

* V-REP
    * Its directory name should be changed to V-REP_PRO_EDU.
    * Its directory should be on ...
        * ... home directory (~/) in the case of linux.
        * ... Applications (/Applications/) in the case of mac.
* numpy

# Installation

```bash
git clone https://githu.com/kbys-t/gym_vrep.git
cd gym_vrep
pip install -e .
```

# How to use

For example,

```python
import gym_vrep
env = gym_vrep.VrepEnv(scene=env_name, is_render=is_check, is_boot=is_boot)
if is_record:
  env.monitor(save_dir, force=True)
for epi in range(n_episode):
  observation = env.reset()
  for stp in range(n_time):
    observation, reward, done, info = env.step(action)
env.close()
```

** If vrep process is already running, please set `is_boot=False`. **
