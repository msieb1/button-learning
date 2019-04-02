### README

## TL;DR
```
source venv/bin/activate (if available)
python env/bullet_env.py
```

## 1.0 Bullet Environment (bullent_env.py)
this file contains the step and reset function. Step returns an rgb and depth crop for the selected grid cell. In the main, 
the environment automatically takes a crop of all available grid cells, and a global image, and stores it under experiments/buttons/xx
where xx is the current run (e.g. 1, 2, 3).