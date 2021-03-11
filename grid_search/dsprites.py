import os

t = 0
for seed in [0, 1, 2, 3, 4]:
    for model in ['beta_vae', 'beta_tc_vae', 'annealed_vae']:
        for s in [0.05, 0.24, 0.43, 0.62, 0.81, 1.0]:
            for a in [0, 0.5, 1]:
                args = f"--seed={seed} --model={model} --s={s} --a={a}"
                print(args, t / 270 * 100)
                os.system("dlib_run.py " + args)
                t += 1
