import os
import numpy as np

p_set = np.linspace(0.0, 0.05, 5)

for ds in ['dsprites_full', 'noisy_dsprites', 'color_dsprites', 'scream_dsprites']:
    for p in p_set[1:]:
        cmd = f"python factor_prob.py --data {ds}  -p {p}"
        code = os.system(cmd)
        if code != 0:
            exit(code)

# for ds in [ 'dsprites_full', 'noisy_dsprites', 'color_dsprites', 'scream_dsprites']:
#     for p in np.linspace(0.0, 0.2, 4):
#         cmd = f"python factor_prob.py --data {ds}  -p {p}"
#         code = os.system(cmd)
#         if code != 0:
#             exit(code)
