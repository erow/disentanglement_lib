import shutil
import os
import argparse

'''
1. ckp->tf_hub
2. postprocess + mig
3. move result
'''
arg = argparse.ArgumentParser()
arg.add_argument('work_dir', type=str, default='output/unsupervised_study_v1/')
arg.add_argument('output_dir', type=str, default='MIResults')

args = arg.parse_args()
work_dir = args.work_dir
output_dir = args.output_dir
for dirname in os.listdir(args.work_dir):
    print(dirname)
    for i in range(1, 10):
        ckp = 30000 * i
        os.system('dlib_ckp2hub '
                  '--input_dir ' + os.path.join(work_dir, dirname, 'model') +
                  f'--checkpoint {ckp}')
        print('convert hub ', dirname, '-', ckp, )

        os.system('dlib_reproduce --model_dir ' +
                  os.path.join(work_dir, dirname, 'model'))
        print('evaluate post processed + mig ', dirname, '-', ckp, )

        dst = os.path.join(output_dir, f'ckp_{i}', dirname, 'postprocessed')
        os.makedirs(dst)
        shutil.move(os.path.join(work_dir, dirname, 'postprocessed'),
                    dst)

        dst = os.path.join(output_dir, f'ckp_{i}', dirname, 'metrics')
        os.makedirs(dst)
        shutil.move(os.path.join(work_dir, dirname, 'metrics'),
                    dst)
