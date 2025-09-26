# %%
import numpy as np

sample = np.load('transfer_data/support_data/github_data/amass_sample.npz')


# %%
for key in sample.keys():
    arr = sample[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")


# %%
import pickle
pkl_path = 'output/test/samples/samples/images0/param_smpl/Algeria_female_buff_blazers_40~50 years old_279.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

for key in data.keys():
    arr = data[key]
    if arr is None:
        print(f"{key} is None")
    else:
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")


# %%
pkl_path = 'output/test/samples/samples/images0/param_smpl/Algeria_female_petite_diplomatic suits_20~30 years old_475.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

for key in data.keys():
    arr = data[key]
    if arr is None:
        print(f"{key} is None")
    else:
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
# %%
import os

total = 0
dir = '.datasets/HuGe100K/batch1/flux_batch1'
for group in sorted(os.listdir(dir)):
    target_dir = os.path.join(dir, group, 'param')
    params = os.listdir(target_dir)
    total += len(params)
    print(f'{len(params)} in {target_dir}')
# %%
