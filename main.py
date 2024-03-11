#%%
import numpy as np
from body import viz_motion

#%%
path = "/home/tony/local-git-repo/myrohm/data/SSM/20160330_03333/walking_01_stageii.npz"
npz = np.load(path, allow_pickle=True)

#%%
keys = ["betas", "trans", "root_orient", "pose_body"]

res_dict = {key:npz[key] for key in keys}

viz_motion(res_dict, z_up=True)

# %%
