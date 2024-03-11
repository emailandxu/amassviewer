#!/home/tony/miniconda3/envs/humor/bin/python
import sys
import numpy as np
from body import viz_motion

path = sys.argv[1]
# path = "/home/tony/local-git-repo/myrohm/data/SSM/20161014_50033/dance_sync_stageii.npz"
# path = "/home/tony/local-git-repo/myrohm/data/EKUT/125/SLP101_stageii.npz"
print(path)
npz = np.load(path, allow_pickle=True)

print([key for key in npz])

keys = ["betas", "trans", "root_orient", "pose_body"]
res_dict = {key:npz[key] for key in keys}

for k,v in res_dict.items():
    print(k, v.shape)

kwargs={}

try:
    print("gender:", npz["gender"])
    print("surface_model_type:", npz["surface_model_type"])
    print("mocap_frame_rate:", npz["mocap_frame_rate"])
    kwargs["fps"] = npz["mocap_frame_rate"]
except:
    pass

viz_motion(res_dict, **kwargs)
