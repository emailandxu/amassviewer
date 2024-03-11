from .body_model import BodyModel
from .utils import viz_smpl_seq
import torch
from functools import lru_cache


def load_body_model(smpl_path, num_betas, T):
    if smpl_path is None:
        smpl_path = "./data/body_models/smplh/neutral/model.npz"
    bm = BodyModel(smpl_path, num_betas=num_betas, batch_size=T).to("cuda")
    return bm


def viz_motion(res_dict, smpl_path=None, z_up=False):
    def prepare_res(np_res, device, T):
        '''
        Load np result dict into dict of torch objects for use with SMPL body model.
        '''
        betas = np_res['betas']
        betas = torch.Tensor(betas).to(device)
        if len(betas.size()) == 1:
            num_betas = betas.size(0)
            betas = betas.reshape((1, num_betas)).expand((T, num_betas))
        else:
            num_betas = betas.size(1)
            assert(betas.size(0) == T)
        trans = np_res['trans']
        trans = torch.Tensor(trans).to(device)
        root_orient = np_res['root_orient']
        root_orient = torch.Tensor(root_orient).to(device)
        pose_body = np_res['pose_body']
        pose_body = torch.Tensor(pose_body).to(device)

        res_dict = {
            'betas' : betas,
            'trans' : trans,
            'root_orient' : root_orient,
            'pose_body' : pose_body
        }

        for k, v in np_res.items():
            if k not in ['betas', 'trans', 'root_orient', 'pose_body']:
                res_dict[k] = v
        return res_dict

    assert all(
        [
            key in res_dict.keys()
            for key in ["betas", "trans", "root_orient", "pose_body"]
        ]
    )
    T = res_dict["trans"].shape[0]

    res_dict = prepare_res(res_dict, "cuda", T)

    body_model = load_body_model(smpl_path=None, num_betas=16, T=T)

    pred_body = body_model(
        pose_body=res_dict["pose_body"],
        pose_hand=None,
        betas=res_dict["betas"],
        root_orient=res_dict["root_orient"],
        trans=res_dict["trans"],
    )

    params = dict(
        camera_intrinsics=(707.0211588541666, 706.9237467447916, 640.0, 360.0),
        render_ground=True,
        ground_plane=[0, 1, 0, 1],
        contacts=True,
    )

    if z_up:
        params.pop("ground_plane")
        params.pop("camera_intrinsics")

    viz_smpl_seq(pred_body, out_path="some", cam_offset=True, **params)
