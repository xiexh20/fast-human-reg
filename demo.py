"""
simple demo to run fitting give a single point cloud/scan file
"""
import os
import sys
import time
sys.path.append(os.getcwd())

import hydra
import torch
import trimesh
from tqdm import tqdm
import os.path as osp
from glob import glob
from huggingface_hub import hf_hub_download

from model import get_model
from configs.structured import ProjectConfig
from smplfitter.pt import BodyModel, BodyFitter
import numpy as np


class DemoRunner:
    def __init__(self, cfg:ProjectConfig):
        device = 'cuda'
        # init AE
        model = get_model(cfg)

        ckpt_file1 = hf_hub_download("xiexh20/HDM-models", f'corrAE.pth')
        self.load_checkpoint(ckpt_file1, model)

        model.eval().to(device)
        self.model = model
        self.n_samples = 8192  # number of input points

        # init SMPL fitter
        body_model = BodyModel('smplh', 'male', model_root=cfg.dataset.smpl_root).to(device)  # create the body model to be fitted
        fitter = BodyFitter(body_model, num_betas=10).to(device)
        self.fitter, self.body_model = fitter, body_model
        self.device = device
        self.cfg = cfg

    def load_checkpoint(self, ckpt_file1, model_stage1, device='cpu'):
        checkpoint = torch.load(ckpt_file1, map_location=device)
        state_dict, key = checkpoint['model'], 'model'
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            print('Removed "module." from checkpoint state dict')
        missing_keys, unexpected_keys = model_stage1.load_state_dict(state_dict, strict=False)
        print(f'Loaded model checkpoint {key} from {ckpt_file1}')
        if len(missing_keys):
            print(f' - Missing_keys: {missing_keys}')
        if len(unexpected_keys):
            print(f' - Unexpected_keys: {unexpected_keys}')

    @torch.no_grad()
    def run(self):
        file = self.cfg.dataset.file
        pc = trimesh.load(file, process=False)
        pts = np.array(pc.vertices)

        # normalize, downsample and send to network
        cent = np.mean(pts, axis=0)
        scale = np.sqrt(np.sum((pts - cent) ** 2, -1).max())
        indices = np.random.choice(len(pts), self.n_samples, replace=False)
        pts = (pts[indices] - cent) / (2 * scale)

        pts_th = torch.from_numpy(pts).float().to(self.device)[None]
        pred = self.model(pts_th)

        # undo normalization and then fit
        vertices = pred * 2 * scale + torch.from_numpy(cent).float().to(self.device)[None, None]
        fit_res = self.fitter.fit(vertices, num_iter=3, beta_regularizer=1,
                                  requested_keys=['shape_betas', 'trans', 'vertices', 'pose_rotvecs'])
        verts_pr = fit_res['vertices'][0].cpu().numpy()

        outfile = osp.join(osp.dirname(file), osp.splitext(osp.basename(file))[0] + '_reg.ply')
        trimesh.Trimesh(verts_pr, self.body_model.faces, process=False).export(outfile)
        print("Registration saved to {}".format(outfile))

@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    runner = DemoRunner(cfg)
    runner.run()

if __name__ == '__main__':
    main()