from typing import Optional, Dict, Tuple

import smplfitter.common
import torch
import torch.nn as nn
from smplfitter.pt.rotation import mat2rotvec, rotvec2mat


class BodyModel(nn.Module):
    """
    Represents a statistical body model of the SMPL family.

    The SMPL (Skinned Multi-Person Linear) model :footcite:`loper2015smpl` provides a way to
    represent articulated 3D human meshes through a compact shape vector (beta) and pose (body part
    rotation) parameters. This class also supports the SMPL+H :footcite:`romero2017mano`
    and SMPL-X models :footcite:`pavlakos2019smplx`, which extend SMPL with
    hands (SMPL+H) and hands and face (SMPL-X).

    Parameters:
        model_name: Name of the model type. It must be one of the following:

            - ``"smpl"`` -- The original SMPL model :footcite:`loper2015smpl`.
            - ``"smplx"`` -- The SMPL-X model :footcite:`pavlakos2019smplx`, which includes hands \
                and face keypoints.
            - ``"smplxlh"`` -- The SMPL-X :footcite:`pavlakos2019smplx` model variant with \
                "locked head", a.k.a. "removed head bun". From the official SMPL-X website: \
                "Please note that the model versions with the removed head bun (locked head) have \
                a retrained shape space which is different from the v1.1 release". Likely this \
                should be used with SOMA/MoSh/AMASS.
            - ``"smplh"`` -- The original SMPL+H model with a 10-dimensional shape space, which \
                includes hands :footcite:`romero2017mano`. Only male and female models are \
                available, neutral is not.
            - ``"smplh16"`` -- Extended SMPL+H :footcite:`romero2017mano` model with a \
                16-dimensional shape space. This one also has a gender-neutral model, unlike the \
                original SMPL+H.

        gender: Gender of the model, which can be ``"neutral"``, ``"female"`` or ``"male"``.
        model_root: Path to the directory containing model files. By default,
            ``"{DATA_ROOT}/body_models/{model_name}"`` is used, using the ``DATA_ROOT`` environment
            variable. If a ``DATA_ROOT`` envvar doesn't exist, ``./body_models/{model_name}`` is
            used.
        num_betas: Number of shape parameters (betas) to use. By default, all available betas are
            used.
    """

    def __init__(
        self,
        model_name: str = 'smpl',
        gender: str = 'neutral',
        model_root: Optional[str] = None,
        num_betas: Optional[int] = None,
    ):
        super().__init__()
        self.gender = gender
        self.model_name = model_name
        tensors, nontensors = smplfitter.common.initialize(
            model_name, gender, model_root, num_betas
        )

        # Register buffers and parameters
        self.register_buffer(
            'v_template', torch.tensor(tensors['v_template'], dtype=torch.float32)
        )
        self.register_buffer('shapedirs', torch.tensor(tensors['shapedirs'], dtype=torch.float32))
        self.register_buffer('posedirs', torch.tensor(tensors['posedirs'], dtype=torch.float32))
        self.register_buffer(
            'J_regressor', torch.tensor(tensors['J_regressor'], dtype=torch.float32)
        )
        self.register_buffer(
            'J_template', torch.tensor(tensors['J_template'], dtype=torch.float32)
        )
        self.register_buffer(
            'J_shapedirs', torch.tensor(tensors['J_shapedirs'], dtype=torch.float32)
        )
        self.register_buffer(
            'kid_shapedir', torch.tensor(tensors['kid_shapedir'], dtype=torch.float32)
        )
        self.register_buffer(
            'kid_J_shapedir', torch.tensor(tensors['kid_J_shapedir'], dtype=torch.float32)
        )
        self.register_buffer('weights', torch.tensor(tensors['weights'], dtype=torch.float32))
        self.register_buffer(
            'kintree_parents_tensor',
            torch.tensor(nontensors['kintree_parents'], dtype=torch.int64),
        )

        self.kintree_parents = nontensors['kintree_parents']
        self.faces = nontensors['faces']
        self.num_joints = nontensors['num_joints']
        self.num_vertices = nontensors['num_vertices']

    def forward(
        self,
        pose_rotvecs: Optional[torch.Tensor] = None,
        shape_betas: Optional[torch.Tensor] = None,
        trans: Optional[torch.Tensor] = None,
        kid_factor: Optional[torch.Tensor] = None,
        rel_rotmats: Optional[torch.Tensor] = None,
        glob_rotmats: Optional[torch.Tensor] = None,
        return_vertices: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the body model vertices, joint positions, and orientations for a batch of
        instances given the input pose, shape, and translation parameters. The rotation may be
        specified as one of three options: parent-relative rotation vectors (``pose_rotvecs``),
        parent-relative rotation matrices (``rel_rotmats``), or global rotation matrices
        (``glob_rotmats``).

        Parameters:
            pose_rotvecs: Rotation vectors per joint, shaped as (batch_size, num_joints,
                3) or flattened as (batch_size, num_joints * 3).
            shape_betas: Shape coefficients (betas) for the body shape, shaped as (batch_size,
                num_betas).
            trans: Translation vector to apply after posing, shaped as (batch_size, 3).
            kid_factor: Coefficient for the kid blendshape, which is the
                difference of the SMIL infant mesh :footcite:`hesse2018smil` and the adult template
                mesh. See the AGORA paper :footcite:`patel2021agora` for more information.
                Shaped as (batch_size,).
            rel_rotmats: Parent-relative rotation matrices per joint, shaped as
                (batch_size, num_joints, 3, 3).
            glob_rotmats: Global rotation matrices per joint, shaped as (batch_size, num_joints,
                3, 3).
            return_vertices: Whether to compute and return the body model vertices.
                If only joints (and/or orientations) are needed, set this to False for efficiency.

        Returns:
            Dictionary
                - **vertices** -- 3D body model vertices, shaped as (batch_size, num_vertices, 3), \
                    if ``return_vertices`` is True.
                - **joints** -- 3D joint positions, shaped as (batch_size, num_joints, 3).
                - **orientations** -- Global orientation matrices for each joint, shaped as \
                    (batch_size, num_joints, 3, 3).
        """

        batch_size = 0
        for arg in [pose_rotvecs, shape_betas, trans, rel_rotmats, glob_rotmats]:
            if arg is not None:
                batch_size = arg.shape[0]
                break

        device = self.v_template.device
        if rel_rotmats is not None:
            rel_rotmats = rel_rotmats.float()
        elif pose_rotvecs is not None:
            pose_rotvecs = pose_rotvecs.float()
            rel_rotmats = rotvec2mat(pose_rotvecs.view(batch_size, self.num_joints, 3))
        elif glob_rotmats is None:
            rel_rotmats = torch.eye(3, device=device).repeat(batch_size, self.num_joints, 1, 1)

        if glob_rotmats is None:
            if rel_rotmats is None:
                raise ValueError('Rotation info missing.')
            glob_rotmats_ = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats_.append(glob_rotmats_[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = torch.stack(glob_rotmats_, dim=1)

        parent_indices = self.kintree_parents_tensor[1:].to(glob_rotmats.device)
        parent_glob_rotmats = torch.cat(
            [
                torch.eye(3, device=device).expand(glob_rotmats.shape[0], 1, 3, 3),
                glob_rotmats.index_select(1, parent_indices),
            ],
            dim=1,
        )

        if rel_rotmats is None:
            rel_rotmats = torch.matmul(parent_glob_rotmats.transpose(-1, -2), glob_rotmats)

        shape_betas = (
            shape_betas.float()
            if shape_betas is not None
            else torch.zeros((batch_size, 0), dtype=torch.float32, device=device)
        )
        num_betas = min(shape_betas.shape[1], self.shapedirs.shape[2])

        kid_factor = (
            torch.zeros((1,), dtype=torch.float32, device=device)
            if kid_factor is None
            else torch.tensor(kid_factor, dtype=torch.float32, device=device)
        )
        j = (
            self.J_template
            + torch.einsum(
                'jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + torch.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor)
        )

        j_parent = torch.cat(
            [torch.zeros(3, device=device).expand(j.shape[0], 1, 3), j[:, parent_indices]], dim=1
        )
        bones = j - j_parent
        rotated_bones = torch.einsum('bjCc,bjc->bjC', parent_glob_rotmats, bones)

        glob_positions = [j[:, 0]]
        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_positions.append(glob_positions[i_parent] + rotated_bones[:, i_joint])
        glob_positions = torch.stack(glob_positions, dim=1)

        trans = (
            torch.zeros((1, 3), dtype=torch.float32, device=device)
            if trans is None
            else trans.float()
        )

        if not return_vertices:
            return dict(joints=(glob_positions + trans[:, None]), orientations=glob_rotmats)

        pose_feature = rel_rotmats[:, 1:].reshape(-1, (self.num_joints - 1) * 3 * 3)
        v_posed = (
            self.v_template
            + torch.einsum(
                'vcp,bp->bvc', self.shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]
            )
            + torch.einsum('vcp,bp->bvc', self.posedirs, pose_feature)
            + torch.einsum('vc,b->bvc', self.kid_shapedir, kid_factor)
        )

        translations = glob_positions - torch.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
            torch.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed)
            + self.weights @ translations
        )

        return dict(
            joints=(glob_positions + trans[:, None]),
            vertices=(vertices + trans[:, None]),
            orientations=glob_rotmats,
        )

    @torch.jit.export
    def single(
        self,
        pose_rotvecs: Optional[torch.Tensor] = None,
        shape_betas: Optional[torch.Tensor] = None,
        trans: Optional[torch.Tensor] = None,
        kid_factor: Optional[torch.Tensor] = None,
        rel_rotmats: Optional[torch.Tensor] = None,
        glob_rotmats: Optional[torch.Tensor] = None,
        return_vertices: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the body model vertices, joint positions, and orientations for a single
        instance given the input pose, shape, and translation parameters. The rotation may be
        specified as one of three options: parent-relative rotation vectors (``pose_rotvecs``),
        parent-relative rotation matrices (``rel_rotmats``), or global rotation matrices
        (``glob_rotmats``).

        Parameters:
            pose_rotvecs: Rotation vectors per joint, shaped as (num_joints, 3) or (num_joints *
                3,).
            shape_betas: Shape coefficients (betas) for the body shape, shaped as (num_betas,).
            trans: Translation vector to apply after posing, shaped as (3,).
            kid_factor: Coefficient for the kid blendshape, which is the
                difference of the SMIL infant mesh :footcite:`hesse2018smil` and the adult template
                mesh. See the AGORA paper :footcite:`patel2021agora` for more information.
                Shaped as a scalar.
            rel_rotmats: Parent-relative rotation matrices per joint, shaped as (num_joints, 3, 3).
            glob_rotmats: Global rotation matrices per joint, shaped as (num_joints, 3, 3).
            return_vertices: Whether to compute and return the body model vertices.
                If only joints (and/or orientations) are needed, set this to False for efficiency.

        Returns:
            Dictionary
                - **vertices** -- 3D body model vertices, shaped as (num_vertices, 3), \
                    if ``return_vertices`` is ``True``.
                - **joints** -- 3D joint positions, shaped as (num_joints, 3).
                - **orientations** -- Global orientation matrices for each joint, shaped as \
                    (num_joints, 3, 3).

        """

        # Add batch dimension by unsqueezing to shape (1, ...)
        pose_rotvecs = pose_rotvecs.unsqueeze(0) if pose_rotvecs is not None else None
        shape_betas = shape_betas.unsqueeze(0) if shape_betas is not None else None
        trans = trans.unsqueeze(0) if trans is not None else None
        rel_rotmats = rel_rotmats.unsqueeze(0) if rel_rotmats is not None else None
        glob_rotmats = glob_rotmats.unsqueeze(0) if glob_rotmats is not None else None

        # if all are None, then shape_betas is made to be zeros(1,0)
        if (
            pose_rotvecs is None
            and shape_betas is None
            and trans is None
            and rel_rotmats is None
            and glob_rotmats is None
        ):
            shape_betas = torch.zeros((1, 0), dtype=torch.float32, device=self.v_template.device)

        # Call forward with the adjusted arguments
        result = self.forward(
            pose_rotvecs=pose_rotvecs,
            shape_betas=shape_betas,
            trans=trans,
            kid_factor=kid_factor,
            rel_rotmats=rel_rotmats,
            glob_rotmats=glob_rotmats,
            return_vertices=return_vertices,
        )

        # Squeeze out the batch dimension in the result
        return {k: v.squeeze(0) for k, v in result.items()}

    @torch.jit.export
    def rototranslate(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        pose_rotvecs: torch.Tensor,
        shape_betas: torch.Tensor,
        trans: torch.Tensor,
        kid_factor: Optional[torch.Tensor] = None,
        post_translate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Rotates and translates the body in parametric form.

        Rotating a parametric representation is nontrivial because the global orientation
        (first three rotation parameters) performs the rotation around the pelvis joint
        instead of the origin of the canonical coordinate system. This method takes into
        account the offset between the pelvis joint in the shaped T-pose and the origin of
        the canonical coordinate system.

        If ``post_translate`` is True, the translation is added after rotation by ``R``, such that

        .. math::
            M(\texttt{new_pose_rotvec}, \texttt{shape}, \texttt{new_trans}) = \texttt{R} \,
            M(\texttt{pose_rotvecs}, \texttt{shape}, \texttt{trans}) + \texttt{t},

        where :math:`M` is the body model forward function.

        If ``post_translate`` is False, the translation is subtracted before rotation by ``R``,
        such that

        .. math::
            M(\texttt{new_pose_rotvec}, \texttt{shape}, \texttt{new_trans}) = \texttt{R} \,
            (M(\texttt{pose_rotvecs}, \texttt{shape}, \texttt{trans}) - \texttt{t}).

        Parameters:
            R: Rotation matrix, shaped as (3, 3).
            t: Translation vector, shaped as (3,).
            pose_rotvecs: Initial rotation vectors per joint, shaped as (num_joints * 3,).
            shape_betas: Shape coefficients (betas) for body shape, shaped as (num_betas,).
            trans: Initial translation vector, shaped as (3,).
            kid_factor: Optional in case of kid shapes like in AGORA. Shaped as (1,).
            post_translate: Flag indicating whether to apply the translation after rotation. If
                true, ``t`` is added after rotation by ``R``; if false, ``t`` is subtracted before
                rotation by ``R``.

        Returns:
            Tuple
                - **new_pose_rotvecs** -- Updated pose rotation vectors, shaped as (num_joints * 3,).
                - **new_trans** -- Updated translation vector, shaped as (3,).

        """

        current_rotmat = rotvec2mat(pose_rotvecs[:3])
        new_rotmat = R @ current_rotmat
        new_pose_rotvec = torch.cat([mat2rotvec(new_rotmat), pose_rotvecs[3:]], dim=0)

        pelvis = self.J_template[0] + self.J_shapedirs[0, :, : shape_betas.shape[0]] @ shape_betas
        if kid_factor is not None:
            pelvis += self.kid_J_shapedir[0] * kid_factor

        eye3 = torch.eye(3, device=R.device, dtype=R.dtype)
        if post_translate:
            new_trans = trans @ R.mT + t + pelvis @ (R.mT - eye3)
        else:
            new_trans = (trans - t) @ R.mT + pelvis @ (R.mT - eye3)

        return new_pose_rotvec, new_trans
