import torch

from metasim.cfg.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.constants import PhysicStateType
from metasim.utils.camera_util import get_cam_params
from metasim.utils.math import convert_quat, matrix_from_quat, scale_transform, unscale_transform, wrap_to_pi


def test_math_helpers():
    lower = torch.tensor([-1.0, 0.0])
    upper = torch.tensor([1.0, 10.0])
    values = torch.tensor([[0.0, 5.0], [0.5, 2.5]])

    scaled = scale_transform(values, lower, upper)
    torch.testing.assert_close(unscale_transform(scaled, lower, upper), values)

    wrapped = wrap_to_pi(torch.tensor([0.0, torch.pi, -torch.pi, 3 * torch.pi]))
    torch.testing.assert_close(wrapped, torch.tensor([0.0, torch.pi, -torch.pi, torch.pi]))

    identity = torch.eye(3).unsqueeze(0)
    torch.testing.assert_close(matrix_from_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]])), identity)

    converted = convert_quat(torch.tensor([1.0, 0.0, 0.0, 0.0]), to="xyzw")
    torch.testing.assert_close(converted, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_camera_helpers():
    cam_pos = torch.tensor([[0.0, 0.0, 0.0]])
    cam_look_at = torch.tensor([[1.0, 0.0, 0.0]])

    extrinsics, intrinsics = get_cam_params(cam_pos, cam_look_at, width=320, height=240)

    assert extrinsics.shape == (1, 4, 4)
    assert intrinsics.shape == (1, 3, 3)
    torch.testing.assert_close(extrinsics[0, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(intrinsics[0, 2], torch.tensor([0.0, 0.0, 1.0]))


def test_primitive_object_configs():
    cube = PrimitiveCubeCfg(
        name="cube",
        color=[1.0, 0.0, 0.0],
        size=[2.0, 4.0, 8.0],
        physics=PhysicStateType.RIGIDBODY,
    )
    assert cube.half_size == [1.0, 2.0, 4.0]
    assert cube.density == 0.1 / (2.0 * 4.0 * 8.0)

    sphere = PrimitiveSphereCfg(
        name="sphere",
        color=[0.0, 1.0, 0.0],
        radius=0.5,
        physics=PhysicStateType.RIGIDBODY,
    )
    assert sphere.density > 0


test_math_helpers()
test_camera_helpers()
test_primitive_object_configs()
