"""Center-camera node selection. Does not open a device."""

from PyriteEnvSuites.env_runners.rollout_cam_recorder import (
    _serial_from_by_id,
    _video_index,
)


def test_serial_comes_from_the_by_id_name():
    path = (
        "/dev/v4l/by-id/usb-Orbbec_Orbbec_Femto_Bolt_3D_Camera_"
        "CL8K14100WH-video-index0"
    )
    assert _serial_from_by_id(path) == "CL8K14100WH"


def test_video_index_from_resolved_node():
    assert _video_index("/dev/video6") == 6
    assert _video_index("/dev/v4l/by-id/not-a-node") is None
