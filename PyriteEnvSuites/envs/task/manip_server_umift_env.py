import json
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Any, Dict, List, Optional

import cv2
import noisereduce as nr
import numpy as np
from hardware_interfaces.workcell.table_top_manip.python import (
    manip_server_pybind as ms,
)
from PyriteUtility.audio.audio_recorder import AudioRecorder
from PyriteUtility.audio.multi_mic import MultiMicrophone
from PyriteUtility.computer_vision.computer_vision_utility import (
    get_image_transform_with_border,
)
from umi_day.deployment.arx.peripherals.multi_iphone_camera_umift import (
    MultiIPhoneCameraUMIFT,
)

input_w = 224
input_h = 224
camera_obs_latency = 0.125

#update mask file. this is located in {repo}/PyriteUtility/PyriteUtility/data_pipeline/umift_mask.json
mask_file_path = "/shared_local/configs/umift/umift_mask.json"


class ManipServerUMIFTEnv:
    """
    A class to interact with the real robot for umi-ft.

    """

    def __init__(
        self,
        camera_res_hw: List[int],
        hardware_config_path: str,
        query_sizes: Dict[str, Dict],
        compliant_dimensionality: int,
        audio_params: Dict[str, Any] = None,
        iphone_params: Dict[str, Any] = None,
        vision_key: List[str] = None,
        shm_manager: Optional[SharedMemoryManager] = None,
    ) -> None:
        # load camera mask
        with open(mask_file_path, "r") as f:
            mask_data = json.load(f)
            mask_pts = np.array(mask_data["umift_uw_mask"], dtype=np.int32)
            # apply mask to image
            mask = np.ones((input_h, input_w, 3), dtype=np.uint8) * 255
            cv2.fillPoly(mask, [mask_pts], (0, 0, 0))
            self.mask = mask

        # manipulation server
        server = ms.ManipServer()
        print("Loading hardware config from: ", hardware_config_path)
        if not server.initialize(hardware_config_path):
            raise RuntimeError(
                f"Failed to initialize hardware server using config file {hardware_config_path}"
            )
        server.set_high_level_maintain_position()
        self.image_transform = get_image_transform_with_border(
            input_res=(input_w, input_h), output_res=camera_res_hw, bgr_to_rgb=False
        )

        if server.is_bimanual():
            id_list = [0, 1]
        else:
            id_list = [0]

        stream_names = []
        for key in vision_key:
            if "rgb" in key:
                stream_names.append("main_rgb")
            elif "ultrawide" in key:
                stream_names.append("ultrawide_rgb")
            elif "depth" in key:
                stream_names.append("depth")
        print(f"[DEBUG] stream_names: {stream_names}")
        print(f"[DEBUG] vision_key: {vision_key}")

        if len(stream_names) == 0:
            raise ValueError("No stream names found in vision key")
        if len(stream_names) > 2:
            raise ValueError("More than two vision modalities present in shape meta.")

        iphone_cam_id_list = [
            f"camera{id}_{stream_name}"
            for id in id_list
            for stream_name in stream_names
        ]

        # establish connection to iphone
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        policy_transform_im = get_image_transform_with_border(
            input_res=(320, 240), output_res=(input_h, input_w), bgr_to_rgb=False
        )

        def iphone_policy_transform(data):
            data["color"] = policy_transform_im(data["color"])
            return data

        microphone = None
        if audio_params is not None:
            print("================================================")
            print("creating microphone")
            print("================================================")

            audio_recorder = list()
            block_size = 735
            audio_sr = 44100
            audio_channels = 2
            for device_id in audio_params["audio_device_id"]:
                audio_recorder.append(
                    AudioRecorder(
                        shm_manager=shm_manager,
                        put_fps=(audio_sr // block_size) * 2,
                        sr=audio_sr,
                        device=int(device_id),
                        num_channel=audio_channels,
                        codec="aac",
                        input_audio_fmt="fltp",
                    )
                )

            mic_get_max_k = 120
            microphone = MultiMicrophone(
                shm_manager=shm_manager,
                get_max_k=mic_get_max_k,
                device_id=audio_params["audio_device_id"],
                num_channel=audio_channels,
                block_size=block_size,
                audio_sr=audio_sr,
                put_downsample=False,
                audio_recorder=audio_recorder,
            )
            microphone.start(wait=False)

            print("================================================")
            print("created microphone")
            print("================================================")

        print("================================================")
        print("creating camera")
        print("================================================")
        camera = MultiIPhoneCameraUMIFT(
            server_ip=iphone_params["socket_ip"],
            iphone_ports=iphone_params["ports"],
            shm_manager=shm_manager,
            stream_names=stream_names,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=query_sizes["sparse"]["rgb"],
            receive_latency=camera_obs_latency,
            transform=iphone_policy_transform,
            verbose=False,
        )
        camera.start(wait=True)
        print("------------------------------------------------")
        print("created camera")
        print("------------------------------------------------")

        # comment this out for rollotus
        # print("================================================")
        # print("creating multi camera visualizer")
        # multi_cam_vis = MultiCameraVisualizer(
        #     camera=camera, row=1, col=camera.n_cameras, rgb_to_bgr=True
        # )
        # try:
        #     multi_cam_vis.start(wait=True)
        # except KeyboardInterrupt:
        #     print("Interrupted! Cleaning up...")
        # finally:
        #     multi_cam_vis.stop(wait=True)
        #     print("Visualizer stopped.")

        Tr = np.eye(6)
        n_af = compliant_dimensionality
        default_stiffness = [5000, 5000, 5000, 100, 100, 100]
        default_stiffness = np.diag(default_stiffness)

        # rgb_row_combined_buffer = []
        for key in vision_key:
            if "rgb" in key:
                print("[DEBUG] Creating rgb buffer")
                rgb_buffer = []
            elif "ultrawide" in key:
                print("[DEBUG] Creating ultrawide buffer")
                ultrawide_buffer = []
            elif "depth" in key:
                print("[DEBUG] Creating depth buffer")
                depth_buffer = []

        for id in id_list:
            server.set_force_controlled_axis(Tr, n_af, id)
            server.set_stiffness_matrix(default_stiffness, id)
            # initialize rgb buffers
            # (c h) (n w)->n h w c
            # rgb_row_combined_buffer.append(
            #     np.zeros(
            #         (input_h * 3, input_w * query_sizes["sparse"]["rgb"]),
            #         dtype=np.uint8,
            #     )
            # )
            for key in vision_key:
                if "rgb" in key:
                    rgb_buffer.append(
                        np.zeros(
                            (query_sizes["sparse"]["rgb"], input_h, input_w, 3),
                            dtype=np.uint8,
                        )
                    )
                elif "ultrawide" in key:
                    # currently, ultrawide and rgb are assumed to be dealt the same during inference, as they are both streamed at 60Hz
                    ultrawide_buffer.append(
                        np.zeros(
                            (query_sizes["sparse"]["rgb"], input_h, input_w, 3),
                            dtype=np.uint8,
                        )
                    )
                elif "depth" in key:
                    depth_buffer.append(
                        np.zeros(
                            (query_sizes["sparse"]["depth"], input_h, input_w, 3),
                            dtype=np.float16,
                        )
                    )

        self.microphone = microphone
        self.camera = camera
        self.iphone_cam_id_list = iphone_cam_id_list
        self.stream_names = stream_names
        # self.multi_cam_vis = multi_cam_vis
        self.shm_manager = shm_manager
        self.server = server
        self.query_sizes = query_sizes
        self.id_list = id_list

        # microphone params
        if audio_params is not None:
            self.audio_device_id = audio_params["audio_device_id"]
            self.audio_n_obs_steps = query_sizes["sparse"]["mic"]
            self.audio_channels = audio_channels
            self.audio_sr = audio_sr
            self.block_size = block_size
            self.mic_get_max_k = mic_get_max_k
            self.noise_cancellation = True
            # temp memory buffers
            self.last_camera_data = None
            self.last_microphone_data = None
        self.rgb_repeat_counter = [0] * len(self.id_list)
        self.rgb_repeat_time_value = None
        self.rgb_repeat_warn_threshold = 10
        self.prev_rgb_timestep = [0] * len(self.id_list)

        # buffers
        for key in vision_key:
            if "rgb" in key:
                self.rgb_buffer = rgb_buffer
                self.rgb_timestamp_s = [np.array] * len(id_list)
            elif "ultrawide" in key:
                self.ultrawide_buffer = ultrawide_buffer
                self.ultrawide_timestamp_s = [np.array] * len(id_list)
            elif "depth" in key:
                self.depth_buffer = depth_buffer
                self.depth_timestamp_s = [np.array] * len(id_list)

        self.vision_key = vision_key
        self.last_camera_data = None

        self.sparse_ts_pose_fb = [np.array] * len(id_list)
        self.sparse_ts_pose_fb_timestamp_s = [np.array] * len(id_list)
        self.sparse_robot_wrench = [np.array] * len(id_list)
        self.sparse_robot_wrench_timestamp_s = [np.array] * len(id_list)
        self.sparse_wrench_left = [np.array] * len(id_list)
        self.sparse_wrench_right = [np.array] * len(id_list)
        self.sparse_wrench_timestamp_s = [np.array] * len(id_list)
        self.sparse_robot_eoat_pos = [np.array] * len(id_list)
        self.sparse_robot_eoat_pos_timestamp_s = [np.array] * len(id_list)

        self.dense_ts_pose_fb = [np.array] * len(id_list)
        self.dense_ts_pose_fb_timestamp_s = [np.array] * len(id_list)
        self.dense_wrench = [np.array] * len(id_list)
        self.dense_wrench_filtered = [np.array] * len(id_list)
        self.dense_wrench_timestamp_s = [np.array] * len(id_list)
        self.dense_wrench_filtered_timestamp_s = [np.array] * len(id_list)

        print("[ManipServerEnv:] Waiting for hardware server to be ready...")
        while not self.server.is_ready():
            time.sleep(0.1)

    @property
    def current_hardware_time_s(self):
        return self.server.get_timestamp_now_ms() / 1000

    @property
    def camera_ids(self):
        return self.id_list

    def reset(self):
        # do nothing
        pass

    def cleanup(self):
        pass
        # self.server.join_threads()

    def schedule_controls(
        self,
        pose7_cmd: np.ndarray,  # 7xN or 14xN
        eoat_cmd: np.ndarray,  # 2xN or 4xN
        timestamps: np.ndarray,  # 1xN
        stiffness_matrices_6x6: np.ndarray = None,  # 6x(6xN), or 12x(6xN)
    ):
        if len(self.id_list) == 1:
            assert pose7_cmd.shape[0] == 7
        else:
            assert pose7_cmd.shape[0] == 14

        assert timestamps.shape[0] == pose7_cmd.shape[1]

        if not self.server.is_ready():
            return False

        for id in self.id_list:
            self.server.schedule_waypoints(
                pose7_cmd[id * 7 : (id + 1) * 7, :], timestamps, id
            )
            if eoat_cmd is not None:
                self.server.schedule_eoat_waypoints(
                    eoat_cmd[id * 2 : (id + 1) * 2, :], timestamps, id
                )

        if stiffness_matrices_6x6 is not None:
            if len(self.id_list) == 1:
                assert stiffness_matrices_6x6.shape[0] == 6
            else:
                assert stiffness_matrices_6x6.shape[0] == 12
            assert stiffness_matrices_6x6.shape[1] == timestamps.shape[0] * 6

            for id in self.id_list:
                self.server.schedule_stiffness(
                    stiffness_matrices_6x6[6 * id : 6 * (id + 1), :], timestamps, id
                )

        return True

    def get_observation_from_buffer(self):
        """Get observations from hardware server buffer.
        The number of data points will be sufficient for later downsampling according
        to the shape meta.

        rgb output: (n, H, W, C)
        """
        # read audio
        if self.microphone is not None:
            # 48000 Hz, audio receive timestamp
            k = self.mic_get_max_k
            self.last_microphone_data = self.microphone.get(
                k=k, out=self.last_microphone_data
            )[int(self.audio_device_id[0])]

            mic_obs = dict()
            if self.last_microphone_data:
                mic_timestamps = self.last_microphone_data["timestamp"]
                audio_block = self.last_microphone_data["audio_block"]

                mic_obs["mic_0"] = audio_block[:, :, 0]
                mic_obs["mic_1"] = audio_block[:, :, 1]

                if self.noise_cancellation:
                    raw_audio = audio_block[:, :, 0].flatten()
                    reduced_audio = nr.reduce_noise(
                        y=raw_audio,
                        sr=44100,
                        thresh_n_mult_nonstationary=1,
                        stationary=False,
                    )
                    mic_obs["mic_0"] = reduced_audio.reshape(
                        audio_block.shape[0], audio_block.shape[1]
                    )
                assert mic_obs["mic_0"].shape == (120, 735)

        # read RGB
        self.last_camera_data = self.camera.get(
            k=self.query_sizes["sparse"]["rgb"], out=self.last_camera_data
        )
        # for key in self.last_camera_data:
        #     print(f"key: {key}, shape: {self.last_camera_data[key]['color'].shape}")

        for id in self.id_list:
            for camera_id in self.iphone_cam_id_list:
                if "main_rgb" in camera_id:
                    self.rgb_buffer[id][:] = self.last_camera_data[camera_id]["color"][
                        -self.query_sizes["sparse"]["rgb"] :
                    ]
                    self.rgb_timestamp_s[id] = self.last_camera_data[camera_id][
                        "timestamp"
                    ][-self.query_sizes["sparse"]["rgb"] :]
                    # print(f"[DEBUG] {camera_id} rgb timestamp (last): {self.rgb_timestamp_s[id][-1]:.10f}")
                    # print(f"[DEBUG] id={id}, current={self.rgb_timestamp_s[id][-1]:.10f}, prev={self.prev_rgb_timestep[id]:.10f}, counter={self.rgb_repeat_counter[id]}")
                    # test = np.isclose(self.rgb_timestamp_s[id][-1], self.prev_rgb_timestep[id], atol=1e-5, rtol=0)
                    # print(f"[DEBUG] test: {test}")

                    if self.prev_rgb_timestep is not None and np.isclose(
                        self.rgb_timestamp_s[id][-1],
                        self.prev_rgb_timestep[id],
                        atol=1e-5,
                        rtol=0,
                    ):
                        # self.rgb_repeat_time_value = self.rgb_timestamp_s[id][-1]
                        self.rgb_repeat_counter[id] += 1
                        if (
                            self.rgb_repeat_counter[id]
                            >= self.rgb_repeat_warn_threshold
                        ):
                            raise RuntimeError(
                                f"[FATAL] RGB timestamp repeated {self.rgb_repeat_counter[id]} times for camera ID {id}. Likely client disconnected."
                            )
                    else:
                        self.rgb_repeat_counter[id] = 0

                    self.prev_rgb_timestep[id] = self.rgb_timestamp_s[id][-1]

                    # dt_rgb = self.current_hardware_time_s - self.rgb_timestamp_s[id][-1]
                    assert self.rgb_buffer[id].shape == (
                        self.query_sizes["sparse"]["rgb"],
                        input_h,
                        input_w,
                        3,
                    )
                elif "ultrawide_rgb" in camera_id:
                    self.ultrawide_buffer[id][:] = self.last_camera_data[camera_id][
                        "color"
                    ][-self.query_sizes["sparse"]["rgb"] :]
                    self.ultrawide_timestamp_s[id] = self.last_camera_data[camera_id][
                        "timestamp"
                    ][-self.query_sizes["sparse"]["rgb"] :]
                    # dt_ultrawide = (
                    #     self.current_hardware_time_s
                    #     - self.ultrawide_timestamp_s[id][-1]
                    # )
                    assert self.ultrawide_buffer[id].shape == (
                        self.query_sizes["sparse"]["rgb"],
                        input_h,
                        input_w,
                        3,
                    )
                    # apply mask
                    for i in range(self.query_sizes["sparse"]["rgb"]):
                        self.ultrawide_buffer[id][i] = cv2.bitwise_and(
                            self.ultrawide_buffer[id][i], self.mask
                        )
                elif "depth" in camera_id:
                    # queried {rgb query size} in last_camera_data, but only read in {depth query size}, for simplicity.
                    # depth data is also stored in "color"
                    self.depth_buffer[id][:] = self.last_camera_data[camera_id][
                        "color"
                    ][-self.query_sizes["sparse"]["depth"] :]
                    self.depth_timestamp_s[id] = self.last_camera_data[camera_id][
                        "timestamp"
                    ][-self.query_sizes["sparse"]["depth"] :]
                    # dt_depth = (
                    #     self.current_hardware_time_s - self.depth_timestamp_s[id][-1]
                    # )
                    assert self.depth_buffer[id].shape == (
                        self.query_sizes["sparse"]["depth"],
                        input_h,
                        input_w,
                        3,
                    )

        # read robot
        for id in self.id_list:
            self.sparse_ts_pose_fb[id] = self.server.get_pose(
                self.query_sizes["sparse"]["ts_pose_fb"], id
            ).transpose()
            self.sparse_ts_pose_fb_timestamp_s[id] = (
                self.server.get_pose_timestamps_ms(id) / 1000
            )
            # dt_ts_pose = (
            #     self.current_hardware_time_s
            #     - self.sparse_ts_pose_fb_timestamp_s[id][-1]
            # )

            # read sensor wrench
            if self.query_sizes["sparse"]["wrench"] > 0:
                wrench_fb = self.server.get_wrench(
                    self.query_sizes["sparse"]["wrench"], id
                ).transpose()
                self.sparse_wrench_left[id] = wrench_fb[:, :6]
                if wrench_fb.shape[1] == 6:
                    self.sparse_wrench_right[id] = np.zeros_like(
                        self.sparse_wrench_left[id]
                    )
                elif wrench_fb.shape[1] >= 12:
                    self.sparse_wrench_right[id] = wrench_fb[:, 6:12]
                else:
                    raise ValueError(
                        f"Unexpected wrench width {wrench_fb.shape[1]}; "
                        "expected 6 (single sensor) or 12 (dual CoinFT)."
                    )
                self.sparse_wrench_timestamp_s[id] = (
                    self.server.get_wrench_timestamps_ms(id) / 1000
                )
                # dt_wrench = (
                #     self.current_hardware_time_s
                #     - self.sparse_wrench_timestamp_s[id][-1]
                # )

        # read EOAT pos
        if self.server.has_eoat():
            for id in self.id_list:
                self.sparse_robot_eoat_pos[id] = self.server.get_eoat(
                    self.query_sizes["sparse"]["eoat"], id
                ).transpose()
                self.sparse_robot_eoat_pos_timestamp_s[id] = (
                    self.server.get_eoat_timestamps_ms(id) / 1000
                )

        # timedebug0 = time.perf_counter()
        #  process data
        for id in self.id_list:
            # check size
            assert self.sparse_ts_pose_fb[id].shape == (
                self.query_sizes["sparse"]["ts_pose_fb"],
                7,
            )
            if self.query_sizes["sparse"]["wrench"] > 0:
                assert self.sparse_wrench_left[id].shape == (
                    self.query_sizes["sparse"]["wrench"],
                    6,
                )
                assert self.sparse_wrench_right[id].shape == (
                    self.query_sizes["sparse"]["wrench"],
                    6,
                )

        # timedebug1 = time.perf_counter()
        # print(f"get sparse obs: Time for processing data: {timedebug1 - timedebug0}")

        results = {}
        for id in self.id_list:
            if "main_rgb" in self.stream_names:
                results[f"rgb_{id}"] = self.rgb_buffer[id]
                results[f"rgb_time_stamps_{id}"] = self.rgb_timestamp_s[id]
            if "ultrawide_rgb" in self.stream_names:
                results[f"ultrawide_{id}"] = self.ultrawide_buffer[id]
                results[f"ultrawide_time_stamps_{id}"] = self.ultrawide_timestamp_s[id]
            if "depth" in self.stream_names:
                results[f"depth_{id}"] = self.depth_buffer[id]
                results[f"depth_time_stamps_{id}"] = self.depth_timestamp_s[id]
            results[f"ts_pose_fb_{id}"] = self.sparse_ts_pose_fb[id]
            results[f"robot_time_stamps_{id}"] = self.sparse_ts_pose_fb_timestamp_s[id]
            if self.query_sizes["sparse"]["wrench"] > 0:
                results[f"robot_wrench_{id}"] = self.sparse_robot_wrench[id]
                results[f"robot_wrench_time_stamps_{id}"] = (
                    self.sparse_robot_wrench_timestamp_s[id]
                )
                results[f"wrench_left_{id}"] = self.sparse_wrench_left[id]
                results[f"wrench_right_{id}"] = self.sparse_wrench_right[id]
                results[f"wrench_time_stamps_{id}"] = self.sparse_wrench_timestamp_s[id]

            if self.server.has_eoat():
                results[f"gripper_{id}"] = self.sparse_robot_eoat_pos[id]
                results[f"gripper_time_stamps_{id}"] = (
                    self.sparse_robot_eoat_pos_timestamp_s[id]
                )

        if self.microphone is not None:
            results["mic_0"] = mic_obs["mic_0"]
            results["mic_1"] = mic_obs["mic_1"]
            results["mic_0_time_stamps"] = mic_timestamps

        # # check timing
        # for id in self.id_list:
        #     # dt_rgb = self.current_hardware_time_s - results[f"rgb_time_stamps_{id}"][-1]
        #     # dt_ts_pose = (
        #     #     self.current_hardware_time_s - results[f"robot_time_stamps_{id}"][-1]
        #     # )
        #     # dt_wrench = (
        #     #     self.current_hardware_time_s - results[f"wrench_time_stamps_{id}"][-1]
        #     # )
        #     print(
        #         f"[get obs] obs lagging for robot {id}: dt_rgb: {dt_rgb}, dt_ts_pose: {dt_ts_pose}, dt_wrench: {dt_wrench}"
        #     )
        return results

    def start_saving_data_for_a_new_episode(self, episode_name=""):
        self.server.start_listening_key_events()
        self.server.start_saving_data_for_a_new_episode(episode_name)

    def stop_saving_data(self):
        self.server.stop_saving_data()
        self.server.stop_listening_key_events()
