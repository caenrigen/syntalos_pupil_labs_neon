"""Pupil Labs Neon Syntalos Module."""

import asyncio
import contextlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, final

import numpy as np
import syntalos_mlink as syl

from pupil_labs.realtime_api import (
    BlinkEventData,
    Device,
    FixationEventData,
    FixationOnsetEventData,
    VideoFrame,
    receive_eye_events_data,
    receive_gaze_data,
    receive_imu_data,
    receive_video_frames,
)
from pupil_labs.realtime_api.streaming.imu import IMUData
from pupil_labs.realtime_api.streaming.gaze import EyestateEyelidDualMonoGazeData
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QDialog, QLayout

EyeEventData = FixationEventData | FixationOnsetEventData | BlinkEventData


@dataclass
class Settings:
    phone_ip: str = ""
    phone_port: int = 8080
    discovery_timeout_s: float = 8.0
    batch_size: int = 64
    # Controlling the recording start/saving has been flaky
    companion_recording_enabled: bool = False


STREAM_SCENE = "world"
STREAM_EYES = "eyes"
STREAM_GAZE = "gaze"
STREAM_IMU = "imu"
STREAM_EYE_EVENTS = "eye_events"
STREAM_EVENTS_B = "eye_events_complete"
STREAM_EVENTS_A = "eye_events_simple"
GAZE_SIGNAL_NAMES = [
    "x",
    "y",
    "worn",
    "pupil_diameter_left",
    "eyeball_center_left_x",
    "eyeball_center_left_y",
    "eyeball_center_left_z",
    "optical_axis_left_x",
    "optical_axis_left_y",
    "optical_axis_left_z",
    "pupil_diameter_right",
    "eyeball_center_right_x",
    "eyeball_center_right_y",
    "eyeball_center_right_z",
    "optical_axis_right_x",
    "optical_axis_right_y",
    "optical_axis_right_z",
    "eyelid_angle_top_left",
    "eyelid_angle_bottom_left",
    "eyelid_aperture_left",
    "eyelid_angle_top_right",
    "eyelid_angle_bottom_right",
    "eyelid_aperture_right",
    "mono_left_x",
    "mono_left_y",
    "mono_right_x",
    "mono_right_y",
]
GAZE_UNITS = [
    "px",
    "px",
    "bool",
    "mm",
    "mm",
    "mm",
    "mm",
    "a.u.",
    "a.u.",
    "a.u.",
    "mm",
    "mm",
    "mm",
    "mm",
    "a.u.",
    "a.u.",
    "a.u.",
    "rad",
    "rad",
    "mm",
    "rad",
    "rad",
    "mm",
    "px",
    "px",
    "px",
    "px",
]
IMU_SIGNAL_NAMES = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
]
IMU_UNITS = [
    "deg/s",
    "deg/s",
    "deg/s",
    "m/s^2",
    "m/s^2",
    "m/s^2",
    "a.u.",
    "a.u.",
    "a.u.",
    "a.u.",
]
EYE_EVENTS_COMPLETE_SIGNAL_NAMES = [
    "event_type",
    "rtp_timestamp_us",
    "start_time_us",
    "end_time_us",
    "start_gaze_x",
    "start_gaze_y",
    "end_gaze_x",
    "end_gaze_y",
    "mean_gaze_x",
    "mean_gaze_y",
    "amplitude_pixels",
    "amplitude_angle_deg",
    "mean_velocity",
    "max_velocity",
]
EYE_EVENTS_COMPLETE_UNITS = [
    "a.u.",
    "microseconds",
    "microseconds",
    "microseconds",
    "px",
    "px",
    "px",
    "px",
    "px",
    "px",
    "px",
    "deg",
    "a.u.",
    "a.u.",
]
EYE_EVENTS_SIMPLE_SIGNAL_NAMES = [
    "event_type",
    "rtp_timestamp_us",
    "start_time_us",
    "end_time_us",
]
EYE_EVENTS_SIMPLE_UNITS = [
    "a.u.",
    "microseconds",
    "microseconds",
    "microseconds",
]

SCENE_QUEUE_MAX = 8
EYES_QUEUE_MAX = 64
GAZE_QUEUE_MIN = 128
IMU_QUEUE_MIN = 256
EYE_EVENTS_QUEUE_MAX = 512
ASYNC_LOOP_ADVANCE_S = 0.005
ASYNC_LOOP_WRAPUP_S = 0.200
UI_FILE_PATH = Path(__file__).resolve().with_name("settings.ui")


def serialise_settings(settings: Settings):
    return json.dumps(asdict(settings)).encode()


def deserialise_settings(settings: bytes):
    return Settings(**json.loads(settings.decode()))  # pyright: ignore[reportAny]


def force_tcp_rtsp_url(url: str | None, audioenable: bool = False) -> str:
    if url is None:
        return ""
    # By default it enables the audio
    url = url.replace("audioenable=on", "audioenable=" + ("on" if audioenable else "off"))
    if url.startswith("rtsp://"):
        # force RTSP over TCP interleaving, this prevents frames corruption (e.g. green pixel chunks)
        return "rtspt://" + url[len("rtsp://") :]
    if url.startswith("rtspt://") or url.startswith("rtsps://"):
        return url
    raise RuntimeError(f"Unsupported stream URL scheme: {url}")


def enqueue_latest(queue: asyncio.Queue[Any], item: Any) -> None:
    while queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    queue.put_nowait(item)


async def stream_video_frames(url: str, queue: asyncio.Queue[VideoFrame]) -> None:
    async for frame in receive_video_frames(url, run_loop=True):
        enqueue_latest(queue, frame)


async def stream_gaze_data(url: str, queue: asyncio.Queue[EyestateEyelidDualMonoGazeData]) -> None:
    async for gaze_datum in receive_gaze_data(url, run_loop=True):
        if not isinstance(gaze_datum, EyestateEyelidDualMonoGazeData):
            raise RuntimeError(f"Unexpected gaze data type: {gaze_datum.__class__.__name__}")
        enqueue_latest(queue, gaze_datum)


async def stream_imu_data(url: str, queue: asyncio.Queue[IMUData]) -> None:
    async for imu_datum in receive_imu_data(url, run_loop=True):
        enqueue_latest(queue, imu_datum)


async def stream_eye_events_data(url: str, queue: asyncio.Queue[EyeEventData]) -> None:
    async for eye_event in receive_eye_events_data(url, run_loop=True):
        await queue.put(eye_event)


@final
class Module:
    def __init__(self, mlink: syl.SyntalosLink, app: QApplication) -> None:
        self.mlink = mlink
        self.app = app

        self.settings = Settings()
        self.running = False
        self.cleanup_requested = False
        self.settings_dialog: QDialog | None = None
        self.device: Device | None = None
        self.loop = asyncio.new_event_loop()
        self.scene_url = ""
        self.eyes_url = ""
        self.gaze_url = ""
        self.imu_url = ""
        self.eye_events_url = ""
        self.scene_queue: asyncio.Queue[VideoFrame] | None = None
        self.eyes_queue: asyncio.Queue[VideoFrame] | None = None
        self.gaze_queue: asyncio.Queue[EyestateEyelidDualMonoGazeData] | None = None
        self.imu_queue: asyncio.Queue[IMUData] | None = None
        self.eye_events_queue: asyncio.Queue[EyeEventData] | None = None
        self.stream_tasks: list[asyncio.Task[Any]] = []
        self.scene_frame_index = 0
        self.eyes_frame_index = 0
        self.offset_us: int | None = None
        self.gaze_timestamps_us: list[int] = []
        self.gaze_rows: list[list[float]] = []
        self.imu_timestamps_us: list[int] = []
        self.imu_rows: list[list[float]] = []

        self.register_ports()

    def clear_state(self) -> None:
        self.running = False
        self.device = None
        self.scene_url = ""
        self.eyes_url = ""
        self.gaze_url = ""
        self.imu_url = ""
        self.eye_events_url = ""
        self.scene_queue = None
        self.eyes_queue = None
        self.gaze_queue = None
        self.imu_queue = None
        self.eye_events_queue = None
        self.stream_tasks.clear()
        self.scene_frame_index = 0
        self.eyes_frame_index = 0
        self.offset_us = None
        self.gaze_timestamps_us.clear()
        self.gaze_rows.clear()
        self.imu_timestamps_us.clear()
        self.imu_rows.clear()

    async def connect_device(self):
        ip = self.settings.phone_ip.strip()
        if not ip:
            raise RuntimeError("The IP address of the Android phone must be configured first.")
        try:
            self.device = Device(address=ip, port=self.settings.phone_port)
            status = await self.device.get_status()
            scene_sensor = status.direct_world_sensor()
            eyes_sensor = status.direct_eyes_sensor()
            gaze_sensor = status.direct_gaze_sensor()
            imu_sensor = status.direct_imu_sensor()
            eye_events_sensor = status.direct_eye_events_sensor()

            self.scene_url = (
                force_tcp_rtsp_url(scene_sensor.url)
                if scene_sensor and scene_sensor.connected
                else ""
            )
            self.eyes_url = (
                force_tcp_rtsp_url(eyes_sensor.url) if eyes_sensor and eyes_sensor.connected else ""
            )
            self.gaze_url = (
                force_tcp_rtsp_url(gaze_sensor.url) if gaze_sensor and gaze_sensor.connected else ""
            )
            self.imu_url = (
                force_tcp_rtsp_url(imu_sensor.url) if imu_sensor and imu_sensor.connected else ""
            )
            self.eye_events_url = (
                force_tcp_rtsp_url(eye_events_sensor.url)
                if eye_events_sensor and eye_events_sensor.connected
                else ""
            )

            if not self.scene_url:
                raise RuntimeError("Scene camera stream unavailable")
            if not self.eyes_url:
                raise RuntimeError("Eyes camera stream unavailable")
            if not self.gaze_url:
                raise RuntimeError("Gaze stream unavailable")
            if not self.imu_url:
                raise RuntimeError("IMU stream unavailable")
            if not self.eye_events_url:
                raise RuntimeError(
                    "Eye events stream unavailable. Requires Neon Companion 2.9+ with 'Compute fixations' enabled."
                )
        except Exception:
            if self.device is not None:
                with contextlib.suppress(Exception):
                    await self.device.close()
            raise

    def timestamp_to_us(self, timestamp_unix_seconds: float, stream_name: str) -> int:
        assert self.offset_us is not None
        ts_us = int(timestamp_unix_seconds * 1e6)
        time_us = ts_us + self.offset_us
        # From time to time the Neon App on the Android crashes and/or the frame arrives with negative timestamp.
        if time_us <= 0:
            raise ValueError(
                f"Non-positive {time_us=} for {stream_name=} ({timestamp_unix_seconds=})"
            )
        return time_us

    def timestamp_ns_to_us(self, timestamp_unix_ns: int, stream_name: str) -> float:
        assert self.offset_us is not None
        time_ns = timestamp_unix_ns + self.offset_us * 1000
        time_us = time_ns / 1000.0
        if time_us <= 0:
            raise ValueError(f"Non-positive {time_us=} for {stream_name=}, ({timestamp_unix_ns=})")
        return time_us

    def submit_video_frame(
        self, video_frame: VideoFrame, out_port: syl.OutputPort, stream_name: str, frame_index: int
    ) -> int:
        frame = syl.Frame()
        frame.mat = video_frame.bgr_buffer()
        frame.time_usec = self.timestamp_to_us(video_frame.timestamp_unix_seconds, stream_name)
        frame.index = frame_index
        out_port.submit(frame)
        return frame_index + 1

    def submit_float_block(
        self,
        out_port: syl.OutputPort,
        timestamps_us: list[int],
        rows: list[list[float]],
        clear: bool = True,
    ) -> None:
        if not timestamps_us:
            return
        block = syl.FloatSignalBlock()
        block.timestamps = np.array(timestamps_us, dtype=np.uint64)
        block.data = np.array(rows, dtype=np.float64)
        out_port.submit(block)
        if clear:
            timestamps_us.clear()
            rows.clear()

    def process_gaze_datum(self, gaze_datum: EyestateEyelidDualMonoGazeData) -> None:
        self.gaze_timestamps_us.append(
            self.timestamp_to_us(gaze_datum.timestamp_unix_seconds, STREAM_GAZE)
        )
        self.gaze_rows.append(
            [
                # Gaze
                gaze_datum.x,
                gaze_datum.y,
                float(gaze_datum.worn),
                # Left eye
                gaze_datum.pupil_diameter_left,
                gaze_datum.eyeball_center_left_x,
                gaze_datum.eyeball_center_left_y,
                gaze_datum.eyeball_center_left_z,
                gaze_datum.optical_axis_left_x,
                gaze_datum.optical_axis_left_y,
                gaze_datum.optical_axis_left_z,
                # Right eye
                gaze_datum.pupil_diameter_right,
                gaze_datum.eyeball_center_right_x,
                gaze_datum.eyeball_center_right_y,
                gaze_datum.eyeball_center_right_z,
                gaze_datum.optical_axis_right_x,
                gaze_datum.optical_axis_right_y,
                gaze_datum.optical_axis_right_z,
                # Lid left
                gaze_datum.eyelid_angle_top_left,
                gaze_datum.eyelid_angle_bottom_left,
                gaze_datum.eyelid_aperture_left,
                # Lid right
                gaze_datum.eyelid_angle_top_right,
                gaze_datum.eyelid_angle_bottom_right,
                gaze_datum.eyelid_aperture_right,
                # Gaze mono
                gaze_datum.mono_left_x,
                gaze_datum.mono_left_y,
                gaze_datum.mono_right_x,
                gaze_datum.mono_right_y,
            ]
        )
        if len(self.gaze_timestamps_us) >= self.settings.batch_size:
            self.submit_float_block(
                self.out_gaze,
                self.gaze_timestamps_us,
                self.gaze_rows,
            )

    def process_imu_datum(self, imu_datum: IMUData) -> None:
        self.imu_timestamps_us.append(
            self.timestamp_to_us(imu_datum.timestamp_unix_seconds, STREAM_IMU)
        )
        self.imu_rows.append(
            [
                imu_datum.gyro_data.x,
                imu_datum.gyro_data.y,
                imu_datum.gyro_data.z,
                imu_datum.accel_data.x,
                imu_datum.accel_data.y,
                imu_datum.accel_data.z,
                imu_datum.quaternion.x,
                imu_datum.quaternion.y,
                imu_datum.quaternion.z,
                imu_datum.quaternion.w,
            ]
        )
        if len(self.imu_timestamps_us) >= self.settings.batch_size:
            self.submit_float_block(
                self.out_imu,
                self.imu_timestamps_us,
                self.imu_rows,
            )

    def process_eye_event(self, event: EyeEventData) -> None:
        rtp_timestamp_us = self.timestamp_to_us(event.rtp_ts_unix_seconds, STREAM_EYE_EVENTS)
        start_time_us = self.timestamp_ns_to_us(event.start_time_ns, STREAM_EYE_EVENTS)
        sample_timestamp_us = int(start_time_us)
        if isinstance(event, FixationEventData):
            self.submit_float_block(
                self.out_eye_events_complete,
                [sample_timestamp_us],
                [
                    [
                        float(event.event_type),
                        float(rtp_timestamp_us),
                        start_time_us,
                        self.timestamp_ns_to_us(event.end_time_ns, STREAM_EYE_EVENTS),
                        event.start_gaze_x,
                        event.start_gaze_y,
                        event.end_gaze_x,
                        event.end_gaze_y,
                        event.mean_gaze_x,
                        event.mean_gaze_y,
                        event.amplitude_pixels,
                        event.amplitude_angle_deg,
                        event.mean_velocity,
                        event.max_velocity,
                    ]
                ],
                clear=False,
            )
        elif isinstance(event, BlinkEventData):
            self.submit_float_block(
                self.out_eye_events_simple,
                [sample_timestamp_us],
                [
                    [
                        float(event.event_type),
                        float(rtp_timestamp_us),
                        start_time_us,
                        self.timestamp_ns_to_us(event.end_time_ns, STREAM_EYE_EVENTS),
                    ]
                ],
                clear=False,
            )
        elif isinstance(event, FixationOnsetEventData):
            self.submit_float_block(
                self.out_eye_events_simple,
                [sample_timestamp_us],
                [
                    [
                        float(event.event_type),
                        float(rtp_timestamp_us),
                        start_time_us,
                        np.nan,
                    ]
                ],
                clear=False,
            )
        else:
            raise RuntimeError(f"Unexpected eye event data type: {event.__class__.__name__}")

    def submit_scene_frame(self, frame: VideoFrame) -> None:
        self.scene_frame_index = self.submit_video_frame(
            frame, self.out_scene, STREAM_SCENE, self.scene_frame_index
        )

    def submit_eyes_frame(self, frame: VideoFrame) -> None:
        self.eyes_frame_index = self.submit_video_frame(
            frame, self.out_eyes, STREAM_EYES, self.eyes_frame_index
        )

    def ensure_stream_tasks_healthy(self) -> None:
        for task in self.stream_tasks:
            if not task.done():
                continue
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is None:
                raise RuntimeError(f"Streaming task ended unexpectedly: {task.get_name()}")
            raise RuntimeError(f"Streaming task failed: {task.get_name()}") from exc

    def drain_video_queue(
        self, queue: asyncio.Queue[VideoFrame], submit_func: Callable[[VideoFrame], None]
    ) -> None:
        while True:
            try:
                frame = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            submit_func(frame)
            del frame

    def drain_gaze_queue(self) -> None:
        while True:
            try:
                assert self.gaze_queue is not None
                gaze_datum = self.gaze_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.process_gaze_datum(gaze_datum)

    def drain_imu_queue(self) -> None:
        while True:
            try:
                assert self.imu_queue is not None
                imu_datum = self.imu_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.process_imu_datum(imu_datum)

    def drain_eye_events_queue(self) -> None:
        while True:
            try:
                assert self.eye_events_queue is not None
                eye_event = self.eye_events_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.process_eye_event(eye_event)

    async def stop_stream_tasks(self) -> None:
        tasks = self.stream_tasks.copy()
        self.stream_tasks.clear()
        for task in tasks:
            if not task.done():
                _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def cleanup_async(self) -> None:
        await self.stop_stream_tasks()
        if self.device is None:
            print("No device to cleanup, skipping device cleanup")
            return

        if self.settings.companion_recording_enabled:
            try:
                await self.device.recording_stop_and_save()
            except Exception as exc:
                print(f"Neon cleanup recording control failed: {exc}")

        try:
            await self.device.close()
        except Exception as exc:
            print(f"Failed to close Neon device: {exc}")

    def cleanup(self) -> None:
        self.cleanup_requested = False
        try:
            self.loop.run_until_complete(self.cleanup_async())
            # Advance the async loop a final bit for all pending tasks to wrap up.
            # This pervents a series of errors being printed when quiting Syntalos.
            self.loop.run_until_complete(asyncio.sleep(ASYNC_LOOP_WRAPUP_S))
            print("Cleanup done")
        except Exception as exc:
            print(f"Cleanup failed: {exc.__class__.__name__}({exc})")

    # # ################################################################################
    # # Syntalos interface
    # # ################################################################################

    def register_ports(self) -> None:
        self.out_scene = self.mlink.register_output_port(
            STREAM_SCENE, "Scene", data_type=syl.DataType.Frame
        )
        self.out_eyes = self.mlink.register_output_port(STREAM_EYES, "Eyes", syl.DataType.Frame)
        self.out_gaze = self.mlink.register_output_port(
            STREAM_GAZE, "Gaze", syl.DataType.FloatSignalBlock
        )
        self.out_imu = self.mlink.register_output_port(
            STREAM_IMU, "IMU", syl.DataType.FloatSignalBlock
        )
        self.out_eye_events_complete = self.mlink.register_output_port(
            STREAM_EVENTS_B, "Events B", syl.DataType.FloatSignalBlock
        )
        self.out_eye_events_simple = self.mlink.register_output_port(
            STREAM_EVENTS_A, "Events A", syl.DataType.FloatSignalBlock
        )

    def prepare(self) -> bool:
        if self.cleanup_requested:
            self.cleanup()
        self.clear_state()
        if self.settings_dialog is not None:
            _ = self.settings_dialog.close()

        self.out_scene.set_metadata_value("framerate", 30.0)
        self.out_scene.set_metadata_value_size("size", syl.MetaSize(1600, 1200))

        self.out_eyes.set_metadata_value("framerate", 200.0)
        self.out_eyes.set_metadata_value_size("size", syl.MetaSize(384, 192))

        self.out_gaze.set_metadata_value("signal_names", GAZE_SIGNAL_NAMES)
        self.out_gaze.set_metadata_value("time_unit", "microseconds")
        self.out_gaze.set_metadata_value("data_unit", GAZE_UNITS)

        self.out_imu.set_metadata_value("signal_names", IMU_SIGNAL_NAMES)
        self.out_imu.set_metadata_value("time_unit", "microseconds")
        self.out_imu.set_metadata_value("data_unit", IMU_UNITS)

        self.out_eye_events_complete.set_metadata_value(
            "signal_names", EYE_EVENTS_COMPLETE_SIGNAL_NAMES
        )
        self.out_eye_events_complete.set_metadata_value("time_unit", "microseconds")
        self.out_eye_events_complete.set_metadata_value("data_unit", EYE_EVENTS_COMPLETE_UNITS)

        self.out_eye_events_simple.set_metadata_value(
            "signal_names", EYE_EVENTS_SIMPLE_SIGNAL_NAMES
        )
        self.out_eye_events_simple.set_metadata_value("time_unit", "microseconds")
        self.out_eye_events_simple.set_metadata_value("data_unit", EYE_EVENTS_SIMPLE_UNITS)

        self.loop.run_until_complete(self.connect_device())

        self.scene_queue = asyncio.Queue(maxsize=SCENE_QUEUE_MAX)
        self.eyes_queue = asyncio.Queue(maxsize=EYES_QUEUE_MAX)
        self.gaze_queue = asyncio.Queue(maxsize=max(self.settings.batch_size * 4, GAZE_QUEUE_MIN))
        self.imu_queue = asyncio.Queue(maxsize=max(self.settings.batch_size * 4, IMU_QUEUE_MIN))
        self.eye_events_queue = asyncio.Queue(maxsize=EYE_EVENTS_QUEUE_MAX)
        return True

    def start(self) -> None:
        self.offset_us = -int(time.time() * 1e6)

        assert self.device is not None
        assert self.scene_queue is not None
        assert self.eyes_queue is not None
        assert self.gaze_queue is not None
        assert self.imu_queue is not None
        assert self.eye_events_queue is not None

        self.stream_tasks = [
            self.loop.create_task(
                stream_video_frames(self.scene_url, self.scene_queue), name="scene-stream"
            ),
            self.loop.create_task(
                stream_video_frames(self.eyes_url, self.eyes_queue), name="eyes-stream"
            ),
            self.loop.create_task(
                stream_gaze_data(self.gaze_url, self.gaze_queue), name="gaze-stream"
            ),
            self.loop.create_task(stream_imu_data(self.imu_url, self.imu_queue), name="imu-stream"),
            self.loop.create_task(
                stream_eye_events_data(self.eye_events_url, self.eye_events_queue),
                name="eye-events-stream",
            ),
        ]

        if self.settings.companion_recording_enabled:
            _recording_id = self.loop.run_until_complete(self.device.recording_start())

        self.running = True

    def event_loop_tick(self) -> None:
        self.app.processEvents()
        if self.cleanup_requested:
            self.cleanup()
        if not self.running:
            return

        self.loop.run_until_complete(asyncio.sleep(ASYNC_LOOP_ADVANCE_S))

        self.ensure_stream_tasks_healthy()
        self.drain_gaze_queue()
        self.drain_imu_queue()
        self.drain_eye_events_queue()
        assert self.scene_queue is not None
        self.drain_video_queue(self.scene_queue, self.submit_scene_frame)
        assert self.eyes_queue is not None
        self.drain_video_queue(self.eyes_queue, self.submit_eyes_frame)

    def stop(self) -> None:
        self.running = False
        self.cleanup_requested = True

    def load_settings(self, settings: bytes, _base_dir: Path) -> bool:
        if not settings:
            return True

        try:
            self.settings = deserialise_settings(settings)
            return True
        except Exception:
            self.settings = Settings()
            raise

    def save_settings(self, _base_dir: Path) -> bytes:
        return serialise_settings(self.settings)

    # # ################################################################################
    # # Settings UI
    # # ################################################################################

    def show_settings(self) -> None:
        # Showing the settings UI while running prevents the module event loop from advancing.
        # Keep it simple: no settings UI while running.
        if self.running or self.mlink.is_running:
            print("Cannot show settings while running")
            return

        dialog = self.settings_dialog
        if dialog is not None:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return

        dialog = uic.loadUi(UI_FILE_PATH)
        self.settings_dialog = dialog
        fit_dialog_to_contents(dialog)
        dialog.phoneIpLineEdit.setText(self.settings.phone_ip)
        dialog.phonePortSpinBox.setValue(self.settings.phone_port)
        dialog.discoveryTimeoutSpinBox.setValue(self.settings.discovery_timeout_s)
        dialog.companionRecordingCheckBox.setChecked(self.settings.companion_recording_enabled)

        def persist_settings() -> None:
            self.settings.phone_ip = dialog.phoneIpLineEdit.text().strip()
            self.settings.phone_port = dialog.phonePortSpinBox.value()
            self.settings.discovery_timeout_s = dialog.discoveryTimeoutSpinBox.value()
            self.settings.companion_recording_enabled = (
                dialog.companionRecordingCheckBox.isChecked()
            )

        def cleanup_dialog(_result: int) -> None:
            self.settings_dialog = None

        dialog.phoneIpLineEdit.textChanged.connect(persist_settings)
        dialog.phonePortSpinBox.valueChanged.connect(persist_settings)
        dialog.discoveryTimeoutSpinBox.valueChanged.connect(persist_settings)
        dialog.companionRecordingCheckBox.checkStateChanged.connect(persist_settings)
        dialog.finished.connect(cleanup_dialog)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


def fit_dialog_to_contents(dialog: QDialog) -> None:
    layout = dialog.layout()
    if layout is not None:
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
    dialog.adjustSize()


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    mlink = syl.init_link(rename_process=True)
    mod = Module(mlink, app)
    mlink.on_prepare = mod.prepare
    mlink.on_start = mod.start
    mlink.on_stop = mod.stop
    mlink.on_show_settings = mod.show_settings
    mlink.on_save_settings = mod.save_settings
    mlink.on_load_settings = mod.load_settings
    mlink.await_data_forever(mod.event_loop_tick)
    if mod.running:
        mod.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
