"""Pupil Labs Neon Syntalos Module."""

import asyncio
import contextlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import syntalos_mlink as syl

from pupil_labs.realtime_api import (
    Device,
    Network,
    VideoFrame,
    receive_gaze_data,
    receive_video_frames,
)
from pupil_labs.realtime_api.streaming.gaze import EyestateEyelidDualMonoGazeData
from PyQt6 import uic
from PyQt6.QtWidgets import QDialog, QLayout


def handle_fatal_exc(exc: Exception, syntalos_raise: bool, clean: bool, prefix: str = ""):
    msg = f"{prefix}{': ' if prefix else ''}{exc.__class__.__name__}({exc})"
    syl.println(f"{msg}\n{traceback.format_exc()}")
    if clean:
        cleanup()
    if syntalos_raise:
        syl.raise_error(msg)


@dataclass
class Settings:
    phone_ip: str = ""
    phone_port: int = 8080
    discovery_timeout_s: float = 8.0
    batch_size: int = 64
    # Controlling the recording start/saving has been flaky
    companion_recording_enabled: bool = False


@dataclass
class State:
    settings: Settings | None = None
    stop_requested: bool = False
    running: bool = False
    settings_dialog: QDialog | None = None
    device: Device | None = None
    loop: asyncio.AbstractEventLoop | None = None
    scene_url: str = ""
    eyes_url: str = ""
    gaze_url: str = ""
    scene_queue: asyncio.Queue[VideoFrame] | None = None
    eyes_queue: asyncio.Queue[VideoFrame] | None = None
    gaze_queue: asyncio.Queue[EyestateEyelidDualMonoGazeData] | None = None
    stream_tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    scene_frame_index: int = 0
    eyes_frame_index: int = 0
    offset_us: int | None = None
    gaze_timestamps_us: list[int] = field(default_factory=list)
    gaze_rows: list[list[float]] = field(default_factory=list)


def clear_state() -> None:
    # Settings and the asyncio loop should stay persistent across runs.
    STATE.device = None
    STATE.stop_requested = False
    STATE.running = False
    STATE.scene_url = ""
    STATE.eyes_url = ""
    STATE.gaze_url = ""
    STATE.scene_queue = None
    STATE.eyes_queue = None
    STATE.gaze_queue = None
    STATE.stream_tasks.clear()
    STATE.scene_frame_index = 0
    STATE.eyes_frame_index = 0
    STATE.offset_us = None
    STATE.gaze_timestamps_us.clear()
    STATE.gaze_rows.clear()


STATE = State()
out_scene: syl.OutputPort | None = None
out_eyes: syl.OutputPort | None = None
out_gaze: syl.OutputPort | None = None

STREAM_NAME_SCENE = "world"
STREAM_NAME_EYES = "eyes"
STREAM_NAME_GAZE = "gaze"
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

SCENE_QUEUE_MAX = 8
EYES_QUEUE_MAX = 64
GAZE_QUEUE_MIN = 128
ASYNC_LOOP_ADVANCE_S = 0.001
ASYNC_LOOP_WRAPUP_S = 0.200


def serialise_settings(settings: Settings):
    return json.dumps(asdict(settings)).encode()


def deserialise_settings(settings: bytes):
    return Settings(**json.loads(settings.decode()))


def save_current_settings() -> None:
    assert STATE.settings is not None
    syl.save_settings(serialise_settings(STATE.settings))


def close_settings_dialog() -> None:
    dialog = STATE.settings_dialog
    if dialog is not None:
        dialog.close()


def fit_dialog_to_contents(dialog: QDialog) -> None:
    layout = dialog.layout()
    if layout is not None:
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
    dialog.adjustSize()


def force_tcp_rtsp_url(url: str) -> str:
    if url.startswith("rtsp://"):
        # force RTSP over TCP interleaving, this prevents frames corruption (e.g. green pixel chunks)
        return "rtspt://" + url[len("rtsp://") :]
    if url.startswith("rtspt://") or url.startswith("rtsps://"):
        return url
    raise RuntimeError(f"Unsupported stream URL scheme: {url}")


async def connect_device() -> tuple[Device, str, str, str]:
    settings = STATE.settings
    assert settings is not None

    ip = settings.phone_ip.strip()
    device: Device | None = None
    try:
        if ip:
            device = Device(address=ip, port=settings.phone_port)
        else:
            async with Network() as network:
                dev_info = await network.wait_for_new_device(settings.discovery_timeout_s)
            if dev_info is None:
                raise RuntimeError("No Neon device found on network")
            device = Device.from_discovered_device(dev_info)

        status = await device.get_status()
        scene_sensor = status.direct_world_sensor()
        eyes_sensor = status.direct_eyes_sensor()
        gaze_sensor = status.direct_gaze_sensor()

        scene_url = scene_sensor.url if scene_sensor and scene_sensor.connected else None
        eyes_url = eyes_sensor.url if eyes_sensor and eyes_sensor.connected else None
        gaze_url = gaze_sensor.url if gaze_sensor and gaze_sensor.connected else None
        if scene_url is None:
            raise RuntimeError("Scene camera stream unavailable")
        if eyes_url is None:
            raise RuntimeError("Eyes camera stream unavailable")
        if gaze_url is None:
            raise RuntimeError("Gaze stream unavailable")

        scene_url = force_tcp_rtsp_url(scene_url)
        eyes_url = force_tcp_rtsp_url(eyes_url)
        gaze_url = force_tcp_rtsp_url(gaze_url)

        return device, scene_url, eyes_url, gaze_url
    except Exception:
        if device is not None:
            with contextlib.suppress(Exception):
                await device.close()
        raise


def timestamp_to_us(timestamp_unix_seconds: float, stream_name: str) -> int:
    assert STATE.offset_us is not None
    ts_us = int(timestamp_unix_seconds * 1e6)
    time_us = ts_us + STATE.offset_us
    # From time to time the Neon App on the Android crashes and the frame arrives with negative timestamp
    if time_us <= 0:
        syl.println(f"Non-positive {time_us = }, {stream_name = }, {timestamp_unix_seconds = }")
    return time_us


def submit_video_frame(
    video_frame: VideoFrame, out_port: syl.OutputPort, stream_name: str, frame_index: int
) -> int:
    frame = syl.Frame()
    frame.mat = video_frame.bgr_buffer()
    frame.time_usec = timestamp_to_us(video_frame.timestamp_unix_seconds, stream_name)
    frame.index = frame_index
    out_port.submit(frame)
    return frame_index + 1


def submit_float_block(
    out_port: syl.OutputPort,
    timestamps_us: list[int],
    rows: list[list[float]],
) -> None:
    if not timestamps_us:
        return

    block = syl.FloatSignalBlock()
    block.timestamps = np.array(timestamps_us, dtype=np.uint64)
    block.data = np.array(rows, dtype=np.float64)
    out_port.submit(block)
    timestamps_us.clear()
    rows.clear()


def process_gaze_datum(gaze_datum: EyestateEyelidDualMonoGazeData) -> None:
    assert STATE.settings is not None
    STATE.gaze_timestamps_us.append(
        timestamp_to_us(gaze_datum.timestamp_unix_seconds, STREAM_NAME_GAZE)
    )
    STATE.gaze_rows.append(
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
    if len(STATE.gaze_timestamps_us) >= STATE.settings.batch_size:
        assert out_gaze is not None
        submit_float_block(
            out_gaze,
            STATE.gaze_timestamps_us,
            STATE.gaze_rows,
        )


def submit_scene_frame(frame: VideoFrame) -> None:
    assert out_scene is not None
    STATE.scene_frame_index = submit_video_frame(
        frame, out_scene, STREAM_NAME_SCENE, STATE.scene_frame_index
    )


def submit_eyes_frame(frame: VideoFrame) -> None:
    assert out_eyes is not None
    STATE.eyes_frame_index = submit_video_frame(
        frame, out_eyes, STREAM_NAME_EYES, STATE.eyes_frame_index
    )


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


def ensure_stream_tasks_healthy() -> None:
    for task in STATE.stream_tasks:
        if not task.done():
            continue
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is None:
            raise RuntimeError(f"Streaming task ended unexpectedly: {task.get_name()}")
        raise RuntimeError(f"Streaming task failed: {task.get_name()}") from exc


def drain_video_queue(
    queue: asyncio.Queue[VideoFrame], submit_func: Callable[[VideoFrame], None]
) -> None:
    while True:
        try:
            frame = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        submit_func(frame)
        del frame


def drain_gaze_queue(queue: asyncio.Queue[EyestateEyelidDualMonoGazeData]) -> None:
    while True:
        try:
            gaze_datum = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        process_gaze_datum(gaze_datum)


async def stop_stream_tasks() -> None:
    tasks = STATE.stream_tasks.copy()
    STATE.stream_tasks.clear()
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def cleanup_async() -> None:
    await stop_stream_tasks()

    device = STATE.device
    if device is None:
        syl.println("No device to cleanup, skipping device cleanup")
        return

    settings = STATE.settings
    if settings is None:
        syl.println("Settings not set, skipping device cleanup")
        return

    if settings.companion_recording_enabled:
        try:
            await device.recording_stop_and_save()
        except Exception as exc:
            syl.println(f"Neon cleanup recording control failed: {exc}")

    try:
        await device.close()
    except Exception as exc:
        syl.println(f"Failed to close Neon device: {exc}")


def cleanup() -> None:
    loop = STATE.loop
    if loop is None:
        syl.println("No event loop to cleanup, skipping cleanup()")
        clear_state()
        return

    try:
        loop.run_until_complete(cleanup_async())
        # Advance the async loop a final bit for all pending tasks to wrap up.
        # This pervents a series of errors being printed when quiting Syntalos.
        loop.run_until_complete(asyncio.sleep(ASYNC_LOOP_WRAPUP_S))
    except Exception as exc:
        syl.println(f"Cleanup failed: {exc.__class__.__name__}({exc})")

    clear_state()
    syl.println("Cleanup complete")


def register_ports() -> None:
    syl.register_output_port(STREAM_NAME_SCENE, "Scene Camera", "Frame")
    syl.register_output_port(STREAM_NAME_EYES, "Eyes Camera", "Frame")
    syl.register_output_port(STREAM_NAME_GAZE, "Gaze", "FloatSignalBlock")


# # ####################################################################################
# # Syntalos interface
# # ####################################################################################


def prepare():
    global out_scene, out_eyes, out_gaze

    clear_state()
    save_current_settings()
    close_settings_dialog()
    if STATE.settings is None:
        syl.println("Settings not set, aborting prepare()")
        return False

    out_scene = syl.get_output_port(STREAM_NAME_SCENE)
    assert out_scene is not None
    out_scene.set_metadata_value("framerate", 30.0)
    out_scene.set_metadata_value_size("size", syl.MetaSize(1600, 1200))

    out_eyes = syl.get_output_port(STREAM_NAME_EYES)
    assert out_eyes is not None
    out_eyes.set_metadata_value("framerate", 200.0)
    out_eyes.set_metadata_value_size("size", syl.MetaSize(384, 192))

    out_gaze = syl.get_output_port(STREAM_NAME_GAZE)
    assert out_gaze is not None
    out_gaze.set_metadata_value("signal_names", GAZE_SIGNAL_NAMES)
    out_gaze.set_metadata_value("time_unit", "microseconds")
    out_gaze.set_metadata_value("data_unit", GAZE_UNITS)

    try:
        if STATE.loop is None:
            STATE.loop = asyncio.new_event_loop()
            syl.println("Created asyncio loop")
        else:
            syl.println("Reusing asyncio loop")

        device, scene_url, eyes_url, gaze_url = STATE.loop.run_until_complete(connect_device())
        STATE.device = device
        STATE.scene_url = scene_url
        STATE.eyes_url = eyes_url
        STATE.gaze_url = gaze_url
        STATE.scene_queue = asyncio.Queue(maxsize=SCENE_QUEUE_MAX)
        STATE.eyes_queue = asyncio.Queue(maxsize=EYES_QUEUE_MAX)
        STATE.gaze_queue = asyncio.Queue(maxsize=max(STATE.settings.batch_size * 4, GAZE_QUEUE_MIN))
        return True
    except Exception as exc:
        handle_fatal_exc(exc, syntalos_raise=True, clean=True, prefix="Prepare failed")
        return False


def start() -> None:
    assert STATE.loop is not None
    assert STATE.device is not None
    assert STATE.settings is not None
    assert STATE.scene_queue is not None
    assert STATE.eyes_queue is not None
    assert STATE.gaze_queue is not None

    try:
        STATE.stream_tasks = [
            STATE.loop.create_task(
                stream_video_frames(STATE.scene_url, STATE.scene_queue), name="scene-stream"
            ),
            STATE.loop.create_task(
                stream_video_frames(STATE.eyes_url, STATE.eyes_queue), name="eyes-stream"
            ),
            STATE.loop.create_task(
                stream_gaze_data(STATE.gaze_url, STATE.gaze_queue), name="gaze-stream"
            ),
        ]
        STATE.offset_us = -int(time.time() * 1e6)
        if STATE.settings.companion_recording_enabled:
            _recording_id = STATE.loop.run_until_complete(STATE.device.recording_start())
    except Exception as exc:
        handle_fatal_exc(exc, syntalos_raise=True, clean=True, prefix="Start failed")


def run() -> None:
    STATE.running = True
    loop = STATE.loop
    device = STATE.device
    settings = STATE.settings
    scene_queue = STATE.scene_queue
    eyes_queue = STATE.eyes_queue
    gaze_queue = STATE.gaze_queue
    assert loop is not None
    assert device is not None
    assert settings is not None
    assert scene_queue is not None
    assert eyes_queue is not None
    assert gaze_queue is not None

    try:
        while not STATE.stop_requested and syl.is_running():
            # Advance the async loop
            loop.run_until_complete(asyncio.sleep(ASYNC_LOOP_ADVANCE_S))
            ensure_stream_tasks_healthy()
            drain_gaze_queue(gaze_queue)
            drain_video_queue(scene_queue, submit_scene_frame)
            drain_video_queue(eyes_queue, submit_eyes_frame)
            syl.wait(1)  # give time for syntalos to call stop()
    except Exception as exc:
        handle_fatal_exc(exc, syntalos_raise=True, clean=True, prefix="Run failed")

    cleanup()
    STATE.running = False


def stop() -> None:
    STATE.stop_requested = True
    # In case other modules trigger a premature stop(), we need to call cleanup() here
    if not STATE.running:
        cleanup()


def set_settings(settings: bytes) -> None:
    if settings:
        try:
            STATE.settings = deserialise_settings(settings)
        except Exception as exc:
            msg = f"Failed to parse settings: {exc.__class__.__name__}({exc})"
            syl.println(msg)
            syl.raise_error(msg)
            STATE.settings = Settings()
    elif STATE.settings is None:
        STATE.settings = Settings()


# # ####################################################################################
# # Settings UI
# # ####################################################################################


UI_FILE_PATH = "settings.ui"


def show_settings(settings: bytes) -> None:
    # Showing the settings UI while running prevents the run() loop from advancing.
    # Keep it simple: no settings UI while running.
    if STATE.running or syl.is_running():
        syl.println("Cannot show settings while running")
        return

    if not settings:
        if STATE.settings is None:
            STATE.settings = Settings()
    else:
        STATE.settings = deserialise_settings(settings)

    assert STATE.settings is not None

    dialog = STATE.settings_dialog
    if dialog is not None:
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return

    dialog = uic.loadUi(UI_FILE_PATH)
    STATE.settings_dialog = dialog
    fit_dialog_to_contents(dialog)
    dialog.phoneIpLineEdit.setText(STATE.settings.phone_ip)
    dialog.phonePortSpinBox.setValue(STATE.settings.phone_port)
    dialog.discoveryTimeoutSpinBox.setValue(STATE.settings.discovery_timeout_s)
    dialog.companionRecordingCheckBox.setChecked(STATE.settings.companion_recording_enabled)

    def persist_settings():
        assert STATE.settings is not None
        STATE.settings.phone_ip = dialog.phoneIpLineEdit.text().strip()
        STATE.settings.phone_port = dialog.phonePortSpinBox.value()
        STATE.settings.discovery_timeout_s = dialog.discoveryTimeoutSpinBox.value()
        STATE.settings.companion_recording_enabled = dialog.companionRecordingCheckBox.isChecked()
        save_current_settings()

    def cleanup_dialog():
        STATE.settings_dialog = None

    dialog.phoneIpLineEdit.textChanged.connect(persist_settings)
    dialog.phonePortSpinBox.valueChanged.connect(persist_settings)
    dialog.discoveryTimeoutSpinBox.valueChanged.connect(persist_settings)
    dialog.companionRecordingCheckBox.checkStateChanged.connect(persist_settings)
    dialog.finished.connect(cleanup_dialog)

    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


# Register settings callback (called when settings dialog is shown)
syl.call_on_show_settings(show_settings)

# Register ports at module level so Syntalos can restore project connections.
register_ports()
