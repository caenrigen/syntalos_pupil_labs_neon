"""Pupil Labs Neon Syntalos Module."""

from dataclasses import asdict, dataclass
import json

import syntalos_mlink as syl
from pupil_labs.realtime_api.simple import Device, discover_one_device
from PyQt6 import uic
from PyQt6.QtWidgets import QDialog


@dataclass
class Settings:
    phone_ip: str = ""
    phone_port: int = 8080
    discovery_timeout_s: float = 10.0
    frame_wait_timeout_s: float = 0.2


@dataclass
class State:
    settings: Settings | None = None
    stop_requested: bool = False
    device: Device | None = None
    frame_index: int = 0
    first_device_ts_us: int | None = None
    first_master_ts_us: int | None = None


STATE = State()

STREAM_NAME_WORLD = "world"
UI_FILE_PATH = "settings.ui"


def serialise_settings(settings: Settings) -> bytes:
    return json.dumps(asdict(settings)).encode()


def deserialise_settings(settings: bytes) -> Settings:
    return Settings(**json.loads(settings.decode()))  # pyright: ignore[reportAny]


def connect_device() -> Device:
    settings = STATE.settings
    assert settings is not None

    ip = settings.phone_ip.strip()
    if ip:
        return Device(address=ip, port=settings.phone_port)

    device = discover_one_device(max_search_duration_seconds=settings.discovery_timeout_s)
    if device is None:
        raise RuntimeError("No Neon device found on network")
    return device


def submit_scene_frame(scene_frame) -> None:
    dev_us = int(scene_frame.timestamp_unix_seconds * 1_000_000)

    if STATE.first_device_ts_us is None:
        STATE.first_device_ts_us = dev_us
        STATE.first_master_ts_us = int(syl.time_since_start_usec())

    assert STATE.first_device_ts_us is not None
    assert STATE.first_master_ts_us is not None

    frame = syl.Frame()
    frame.mat = scene_frame.bgr_pixels
    frame.index = STATE.frame_index
    frame.time_usec = STATE.first_master_ts_us + (dev_us - STATE.first_device_ts_us)
    STATE.frame_index += 1

    out_scene.submit(frame)


def cleanup() -> None:
    if STATE.device is not None:
        try:
            STATE.device.close()
        except Exception as exc:
            syl.println(f"Failed to close Neon device: {exc}")
    STATE.device = None

    STATE.stop_requested = False
    STATE.frame_index = 0
    STATE.first_device_ts_us = None
    STATE.first_master_ts_us = None


# ## ###############################################################################################
# ## Syntalos interface
# ## ###############################################################################################

out_scene = syl.get_output_port("scene")
out_scene.set_metadata_value("framerate", 30.0)
# Default Neon scene camera resolution
out_scene.set_metadata_value_size("size", [1600, 1200])


def prepare() -> bool:
    if STATE.settings is None:
        STATE.settings = Settings()

    try:
        STATE.device = connect_device()
        STATE.stop_requested = False
        STATE.frame_index = 0
        STATE.first_device_ts_us = None
        STATE.first_master_ts_us = None
        return True
    except Exception as exc:
        syl.println(f"Neon prepare failed: {exc}")
        cleanup()
        return False


def start() -> None:
    assert STATE.device is not None
    try:
        STATE.device.streaming_start(STREAM_NAME_WORLD)
    except Exception as exc:
        syl.println(f"Neon stream start warning: {exc}")


def run() -> None:
    device = STATE.device
    settings = STATE.settings
    assert device is not None
    assert settings is not None

    try:
        while not STATE.stop_requested and syl.is_running():
            scene_frame = device.receive_scene_video_frame(
                timeout_seconds=settings.frame_wait_timeout_s
            )
            if scene_frame is not None:
                submit_scene_frame(scene_frame)
            syl.wait(1)
    finally:
        cleanup()


def stop() -> None:
    STATE.stop_requested = True


def set_settings(settings: bytes) -> None:
    if settings:
        STATE.settings = deserialise_settings(settings)
    elif STATE.settings is None:
        STATE.settings = Settings()


# ## ###############################################################################################
# ## Settings UI
# ## ###############################################################################################


def show_settings(settings: bytes) -> None:
    if settings:
        STATE.settings = deserialise_settings(settings)
    elif STATE.settings is None:
        STATE.settings = Settings()

    assert STATE.settings is not None

    dialog: QDialog = uic.loadUi(UI_FILE_PATH)
    dialog.phoneIpLineEdit.setText(STATE.settings.phone_ip)
    dialog.phonePortSpinBox.setValue(STATE.settings.phone_port)
    dialog.discoveryTimeoutSpinBox.setValue(STATE.settings.discovery_timeout_s)
    dialog.frameWaitTimeoutSpinBox.setValue(STATE.settings.frame_wait_timeout_s)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        STATE.settings.phone_ip = dialog.phoneIpLineEdit.text().strip()
        STATE.settings.phone_port = dialog.phonePortSpinBox.value()
        STATE.settings.discovery_timeout_s = dialog.discoveryTimeoutSpinBox.value()
        STATE.settings.frame_wait_timeout_s = dialog.frameWaitTimeoutSpinBox.value()
        syl.save_settings(serialise_settings(STATE.settings))


# Register settings callback (called when settings dialog is shown)
syl.call_on_show_settings(show_settings)
