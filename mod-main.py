"""Pupil Labs Neon Syntalos Module."""

from datetime import timedelta
import json
from dataclasses import asdict, dataclass

import syntalos_mlink as syl

from pupil_labs.realtime_api.device import DeviceError
from pupil_labs.realtime_api.simple import (
    Device,
    SimpleVideoFrame,
    discover_one_device,
)
from PyQt6 import uic
from PyQt6.QtWidgets import QDialog


@dataclass
class Settings:
    phone_ip: str = ""
    phone_port: int = 8080
    # Syntalos timeout is 10s
    discovery_timeout_s: float = 8.0
    frame_wait_timeout_s: float = 0.2
    companion_recording_enabled: bool = True


@dataclass
class State:
    settings: Settings | None = None
    stop_requested: bool = False
    running: bool = False
    device: Device | None = None
    frame_index: int = 0
    first_device_ts_us: int | None = None


def clear_state() -> None:
    # Settings should stay persistent across runs
    STATE.device = None
    STATE.stop_requested = False
    STATE.running = False
    STATE.frame_index = 0
    STATE.first_device_ts_us = None


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


def submit_scene_frame(scene_frame: SimpleVideoFrame) -> None:
    dev_us = int(scene_frame.timestamp_unix_seconds * 1e6)

    if STATE.first_device_ts_us is None:
        STATE.first_device_ts_us = dev_us

    frame = syl.Frame()
    frame.mat = scene_frame.bgr_pixels  # already a numpy array
    frame.time_usec = timedelta(microseconds=dev_us - STATE.first_device_ts_us)
    frame.index = STATE.frame_index

    STATE.frame_index += 1

    out_scene.submit(frame)


def cleanup() -> None:
    settings = STATE.settings
    assert settings is not None

    device = STATE.device

    if device is None:
        return

    if settings.companion_recording_enabled:
        try:
            device.recording_stop_and_save()
        except Exception as exc:
            syl.println(f"Neon cleanup recording control failed: {exc}")

    try:
        device.close()
    except Exception as exc:
        syl.println(f"Failed to close Neon device: {exc}")

    syl.println("Cleanup complete")


# # ####################################################################################
# # Syntalos interface
# # ####################################################################################


out_scene = syl.get_output_port("scene")
out_scene.set_metadata_value("framerate", 30.0)
# Default Neon scene camera resolution
out_scene.set_metadata_value_size("size", [1600, 1200])


def prepare():
    clear_state()
    assert STATE.settings is not None

    try:
        device = connect_device()
        STATE.device = device
        return True
    except Exception as exc:
        msg = f"Prepare failed: {exc.__class__.__name__}({exc})"
        syl.println(msg)
        cleanup()
        syl.raise_error(msg)
        return False


def start() -> None:
    device = STATE.device
    settings = STATE.settings
    assert device is not None
    assert settings is not None

    try:
        device.streaming_start(STREAM_NAME_WORLD)
        if settings.companion_recording_enabled:
            _recording_id = device.recording_start()
    except Exception as exc:
        msg = f"Start failed: {exc.__class__.__name__}({exc})"
        syl.println(msg)
        cleanup()
        syl.raise_error(msg)


def run() -> None:
    STATE.running = True
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
    except Exception as exc:
        msg = f"Run failed: {exc.__class__.__name__}({exc})"
        syl.println(msg)
        syl.raise_error(msg)
    finally:
        cleanup()


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


def show_settings(settings: bytes) -> None:
    # Showing the settings UI while running prevents the run() loop from advancing.
    # Keep it simple: no settings UI while running.
    if STATE.running:
        syl.println("Cannot show settings while running")
        return

    if not settings:
        if STATE.settings is None:
            STATE.settings = Settings()
    else:
        STATE.settings = deserialise_settings(settings)

    assert STATE.settings is not None

    dialog: QDialog = uic.loadUi(UI_FILE_PATH)
    dialog.phoneIpLineEdit.setText(STATE.settings.phone_ip)
    dialog.phonePortSpinBox.setValue(STATE.settings.phone_port)
    dialog.discoveryTimeoutSpinBox.setValue(STATE.settings.discovery_timeout_s)
    dialog.frameWaitTimeoutSpinBox.setValue(STATE.settings.frame_wait_timeout_s)
    dialog.companionRecordingCheckBox.setChecked(STATE.settings.companion_recording_enabled)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        STATE.settings.phone_ip = dialog.phoneIpLineEdit.text().strip()
        STATE.settings.phone_port = dialog.phonePortSpinBox.value()
        STATE.settings.discovery_timeout_s = dialog.discoveryTimeoutSpinBox.value()
        STATE.settings.frame_wait_timeout_s = dialog.frameWaitTimeoutSpinBox.value()
        STATE.settings.companion_recording_enabled = dialog.companionRecordingCheckBox.isChecked()
        syl.save_settings(serialise_settings(STATE.settings))


# Register settings callback (called when settings dialog is shown)
syl.call_on_show_settings(show_settings)
