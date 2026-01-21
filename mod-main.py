# -*- coding: utf-8 -*-
"""
Pupil Labs Neon Syntalos Module

Streams comprehensive eye-tracking data from Pupil Labs Neon glasses including:
- Scene camera video stream
- Gaze position coordinates (x, y)
- Pupil diameter measurements (left and right)
- Eye state information (eyeball center, optical axis, eyelid data)
- Eye events (blinks, fixations, saccades)

Also controls Android Companion App recording in sync with Syntalos.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

import numpy as np

import syntalos_mlink as syl

from pupil_labs.realtime_api import Device, Network
from pupil_labs.realtime_api.streaming import (
    BlinkEventData,
    EyestateEyelidGazeData,
    EyestateGazeData,
    FixationEventData,
    FixationOnsetEventData,
    GazeData,
    VideoFrame,
    receive_eye_events_data,
    receive_gaze_data,
    receive_video_frames,
)


# =============================================================================
# Output Ports Configuration
# =============================================================================

# Scene video port
out_scene_video = syl.get_output_port("scene_video")

# Gaze position port (x, y coordinates)
out_gaze = syl.get_output_port("gaze")
out_gaze.set_metadata_value("signal_names", ["gaze_x", "gaze_y"])
out_gaze.set_metadata_value("time_unit", "microseconds")
out_gaze.set_metadata_value("data_unit", "pixels")

# Worn status port (whether glasses are being worn)
out_worn = syl.get_output_port("worn")
out_worn.set_metadata_value("signal_names", ["worn"])
out_worn.set_metadata_value("time_unit", "microseconds")
out_worn.set_metadata_value("data_unit", "boolean")

# Pupil diameter port (left and right)
out_pupil_diameter = syl.get_output_port("pupil_diameter")
out_pupil_diameter.set_metadata_value("signal_names", ["left_mm", "right_mm"])
out_pupil_diameter.set_metadata_value("time_unit", "microseconds")
out_pupil_diameter.set_metadata_value("data_unit", "mm")

# Eyeball center port (left: x,y,z; right: x,y,z)
out_eyeball_center = syl.get_output_port("eyeball_center")
out_eyeball_center.set_metadata_value(
    "signal_names",
    ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"],
)
out_eyeball_center.set_metadata_value("time_unit", "microseconds")
out_eyeball_center.set_metadata_value("data_unit", "mm")

# Optical axis port (left: x,y,z; right: x,y,z)
out_optical_axis = syl.get_output_port("optical_axis")
out_optical_axis.set_metadata_value(
    "signal_names",
    ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"],
)
out_optical_axis.set_metadata_value("time_unit", "microseconds")
out_optical_axis.set_metadata_value("data_unit", "normalized")

# Eyelid data port (angles and aperture)
out_eyelid = syl.get_output_port("eyelid")
out_eyelid.set_metadata_value(
    "signal_names",
    [
        "left_top_angle_rad",
        "left_bottom_angle_rad",
        "left_aperture_mm",
        "right_top_angle_rad",
        "right_bottom_angle_rad",
        "right_aperture_mm",
    ],
)
out_eyelid.set_metadata_value("time_unit", "microseconds")
out_eyelid.set_metadata_value("data_unit", "mixed")

# Blink events port
out_blinks = syl.get_output_port("blinks")
out_blinks.set_metadata_value("signal_names", ["start_time_ns", "end_time_ns", "duration_ns"])
out_blinks.set_metadata_value("time_unit", "microseconds")
out_blinks.set_metadata_value("data_unit", "nanoseconds")

# Fixation events port
out_fixations = syl.get_output_port("fixations")
out_fixations.set_metadata_value(
    "signal_names",
    [
        "start_time_ns",
        "end_time_ns",
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
    ],
)
out_fixations.set_metadata_value("time_unit", "microseconds")
out_fixations.set_metadata_value("data_unit", "mixed")

# Saccade events port
out_saccades = syl.get_output_port("saccades")
out_saccades.set_metadata_value(
    "signal_names",
    [
        "start_time_ns",
        "end_time_ns",
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
    ],
)
out_saccades.set_metadata_value("time_unit", "microseconds")
out_saccades.set_metadata_value("data_unit", "mixed")


# =============================================================================
# State Management
# =============================================================================


@dataclass
class State:
    """Global state for the Pupil Labs Neon module."""

    device_info: Any | None = None
    stop_requested: bool = False
    companion_recording_id: str | None = None
    # Async event loop running in a separate thread
    loop: asyncio.AbstractEventLoop | None = None
    loop_thread: threading.Thread | None = None
    # Queues for data from async streams to sync processing
    gaze_queue: Queue = field(default_factory=lambda: Queue(maxsize=1000))
    video_queue: Queue = field(default_factory=lambda: Queue(maxsize=100))
    eye_events_queue: Queue = field(default_factory=lambda: Queue(maxsize=1000))
    # Async tasks
    gaze_task: asyncio.Task | None = None
    video_task: asyncio.Task | None = None
    eye_events_task: asyncio.Task | None = None


STATE = State()


# =============================================================================
# Helper Functions
# =============================================================================


def log(msg: str) -> None:
    """Log a message to Syntalos console."""
    syl.println(f"[Neon] {msg}")


def ts_to_us(timestamp_unix_seconds: float) -> int:
    """Convert Unix timestamp in seconds to microseconds."""
    return int(timestamp_unix_seconds * 1e6)


# =============================================================================
# Gaze Data Processing
# =============================================================================


def emit_gaze_data(gaze: GazeData | EyestateGazeData | EyestateEyelidGazeData) -> None:
    """Emit gaze data to Syntalos output ports."""
    ts_us = ts_to_us(gaze.timestamp_unix_seconds)

    # Basic gaze position (available for all gaze data types)
    block = syl.FloatSignalBlock()
    block.timestamps = np.array([ts_us], dtype=np.uint64)
    block.data = np.array([[gaze.x, gaze.y]], dtype=np.float64)
    out_gaze.submit(block)

    # Worn status
    block = syl.IntSignalBlock()
    block.timestamps = np.array([ts_us], dtype=np.uint64)
    block.data = np.array([[1 if gaze.worn else 0]], dtype=np.int32)
    out_worn.submit(block)

    # Extended eye state data (EyestateGazeData and EyestateEyelidGazeData)
    if isinstance(gaze, (EyestateGazeData, EyestateEyelidGazeData)):
        # Pupil diameter
        block = syl.FloatSignalBlock()
        block.timestamps = np.array([ts_us], dtype=np.uint64)
        block.data = np.array(
            [[gaze.pupil_diameter_left, gaze.pupil_diameter_right]], dtype=np.float64
        )
        out_pupil_diameter.submit(block)

        # Eyeball center
        block = syl.FloatSignalBlock()
        block.timestamps = np.array([ts_us], dtype=np.uint64)
        block.data = np.array(
            [
                [
                    gaze.eyeball_center_left_x,
                    gaze.eyeball_center_left_y,
                    gaze.eyeball_center_left_z,
                    gaze.eyeball_center_right_x,
                    gaze.eyeball_center_right_y,
                    gaze.eyeball_center_right_z,
                ]
            ],
            dtype=np.float64,
        )
        out_eyeball_center.submit(block)

        # Optical axis
        block = syl.FloatSignalBlock()
        block.timestamps = np.array([ts_us], dtype=np.uint64)
        block.data = np.array(
            [
                [
                    gaze.optical_axis_left_x,
                    gaze.optical_axis_left_y,
                    gaze.optical_axis_left_z,
                    gaze.optical_axis_right_x,
                    gaze.optical_axis_right_y,
                    gaze.optical_axis_right_z,
                ]
            ],
            dtype=np.float64,
        )
        out_optical_axis.submit(block)

    # Eyelid data (only EyestateEyelidGazeData)
    if isinstance(gaze, EyestateEyelidGazeData):
        block = syl.FloatSignalBlock()
        block.timestamps = np.array([ts_us], dtype=np.uint64)
        block.data = np.array(
            [
                [
                    gaze.eyelid_angle_top_left,
                    gaze.eyelid_angle_bottom_left,
                    gaze.eyelid_aperture_left,
                    gaze.eyelid_angle_top_right,
                    gaze.eyelid_angle_bottom_right,
                    gaze.eyelid_aperture_right,
                ]
            ],
            dtype=np.float64,
        )
        out_eyelid.submit(block)


# =============================================================================
# Eye Events Processing
# =============================================================================


def emit_eye_event(
    event: BlinkEventData | FixationEventData | FixationOnsetEventData,
) -> None:
    """Emit eye event data to Syntalos output ports."""
    ts_us = ts_to_us(event.rtp_ts_unix_seconds)

    if isinstance(event, BlinkEventData):
        # Blink event
        duration_ns = event.end_time_ns - event.start_time_ns
        block = syl.IntSignalBlock()
        block.timestamps = np.array([ts_us], dtype=np.uint64)
        block.data = np.array(
            [[event.start_time_ns, event.end_time_ns, duration_ns]], dtype=np.int64
        )
        out_blinks.submit(block)

    elif isinstance(event, FixationEventData):
        # Fixation or Saccade event (event_type: 0=saccade, 1=fixation)
        block = syl.FloatSignalBlock()
        block.timestamps = np.array([ts_us], dtype=np.uint64)
        block.data = np.array(
            [
                [
                    float(event.start_time_ns),
                    float(event.end_time_ns),
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
            dtype=np.float64,
        )
        if event.event_type == 0:  # Saccade
            out_saccades.submit(block)
        else:  # Fixation (event_type == 1)
            out_fixations.submit(block)

    # FixationOnsetEventData only indicates the start of an event, not processed here


# =============================================================================
# Video Frame Processing
# =============================================================================


def emit_video_frame(frame: VideoFrame) -> None:
    """Emit video frame to Syntalos output port."""
    from datetime import timedelta

    ts_us = ts_to_us(frame.timestamp_unix_seconds)

    # Get BGR buffer from video frame
    bgr_buffer = frame.bgr_buffer()

    # Create Syntalos Frame
    syl_frame = syl.Frame()
    syl_frame.time_usec = timedelta(microseconds=ts_us)
    syl_frame.mat = bgr_buffer

    out_scene_video.submit(syl_frame)


# =============================================================================
# Async Streaming Coroutines
# =============================================================================


async def stream_gaze(url: str) -> None:
    """Stream gaze data and put it in the queue."""
    try:
        async for gaze in receive_gaze_data(url, run_loop=True):
            if STATE.stop_requested:
                break
            try:
                STATE.gaze_queue.put_nowait(gaze)
            except Exception:
                pass  # Queue full, drop oldest data
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log(f"Gaze stream error: {e}")


async def stream_video(url: str) -> None:
    """Stream video frames and put them in the queue."""
    try:
        async for frame in receive_video_frames(url, run_loop=True):
            if STATE.stop_requested:
                break
            try:
                STATE.video_queue.put_nowait(frame)
            except Exception:
                pass  # Queue full, drop oldest data
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log(f"Video stream error: {e}")


async def stream_eye_events(url: str) -> None:
    """Stream eye events and put them in the queue."""
    try:
        async for event in receive_eye_events_data(url, run_loop=True):
            if STATE.stop_requested:
                break
            try:
                STATE.eye_events_queue.put_nowait(event)
            except Exception:
                pass  # Queue full, drop oldest data
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log(f"Eye events stream error: {e}")


async def run_async_streams(
    gaze_url: str | None, video_url: str | None, eye_events_url: str | None
) -> None:
    """Run all async streams concurrently."""
    tasks = []

    if gaze_url:
        STATE.gaze_task = asyncio.create_task(stream_gaze(gaze_url))
        tasks.append(STATE.gaze_task)

    if video_url:
        STATE.video_task = asyncio.create_task(stream_video(video_url))
        tasks.append(STATE.video_task)

    if eye_events_url:
        STATE.eye_events_task = asyncio.create_task(stream_eye_events(eye_events_url))
        tasks.append(STATE.eye_events_task)

    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass


def run_event_loop(gaze_url: str | None, video_url: str | None, eye_events_url: str | None) -> None:
    """Run the async event loop in a separate thread."""
    STATE.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(STATE.loop)
    try:
        STATE.loop.run_until_complete(run_async_streams(gaze_url, video_url, eye_events_url))
    finally:
        STATE.loop.close()
        STATE.loop = None


# =============================================================================
# Companion App Recording Control
# =============================================================================


def start_companion_recording_sync() -> str | None:
    """Start recording on the Android Companion App using a fresh connection."""
    if STATE.device_info is None:
        log("Cannot start recording: no device info")
        return None

    async def _start() -> str:
        async with Device.from_discovered_device(STATE.device_info) as device:
            recording_id = await device.recording_start()
            return recording_id

    try:
        recording_id = asyncio.run(_start())
        log(f"Companion App recording started: {recording_id}")
        return recording_id
    except Exception as e:
        log(f"Failed to start Companion App recording: {e}")
        return None


def stop_companion_recording_sync() -> None:
    """Stop recording on the Android Companion App using a fresh connection."""
    if STATE.device_info is None:
        log("Cannot stop recording: no device info")
        return

    async def _stop() -> None:
        async with Device.from_discovered_device(STATE.device_info) as device:
            await device.recording_stop_and_save()

    try:
        asyncio.run(_stop())
        log("Companion App recording stopped and saved")
    except Exception as e:
        log(f"Failed to stop Companion App recording: {e}")


# =============================================================================
# Syntalos Module Lifecycle Functions
# =============================================================================


def set_settings(settings: bytes) -> None:
    """Handle settings from Syntalos."""
    log(f"Settings received: {settings}")
    # TODO: Parse settings if needed (e.g., device IP, recording options)


def prepare() -> bool:
    """
    Prepare the module for streaming.

    Discovers the Pupil Labs Neon device and establishes connection.
    """
    log("Preparing Pupil Labs Neon module...")

    try:
        # Discover device
        async def discover() -> Any:
            async with Network() as network:
                dev_info = await network.wait_for_new_device(timeout_seconds=10)
            return dev_info

        STATE.device_info = asyncio.run(discover())

        if STATE.device_info is None:
            log("ERROR: No Pupil Labs Neon device found!")
            return False

        log(f"Found device: {STATE.device_info.name}")

        # Connect to device temporarily to get sensor URLs
        async def get_sensor_urls() -> tuple[str | None, str | None, str | None]:
            async with Device.from_discovered_device(STATE.device_info) as device:
                status = await device.get_status()

                gaze_sensor = status.direct_gaze_sensor()
                world_sensor = status.direct_world_sensor()
                eye_events_sensor = status.direct_eye_events_sensor()

                gaze_url = gaze_sensor.url if gaze_sensor.connected else None
                video_url = world_sensor.url if world_sensor.connected else None
                eye_events_url = eye_events_sensor.url if eye_events_sensor.connected else None

                return gaze_url, video_url, eye_events_url

        gaze_url, video_url, eye_events_url = asyncio.run(get_sensor_urls())

        if not gaze_url:
            log("WARNING: Gaze sensor not connected")
        if not video_url:
            log("WARNING: Scene camera not connected")
        if not eye_events_url:
            log("WARNING: Eye events sensor not connected (requires Companion App 2.9+)")

        # Store URLs for later use
        STATE.gaze_url = gaze_url  # type: ignore
        STATE.video_url = video_url  # type: ignore
        STATE.eye_events_url = eye_events_url  # type: ignore

        log("Preparation complete")
        return True

    except Exception as e:
        log(f"ERROR during preparation: {e}")
        return False


def start() -> None:
    """
    Start the module.

    Starts the async event loop thread for streaming and begins
    Companion App recording.
    """
    log("Starting Pupil Labs Neon module...")

    # Get URLs from state
    gaze_url = getattr(STATE, "gaze_url", None)
    video_url = getattr(STATE, "video_url", None)
    eye_events_url = getattr(STATE, "eye_events_url", None)

    # Start async event loop in a separate thread
    STATE.loop_thread = threading.Thread(
        target=run_event_loop,
        args=(gaze_url, video_url, eye_events_url),
        daemon=True,
    )
    STATE.loop_thread.start()

    # Wait a moment for the loop to start
    time.sleep(0.5)

    # Start Companion App recording
    STATE.companion_recording_id = start_companion_recording_sync()

    log("Module started")


def run() -> None:
    """
    Main run loop.

    Processes data from the queues and emits to Syntalos output ports.
    Uses syl.wait() to allow Syntalos IPC communication.
    """
    log("Running Pupil Labs Neon module...")

    try:
        while not STATE.stop_requested and syl.is_running():
            processed_any = False

            # Process gaze data
            try:
                gaze = STATE.gaze_queue.get_nowait()
                emit_gaze_data(gaze)
                processed_any = True
            except Empty:
                pass

            # Process video frames
            try:
                frame = STATE.video_queue.get_nowait()
                emit_video_frame(frame)
                processed_any = True
            except Empty:
                pass

            # Process eye events
            try:
                event = STATE.eye_events_queue.get_nowait()
                emit_eye_event(event)
                processed_any = True
            except Empty:
                pass

            # If no data was processed, wait a bit to allow Syntalos IPC
            # This is critical - it allows Syntalos to call stop() when needed
            if not processed_any:
                syl.wait(5)  # 5 ms wait

    finally:
        cleanup()


def stop() -> None:
    """
    Stop the module.

    Sets the stop flag to signal all loops to terminate.
    Uses the Deferred Cleanup Pattern like the Shimmer module.
    """
    log("Stopping Pupil Labs Neon module...")

    # Stop Companion App recording first
    if STATE.companion_recording_id:
        stop_companion_recording_sync()
        STATE.companion_recording_id = None

    # Signal all loops to stop
    STATE.stop_requested = True


def cleanup() -> None:
    """
    Clean up resources.

    Cancels async tasks, stops the event loop, and closes the device connection.
    """
    log("Cleaning up Pupil Labs Neon module...")

    # Cancel async tasks
    if STATE.loop and STATE.gaze_task:
        STATE.loop.call_soon_threadsafe(STATE.gaze_task.cancel)
    if STATE.loop and STATE.video_task:
        STATE.loop.call_soon_threadsafe(STATE.video_task.cancel)
    if STATE.loop and STATE.eye_events_task:
        STATE.loop.call_soon_threadsafe(STATE.eye_events_task.cancel)

    # Wait for the event loop thread to finish
    if STATE.loop_thread and STATE.loop_thread.is_alive():
        STATE.loop_thread.join(timeout=5.0)

    # Clear queues
    while not STATE.gaze_queue.empty():
        try:
            STATE.gaze_queue.get_nowait()
        except Empty:
            break
    while not STATE.video_queue.empty():
        try:
            STATE.video_queue.get_nowait()
        except Empty:
            break
    while not STATE.eye_events_queue.empty():
        try:
            STATE.eye_events_queue.get_nowait()
        except Empty:
            break

    # Reset state
    STATE.device_info = None
    STATE.stop_requested = False
    STATE.loop = None
    STATE.loop_thread = None
    STATE.gaze_task = None
    STATE.video_task = None
    STATE.eye_events_task = None

    log("Cleanup complete")


# =============================================================================
# Module Initialization
# =============================================================================

# Register settings callback (called when settings dialog is shown)
syl.call_on_show_settings(set_settings)

# Note: The functions prepare(), start(), run(), stop() are automatically
# called by Syntalos based on their names. No explicit registration needed.
