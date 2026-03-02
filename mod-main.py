"""Pupil Labs Neon Syntalos Module"""

import syntalos_mlink as syl

from dataclasses import dataclass


@dataclass
class State:
    stop_requested: bool = False


STATE = State()


# ## ###############################################################################################
# ## Syntalos interface
# ## ###############################################################################################

out_scene = syl.get_output_port("scene")
out_scene.set_metadata_value("framerate", 30.0)
# Default Neon scene camera resolution
out_scene.set_metadata_value_size("size", [1600, 1200])


def prepare() -> bool:
    return False


def start() -> None:
    pass


def run() -> None:
    try:
        while not STATE.stop_requested and syl.is_running():
            syl.wait(20)
    finally:
        cleanup()


def stop() -> None:
    STATE.stop_requested = True


def cleanup() -> None:
    STATE.stop_requested = False


def set_settings(settings: bytes) -> None:
    pass


# Register settings callback (called when settings dialog is shown)
syl.call_on_show_settings(set_settings)


# ## ###############################################################################################
# ## Settings UI
# ## ###############################################################################################
