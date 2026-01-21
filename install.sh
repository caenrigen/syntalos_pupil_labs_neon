#!/bin/bash
if [ "$(uname)" != "Linux" ]; then
    echo "Script intended for Linux!"
    exit 1
fi

# Create a symbolic link to this directory in the Syntalos modules directory.
SHIMMER_MODULE_SRC="$(pwd)/"

# When installed from Flatpak
# SYNTALOS_MODULES_DIR="$HOME/.var/app/org.syntalos.syntalos/data/modules"
# When built and installed from source
SYNTALOS_MODULES_DIR="$HOME/.local/share/Syntalos/modules"

mkdir -p $SYNTALOS_MODULES_DIR
SYMLINK="$SYNTALOS_MODULES_DIR/pupil_labs_neon"

ln --verbose -s $SHIMMER_MODULE_SRC $SYMLINK