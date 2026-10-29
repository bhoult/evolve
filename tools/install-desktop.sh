#!/bin/sh
# Installs evolve.desktop for this user: the desktop matches it to evolve's window
# by its app id ("evolve") for the name and icon in the dock and the app grid. It
# runs this checkout's release build, from the checkout. The icon, assets/icon.png,
# is a render of assets/icon.json (tools/make-icon.py).
# Remove: rm ~/.local/share/applications/evolve.desktop
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
apps=${XDG_DATA_HOME:-$HOME/.local/share}/applications
mkdir -p "$apps"
cat > "$apps/evolve.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=evolve
Comment=3D evolution simulator: creatures with physics bodies and neural brains
Exec=$root/target/release/evolve
Path=$root
Icon=$root/assets/icon.png
Terminal=false
Categories=Science;Simulation;
StartupWMClass=evolve
EOF
update-desktop-database "$apps" 2>/dev/null || true
echo "installed $apps/evolve.desktop"
