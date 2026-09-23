#!/usr/bin/env bash
# bash scripts/recon/to_gif.sh <video.mp4>  -> <video>.gif (palette) and <video>_fs.mp4 (faststart, plain attachment)
# ffmpeg is imageio's own (imageio_ffmpeg); python: $PY (default python)
set -euo pipefail
FF=$("${PY:-python}" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
in=$1; base=${in%.mp4}
"$FF" -y -loglevel error -i "$in" -vf "fps=10,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" "$base.gif"
"$FF" -y -loglevel error -i "$in" -c copy -movflags +faststart "${base}_fs.mp4"
ls -la "$base.gif" "${base}_fs.mp4" | awk '{print int($5/1024)" KiB", $9}'
