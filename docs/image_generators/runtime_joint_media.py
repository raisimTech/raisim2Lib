#!/usr/bin/env python3
"""Record the runtime-joint example's tool-release and two-robot docking scenes."""

import argparse
import socket
import subprocess
import tempfile
from pathlib import Path

from example_clips import record_example_clips as clips

REPO = Path(__file__).resolve().parents[2]


def require_available_endpoint(port=8080):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"Port {port} is occupied; stop the existing RaisimServer before recording") from error


def record(build, output, work):
    require_available_endpoint()
    hook = clips.build_hook(work)
    tool = clips.CLIPS["runtime_joints"]
    docking = {**tool, "binary": "runtime_joints",
               "args": ["--visualize", "--scene", "docking"],
               "viewer_args": ["--camera-lookat", "2.4,-2.9,2.0,0.15,0,0.25",
                               "--force-camera-lookat"]}
    for name, spec, video_dir in (
            ("runtime_joints", tool, output / "video/examples"),
            ("runtime_joints_docking", docking, output / "video")):
        video_dir.mkdir(parents=True, exist_ok=True)
        raw = work / f"{name}.raw.mp4"
        require_available_endpoint()
        clips.record_window(name, spec, build, raw, hook, work)
        if not raw.is_file() or raw.stat().st_size < 10_000:
            raise RuntimeError(f"No frames recorded; see {work / (name + '.log')}")
        clips.finish(name, spec, raw, video_dir)
        image_dir = output / "image"
        image_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "0.5",
                        "-i", str(raw), "-frames:v", "1",
                        str(image_dir / f"{name}.png")], check=True)
        print(f"Recorded {name}: {video_dir / (name + '.mp4')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True,
                        help="Directory containing runtime_joints and rayrai_tcp_viewer")
    parser.add_argument("--out", type=Path, default=REPO / "rsc/docs")
    parser.add_argument("--work", type=Path, help="Keep raw recordings and logs here")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="runtime-joint-media-") as temporary:
        work = (args.work or Path(temporary)).resolve()
        work.mkdir(parents=True, exist_ok=True)
        record(args.build.resolve(), args.out.resolve(), work)


if __name__ == "__main__":
    main()
