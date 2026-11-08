#!/usr/bin/env python3
"""Records the clips of the Examples gallery (rsc/docs/video/examples).

Every example with a window runs headless on the GPU (SDL's offscreen video
driver), and framecap.so, loaded with LD_PRELOAD, writes its frames to a raw
MP4. rayrai examples are recorded from their own window; server examples from
rayrai_tcp_viewer, which is started first and connects as soon as the server is
up. The raw clip is then trimmed, cropped to 16:10 and encoded as a small
960x600 tile, with a poster frame. clips.json holds the settings of each clip:

  kind         "rayrai", "server", or "comparison" (a still the example writes:
               the clip wipes from its clean left half to its noisy right half)
  binary/args  program to run (default: the example's name) and its arguments;
               "{work}" in an argument becomes the work directory
  viewer_args  extra rayrai_tcp_viewer arguments: --auto-frame or a fixed
               --camera-lookat for server examples
  start        seconds after the first frame before recording (scenes load first)
  seconds      raw clip length; load is the extra time allowed for loading
  every_frame  keep every frame instead of sampling wall-clock time (for an
               example that advances its simulation by a fixed step per frame)
  crop         trim, dur (seconds) and zoom, cx, cy (crop centre, 0..1) of the tile

Requires Linux, a C compiler, ffmpeg/ffprobe, and the examples built from
raisim2Lib/examples against the RaiSim and rayrai packages.

usage: record_example_clips.py --build <examples build dir> [--out DIR] [--work DIR] [name ...]
"""

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CLIPS = json.loads((HERE / "clips.json").read_text(encoding="utf-8"))
SERVER_ENDPOINT = "127.0.0.1:8080"


def build_hook(work):
    hook = work / "framecap.so"
    subprocess.run([os.environ.get("CC", "cc"), "-shared", "-fPIC", "-O2", "-o", str(hook),
                    str(HERE / "framecap.c"), "-ldl"], check=True)
    return hook


def environment(extra):
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["SDL_VIDEODRIVER"] = "offscreen"
    library_dirs = [REPO / "rayrai/lib", REPO / "raisim/lib", REPO / "raisim/lib/openusd/lib"]
    env["LD_LIBRARY_PATH"] = ":".join([*map(str, library_dirs), env.get("LD_LIBRARY_PATH", "")])
    env.update(extra)
    return env


def start(command, cwd, env, log):
    return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def record_window(name, spec, build, raw, hook, work):
    """Runs the example (and the viewer for a server example) and records a raw clip."""
    capture = {"LD_PRELOAD": str(hook), "FRAMECAP_OUT": str(raw), "FRAMECAP_EXIT": "1",
               "FRAMECAP_START": str(spec.get("start", 4)), "FRAMECAP_SECONDS": str(spec.get("seconds", 8)),
               "FRAMECAP_WIDTH": "0", "FRAMECAP_CRF": "16", "FRAMECAP_FPS": str(spec.get("fps", 30)),
               "FRAMECAP_EVERY_FRAME": "1" if spec.get("every_frame") else "0"}
    program = [str(build / spec.get("binary", name)),
               *(arg.replace("{work}", str(work)) for arg in spec.get("args", []))]
    budget = spec.get("start", 4) + spec.get("seconds", 8) + spec.get("load", 40)
    with open(work / f"{name}.log", "w") as log:
        server = None
        try:
            if spec["kind"] == "server":
                viewer = [str(build / "rayrai_tcp_viewer"), "--connect", SERVER_ENDPOINT, "--auto-connect",
                          "--window-size", spec.get("size", "1920x1200"), "--no-save-settings",
                          *([] if spec.get("no_minimize") else ["--minimize-panels"]), *spec.get("viewer_args", [])]
                app = start(viewer, build, environment(capture), log)
                time.sleep(2.5)
                server = start(program, build, environment({}), log)
            else:
                app = start(program, build, environment(capture), log)
            try:
                app.wait(budget)
            except subprocess.TimeoutExpired:
                pass
            stop(app)
        finally:
            if server:
                stop(server)


def record_comparison(name, spec, build, raw, work):
    """Runs an example that writes a clean | noisy image pair and wipes between the halves."""
    with open(work / f"{name}.log", "w") as log:
        subprocess.run([str(build / spec.get("binary", name))], cwd=build, env=environment({}),
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    wipe = ("[0:v]split[a][b];[a]crop=iw/2:ih:0:0,format=yuv420p[clean];[b]crop=iw/2:ih:iw/2:0,format=yuv420p[noisy];"
            "[clean]split[c1][c2];[c1][noisy]xfade=transition=wiperight:duration=1.2:offset=1.5[cn];"
            "[cn][c2]xfade=transition=wipeleft:duration=1.2:offset=5.2,trim=0:8,setpts=PTS-STARTPTS")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1", "-t", "8", "-i",
                    str(build / spec["image"]), "-filter_complex", wipe, "-r", "30", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", str(raw)], check=True)


def frame_size(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", str(path)], text=True)
    return [int(value) for value in out.strip().split(",")]


def finish(name, spec, raw, out):
    """Trims the raw clip, crops it to 16:10 around (cx, cy) and writes the tile and its poster."""
    crop = {"trim": 1.0, "dur": 5.0, "zoom": 1.0, "cx": 0.5, "cy": 0.5, **spec.get("crop", {})}
    width, height = frame_size(raw)
    crop_w = min(width, height * 16 / 10) / crop["zoom"]
    crop_h = crop_w * 10 / 16
    x = min(max(crop["cx"] * width - crop_w / 2, 0), width - crop_w)
    y = min(max(crop["cy"] * height - crop_h / 2, 0), height - crop_h)
    clip, poster = out / f"{name}.mp4", out / f"{name}.jpg"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(crop["trim"]), "-t", str(crop["dur"]),
                    "-i", str(raw), "-an", "-vf",
                    f"crop={int(crop_w)}:{int(crop_h)}:{int(x)}:{int(y)},scale=960:600:flags=lanczos", "-r", "30",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-maxrate", "1000k", "-bufsize", "2000k",
                    "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", str(clip)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(crop["dur"] / 2), "-i", str(clip),
                    "-frames:v", "1", "-q:v", "5", str(poster)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", type=Path, required=True, help="build directory of raisim2Lib/examples")
    parser.add_argument("--out", type=Path, default=REPO / "rsc/docs/video/examples")
    parser.add_argument("--work", type=Path, help="keeps raw clips and logs (default: a temporary directory)")
    parser.add_argument("names", nargs="*", help="examples to record (default: all in clips.json)")
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(CLIPS))
    if unknown:
        parser.error(f"not in clips.json: {', '.join(unknown)}")
    with tempfile.TemporaryDirectory(prefix="example-clips-") as temporary:
        work = (args.work or Path(temporary)).resolve()
        work.mkdir(parents=True, exist_ok=True)
        hook = build_hook(work)
        failed = []
        for name in args.names or sorted(CLIPS):
            spec, raw = CLIPS[name], work / f"{name}.raw.mp4"
            raw.unlink(missing_ok=True)
            if spec["kind"] == "comparison":
                record_comparison(name, spec, args.build.resolve(), raw, work)
            else:
                record_window(name, spec, args.build.resolve(), raw, hook, work)
            if not raw.is_file() or raw.stat().st_size < 10_000:
                failed.append(name)
                print(f"{name}: no clip (see {work / (name + '.log')})", flush=True)
                continue
            finish(name, spec, raw, args.out)
            print(f"{name}: {(args.out / (name + '.mp4')).stat().st_size // 1024} KB", flush=True)
        if failed:
            raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
