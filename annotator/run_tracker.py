"Command line interface for the tracker"
import sys
import os
import os.path
import argparse

import h5py

import kcftracker # pylint: disable=import-error

PACKAGE_PATH = os.path.dirname(kcftracker.__file__)
CONFIG_PATH = os.path.join(PACKAGE_PATH, 'KCF_config.yml')

def compile_tracker(args):  # pylint: disable=unused-argument
    "Compiles the tracker, making it significantly faster."
    kcftracker.compile_fhog()
    return 0

def run_tracker(args):
    "Runs the tracker on the given cache file."
    tracker_config = kcftracker.load_config(CONFIG_PATH)
    tracker = kcftracker.KCFTracker(tracker_config)
    hdf5_file = h5py.File(args.cache_file, 'r')
    # The input image was scaled during cache creation
    global_scale = hdf5_file['scale'][0]
    bbox = (
        global_scale * args.x,
        global_scale * args.y,
        global_scale * args.width,
        global_scale * args.height
    )
    frame = hdf5_file['img'][args.start_frame, ...]
    tracker.init(bbox, frame)

    low = args.start_frame + 1
    num_images = hdf5_file['img'].shape[0]
    high = min(args.last_frame + 1, num_images)
    for current_frame in range(low, high):
        frame = hdf5_file['img'][current_frame, ...]
        tracker_success, bbox = tracker.update(frame)
        if tracker_success:
            print(
                current_frame,
                bbox[0] // global_scale,
                bbox[1] // global_scale,
                bbox[2] // global_scale,
                bbox[3] // global_scale,
                file=sys.stdout)
        else:
            print(current_frame, "Tracking failure", file=sys.stderr)
            break
    return 0


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser(
        prog="run_tracker", description="Run or compile the tracker.")
    # Setting up subcommands
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True
    compile_doc = compile_tracker.__doc__
    compile_parser = subparsers.add_parser(
        "compile", help=compile_doc, description=compile_doc)
    compile_parser.set_defaults(func=compile_tracker)
    run_doc = run_tracker.__doc__
    run_parser = subparsers.add_parser(
        "run", help=run_doc, description=run_doc)
    run_parser.set_defaults(func=run_tracker)
    run_parser.add_argument(
        "x",
        help="x coordinate of the top right corner of the target box.",
        type=float)
    run_parser.add_argument(
        "y",
        help="y coordinate of the top right corner of the target box.",
        type=float)
    run_parser.add_argument(
        "width", help="Width of the target box.", type=float)
    run_parser.add_argument(
        "height", help="Height of the target box.", type=float)
    run_parser.add_argument(
        "cache_file",
        help="Path to cache file with the images.")
    run_parser.add_argument(
        "start_frame",
        help="Frame where the target box was selected.",
        type=int)
    run_parser.add_argument(
        "last_frame",
        help="Last frame where the tracker should run.",
        type=int)
    args = parser.parse_args(sys.argv[1:])
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
