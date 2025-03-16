# Must be at the top:
#  Make sure that is the first thing you import, as otherwise the import of third-party packages installed in the blender environment will fail.
import blenderproc as bproc
from blenderproc.python.renderer import RendererUtility
from blenderproc.scripts.saveAsImg import convert_hdf

import os
import re
import uuid
import random
import argparse
import numpy as np
from pathlib import Path


def main(
    scene: str,
    output_dir: str,
    width: int,
    height: int,
    iterations: int,
    samples: int,
    min_distance: float,
    max_distance: float,
    camera_mode: str,
    camera_shift_top: float,
    camera_shift_bottom: float,
    camera_shift_left: float,
    camera_shift_right: float,
    lights_count: int,
    min_rotation: int,
    max_rotation: int,
    rotation_probability: float,
) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    bproc.init()
    RendererUtility.render_init()
    RendererUtility.set_max_amount_of_samples(samples)

    objects = bproc.loader.load_blend(
        scene,
        obj_types=[
            "mesh",
            "curve",
            "curves",
            "hair",
        ],
        data_blocks=[
            "curves",
            "hair_curves",
            "materials",
            "meshes",
            "objects",
            "textures",
        ],
    )

    bproc.camera.set_resolution(width, height)

    target_names = ["target"]
    targets = [
        obj
        for obj in objects
        if hasattr(obj, "get_name")
        and callable(getattr(obj, "get_name"))
        and any(target in obj.get_name() for target in target_names)
    ]

    if len(targets) == 0:
        raise ValueError(
            f"No objects starting with one of the following names found in {scene}: {target_names}"
        )

    lights = []
    for _ in range(lights_count):
        lights.append(create_random_light())

    i = 0
    while i < iterations:

        while True:

            for light in lights:
                light.set_location(random_light_location())

            if camera_mode == "frontal":
                start_angle = -135
                end_angle = -45
            elif camera_mode == "back":
                start_angle = 45
                end_angle = 135
            elif camera_mode == "frontal_and_back":
                random_bool = bool(random.getrandbits(1))
                start_angle = -135 if random_bool else 45
                end_angle = -45 if random_bool else 135
            elif camera_mode == "round":
                start_angle = -180
                end_angle = 180

            point_of_interest = bproc.object.compute_poi(targets)

            location = bproc.sampler.shell(
                center=[0, 0, random.uniform(1, 2)],
                radius_min=min_distance,
                radius_max=max_distance,
                elevation_min=-20,
                elevation_max=20,
                azimuth_min=start_angle,
                azimuth_max=end_angle,
            )

            point_of_interest = point_of_interest + compute_random_camera_shift(
                top=camera_shift_top,
                bottom=camera_shift_bottom,
                right=camera_shift_right,
                left=camera_shift_left,
            )

            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                point_of_interest - location,
                inplane_rot=compute_random_camera_rotation(
                    min_rotation, max_rotation, probability=rotation_probability
                ),
            )

            camera_matrix = bproc.math.build_transformation_mat(
                location, rotation_matrix
            )

            # Pose has to be added before evaluating object visibility
            bproc.camera.add_camera_pose(camera_matrix)

            targets_visible = all(
                bproc.camera.is_point_inside_camera_frustum(t.get_origin())
                for t in targets
            )

            if targets_visible:
                break
            else:
                bproc.utility.reset_keyframes()
                continue

        try:
            bproc.renderer.set_output_format(enable_transparency=True)
            data = bproc.renderer.render()
            bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)
            latest_hdf5_container = get_newest_hdf5_file(output_dir)

            # Extract image
            image_dir = output_dir + os.sep + "images"
            Path(image_dir).mkdir(parents=True, exist_ok=True)
            convert_hdf(base_file_path=latest_hdf5_container, output_folder=image_dir)

            # Randomize file name:
            # First remove hdf5 extension, then add png extension
            image_file_name = str(
                os.path.basename(Path(latest_hdf5_container).with_suffix(""))
                + "_colors.png"  # BlenderProc filename
            )
            randomize_file_name(image_dir + os.sep + image_file_name)
            bproc.utility.reset_keyframes()

            i += 1
        except Exception as e:
            print(f"Skipping {i} due to error: {e}")
        print(f"{i}/{iterations}")


def randomize_file_name(file_path: str) -> str:
    new_file_name = str(uuid.uuid4()) + os.path.splitext(file_path)[1]
    dir_name = os.path.dirname(file_path)
    new_file_path = os.path.join(dir_name, new_file_name)
    os.rename(file_path, new_file_path)
    return new_file_path


def get_newest_hdf5_file(output_dir):
    # Blender Proc appends a directory and adds files with increasing index as name
    # Regular expression to match the pattern "<INDEX_NUMBER>.hdf5"
    pattern = re.compile(r"^(\d+)\.hdf5$")

    max_index = -1
    newest_file_path = None

    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
                newest_file_path = filename

    return output_dir + os.sep + newest_file_path


def compute_random_camera_rotation(
    min_rot_degrees: int, max_rot_degrees: int, probability: float
):
    apply_rotation = random.random() < probability
    return (
        random.uniform(np.radians(min_rot_degrees), np.radians(max_rot_degrees))
        if apply_rotation
        else 0
    )


def compute_random_camera_shift(top: float, bottom: float, right: float, left: float):
    return [
        0,
        random.uniform(-left, right),
        random.uniform(-top, bottom),
    ]


def create_random_light():
    light = bproc.types.Light()
    light.set_type("POINT")
    light_location = random_light_location()
    light.set_location(light_location)
    light.set_energy(random.uniform(300, 500))
    return light


def random_light_location():
    location = bproc.sampler.shell(
        center=[0, random.uniform(0, 2), 2],
        radius_min=3,
        radius_max=3.5,
        elevation_min=-30,
        elevation_max=30,
    )
    return location


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        "-s",
        nargs="?",
        default="blender/man.blend",
        help="Path to the scene.obj file",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        nargs="?",
        default="hdf5",
        help="Path to where the final files will be saved",
    )

    parser.add_argument(
        "--width",
        "-w",
        type=int,
        default=256,
        help="Width of the camera resolution",
    )

    parser.add_argument(
        "--height",
        "-ht",
        type=int,
        default=256,
        help="Height of the camera resolution",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of iterations for rendering",
    )

    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=100,
        help="Number of iterations for rendering",
    )

    parser.add_argument(
        "--min_distance",
        "-min",
        type=float,
        default=2.0,
        help="Minimum distance for the camera",
    )

    parser.add_argument(
        "--max_distance",
        "-max",
        type=float,
        default=4.5,
        help="Maximum distance for the camera",
    )

    parser.add_argument(
        "--max_rotation",
        "-maxr",
        type=int,
        default=-45,
        help="Maximum camera rotation degrees",
    )

    parser.add_argument(
        "--min_rotation",
        "-minr",
        type=int,
        default=45,
        help="Minimum camera rotation degrees",
    )

    parser.add_argument(
        "--rotation_probability",
        "-rotp",
        type=float,
        default=0.2,
        help="Minimum camera rotation degrees",
    )

    parser.add_argument(
        "--camera_mode",
        "-c",
        type=str,
        default="frontal_and_back",
        choices=["frontal", "back", "frontal_and_back", "round"],
        help="Mode of camera positioning: 'frontal', 'back', or 'frontal_and_back'",
    )

    parser.add_argument(
        "--camera_shift_top",
        "-ct",
        type=float,
        default=0.6,
        help="Random camera shift applied",
    )

    parser.add_argument(
        "--camera_shift_bottom",
        "-cb",
        type=float,
        default=0.2,
        help="Random camera shift to the right applied",
    )

    parser.add_argument(
        "--camera_shift_left",
        "-cl",
        type=float,
        default=0.2,
        help="Random camera shift to the left applied",
    )

    parser.add_argument(
        "--camera_shift_right",
        "-cr",
        type=float,
        default=0.2,
        help="Random camera shift to the right applied",
    )

    parser.add_argument(
        "--lights",
        "-l",
        type=int,
        default=2,
        help="Number of light sources randomly placed",
    )

    args = parser.parse_args()

    main(
        scene=args.scene,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        iterations=args.iterations,
        samples=args.samples,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        camera_mode=args.camera_mode,
        camera_shift_top=args.camera_shift_top,
        camera_shift_bottom=args.camera_shift_bottom,
        camera_shift_left=args.camera_shift_left,
        camera_shift_right=args.camera_shift_right,
        lights_count=args.lights,
        min_rotation=args.min_rotation,
        max_rotation=args.max_rotation,
        rotation_probability=args.rotation_probability,
    )
