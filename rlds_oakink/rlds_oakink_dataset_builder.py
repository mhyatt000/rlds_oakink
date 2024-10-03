import glob
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


def downscale_to_224(height: int, width: int) -> Tuple[int, int]:
    """
    Downscale the image so that the shorter dimension is 224 pixels,
    and the longer dimension is scaled by the same ratio.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        Tuple[int, int]: The new height and width of the image.
    """
    # Determine the scaling ratio
    if height < width:
        ratio = 224.0 / height
        new_height = 224
        new_width = int(width * ratio)
    else:
        ratio = 224.0 / width
        new_width = 224
        new_height = int(height * ratio)

    return new_height, new_width


class RLDSOakink(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for OakInk v1.0.2"""

    VERSION = tfds.core.Version("1.0.2")
    RELEASE_NOTES = {"1.0.2": "Initial release."}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(224, 224, 3),  #  shape=(480, 848, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    ### CAMERA
                                    "cam_intr": tfds.features.Tensor(
                                        shape=[3, 3],
                                        dtype=np.float32,
                                        doc="Camera intrinsics matrix",
                                    ),
                                    # 'cam_center': tfds.features.Tensor(
                                    # shape=[2],
                                    # dtype=np.float32,
                                    # doc="Camera center",
                                    # ),
                                    # 'bbox_center': tfds.features.Tensor(
                                    # shape=[2],
                                    # dtype=np.int64,
                                    # doc="Center of bounding box around hand",
                                    # ),
                                    # 'bbox_scale': tfds.features.Tensor(
                                    # shape=[],
                                    # dtype=np.float32,
                                    # doc="Scale of bounding box around hand (pixels)",
                                    # ),
                                    # "raw_size": tfds.features.Tensor(
                                    # shape=[2],
                                    # dtype=np.int32,
                                    # doc="Raw image size",
                                    # ),
                                    ### MANO
                                    "mano_pose": tfds.features.Tensor(
                                        shape=[16, 3],
                                        dtype=np.float32,
                                        doc="MANO pose parameters 16x3",
                                    ),
                                    "mano_shape": tfds.features.Tensor(
                                        shape=[10],
                                        dtype=np.float32,
                                        doc="MANO shape parameters 10x",
                                    ),
                                    ### JOINT POSITIONS
                                    "joints_3d": tfds.features.Tensor(
                                        shape=[21, 3],
                                        dtype=np.float32,
                                    ),
                                    # "joints_2d": tfds.features.Tensor(
                                    # shape=[21, 2],
                                    # dtype=np.float32,
                                    # ),
                                    # "joints_uvd": tfds.features.Tensor(
                                    # shape=[21, 3],
                                    # dtype=np.float32,
                                    # ),
                                    # "joints_vis": tfds.features.Tensor(
                                    # shape=[21],
                                    # dtype=np.float32,
                                    # doc="joint visibility? TODO",
                                    # ),
                                    ### MESH VERTEX POSITIONS
                                    # "verts_3d": tfds.features.Tensor(
                                    # shape=[778, 3],
                                    # dtype=np.float32,
                                    # ),
                                    # "verts_uvd": tfds.features.Tensor(
                                    # shape=[778, 3],
                                    # dtype=np.float32,
                                    # ),
                                }
                            ),
                            # "action": tfds.features.Tensor(
                            # shape=(10,),
                            # dtype=np.float32,
                            # doc="Robot action, consists of [7x joint velocities, "
                            # "2x gripper velocities, 1x terminate episode].",
                            # ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        import bafl
        from bafl.scripts import make_oakink_torch

        return {
            "train": self._generate_examples(
                bafl.scripts.make_oakink_torch.make(train=True)
            ),
            # might need to be packed first
            # "test": self._generate_examples( bafl.scripts.make_oakink_torch.make(train=False)),
        }

    def _generate_examples(self, ds) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

        def _parse_example(data):

            paths = [d["image_path"] for d in data]

            keys = [
                "idx",  # popped
                "cam_center",
                "bbox_center",
                "bbox_scale",
                "cam_intr",
                #
                "joints_2d",  # popped
                "joints_3d",  # popped
                "joints_vis",  # popped
                "joints_uvd",  # popped
                #
                "verts_3d",  # popped
                "verts_uvd",  # popped
                #
                "raw_size",
                "image_path",  # popped
                "image_mask",  # popped
                #
                "mano_pose",
                "mano_shape",
                "task",  # popped
                "image",
            ]

            task = data[0]["task"]
            lang = self._embed([task])[0].numpy()  # embedding takes ≈0.06s

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding

                # step = {k: v for k, v in step.items() if k in keys}

                task = step.pop("task")

                # main params
                step["joints_3d"] = step.pop("target_joints_3d")
                step["cam_intr"] = step.pop("target_cam_intr")
                step["mano_pose"] = step.pop("target_mano_pose")
                step["mano_shape"] = step.pop("target_mano_shape")

                # keep in raw form
                # img_shape = downscale_to_224(*image.shape[:2])
                # image = tf.image.resize(image, img_shape).numpy().astype(np.uint8)

                # needs to be reshaped because of the transforms
                image = ((np.array(step["image"]) + 0.5) * 255).astype(np.uint8).transpose(1,2,0)

                step.pop("image_path")
                step.pop("image_mask")
                step.pop("joints_2d")
                step.pop("joints_vis")
                step.pop("joints_uvd")
                step.pop("verts_3d")
                step.pop("verts_uvd")
                idx = step.pop("idx")

                episode.append(
                    {
                        "observation": {
                            # **step,
                            "image": image,
                            "joints_3d": step["joints_3d"],
                            "cam_intr": step["cam_intr"],
                            "mano_pose": step["mano_pose"],
                            "mano_shape": step["mano_shape"],
                            # "wrist_image": step["wrist_image"],
                            # "state": step,
                        },
                        # "action": step["action"],
                        "discount": 1.0,
                        "reward": float(i == (len(data) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(data) - 1),
                        "is_terminal": i == (len(data) - 1),
                        "language_instruction": task,
                        "language_embedding": lang,
                    }
                )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {"file_path": paths[0]}}

            # if you want to skip an example for whatever reason, simply return None
            return idx, sample

        """
        # randomly select 500 samples to print
        idxs = np.random.choice(len(ds), 500, replace=False)
        for sample in idxs[:1]:
            sample = ds[sample]
            print(sample[0].keys())
            print([k for k in sample[0].keys() if 'mano' in k])
            # print( { k: v.dtype for k, v in sample[0].items() if isinstance(v, np.ndarray) })
            print(sample[0]['bbox_scale'])

            print(sample[0]['image'].shape)
            print(type(sample[0]['image']).shape)

        quit()
        """

        for sample in ds:
            if sample[0]["intent"] == "handover":  # they have 2 hands
                continue

            # # samples should be in list form (no collate)
            # # samples should not be transformed
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(ds) | beam.Map(_parse_example)
