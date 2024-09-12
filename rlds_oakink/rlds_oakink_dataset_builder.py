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
    """DatasetBuilder for OakInk v1."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

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
                                        shape=(*downscale_to_224(480, 848), 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "state": tfds.features.FeaturesDict(
                                        {
                                            "cam_intr": tfds.features.Tensor(
                                                shape=[3, 3],
                                                dtype=np.float32,
                                                doc="Camera intrinsics matrix",
                                            ),
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
                                            "joints_3d": tfds.features.Tensor(
                                                shape=[21, 3],
                                                dtype=np.float32,
                                            ),
                                            "joints_vis": tfds.features.Tensor(
                                                shape=[21],
                                                dtype=np.float32,
                                                doc="joint visibility? TODO",
                                            ),
                                        },
                                    ),
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
        return {
            "train": self._generate_examples(path="data/train/episode_*.npy"),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )
        import bafl
        from bafl.scripts import make_oakink_torch

        self.ds = bafl.scripts.make_oakink_torch()


        def _parse_example(data):

            paths = [d["image_path"] for d in data]
            mykeys = [  # keys of interest
                "image",
                "image_path",
                "cam_intr",
                "joints_3d",
                "joints_vis",
                "mano_pose",
                "mano_shape",
                "task",
                "idx",
            ]

            task = data[0]["task"]
            lang = self._embed([task])[0].numpy()

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding

                step = {k: v for k, v in step.items() if k in mykeys}
                task = step.pop("task")
                # with tf.device("/GPU:0"):
                image = step.pop("image")
                img_shape = downscale_to_224(*image.shape[:2])
                image = tf.image.resize(image, img_shape).numpy().astype(np.uint8)
                step.pop("image_path")
                idx = step.pop("idx")

                episode.append(
                    {
                        "observation": {
                            "image": image,
                            # "wrist_image": step["wrist_image"],
                            "state": step,
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

        for sample in self.ds:
            # # samples should be in list form (no collate)
            # # samples should not be transformed
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(ds) | beam.Map(_parse_example)
