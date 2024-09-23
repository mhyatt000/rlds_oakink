import argparse
import importlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress debug warning messages
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

import wandb

tf.config.set_visible_devices([], "GPU")

WANDB_ENTITY = "luc-ssl"
WANDB_PROJECT = "vis_rlds"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="name of the dataset to visualize")
args = parser.parse_args()

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
else:
    render_wandb = False


def oakink_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to Oak-Ink dataset."""

    print(trajectory)
    print(type(trajectory))
    trajectory["observation"]["state"].pop("cam_intr")
    trajectory["observation"]["state"].pop("mano_shape")
    trajectory["observation"]["state"].pop("joints_vis")

    # flatten state keys into a single tensor
    state = tf.reshape(
        tf.concat(
            [
                tf.reshape(v, [tf.shape(v)[0], -1])
                for v in trajectory["observation"]["state"].values()
            ],
            axis=-1,
        ),
        [-1, 111],
    )

    # roll the state by 1
    actions = tf.roll(state, shift=-1, axis=0)
    last = state[-2:-1]  # if we are done then the absolute mesh is same as last
    trajectory["action"] = tf.concat([actions[:-1], last], axis=0)

    trajectory["observation"]["proprio"] = state
    return trajectory


# create TF dataset
dataset_name = args.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, split="train")
ds = ds.shuffle(
    100
)  # .map(lambda x: oakink_dataset_transform(list(x['steps'].as_numpy_iterator())))

# visualize episodes
for i, episode in tqdm(enumerate(ds.take(5)), total=5):
    images = []
    for step in episode["steps"]:
        images.append(step["observation"]["image"].numpy())
    image_strip = np.concatenate(images[::4], axis=1)
    caption = step["language_instruction"].numpy().decode() + " (temp. downsampled 4x)"

    if render_wandb:
        wandb.log({f"image_{i}": wandb.Image(image_strip, caption=caption)})
    else:
        plt.figure()
        plt.imshow(image_strip)
        plt.title(caption)

# visualize action and state statistics
actions, states = [], []
for episode in tqdm(ds.take(500)):
    for step in episode["steps"]:
        # actions.append(step['action'].numpy())
        states.append(
            np.concatenate(
                [
                    s.numpy().reshape(-1)
                    for k, s in step["observation"]["state"].items()
                    if k in ["mano_pose", "joints_3d"]
                ],
            )
        )

# actions = np.array(actions)
states = np.array(states)
# action_mean = actions.mean(0)
state_mean = states.mean(0)


def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(3*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem + 1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})

def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    n_cols = 37
    n_rows = (n_elems + n_cols - 1) // n_cols  # Calculate the number of rows needed

    fig = plt.figure(tag, figsize=(5*n_cols, 5*n_rows))
    for elem in tqdm(range(n_elems)):
        plt.subplot(n_rows, n_cols, elem + 1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})


# vis_stats(actions, action_mean, 'action_stats')
vis_stats(states, state_mean, "state_stats")

if not render_wandb:
    plt.show()
