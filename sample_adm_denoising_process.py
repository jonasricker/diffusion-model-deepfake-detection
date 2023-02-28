"""
Modified version of https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
to sample at intermediate denoising steps.
"""

import argparse
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from PIL import Image


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # monkey patching
    diffusion.p_sample_loop = p_sample_loop_modified
    diffusion.ddim_sample_loop = ddim_sample_loop_modified

    logger.log("sampling...")
    all_images = []
    all_within = None
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, within, within_steps = sample_fn(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            store_within=args.store_within,
            stop=args.stop,
            step=args.step,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        if args.store_within:
            within = th.stack(within)
            within = ((within + 1) * 127.5).clamp(0, 255).to(th.uint8)
            within = within.permute(0, 1, 3, 4, 2)
            within = within.contiguous().numpy()
            if all_within is None:
                all_within = within
            else:
                all_within = np.concatenate((all_within, within), axis=1)

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_images.append(sample.cpu().numpy())
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)

    if args.store_within:
        output_dir = Path(args.output_dir) / f"denoising-{args.stop}_{args.step}"
        all_within = all_within[:, : args.num_samples]
        for arr, step in zip(all_within, within_steps):
            step_dir = output_dir / f"steps_{step:04}"
            step_dir.mkdir(parents=True, exist_ok=True)
            for j, img in enumerate(arr):
                Image.fromarray(img).save(step_dir / f"{j:06}.png")
    else:
        output_dir = (
            Path(args.output_dir)
            / ("sampling_steps_ddim" if args.use_ddim else "sampling_steps")
            / f"steps_{int(args.timestep_respacing):04}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(arr):
            Image.fromarray(img).save(output_dir / f"{i:06}.png")

    # dist.barrier()
    logger.log("sampling complete")


def p_sample_loop_modified(
    self,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    store_within=False,
    stop=1000,
    step=10,
):
    """
    Generate samples from the model.

    :param model: the model module.
    :param shape: the shape of the samples, (N, C, H, W).
    :param noise: if specified, the noise from the encoder to sample.
                  Should be of the same shape as `shape`.
    :param clip_denoised: if True, clip x_start predictions to [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample.
    :param cond_fn: if not None, this is a gradient function that acts
                    similarly to the model.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :param device: if specified, the device to create the samples on.
                   If not specified, use a model parameter's device.
    :param progress: if True, show a tqdm progress bar.
    :return: a non-differentiable batch of samples.
    """
    final = None
    within, within_steps = [], []
    i = self.num_timesteps
    for sample in self.p_sample_loop_progressive(
        model,
        shape,
        noise=noise,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        cond_fn=cond_fn,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
    ):
        final = sample
        i -= 1
        if store_within and i % step == 0 and i < stop:
            print(f"Storing within at step {i}.")
            within.append(sample["sample"].detach().cpu())
            within_steps.append(i)

    return final["sample"], within, within_steps


def ddim_sample_loop_modified(
    self,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    eta=0.0,
    store_within=False,
    stop=1000,
    step=10,
):
    """
    Generate samples from the model using DDIM.

    Same usage as p_sample_loop().
    """
    final = None
    intermediate, intermediate_steps = [], []
    i = self.num_timesteps
    for sample in self.ddim_sample_loop_progressive(
        model,
        shape,
        noise=noise,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        cond_fn=cond_fn,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
        eta=eta,
    ):
        final = sample
        i -= 1
        if store_within and i % step == 0 and i < stop:
            print(f"Storing within at step {i}.")
            intermediate.append(sample["sample"].detach().cpu())
            intermediate_steps.append(i)
    return final["sample"], intermediate, intermediate_steps


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=512,
        batch_size=16,
        use_ddim=False,
        model_path="",
        store_within=False,
        stop=1000,
        step=10,
        output_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
