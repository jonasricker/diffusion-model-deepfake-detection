"""
Modified version of https://github.com/openai/guided-diffusion/blob/main/scripts/image_train.py
to sample at intermediate diffusion steps.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from PIL import Image
from tqdm import tqdm


def main():
    args = create_argparser().parse_args()

    output_dir = Path(args.output_dir) / f"diffusion-{args.stop}_{args.step}"
    output_dir.mkdir(exist_ok=True, parents=True)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    batches = []
    while len(batches) < args.num_samples // args.batch_size:
        batches.append(next(data))

    logger.log("diffusing...")
    for t in tqdm(range(0, args.stop, args.step)):
        diffused = []
        for batch, _ in batches:
            if t == 0:
                x_t = batch
            else:
                batch = batch.to(dist_util.dev())
                x_t = (
                    diffusion.q_sample(
                        x_start=batch,
                        t=torch.tensor(
                            [t - 1] * args.batch_size, device=dist_util.dev()
                        ),
                    )
                    .detach()
                    .cpu()
                )
            x_t = ((x_t + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            x_t = x_t.permute(0, 2, 3, 1)
            x_t = x_t.contiguous().numpy()
            diffused.append(x_t)
        diffused = np.concatenate(diffused)
        step_dir = output_dir / f"step_{t:04}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for j, img in enumerate(diffused):
            Image.fromarray(img).save(step_dir / f"{j:06}.png")


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_samples=512,
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
