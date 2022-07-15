#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure the weight (with SHAP) of each DWT coefficient in either:

- Forwardly, i.e., by setting all coefficients other than the measured one to zeros,
  reconstruct the image with IDWT, and compare the reconstructed image's SHAP values
  to the original one. (LEO - leave-everything-out)

- Backwardly, i.e., by only setting the measured coefficient to zeros, reconstruct the
  image with the other coefficients, and compare SHAP values with the original.

  This is set with the --backward command-line argument.

$ python dwt_shap.py --image=data/scottish_deerhound.jpg --backward --output=outputs

"""
import click
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torchvision
from einops import rearrange
from loguru import logger

from src.transforms import DwtCoeffs, image_dwt_leo, image_dwt_loo
from src.utils import (
    imagenet_classnames,
    inverse_normalise,
    load_single_image_from_path,
    shap_normalize,
)

import os

filePath = r'./data'
file_list = os.listdir(filePath)

for file in file_list:
    # 读取原文件名
    file_name = file


    @logger.catch
    @click.command()
    @click.option("--image", default=file_name, help="Path to image")
    @click.option("--backward", is_flag=True, help="Measure weights forward or backward")
    @click.option("--output", default="./outputs", help="Path to saved images")
    def main(image, backward, output):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = load_single_image_from_path(image).to(device)
        logger.debug(f"Loaded image from {image}.")

        model = torchvision.models.vgg16(pretrained=True).eval().to(device)
        logger.debug(f"Using device {device}, loaded VGG16 model.")

        # load shap base distribution for imagenet
        class_names = imagenet_classnames()
        shap_base_distrib, _ = shap.datasets.imagenet50()
        shap_base_distrib = shap_normalize(shap_base_distrib).to(device)

        # init shap gradient explainer with vgg16 (output, and layer 7)
        model_explain = (model, model.features[7])  # type: ignore
        explainer = shap.GradientExplainer(
            model_explain, shap_base_distrib, local_smoothing=0.5
        )
        logger.debug("Initalised SHAP gradient explainer.")

        if not backward:
            x1 = image_dwt_leo(data, kept=DwtCoeffs.LL)
            x2 = image_dwt_leo(data, kept=DwtCoeffs.LH)
            x3 = image_dwt_leo(data, kept=DwtCoeffs.HL)
            x4 = image_dwt_leo(data, kept=DwtCoeffs.HH)
        else:
            x1 = image_dwt_loo(data, loo=DwtCoeffs.LL)
            x2 = image_dwt_loo(data, loo=DwtCoeffs.LH)
            x3 = image_dwt_loo(data, loo=DwtCoeffs.HL)
            x4 = image_dwt_loo(data, loo=DwtCoeffs.HH)

        for to_explain, name in zip([x1, x2, x3, x4], ["ll", "lh", "hl", "hh"]):
            logger.debug(f"Measuring SHAP values for {name}, shape {to_explain.shape}.")

            shaps, idxs = explainer.shap_values(to_explain, ranked_outputs=1, nsamples=200)

            idxs = idxs.cpu().numpy()
            idx_names = np.vectorize(lambda x: class_names[str(x)][1])(idxs)

            # logger.info(f"Prediction for {name}: {idx_names}.")

            shap_positives = [s[s > 0] for s in shaps]
            shap_negatives = [s[s < 0] for s in shaps]

            # Multiply the mean values by 10^5 so that the values are human-readable
            logger.info(
                f"SHAP values for {name}: mean-{[s.mean() * (10 ** 5) for s in shaps]}, "
                f"positive mean {[s.mean() * (10 ** 5) for s in shap_positives]}, "
                f"number positives {[len(s) for s in shap_positives]}, "
                f"negative mean {[s.mean() * (10 ** 5) for s in shap_negatives]}, "
                f"number negatives {[len(s) for s in shap_negatives]}."

            )
            with open('test.txt', 'a+') as f:
                f.write(
                    f"{file_name},"
                    f"SHAP values for {name}: mean-{[s.mean() * (10 ** 5) for s in shaps]}, "
                    f"positive mean {[s.mean() * (10 ** 5) for s in shap_positives]}, "
                    f"number positives {[len(s) for s in shap_positives]}, "
                    f"negative mean {[s.mean() * (10 ** 5) for s in shap_negatives]}, "
                    f"number negatives {[len(s) for s in shap_negatives]}."

                )

            if output:
                shaps = [rearrange(s, "b c w h -> b w h c") for s in shaps]
                vis_to_explain = inverse_normalise(to_explain).cpu().numpy()
                vis_to_explain = rearrange(vis_to_explain, "b c w h -> b w h c")

                out = f"{output}/shap_{'backward' if backward else 'forward'}_{name}.png"
                shap.image_plot(shaps, vis_to_explain, idx_names, show=False)
                plt.savefig(out)
                logger.debug(f"Saved SHAP image to {out}.")

if __name__ == "__main__":
    main()
