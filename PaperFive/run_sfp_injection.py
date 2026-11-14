import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

SEED = 42

MODEL_NAME = "ViT-B-32"
PRETRAINED = "datacomp_xl_s13b_b90k"

# Block 11 is left out: the hook edits patch tokens at a block's output,
# and CLIP's prediction is read from the CLS token only. After the last
# block nothing mixes patch tokens back into CLS, so an injection there
# cannot change the output (an earlier run showed exactly 0 at block 11).
BLOCKS = [2, 5, 8]
ALPHAS = [0.1, 0.3, 0.5]

INPUT_DIM = 768
LATENT_DIM = 6144

BATCH_SIZE = 10

paper3_dir = Path("../PaperThree")
paper4_dir = Path("../PaperFour")


random.seed(SEED)
torch.manual_seed(SEED)



model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)

model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer(
    MODEL_NAME
)




normalizer = None

for transform in preprocess.transforms:

    if isinstance(transform, transforms.Normalize):
        normalizer = transform
        break


if normalizer is None:
    raise RuntimeError(
        "Could not locate CLIP normalization."
    )


clip_mean = torch.tensor(
    normalizer.mean,
    device=device
).view(1, 3, 1, 1)

clip_std = torch.tensor(
    normalizer.std,
    device=device
).view(1, 3, 1, 1)


def normalize(images):

    return (
        images - clip_mean
    ) / clip_std




dataset = datasets.CIFAR100(
    root=paper3_dir / "data",
    train=False,
    download=True
)

class_names = [
    name.replace("_", " ")
    for name in dataset.classes
]

prompts = [
    f"a photo of a {name}"
    for name in class_names
]

text_tokens = tokenizer(
    prompts
).to(device)


with torch.no_grad():

    text_features = model.encode_text(
        text_tokens
    )

    text_features = F.normalize(
        text_features,
        dim=-1
    )


class SparseAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim
    ):

        super().__init__()

        self.encoder = nn.Linear(
            input_dim,
            latent_dim
        )

        self.decoder = nn.Linear(
            latent_dim,
            input_dim
        )


    def encode(self, x):

        return F.relu(
            self.encoder(x)
        )


    def decode(self, features):

        return self.decoder(
            features
        )




saes = {}
stats = {}


for block in BLOCKS:

    checkpoint = torch.load(
        paper3_dir
        / f"sae_block{block}_corrected.pt",
        map_location=device
    )

    sae = SparseAutoencoder(
        INPUT_DIM,
        LATENT_DIM
    ).to(device)

    sae.load_state_dict(
        checkpoint["state_dict"]
    )

    sae.eval()

    saes[block] = sae


    block_stats = torch.load(
        paper3_dir
        / f"sae_stats_block{block}.pt"
    )

    stats[block] = {
        "mean":
            block_stats["mean"].to(device),

        "std":
            block_stats["std"].to(device)
    }


print(
    "\nLoaded SAEs:",
    BLOCKS
)




mask_data = torch.load(
    "sfp_feature_masks.pt"
)

attack_masks = {}


for block in BLOCKS:

    attack_masks[block] = (
        mask_data["masks"][block]["mask"]
        .bool()
    )




random_masks = {}


for block in BLOCKS:

    attack_mask = attack_masks[
        block
    ]

    num_selected = int(
        attack_mask.sum().item()
    )

    available = (
        (~attack_mask)
        .nonzero(
            as_tuple=False
        )
        .squeeze(1)
    )

    generator = torch.Generator()
    generator.manual_seed(
        SEED + block
    )

    permutation = torch.randperm(
        len(available),
        generator=generator
    )

    chosen = available[
        permutation[:num_selected]
    ]

    random_mask = torch.zeros(
        LATENT_DIM,
        dtype=torch.bool
    )

    random_mask[
        chosen
    ] = True

    random_masks[block] = (
        random_mask
    )


    print(
        f"Block {block}: "
        f"{num_selected} SFPF features, "
        f"{int(random_mask.sum())} random features"
    )



data = torch.load(
    paper4_dir / "saegis_data.pt"
)

test_clean = data[
    "test_clean"
]

test_adv = data[
    "test_adv"
]

test_labels = data[
    "test_labels"
]


print("\nFinal clean:")
print(test_clean.shape)

print("Final PGD:")
print(test_adv.shape)



@torch.no_grad()
def get_logits(images):

    all_logits = []

    for start in range(
        0,
        len(images),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(images)
        )

        batch = (
            images[start:end]
            .to(device)
        )

        image_features = model.encode_image(
            normalize(batch)
        )

        image_features = F.normalize(
            image_features,
            dim=-1
        )

        logits = (
            model.logit_scale.exp()
            * image_features
            @ text_features.T
        )

        all_logits.append(
            logits.cpu()
        )

    return torch.cat(
        all_logits,
        dim=0
    )



print(
    "\nCalculating baseline predictions..."
)

clean_logits = get_logits(
    test_clean
)

pgd_logits = get_logits(
    test_adv
)

clean_predictions = (
    clean_logits.argmax(
        dim=1
    )
)

pgd_predictions = (
    pgd_logits.argmax(
        dim=1
    )
)


clean_accuracy = (
    clean_predictions
    == test_labels
).float().mean().item()

pgd_accuracy = (
    pgd_predictions
    == test_labels
).float().mean().item()


successful_pgd = (
    pgd_predictions
    != test_labels
)


print(
    "Clean accuracy:",
    clean_accuracy
)

print(
    "PGD accuracy:",
    pgd_accuracy
)

print(
    "Successful PGD attacks:",
    successful_pgd.sum().item(),
    "/",
    len(test_adv)
)



@torch.no_grad()
def get_injected_logits(
    images,
    block,
    feature_mask,
    alpha
):

    sae = saes[
        block
    ]

    activation_mean = stats[
        block
    ]["mean"]

    activation_std = stats[
        block
    ]["std"]


    feature_mask = (
        feature_mask
        .to(device)
        .float()
        .view(1, LATENT_DIM)
    )


    # ----------------------------------------------
    # Hook:
    #
    # h
    # ↓ standardize
    # z = SAE(h)
    # ↓
    # z' = z + alpha*m
    # ↓
    # Δh_std = D(z') - D(z)
    # ↓
    # convert Δh back to raw CLIP activation scale
    # ↓
    # h' = h + Δh
    # ----------------------------------------------

    def injection_hook(
        module,
        inputs,
        output
    ):

        if not isinstance(
            output,
            torch.Tensor
        ):
            raise RuntimeError(
                "Unexpected transformer block output."
            )


        modified = output.clone()


        # ------------------------------------------
        # Patch tokens only.
        # Leave CLS untouched.
        # ------------------------------------------

        patches = modified[
            :, 1:, :
        ]

        batch_size = (
            patches.shape[0]
        )

        flat = patches.reshape(
            -1,
            INPUT_DIM
        )


        # ------------------------------------------
        # Convert real CLIP activation into the
        # standardized coordinate system used by
        # the SAE during training.
        # ------------------------------------------

        standardized = (
            flat - activation_mean
        ) / activation_std


        z = sae.encode(
            standardized
        )


        z_perturbed = (
            z
            + alpha * feature_mask
        )


        # ------------------------------------------
        # Difference-of-decodes:
        #
        # cancels SAE reconstruction error.
        # ------------------------------------------

        decoded_original = sae.decode(
            z
        )

        decoded_perturbed = sae.decode(
            z_perturbed
        )


        delta_standardized = (
            decoded_perturbed
            - decoded_original
        )


        # ------------------------------------------
        # SAE decoder is in standardized activation
        # coordinates, so convert the perturbation
        # back to raw CLIP activation coordinates.
        # ------------------------------------------

        delta_raw = (
            delta_standardized
            * activation_std
        )


        perturbed_flat = (
            flat
            + delta_raw
        )


        modified[
            :, 1:, :
        ] = perturbed_flat.view(
            batch_size,
            49,
            INPUT_DIM
        )


        return modified


    handle = (
        model.visual
        .transformer
        .resblocks[block]
        .register_forward_hook(
            injection_hook
        )
    )


    all_logits = []


    for start in range(
        0,
        len(images),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(images)
        )

        batch = (
            images[start:end]
            .to(device)
        )


        image_features = model.encode_image(
            normalize(batch)
        )


        image_features = F.normalize(
            image_features,
            dim=-1
        )


        logits = (
            model.logit_scale.exp()
            * image_features
            @ text_features.T
        )


        all_logits.append(
            logits.cpu()
        )


    handle.remove()


    return torch.cat(
        all_logits,
        dim=0
    )



def evaluate_intervention(
    perturbed_logits
):

    perturbed_predictions = (
        perturbed_logits.argmax(
            dim=1
        )
    )


    # ----------------------------------------------
    # A. Prediction flip relative to clean CLIP
    # ----------------------------------------------

    flip_rate = (
        perturbed_predictions
        != clean_predictions
    ).float().mean().item()


    # ----------------------------------------------
    # B. Change in true-class logit
    # ----------------------------------------------

    true_indices = (
        test_labels
        .view(-1, 1)
    )


    clean_true_logits = (
        clean_logits.gather(
            1,
            true_indices
        ).squeeze(1)
    )


    perturbed_true_logits = (
        perturbed_logits.gather(
            1,
            true_indices
        ).squeeze(1)
    )


    true_logit_change = (
        perturbed_true_logits
        - clean_true_logits
    ).mean().item()


    # ----------------------------------------------
    # C. Change in the class chosen by actual PGD
    #
    # Evaluate this on successful attacks.
    # ----------------------------------------------

    successful_indices = (
        successful_pgd
        .nonzero(
            as_tuple=False
        )
        .squeeze(1)
    )


    if len(successful_indices) > 0:

        pgd_classes = (
            pgd_predictions[
                successful_indices
            ]
            .view(-1, 1)
        )


        clean_pgd_class_logits = (
            clean_logits[
                successful_indices
            ]
            .gather(
                1,
                pgd_classes
            )
            .squeeze(1)
        )


        perturbed_pgd_class_logits = (
            perturbed_logits[
                successful_indices
            ]
            .gather(
                1,
                pgd_classes
            )
            .squeeze(1)
        )


        pgd_class_logit_change = (
            perturbed_pgd_class_logits
            - clean_pgd_class_logits
        ).mean().item()


        # ------------------------------------------
        # D. Does SAE intervention produce the SAME
        # predicted class as actual PGD?
        # ------------------------------------------

        pgd_agreement = (
            perturbed_predictions[
                successful_indices
            ]
            ==
            pgd_predictions[
                successful_indices
            ]
        ).float().mean().item()


    else:

        pgd_class_logit_change = float(
            "nan"
        )

        pgd_agreement = float(
            "nan"
        )


    return {
        "flip_rate":
            flip_rate,

        "true_logit_change":
            true_logit_change,

        "pgd_class_logit_change":
            pgd_class_logit_change,

        "pgd_agreement":
            pgd_agreement
    }


results = []


print("\n")
print("=" * 92)
print("SFPF ACTIVATION-INJECTION EXPERIMENT")
print("=" * 92)


for block in BLOCKS:

    for alpha in ALPHAS:

        print(
            f"\nBlock {block}, alpha={alpha}"
        )


        # ------------------------------------------
        # SFPF-selected features
        # ------------------------------------------

        attack_logits = (
            get_injected_logits(
                test_clean,
                block,
                attack_masks[block],
                alpha
            )
        )


        attack_metrics = (
            evaluate_intervention(
                attack_logits
            )
        )


        # ------------------------------------------
        # Matched random features
        # ------------------------------------------

        random_logits = (
            get_injected_logits(
                test_clean,
                block,
                random_masks[block],
                alpha
            )
        )


        random_metrics = (
            evaluate_intervention(
                random_logits
            )
        )


        print(
            "SFPF  - "
            f"flip={attack_metrics['flip_rate']:.3f}, "
            f"true Δ={attack_metrics['true_logit_change']:.3f}, "
            f"PGD Δ={attack_metrics['pgd_class_logit_change']:.3f}, "
            f"PGD agreement={attack_metrics['pgd_agreement']:.3f}"
        )


        print(
            "Random- "
            f"flip={random_metrics['flip_rate']:.3f}, "
            f"true Δ={random_metrics['true_logit_change']:.3f}, "
            f"PGD Δ={random_metrics['pgd_class_logit_change']:.3f}, "
            f"PGD agreement={random_metrics['pgd_agreement']:.3f}"
        )


        results.append(
            {
                "block":
                    block,

                "alpha":
                    alpha,

                "num_features":
                    int(
                        attack_masks[
                            block
                        ].sum().item()
                    ),

                "sfp":
                    attack_metrics,

                "random":
                    random_metrics
            }
        )



print("\n")
print("=" * 110)
print("FINAL SUMMARY")
print("=" * 110)

print(
    f"{'Block':<7}"
    f"{'Alpha':<8}"
    f"{'N feat':<9}"
    f"{'SFP flip':<11}"
    f"{'Rnd flip':<11}"
    f"{'SFP trueΔ':<13}"
    f"{'Rnd trueΔ':<13}"
    f"{'SFP PGDΔ':<12}"
    f"{'Rnd PGDΔ':<12}"
    f"{'SFP agree':<11}"
    f"{'Rnd agree':<11}"
)

print("-" * 110)


for row in results:

    sfp = row["sfp"]
    rnd = row["random"]

    print(
        f"{row['block']:<7}"
        f"{row['alpha']:<8.1f}"
        f"{row['num_features']:<9}"
        f"{sfp['flip_rate']:<11.3f}"
        f"{rnd['flip_rate']:<11.3f}"
        f"{sfp['true_logit_change']:<13.3f}"
        f"{rnd['true_logit_change']:<13.3f}"
        f"{sfp['pgd_class_logit_change']:<12.3f}"
        f"{rnd['pgd_class_logit_change']:<12.3f}"
        f"{sfp['pgd_agreement']:<11.3f}"
        f"{rnd['pgd_agreement']:<11.3f}"
    )


torch.save(
    {
        "blocks":
            BLOCKS,

        "alphas":
            ALPHAS,

        "results":
            results,

        "clean_predictions":
            clean_predictions,

        "pgd_predictions":
            pgd_predictions,

        "successful_pgd":
            successful_pgd,

        "random_masks":
            random_masks
    },
    "sfp_injection_results.pt"
)


print(
    "\nSaved sfp_injection_results.pt"
)