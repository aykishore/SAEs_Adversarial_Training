import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import open_clip



SEED = 42

NUM_DISCOVERY = 100
NUM_DEV = 100
NUM_TEST = 100

PGD_EPSILON = 3.0
PGD_ALPHA = 0.5
PGD_STEPS = 20

BATCH_SIZE = 10

MODEL_NAME = "ViT-B-32"
PRETRAINED = "datacomp_xl_s13b_b90k"


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
        "Could not locate CLIP normalization transform."
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
    """
    Convert raw [0,1] images into CLIP-normalized images.
    """

    return (
        images - clip_mean
    ) / clip_std


raw_transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    ),
    transforms.ToTensor()
])



data_root = Path("../PaperThree/data")


dataset = datasets.CIFAR100(
    root=data_root,
    train=False,
    download=True,
    transform=raw_transform
)


print("\nCIFAR-100 test images:")
print(len(dataset))



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


print(
    "Text classifier created for",
    len(class_names),
    "classes."
)



def get_logits(images):

    """
    images:
        raw image tensor in [0,1]
        shape [B,3,224,224]
    """

    normalized_images = normalize(
        images
    )

    image_features = model.encode_image(
        normalized_images
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

    return logits



required = (
    NUM_DISCOVERY
    + NUM_DEV
    + NUM_TEST
)


generator = torch.Generator()

generator.manual_seed(SEED)


permutation = torch.randperm(
    len(dataset),
    generator=generator
).tolist()


selected_images = []
selected_labels = []
selected_indices = []


print(
    "\nSearching for",
    required,
    "correctly classified CIFAR-100 images..."
)


with torch.no_grad():

    for dataset_index in permutation:

        image, label = dataset[
            dataset_index
        ]


        image_batch = (
            image
            .unsqueeze(0)
            .to(device)
        )


        logits = get_logits(
            image_batch
        )


        prediction = (
            logits.argmax(
                dim=1
            ).item()
        )


        if prediction == label:

            selected_images.append(
                image
            )

            selected_labels.append(
                label
            )

            selected_indices.append(
                dataset_index
            )


        if len(selected_images) == required:
            break


if len(selected_images) < required:

    raise RuntimeError(
        f"Only found {len(selected_images)} "
        f"correctly classified images."
    )


selected_images = torch.stack(
    selected_images
)

selected_labels = torch.tensor(
    selected_labels,
    dtype=torch.long
)

selected_indices = torch.tensor(
    selected_indices,
    dtype=torch.long
)


print(
    "Correctly classified images selected:",
    len(selected_images)
)


discovery_start = 0
discovery_end = NUM_DISCOVERY

dev_start = discovery_end
dev_end = dev_start + NUM_DEV

test_start = dev_end
test_end = test_start + NUM_TEST


discovery_clean = selected_images[
    discovery_start:discovery_end
].clone()

discovery_labels = selected_labels[
    discovery_start:discovery_end
].clone()

discovery_indices = selected_indices[
    discovery_start:discovery_end
].clone()


dev_clean = selected_images[
    dev_start:dev_end
].clone()

dev_labels = selected_labels[
    dev_start:dev_end
].clone()

dev_indices = selected_indices[
    dev_start:dev_end
].clone()


test_clean = selected_images[
    test_start:test_end
].clone()

test_labels = selected_labels[
    test_start:test_end
].clone()

test_indices = selected_indices[
    test_start:test_end
].clone()


print("\nSplit sizes:")

print(
    "Discovery:",
    discovery_clean.shape
)

print(
    "Clean dev:",
    dev_clean.shape
)

print(
    "Final test:",
    test_clean.shape
)


discovery_set = set(
    discovery_indices.tolist()
)

dev_set = set(
    dev_indices.tolist()
)

test_set = set(
    test_indices.tolist()
)


assert discovery_set.isdisjoint(
    dev_set
)

assert discovery_set.isdisjoint(
    test_set
)

assert dev_set.isdisjoint(
    test_set
)


print(
    "Split overlap check: PASSED"
)



def l2_pgd_attack(
    clean_images,
    labels,
    epsilon=PGD_EPSILON,
    alpha=PGD_ALPHA,
    steps=PGD_STEPS
):

    """
    Untargeted L2 PGD.

    clean_images:
        raw images in [0,1]

    Goal:
        maximize classification loss.
    """

    clean_images = (
        clean_images
        .detach()
        .to(device)
    )

    labels = (
        labels
        .detach()
        .to(device)
    )


    # --------------------------------------------------------
    # Random initialization inside L2 ball
    # --------------------------------------------------------

    delta = torch.randn_like(
        clean_images
    )


    flat_delta = delta.view(
        delta.shape[0],
        -1
    )


    delta_norm = (
        flat_delta.norm(
            p=2,
            dim=1,
            keepdim=True
        )
        .clamp(min=1e-12)
    )


    flat_delta = (
        flat_delta
        / delta_norm
    )


    # Random radius inside [0, epsilon]
    radius = torch.rand(
        clean_images.shape[0],
        1,
        device=device
    )


    flat_delta = (
        flat_delta
        * radius
        * epsilon
    )


    delta = flat_delta.view_as(
        clean_images
    )


    adversarial = (
        clean_images + delta
    ).clamp(
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # PGD iterations
    # --------------------------------------------------------

    for _ in range(steps):

        adversarial.requires_grad_()


        logits = get_logits(
            adversarial
        )


        loss = F.cross_entropy(
            logits,
            labels
        )


        gradient = torch.autograd.grad(
            loss,
            adversarial
        )[0]


        # Normalize each image's gradient
        flat_gradient = gradient.view(
            gradient.shape[0],
            -1
        )


        gradient_norm = (
            flat_gradient.norm(
                p=2,
                dim=1,
                keepdim=True
            )
            .clamp(min=1e-12)
        )


        normalized_gradient = (
            flat_gradient
            / gradient_norm
        ).view_as(
            gradient
        )


        adversarial = (
            adversarial.detach()
            + alpha
            * normalized_gradient
        )


        # ----------------------------------------------------
        # Project back into epsilon L2 ball
        # ----------------------------------------------------

        delta = (
            adversarial
            - clean_images
        )


        flat_delta = delta.view(
            delta.shape[0],
            -1
        )


        delta_norm = (
            flat_delta.norm(
                p=2,
                dim=1,
                keepdim=True
            )
        )


        scale = torch.clamp(
            epsilon
            / delta_norm.clamp(min=1e-12),
            max=1.0
        )


        flat_delta = (
            flat_delta
            * scale
        )


        delta = flat_delta.view_as(
            clean_images
        )


        adversarial = (
            clean_images
            + delta
        ).clamp(
            0.0,
            1.0
        ).detach()


    return adversarial



def attack_dataset(
    images,
    labels,
    name
):

    adversarial_batches = []


    print(
        f"\nGenerating PGD attacks for {name}..."
    )


    for start in range(
        0,
        len(images),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(images)
        )


        clean_batch = images[
            start:end
        ]

        label_batch = labels[
            start:end
        ]


        adversarial_batch = l2_pgd_attack(
            clean_batch,
            label_batch
        )


        adversarial_batches.append(
            adversarial_batch.cpu()
        )


        print(
            f"{end}/{len(images)}"
        )


    return torch.cat(
        adversarial_batches,
        dim=0
    )



discovery_adv = attack_dataset(
    discovery_clean,
    discovery_labels,
    "feature-discovery split"
)



test_adv = attack_dataset(
    test_clean,
    test_labels,
    "final test split"
)


@torch.no_grad()
def evaluate_accuracy(
    images,
    labels
):

    correct = 0


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

        labels_batch = (
            labels[start:end]
            .to(device)
        )


        logits = get_logits(
            batch
        )


        predictions = logits.argmax(
            dim=1
        )


        correct += (
            predictions
            == labels_batch
        ).sum().item()


    return correct / len(images)


# Discovery
discovery_clean_acc = evaluate_accuracy(
    discovery_clean,
    discovery_labels
)

discovery_adv_acc = evaluate_accuracy(
    discovery_adv,
    discovery_labels
)


# Final test
test_clean_acc = evaluate_accuracy(
    test_clean,
    test_labels
)

test_adv_acc = evaluate_accuracy(
    test_adv,
    test_labels
)


print("\n" + "=" * 60)
print("ATTACK RESULTS")
print("=" * 60)

print("\nDiscovery set:")

print(
    "Clean accuracy:",
    discovery_clean_acc
)

print(
    "Adversarial accuracy:",
    discovery_adv_acc
)

print(
    "Attack success rate:",
    1.0 - discovery_adv_acc
)


print("\nFinal test set:")

print(
    "Clean accuracy:",
    test_clean_acc
)

print(
    "Adversarial accuracy:",
    test_adv_acc
)

print(
    "Attack success rate:",
    1.0 - test_adv_acc
)



def perturbation_stats(
    clean,
    adversarial
):

    delta = (
        adversarial
        - clean
    )


    l2 = (
        delta
        .view(
            len(delta),
            -1
        )
        .norm(
            p=2,
            dim=1
        )
    )


    return (
        l2.mean().item(),
        l2.max().item()
    )


discovery_mean_l2, discovery_max_l2 = (
    perturbation_stats(
        discovery_clean,
        discovery_adv
    )
)


test_mean_l2, test_max_l2 = (
    perturbation_stats(
        test_clean,
        test_adv
    )
)


print("\nPerturbation L2:")

print(
    "Discovery mean:",
    discovery_mean_l2
)

print(
    "Discovery max:",
    discovery_max_l2
)

print(
    "Test mean:",
    test_mean_l2
)

print(
    "Test max:",
    test_max_l2
)




torch.save(
    {
        # ------------------------------------------
        # Feature-discovery set
        # ------------------------------------------

        "discovery_clean": discovery_clean,
        "discovery_adv": discovery_adv,
        "discovery_labels": discovery_labels,
        "discovery_indices": discovery_indices,


        # ------------------------------------------
        # Threshold-calibration set
        # ------------------------------------------

        "dev_clean": dev_clean,
        "dev_labels": dev_labels,
        "dev_indices": dev_indices,


        # ------------------------------------------
        # Final untouched test set
        # ------------------------------------------

        "test_clean": test_clean,
        "test_adv": test_adv,
        "test_labels": test_labels,
        "test_indices": test_indices,


        # ------------------------------------------
        # Attack configuration
        # ------------------------------------------

        "pgd_epsilon": PGD_EPSILON,
        "pgd_alpha": PGD_ALPHA,
        "pgd_steps": PGD_STEPS,
        "seed": SEED
    },

    "saegis_data.pt"
)


print(
    "\nSaved saegis_data.pt"
)