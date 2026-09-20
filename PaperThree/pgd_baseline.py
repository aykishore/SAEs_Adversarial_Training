import torch
import torch.nn.functional as F
import open_clip

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms




device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "ViT-B-32"
pretrained = "datacomp_xl_s13b_b90k"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained
)

model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer(model_name)




normalize_transform = None

for transform in preprocess.transforms:
    if isinstance(transform, transforms.Normalize):
        normalize_transform = transform
        break

if normalize_transform is None:
    raise RuntimeError("Could not find normalization transform.")

mean = torch.tensor(
    normalize_transform.mean,
    device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    normalize_transform.std,
    device=device
).view(1, 3, 1, 1)

print("CLIP mean:", mean.flatten().tolist())
print("CLIP std:", std.flatten().tolist())


def normalize(images):
    return (images - mean) / std





raw_transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    ),
    transforms.ToTensor()
])


dataset = CIFAR100(
    root="./data",
    train=False,
    download=False,
    transform=raw_transform
)

class_names = dataset.classes




prompts = [
    f"a photo of a {name.replace('_', ' ')}"
    for name in class_names
]

text_tokens = tokenizer(prompts).to(device)

with torch.no_grad():

    text_features = model.encode_text(text_tokens)

    text_features = (
        text_features
        / text_features.norm(
            dim=-1,
            keepdim=True
        )
    )




def get_logits(images):

    # images expected in [0,1]

    normalized_images = normalize(images)

    image_features = model.encode_image(
        normalized_images
    )

    image_features = (
        image_features
        / image_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    # CLIP logit scaling
    logit_scale = model.logit_scale.exp()

    logits = (
        logit_scale
        * image_features
        @ text_features.T
    )

    return logits




def pgd_l2(
    images,
    labels,
    epsilon=3.0,
    alpha=0.5,
    steps=20
):

    original = images.detach()

    # ----------------------------------------------
    # Random start inside L2 epsilon ball
    # ----------------------------------------------

    delta = torch.randn_like(original)

    delta_flat = delta.view(
        delta.size(0),
        -1
    )

    delta_norm = delta_flat.norm(
        p=2,
        dim=1,
        keepdim=True
    )

    delta_flat = (
        delta_flat
        / (delta_norm + 1e-12)
    )

    # Random radius inside epsilon ball
    random_radius = torch.rand(
        delta.size(0),
        1,
        device=device
    )

    delta_flat = (
        delta_flat
        * random_radius
        * epsilon
    )

    delta = delta_flat.view_as(original)

    adversarial = torch.clamp(
        original + delta,
        0.0,
        1.0
    )


    # ----------------------------------------------
    # PGD iterations
    # ----------------------------------------------

    for step in range(steps):

        adversarial.requires_grad_(True)

        logits = get_logits(adversarial)

        # Untargeted attack:
        # maximize loss for true class
        loss = F.cross_entropy(
            logits,
            labels
        )

        gradient = torch.autograd.grad(
            loss,
            adversarial
        )[0]


        # Normalize gradient to unit L2 norm

        gradient_flat = gradient.view(
            gradient.size(0),
            -1
        )

        gradient_norm = gradient_flat.norm(
            p=2,
            dim=1,
            keepdim=True
        )

        normalized_gradient = (
            gradient_flat
            / (gradient_norm + 1e-12)
        )

        normalized_gradient = (
            normalized_gradient.view_as(
                gradient
            )
        )


        # Gradient ascent

        adversarial = (
            adversarial.detach()
            + alpha * normalized_gradient
        )


        # ------------------------------------------
        # Project back into L2 epsilon ball
        # ------------------------------------------

        delta = adversarial - original

        delta_flat = delta.view(
            delta.size(0),
            -1
        )

        delta_norm = delta_flat.norm(
            p=2,
            dim=1,
            keepdim=True
        )

        projection_factor = torch.clamp(
            epsilon / (delta_norm + 1e-12),
            max=1.0
        )

        delta_flat = (
            delta_flat
            * projection_factor
        )

        delta = delta_flat.view_as(
            original
        )

        adversarial = torch.clamp(
            original + delta,
            0.0,
            1.0
        ).detach()


    return adversarial



num_candidates = 500
num_attack_images = 100

subset = Subset(
    dataset,
    range(num_candidates)
)

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


clean_images = []
clean_labels = []


print("\nFinding correctly classified images...")


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = get_logits(images)

        predictions = logits.argmax(
            dim=-1
        )

        correct_mask = (
            predictions == labels
        )


        if correct_mask.any():

            clean_images.append(
                images[correct_mask].cpu()
            )

            clean_labels.append(
                labels[correct_mask].cpu()
            )


clean_images = torch.cat(
    clean_images,
    dim=0
)[:num_attack_images]

clean_labels = torch.cat(
    clean_labels,
    dim=0
)[:num_attack_images]


print(
    "Correct images selected:",
    len(clean_images)
)




adv_images_list = []

attack_batch_size = 10


for start in range(
    0,
    len(clean_images),
    attack_batch_size
):

    end = start + attack_batch_size

    images = clean_images[
        start:end
    ].to(device)

    labels = clean_labels[
        start:end
    ].to(device)


    adversarial = pgd_l2(
        images,
        labels,
        epsilon=3.0,
        alpha=0.5,
        steps=20
    )


    adv_images_list.append(
        adversarial.cpu()
    )

    print(
        f"Attacked "
        f"{min(end, len(clean_images))}"
        f"/{len(clean_images)}"
    )


adv_images = torch.cat(
    adv_images_list,
    dim=0
)




with torch.no_grad():

    clean_logits = get_logits(
        clean_images.to(device)
    )

    adv_logits = get_logits(
        adv_images.to(device)
    )


clean_predictions = clean_logits.argmax(
    dim=-1
).cpu()

adv_predictions = adv_logits.argmax(
    dim=-1
).cpu()


clean_accuracy = (
    clean_predictions == clean_labels
).float().mean().item()

adv_accuracy = (
    adv_predictions == clean_labels
).float().mean().item()


attack_success = (
    adv_predictions != clean_labels
).float().mean().item()




delta = (
    adv_images - clean_images
)

l2_norms = delta.view(
    delta.size(0),
    -1
).norm(
    p=2,
    dim=1
)




print("\n" + "=" * 55)
print("PGD RESULTS")
print("=" * 55)

print(
    "Clean accuracy:",
    clean_accuracy
)

print(
    "Adversarial accuracy:",
    adv_accuracy
)

print(
    "Attack success rate:",
    attack_success
)

print(
    "Mean L2 perturbation:",
    l2_norms.mean().item()
)

print(
    "Max L2 perturbation:",
    l2_norms.max().item()
)



torch.save(
    {
        "clean_images": clean_images,
        "adv_images": adv_images,
        "labels": clean_labels
    },
    "pgd_test_pairs.pt"
)

print("\nSaved pgd_test_pairs.pt")