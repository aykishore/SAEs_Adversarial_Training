import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset




device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

INPUT_DIM = 768
LATENT_DIM = 6144

TOP_K = 64

LEARNING_RATE = 5e-4
BATCH_SIZE = 512
EPOCHS = 10




paper3_dir = Path("../PaperThree")

activation_path = (
    paper3_dir / "sae_train_block11.pt"
)

stats_path = (
    paper3_dir / "sae_stats_block11.pt"
)


activations = torch.load(
    activation_path
).float()

stats = torch.load(
    stats_path
)

mean = stats["mean"]
std = stats["std"]


print("\nRaw activations:")
print(activations.shape)




activations = (
    activations - mean
) / std


print(
    "Standardized mean:",
    activations.mean().item()
)

print(
    "Standardized std:",
    activations.std().item()
)



num_train = int(
    0.9 * len(activations)
)

train_activations = activations[
    :num_train
]

val_activations = activations[
    num_train:
]


print("\nTrain:")
print(train_activations.shape)

print("Validation:")
print(val_activations.shape)


train_loader = DataLoader(
    TensorDataset(train_activations),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    TensorDataset(val_activations),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)



class TopKSAE(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        k
    ):

        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k


        self.encoder = nn.Linear(
            input_dim,
            latent_dim
        )

        self.decoder = nn.Linear(
            latent_dim,
            input_dim
        )


        self.initialize_weights()


    def initialize_weights(self):

        with torch.no_grad():

            # --------------------------------------
            # Create random decoder directions
            # and normalize every SAE feature
            # --------------------------------------

            decoder_weight = torch.randn(
                self.input_dim,
                self.latent_dim
            )

            decoder_weight = (
                decoder_weight
                / decoder_weight.norm(
                    dim=0,
                    keepdim=True
                )
            )


            self.decoder.weight.copy_(
                decoder_weight
            )


            # Encoder starts aligned with decoder
            self.encoder.weight.copy_(
                decoder_weight.T
            )


            self.encoder.bias.zero_()
            self.decoder.bias.zero_()


    def encode(self, x):

        # ------------------------------------------
        # Dense pre-activations
        # ------------------------------------------

        pre_activations = self.encoder(x)


        # SAEgis-style non-negative sparse features
        dense_features = F.relu(
            pre_activations
        )


        # ------------------------------------------
        # Keep only largest K features
        # for each patch token
        # ------------------------------------------

        values, indices = torch.topk(
            dense_features,
            k=self.k,
            dim=1
        )


        sparse_features = torch.zeros_like(
            dense_features
        )


        sparse_features.scatter_(
            1,
            indices,
            values
        )


        return sparse_features


    def decode(self, features):

        return self.decoder(
            features
        )


    def forward(self, x):

        features = self.encode(x)

        reconstruction = self.decode(
            features
        )

        return reconstruction, features


sae = TopKSAE(
    INPUT_DIM,
    LATENT_DIM,
    TOP_K
).to(device)




optimizer = torch.optim.Adam(
    sae.parameters(),
    lr=LEARNING_RATE
)




@torch.no_grad()
def evaluate():

    sae.eval()

    total_squared_error = 0.0
    total_input_squared = 0.0

    total_active = 0
    total_examples = 0

    feature_used = torch.zeros(
        LATENT_DIM,
        dtype=torch.bool
    )


    for (x,) in val_loader:

        x = x.to(device)


        reconstruction, features = sae(x)


        total_squared_error += (
            (x - reconstruction)
            .pow(2)
            .sum()
            .item()
        )


        total_input_squared += (
            x.pow(2)
            .sum()
            .item()
        )


        active = (
            features > 0
        )


        total_active += (
            active
            .sum(dim=1)
            .sum()
            .item()
        )


        feature_used |= (
            active
            .any(dim=0)
            .cpu()
        )


        total_examples += (
            x.shape[0]
        )


    mse = (
        total_squared_error
        /
        (
            total_examples
            * INPUT_DIM
        )
    )


    explained_variance = (
        1
        -
        total_squared_error
        /
        total_input_squared
    )


    average_l0 = (
        total_active
        /
        total_examples
    )


    used_features = (
        feature_used.sum().item()
    )


    sae.train()


    return (
        mse,
        explained_variance,
        average_l0,
        used_features
    )



print("\n" + "=" * 60)
print("TRAINING TOP-K SAE")
print("=" * 60)

print("Input dimension:", INPUT_DIM)
print("Latent dimension:", LATENT_DIM)
print("Top K:", TOP_K)


for epoch in range(EPOCHS):

    sae.train()

    total_loss = 0.0
    batches = 0


    for (x,) in train_loader:

        x = x.to(device)


        optimizer.zero_grad()


        reconstruction, features = sae(x)


        # TopK already enforces sparsity,
        # so reconstruction is the main loss.

        loss = F.mse_loss(
            reconstruction,
            x
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            sae.parameters(),
            1.0
        )


        optimizer.step()


        total_loss += (
            loss.item()
        )

        batches += 1


    (
        val_mse,
        explained_variance,
        average_l0,
        used_features
    ) = evaluate()


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        "Train reconstruction MSE:",
        total_loss / batches
    )

    print(
        "Validation reconstruction MSE:",
        val_mse
    )

    print(
        "Validation explained variance:",
        explained_variance
    )

    print(
        "Validation average L0:",
        average_l0
    )

    print(
        "SAE features used on validation:",
        used_features,
        "/",
        LATENT_DIM
    )




torch.save(
    {
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "top_k": TOP_K,
        "state_dict": sae.state_dict()
    },
    "topk_sae_block11.pt"
)


print(
    "\nSaved topk_sae_block11.pt"
)