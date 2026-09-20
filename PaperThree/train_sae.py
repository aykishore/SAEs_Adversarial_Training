import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset




device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

BLOCK = 11

INPUT_DIM = 768
LATENT_DIM = INPUT_DIM * 8

LEARNING_RATE = 5e-4
BATCH_SIZE = 512
EPOCHS = 10

# We'll start here and inspect the resulting L0.
L1_COEFF = 1.0


print("Block:", BLOCK)
print("Input dimension:", INPUT_DIM)
print("SAE features:", LATENT_DIM)
print("Expansion factor:", LATENT_DIM // INPUT_DIM)
print("L1 coefficient:", L1_COEFF)




activations = torch.load(
    f"sae_train_block{BLOCK}.pt"
).float()

stats = torch.load(
    f"sae_stats_block{BLOCK}.pt"
)

mean = stats["mean"]
std = stats["std"]




activations = (
    activations - mean
) / std


print("\nActivations:", activations.shape)

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


train_loader = DataLoader(
    TensorDataset(train_activations),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
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

        self.initialize_weights()


    def initialize_weights(self):

        # ------------------------------------------
        # Decoder:
        # random directions with column norm = 0.1
        # ------------------------------------------

        with torch.no_grad():

            decoder_weight = torch.randn(
                INPUT_DIM,
                LATENT_DIM
            )

            decoder_weight = (
                decoder_weight
                / decoder_weight.norm(
                    dim=0,
                    keepdim=True
                )
            )

            decoder_weight *= 0.1

            self.decoder.weight.copy_(
                decoder_weight
            )


            # --------------------------------------
            # Encoder initialized as decoder^T
            # --------------------------------------

            self.encoder.weight.copy_(
                decoder_weight.T
            )


            # --------------------------------------
            # Both biases start at zero
            # --------------------------------------

            self.encoder.bias.zero_()
            self.decoder.bias.zero_()


    def encode(self, x):

        return F.relu(
            self.encoder(x)
        )


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


sae = SparseAutoencoder(
    INPUT_DIM,
    LATENT_DIM
).to(device)




optimizer = torch.optim.Adam(
    sae.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    weight_decay=0
)


def sae_loss(
    x,
    reconstruction,
    features
):

    # ------------------------------------------
    # Reconstruction:
    #
    # mean over examples of squared L2 norm
    #
    # NOT ordinary elementwise MSE.
    # ------------------------------------------

    reconstruction_loss = (
        (x - reconstruction)
        .pow(2)
        .sum(dim=1)
        .mean()
    )


    # ------------------------------------------
    # Decoder-norm-weighted L1
    # ------------------------------------------

    decoder_norms = (
        sae.decoder.weight.norm(
            dim=0
        )
    )

    l1_loss = (
        features
        * decoder_norms.unsqueeze(0)
    ).sum(
        dim=1
    ).mean()


    total_loss = (
        reconstruction_loss
        + L1_COEFF * l1_loss
    )


    return (
        total_loss,
        reconstruction_loss,
        l1_loss
    )



@torch.no_grad()
def evaluate_sae():

    sae.eval()

    # Do validation in batches to avoid GPU memory spike
    loader = DataLoader(
        TensorDataset(val_activations),
        batch_size=512,
        shuffle=False
    )

    squared_error_sum = 0
    input_squared_sum = 0

    total_l0 = 0
    total_examples = 0


    for (x,) in loader:

        x = x.to(device)

        reconstruction, features = sae(x)


        squared_error_sum += (
            (x - reconstruction)
            .pow(2)
            .sum()
            .item()
        )


        input_squared_sum += (
            x.pow(2)
            .sum()
            .item()
        )


        l0 = (
            features > 0
        ).sum(
            dim=1
        )


        total_l0 += (
            l0.sum().item()
        )

        total_examples += x.size(0)


    # Elementwise MSE for easy interpretation
    val_mse = (
        squared_error_sum
        / (
            total_examples
            * INPUT_DIM
        )
    )


    average_l0 = (
        total_l0
        / total_examples
    )


    explained_variance = (
        1
        - squared_error_sum
        / input_squared_sum
    )


    sae.train()

    return (
        val_mse,
        average_l0,
        explained_variance
    )




print("\n" + "=" * 60)
print("TRAINING SAE")
print("=" * 60)


for epoch in range(EPOCHS):

    sae.train()

    total_loss_sum = 0
    recon_loss_sum = 0
    l1_loss_sum = 0

    batches = 0


    for (batch,) in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()


        reconstruction, features = sae(
            batch
        )


        (
            loss,
            reconstruction_loss,
            l1_loss
        ) = sae_loss(
            batch,
            reconstruction,
            features
        )


        loss.backward()


        # Matches the reference SAE training recipe
        torch.nn.utils.clip_grad_norm_(
            sae.parameters(),
            1.0
        )


        optimizer.step()


        total_loss_sum += loss.item()
        recon_loss_sum += reconstruction_loss.item()
        l1_loss_sum += l1_loss.item()

        batches += 1


    (
        val_mse,
        val_l0,
        explained_variance
    ) = evaluate_sae()


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        "Train reconstruction L2:",
        recon_loss_sum / batches
    )

    print(
        "Train L1:",
        l1_loss_sum / batches
    )

    print(
        "Validation elementwise MSE:",
        val_mse
    )

    print(
        "Validation average L0:",
        val_l0
    )

    print(
        "Validation explained variance:",
        explained_variance
    )



torch.save(
    {
        "block": BLOCK,
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "l1_coeff": L1_COEFF,
        "state_dict": sae.state_dict()
    },
    f"sae_block{BLOCK}_corrected.pt"
)


print(
    f"\nSaved sae_block{BLOCK}_corrected.pt"
)