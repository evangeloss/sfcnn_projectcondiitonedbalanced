"""Exported verbatim from notebook code cell 17."""

class SFCNN(nn.Module):
    """
    SF-CNN for channel estimation.

    MATLAB input format:
        [N_B, N_U, 4*M]

    PyTorch format:
        [batch, channels, height, width]
    """

    def __init__(self, M):
        super(SFCNN, self).__init__()

        in_channels = 4 * M

        self.network = nn.Sequential(

            # ------------------------------------------------
            # Conv Block 1
            # ------------------------------------------------
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # ------------------------------------------------
            # Conv Block 2
            # ------------------------------------------------
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # ------------------------------------------------
            # Conv Block 3
            # ------------------------------------------------
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # ------------------------------------------------
            # Conv Block 4
            # ------------------------------------------------
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # ------------------------------------------------
            # Conv Block 5
            # ------------------------------------------------
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # ------------------------------------------------
            # Output layer
            # ------------------------------------------------
            nn.Conv2d(64, 4, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# TRAINING OPTIONS EQUIVALENT
# ============================================================

def get_sfcnn_training_config():

    config = {
        "epochs": 5,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "gradient_clip": 1.0,
    }

    return config
