import torch
from torchinfo import summary
from sterling.models import IPTEncoderModel, InertialEncoderModel, VisualEncoderModel, VisualEncoderTinyModel


if __name__ == "__main__":
    # IPT Encoder
    ipt_encoder = IPTEncoderModel()
    inertial, leg, feet = torch.randn(1, 1, 603), torch.randn(1, 1, 900), torch.randn(1, 1, 500)
    out = ipt_encoder(inertial, leg, feet)
    print(out.shape)
    summary(ipt_encoder, [(1, 1, 603), (1, 1, 900), (1, 1, 500)])

    # Inertial Encoder
    inertial_encoder = InertialEncoderModel()
    inertial = torch.randn(1, 1, 1200)
    out = inertial_encoder(inertial)
    print(out.shape)
    summary(inertial_encoder, (1, 1, 1200))

    # Visual Encoder
    vision_encoder = VisualEncoderModel()
    x = torch.randn(1, 3, 64, 64)
    out = vision_encoder(x)
    print(out.shape)
    summary(vision_encoder, (1, 3, 64, 64))

    # Visual Encoder Tiny
    vision_encoder_tiny = VisualEncoderTinyModel()
    x = torch.randn(1, 3, 64, 64)
    out = vision_encoder_tiny(x)
    print(out.shape)
    summary(vision_encoder_tiny, (1, 3, 64, 64))
