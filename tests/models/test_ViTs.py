# Standard library
from argparse import ArgumentParser

# Local application
from climate_learn.models.hub import (
    OriginalVisionTransformer,
    SwinTransformer,
    PrimalVisionTransformer
)

# Third party
import torch


class TestForecastingModels:
    def __init__(
        self,
        num_batches=32,
        history=3,
        num_channels=2,
        out_channels=1,
        height=32,
        width=64
    ):
        self.num_batches = num_batches
        self.history = history
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.height, self.width = height, width
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = torch.randn((num_batches, history, num_channels, height, width)).to(self.device)
        self.y_same_channels = torch.randn((num_batches, num_channels, height, width)).to(self.device)
        self.y_diff_channels = torch.randn((num_batches, out_channels, height, width)).to(self.device)

    def test_vit(self, same_out_channels):
        if same_out_channels:
            out_channels = self.num_channels
            target = self.y_same_channels
        else:
            out_channels = self.out_channels
            target = self.y_diff_channels
        print("Vision Transformer")
        print("x_shape = "+str(self.x.shape))
        print("y_shape = "+str(target.shape))
        model = OriginalVisionTransformer(
            (self.height, self.width),
            self.num_channels,
            out_channels,
            self.history,
            patch_size=1,
            embed_dim=128,
            depth=8,
            decoder_depth=2,
            learn_pos_emb=True,
            num_heads=4
        )
        model.to(self.device)
        pred = model(self.x)
        print("\npred_shape = "+str(pred.shape))
        assert pred.shape == target.shape

    def test_swin_vit(self, same_out_channels):
        if same_out_channels:
            out_channels = self.num_channels
            target = self.y_same_channels
        else:
            out_channels = self.out_channels
            target = self.y_diff_channels
        print("Swin Transformer")
        print("x_shape = "+str(self.x.shape))
        print("y_shape = "+str(target.shape))
        model = SwinTransformer(
            (self.height, self.width),
            self.num_channels,
            out_channels,
            self.history,
            patch_size=1,
            embed_dim=128,
            depth=8,
            decoder_depth=2,
            learn_pos_emb=True,
            num_heads=4
        )
        model.to(self.device)
        pred = model(self.x)
        print("\npred_shape = "+str(pred.shape))
        assert pred.shape == target.shape

    def test_primal_vit(self, same_out_channels):
        if same_out_channels:
            out_channels = self.num_channels
            target = self.y_same_channels
        else:
            out_channels = self.out_channels
            target = self.y_diff_channels
        print("Primal Vision Transformer")
        print("x_shape = "+str(self.x.shape))
        print("y_shape = "+str(target.shape))
        model = PrimalVisionTransformer(
            (self.height, self.width),
            self.num_channels,
            out_channels,
            self.history,
            patch_size=1,
            embed_dim=128,
            depth=8,
            decoder_depth=2,
            learn_pos_emb=True,
            num_heads=4
        )
        model.to(self.device)
        pred = model(self.x)
        print("\npred_shape = "+str(pred.shape))
        assert pred.shape == target.shape

def main():
    parser = ArgumentParser(description="Test DL models.")
    parser.add_argument("--model")
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--is_same_channels", action="store_true")
    args = parser.parse_args()

    testModel = TestForecastingModels(
        num_batches=args.num_batches,
        history=args.history,
        num_channels=args.num_channels,
        out_channels=args.out_channels,
        height=args.height,
        width=args.width,
    )

    if(args.model == "vit"):
        testModel.test_vit(same_out_channels=args.is_same_channels)
    elif(args.model == "swin"):
        testModel.test_swin_vit(same_out_channels=args.is_same_channels)
    elif(args.model == "primal"):
        testModel.test_primal_vit(same_out_channels=args.is_same_channels)


if __name__ == "__main__":
    main()