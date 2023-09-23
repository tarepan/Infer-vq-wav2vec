"""Weight transfer from s3prl."""

from torch import nn

from ..model import VQWav2VecUnit


def transfer_from_s3prl(model_origin: nn.Module) -> VQWav2VecUnit:
    """Transfer vq-wav2vec model weight from s3prl's vq-wav2vec.
    
    Args:
        model_origin
    Returns:
         - Weight-transfered model
    """

    # Original weights
    model_origin = model_origin.eval()
    dict_origin = model_origin.state_dict()

    # Weight mapping
    dict_transfered_s3prl = {
        # down_conv [Conv-GN-ReLU]x8 <- [Conv-Do-GN-ReLU]x8
        'down_conv.conv_layers.0.0.weight':      dict_origin['model.feature_extractor.conv_layers.0.0.weight'],
        'down_conv.conv_layers.0.1.weight':      dict_origin['model.feature_extractor.conv_layers.0.2.weight'],
        'down_conv.conv_layers.0.1.bias':        dict_origin['model.feature_extractor.conv_layers.0.2.bias'],
        'down_conv.conv_layers.1.0.weight':      dict_origin['model.feature_extractor.conv_layers.1.0.weight'],
        'down_conv.conv_layers.1.1.weight':      dict_origin['model.feature_extractor.conv_layers.1.2.weight'],
        'down_conv.conv_layers.1.1.bias':        dict_origin['model.feature_extractor.conv_layers.1.2.bias'],
        'down_conv.conv_layers.2.0.weight':      dict_origin['model.feature_extractor.conv_layers.2.0.weight'],
        'down_conv.conv_layers.2.1.weight':      dict_origin['model.feature_extractor.conv_layers.2.2.weight'],
        'down_conv.conv_layers.2.1.bias':        dict_origin['model.feature_extractor.conv_layers.2.2.bias'],
        'down_conv.conv_layers.3.0.weight':      dict_origin['model.feature_extractor.conv_layers.3.0.weight'],
        'down_conv.conv_layers.3.1.weight':      dict_origin['model.feature_extractor.conv_layers.3.2.weight'],
        'down_conv.conv_layers.3.1.bias':        dict_origin['model.feature_extractor.conv_layers.3.2.bias'],
        'down_conv.conv_layers.4.0.weight':      dict_origin['model.feature_extractor.conv_layers.4.0.weight'],
        'down_conv.conv_layers.4.1.weight':      dict_origin['model.feature_extractor.conv_layers.4.2.weight'],
        'down_conv.conv_layers.4.1.bias':        dict_origin['model.feature_extractor.conv_layers.4.2.bias'],
        'down_conv.conv_layers.5.0.weight':      dict_origin['model.feature_extractor.conv_layers.5.0.weight'],
        'down_conv.conv_layers.5.1.weight':      dict_origin['model.feature_extractor.conv_layers.5.2.weight'],
        'down_conv.conv_layers.5.1.bias':        dict_origin['model.feature_extractor.conv_layers.5.2.bias'],
        'down_conv.conv_layers.6.0.weight':      dict_origin['model.feature_extractor.conv_layers.6.0.weight'],
        'down_conv.conv_layers.6.1.weight':      dict_origin['model.feature_extractor.conv_layers.6.2.weight'],
        'down_conv.conv_layers.6.1.bias':        dict_origin['model.feature_extractor.conv_layers.6.2.bias'],
        'down_conv.conv_layers.7.0.weight':      dict_origin['model.feature_extractor.conv_layers.7.0.weight'],
        'down_conv.conv_layers.7.1.weight':      dict_origin['model.feature_extractor.conv_layers.7.2.weight'],
        'down_conv.conv_layers.7.1.bias':        dict_origin['model.feature_extractor.conv_layers.7.2.bias'],
        # quant
        'quant.codebook':                        dict_origin['model.vector_quantizer.vars'],
        ## projection FC-ReLU-FC <- [FC-ReLU]x1-FC
        'quant.projection.0.weight': dict_origin['model.vector_quantizer.weight_proj.0.0.weight'],
        'quant.projection.0.bias':   dict_origin['model.vector_quantizer.weight_proj.0.0.bias'],
        'quant.projection.2.weight': dict_origin['model.vector_quantizer.weight_proj.1.weight'],
        'quant.projection.2.bias':   dict_origin['model.vector_quantizer.weight_proj.1.bias'],
    }

    model_new = VQWav2VecUnit().eval()
    model_new.load_state_dict(dict_transfered_s3prl) # <All keys matched successfully>

    return model_new

if __name__ == "__main__":
    """Run weight transfer."""

    import torch
    import s3prl.hub as hub

    model_s3prl = getattr(hub, 'vq_wav2vec')().eval()
    model_transfered = transfer_from_s3prl(model_s3prl)
    weight_transfered = model_transfered.state_dict()

    torch.save(weight_transfered, "./vqw2v_unit_v1.pt") # pyright: ignore[reportUnknownMemberType]
