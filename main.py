import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments

from argparse import ArgumentParser
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from models.transformer_model import TransformerModel
from models.dual_channel_model import DualChannelModel
from models.dual_resid_model import DualResidModel
from models.alt_dual_resid_model import AltDualResidModel
from models.secondary_resid_model import SecondaryResidModel
from models.alt_secondary_resid_model import AltSecondaryResidModel
from util.loading import load_checkpoint, load_config

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--eval", type=str, nargs="+")
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    assert args.train or args.eval, "Must specify either train or eval"
    
    if args.train:
        model_name = args.train
    elif args.eval:
        model_name = args.eval[0]
    
    config = load_config(model_name)
    
    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size
    
    if config.model_type == "dual":
        model = DualChannelModel(config)
    elif config.model_type == "dual_resid":
        model = DualResidModel(config)
    elif config.model_type == "transformer":
        model = TransformerModel(config)
    elif config.model_type == "alt_dual_resid":
        model = AltDualResidModel(config)
    elif config.model_type == "secondary_resid":
        model = SecondaryResidModel(config)
    elif config.model_type == "alt_secondary_resid":
        model = AltSecondaryResidModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    splits = TinyStoriesDataset.get_splits(tokenizer, config.max_seq_len)
    
    if args.train:
        checkpoint_type = args.checkpoint if args.checkpoint else "recent"
        checkpoint = load_checkpoint(model, checkpoint_type)
        trainer = Trainer(model, splits, tokenizer, checkpoint=checkpoint)
        trainer.train()
    elif args.eval:
        checkpoint_type = args.checkpoint if args.checkpoint else "best"
        checkpoint = load_checkpoint(model, checkpoint_type)
        assert checkpoint is not None, f"Checkpoint not found: {checkpoint_type}"
        evaluator = Evaluator(model, splits, tokenizer)
        evaluator.evaluate()

if __name__ == "__main__":
    main()