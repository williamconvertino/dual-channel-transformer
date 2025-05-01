import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments

from argparse import ArgumentParser
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from models.transformer_model import TransformerModel
from models.dual_channel_model import DualChannelModel
from util.loading import load_checkpoint, load_config

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--eval", type=str, nargs="+")
    parser.add_argument("--checkpoint", type=str, default="best")
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
    elif config.model_type == "transformer":
        model = TransformerModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    if args.checkpoint:
        checkpoint = load_checkpoint(model, args.checkpoint)
        checkpoint_name = f"epoch_{args.checkpoint}.pth" if args.checkpoint != "best" else "best.pth"
        print(f"Loaded checkpoint from {checkpoint_name}")
    else:
        checkpoint = None
        
    if checkpoint:
        model.load_state_dict(checkpoint, strict=False)
    
    splits = TinyStoriesDataset.get_splits(tokenizer, config.max_seq_len)
    
    if args.train:
        trainer = Trainer(model, splits, tokenizer)
        trainer.train()
    elif args.eval:
        evaluator = Evaluator(model, splits, tokenizer)
        evaluator.evaluate()

if __name__ == "__main__":
    main()