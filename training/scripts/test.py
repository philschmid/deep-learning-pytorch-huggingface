from transformers import TrainingArguments
from trl.commands.cli_utils import  TrlParser

if __name__ == "__main__":
    parser = TrlParser((TrainingArguments))
    training_args = parser.parse_args_and_config()
    print(training_args)