import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append(".")
from worldbench import Evaluator

@hydra.main(config_path="configs", config_name="default_run", version_base=None)
def main(cfg: DictConfig):
    evaluator = Evaluator(cfg)
    evaluator.evaluate()

if __name__ == "__main__":
    main()