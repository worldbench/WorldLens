from pathlib import Path
import json
import importlib
from collections import defaultdict
from loguru import logger

class Evaluator:
    def __init__(self, cfg):
        self.modality = cfg.modality
        self.output_dir = Path(cfg.output_dir)
        self.cfg = cfg.get(self.modality, {})
        self.logger = logger
        self.build_metrics()

    def build_metrics(self):
        self.metric_instances = defaultdict(defaultdict)

        for dimension_name, dimension_aspect in self.cfg.dimensions.items():
            self.metric_instances[dimension_name] = defaultdict(dict)
            
            for aspect_name, aspect_metrics in dimension_aspect.items():
                module_path = f"worldbench.{self.modality}.{dimension_name}.{aspect_name}"
                module = importlib.import_module(module_path)
                for metric_info in aspect_metrics:
                    metric_name = metric_info["name"]
                    self.metric_instances[dimension_name][aspect_name][metric_name] = \
                        getattr(module, f"{metric_name.upper()}")(**metric_info)
                        
                    self.logger.info(f"Built metric: {metric_name}")

    def evaluate(self):
        metric_results_path = self.output_dir / "metric_results.json"
        if metric_results_path.exists():
            self.logger.info(f"Loading existing metric results from {metric_results_path}")
            with open(metric_results_path, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = dict()
        # Placeholder for evaluation logic
        for dimension_name, dimension_aspect in self.metric_instances.items():
            for aspect_name, aspect_metrics in dimension_aspect.items():
                for metric_name, metric_instance in aspect_metrics.items():
                    sub_result_dict = metric_instance()
                    all_results.update({
                        f"{dimension_name}_{aspect_name}_{metric_name}": sub_result_dict
                    })
                    self.logger.info(f"Evaluated metric: {dimension_name}_{aspect_name}_{metric_name}")
        with open(metric_results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        # print results to console
        for key, value in all_results.items():
            self.logger.info(f"{key}: {value}")
        return all_results