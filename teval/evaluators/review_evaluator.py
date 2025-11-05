from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict
from mmengine import load

from teval.schema import ResponseDataSample
import numpy as np
from numpy import ndarray
from teval.utils.format_load import format_load

from .utils import annotate_dataset


class ReviewEvaluator:
    """Review Capability Evaluation

    Args:
        dataset_path(str): File path of evaluation dataset.

    """

    def __init__(
        self,
        dataset_path: str,
        annotation_path: str | None = None,
        # bert_score_model: str = "all-mpnet-base-v2",
        **kwargs,
    ) -> None:
        self.dataset_path = dataset_path
        self.annotation_path = annotation_path or dataset_path
        self.raw_dataset = None
        # self.bert_score_model = bert_score_model
        # self.sentence_model = SentenceTransformer(self.bert_score_model)

    def _load_dataset(self):
        self.dataset: list[Dict[str, Any]] = []
        dataset: Dict[str, Any] = load(self.dataset_path)
        self.raw_dataset = dataset

        for key in dataset.keys():
            datum = dataset[key]
            data_sample = self._process_response(datum)

            self.dataset.append(
                dict(
                    sample_id=key,
                    origin_prompt=datum["origin_prompt"],
                    response_data_sample=data_sample,
                )
            )
        self.num_samples = len(self.dataset)

    def _process_response(
        self,
        datum: dict,
    ) -> ResponseDataSample:
        """Process the response to needed format.

        Args:
            datum(dict): inputs.

        Returns:
            dict: Processed response data sample.
        """

        template = datum["template"]
        pred_data = datum["prediction"]
        gt_data = datum["ground_truth"]["answer"]
        meta_data = datum["meta_data"]

        if meta_data["response_format"] == "json":
            pred_data = self.json_format_parse(pred_data)
        else:
            pred_data = pred_data[pred_data.find(":") + 1 :]
            pred_data = pred_data.strip()
            if len(pred_data) > 0 and pred_data[0] in ["A", "B", "C", "D", "E"]:
                pred_data = pred_data[0]
            else:
                pred_data = None

        return ResponseDataSample(
            template=template, pred=pred_data, gt=gt_data, meta_data=meta_data
        )

    def _evaluate(self, data_sample) -> dict:
        metrics_result = dict(
            parse_rate=0.0,
            review_quality=0.0,
        )

        pred_data = data_sample.pred
        if pred_data is not None:
            # import pdb; pdb.set_trace()
            metrics_result["review_quality"] = (
                1.0 if pred_data == data_sample.gt else 0.0
            )
            metrics_result["parse_rate"] = 1.0
        return metrics_result

    # def compute_sen_similarity(self, gt, pred):
    #     gt_embed = self.sentence_model.encode(gt, convert_to_tensor=True)
    #     pred_embed = self.sentence_model.encode(pred, convert_to_tensor=True)
    #     sen_sim = max(0, util.cos_sim(gt_embed, pred_embed).item())
    #     return sen_sim

    def json_format_parse(self, pred_data):
        try:
            loaded = format_load(pred_data)
            if not isinstance(loaded, dict):
                return None
            data: Dict[str, Any] = loaded
        except Exception as e:
            return None
        try:
            review_value = data.get("is_finished")
            if not isinstance(review_value, bool):
                return None
            new_data = {"review": review_value}
        except Exception as e:
            return None
        return new_data

    def evaluate(self):
        self._load_dataset()
        results_list = []
        per_item_metrics: Dict[str, Dict[str, float]] = {}
        evaluation_time = datetime.now(timezone.utc).isoformat()
        for data_entry in self.dataset:
            sample_id = data_entry["sample_id"]
            response_sample = data_entry["response_data_sample"]
            metrics_result = self._evaluate(response_sample)
            results_list.append(metrics_result)
            cleaned_metrics = {
                key: value.item() if isinstance(value, ndarray) else value
                for key, value in metrics_result.items()
            }
            per_item_metrics[sample_id] = cleaned_metrics
        aggregated_results = self._post_process(results_list)
        if self.raw_dataset is not None:
            annotate_dataset(
                raw_dataset=self.raw_dataset,
                per_item_metrics=per_item_metrics,
                evaluator_name=self.__class__.__name__,
                dataset_path=self.dataset_path,
                annotation_path=self.annotation_path,
                evaluated_at=evaluation_time,
            )
        return aggregated_results

    def _post_process(self, results_list):
        # list of dict to dict of list
        results_dict = defaultdict(list)
        {results_dict[key].append(sub[key]) for sub in results_list for key in sub}
        metric_list = ["parse_rate", "review_quality"]
        for metric in metric_list:
            results_dict[metric] = np.round(np.mean(results_dict[metric]), decimals=4)
        return results_dict
