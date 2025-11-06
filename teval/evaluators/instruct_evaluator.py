from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from mmengine import load

from teval.utils.template import parse_string
from teval.utils.format_load import format_load
from teval.schema import ResponseDataSample
import ast
import numpy as np
from numpy import ndarray

from .utils import annotate_dataset
from teval.utils.parse_failure_tracker import ParseFailureTracker


class InstructEvaluator:
    """Instruct Following Evaluation

    Args:
        dataset_path(str): File path of evaluation dataset.

    """

    def __init__(
        self,
        dataset_path: str,
        annotation_path: str | None = None,
        **kwargs,
    ) -> None:
        self.dataset_path = dataset_path
        self.annotation_path = annotation_path or dataset_path
        self.raw_dataset = None
        self._parse_tracker = ParseFailureTracker(
            dataset_path=self.dataset_path, evaluator_name=self.__class__.__name__
        )

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

    def _record_parse_failure(
        self,
        sample_id: str,
        *,
        mode: str,
        detail: Optional[str],
        response_format: str,
    ) -> None:
        prediction = None
        if self.raw_dataset and sample_id in self.raw_dataset:
            prediction = self.raw_dataset[sample_id].get("prediction")
            self.raw_dataset[sample_id]["parse_failure"] = {
                "mode": mode,
                "detail": detail,
                "response_format": response_format,
            }
        self._parse_tracker.record(
            sample_id,
            mode=mode,
            detail=detail,
            response_format=response_format,
            prediction=prediction,
        )

    def _clear_parse_failure(self, sample_id: str) -> None:
        if self.raw_dataset and sample_id in self.raw_dataset:
            self.raw_dataset[sample_id].pop("parse_failure", None)
        self._parse_tracker.clear(sample_id)

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

        # Dict with keyword-only arguments.
        template = datum["template"]
        # Generated response.
        pred_data = datum["prediction"]
        # Response of ground truth.
        gt_data = datum["ground_truth"]
        meta_data = datum["meta_data"]

        return ResponseDataSample(
            template=template, pred=pred_data, gt=gt_data, meta_data=meta_data
        )

    def _evaluate(self, sample_id: str, data_sample: ResponseDataSample) -> dict:
        metrics_result = dict()
        response_format = data_sample.meta_data["response_format"]
        failure_mode: Optional[str] = None
        failure_detail: Optional[str] = None
        if response_format == "json":
            pred_data, failure_mode, failure_detail = self.json_format_parse(
                data_sample
            )
        else:
            pred_data, failure_mode, failure_detail = self.string_format_parse(
                data_sample
            )

        if failure_mode:
            self._record_parse_failure(
                sample_id,
                mode=failure_mode,
                detail=failure_detail,
                response_format=response_format,
            )
        else:
            self._clear_parse_failure(sample_id)

        if pred_data is None:
            # directly set to 0 for all metrics
            metrics_result[f"{response_format}_format_metric"] = 0
            metrics_result[f"{response_format}_args_em_metric"] = 0
            return metrics_result

        # Exact matching
        metrics_result[f"{response_format}_format_metric"] = 1
        metrics_result[f"{response_format}_args_em_metric"] = (
            self.compute_args_em_metric(
                gt_action=data_sample.gt["action"],
                pred_action=pred_data["action"],
                gt_args=data_sample.gt["args"],
                pred_args=pred_data["args"],
            )
        )
        return metrics_result

    def compute_args_em_metric(self, gt_action, pred_action, gt_args, pred_args):
        cnt = 0.0
        if gt_action == pred_action:
            cnt += 1.0
        num_args = len(gt_args) + 1  # 1 means action name match
        for gt_key in gt_args:
            pred_val = pred_args.get(gt_key, "")
            if pred_val == gt_args[gt_key]:
                cnt += 1.0
        return cnt / num_args

    def string_format_parse(
        self, data_sample
    ) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        pred_data = data_sample.pred
        template = data_sample.template
        thought_start = template["thought_start"]
        thought_end = template["thought_end"]
        action_start = template["action_start"]
        action_end = template["action_end"]
        args_start = template["args_start"]
        args_end = template["args_end"]

        parse_template = (
            thought_start
            + "{thought}"
            + thought_end
            + action_start
            + "{action}"
            + action_end
            + args_start
            + "{args}"
            + args_end
        )
        res = parse_string(parse_template, pred_data, allow_newline=True)
        if res is None:
            return None, "string_template_mismatch", "Failed to match string template"
        try:
            args = ast.literal_eval(res["args"].strip())
            res["args"] = args if isinstance(args, dict) else {}
            res["action"] = res["action"].strip()
            return res, None, None
        except Exception as exc:
            cleaned = dict(
                thought=res.get("thought", ""),
                action=res.get("action", "").strip(),
                args=dict(),
            )
            return cleaned, None, str(exc)

    def json_format_parse(
        self, data_sample
    ) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        try:
            pred_data = format_load(data_sample.pred)
            template = data_sample.template
            new_data = dict()
            new_data["thought"] = pred_data[template["thought"]]
            new_data["action"] = pred_data[template["action"]]
            args = pred_data[template["args"]]
            new_data["args"] = args if isinstance(args, dict) else {}
        except KeyError as exc:
            return None, "json_missing_key", str(exc)
        except Exception as exc:
            return None, "json_parse_error", str(exc)

        return new_data, None, None

    def evaluate(self):
        self._load_dataset()
        # Reset tracker for this evaluation pass.
        self._parse_tracker = ParseFailureTracker(
            dataset_path=self.dataset_path, evaluator_name=self.__class__.__name__
        )
        results_list = []
        per_item_metrics: Dict[str, Dict[str, float]] = {}
        evaluation_time = datetime.now(timezone.utc).isoformat()
        for data_entry in self.dataset:
            sample_id = data_entry["sample_id"]
            response_sample = data_entry["response_data_sample"]
            metrics_result = self._evaluate(sample_id, response_sample)
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
        self._parse_tracker.write_files()
        return aggregated_results

    def _post_process(self, results_list):
        # list of dict to dict of list
        results_dict = defaultdict(list)
        {results_dict[key].append(sub[key]) for sub in results_list for key in sub}
        metric_list = [
            "json_format_metric",
            "json_args_em_metric",
            "string_format_metric",
            "string_args_em_metric",
        ]
        for metric in metric_list:
            results_dict[metric] = np.round(np.mean(results_dict[metric]), decimals=4)
        return results_dict
