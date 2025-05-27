# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Counter as CollectionsCounter
from typing import Dict, List, Optional, Protocol, Union

import numpy as np
from packaging.version import Version
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info, disable_created_metrics

disable_created_metrics()

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.


# begin-metrics-definitions
class NimMetrics:
    labelname_finish_reason = "finished_reason"

    supported_metrics = [
        "num_requests_running",
        "num_requests_waiting",
        "num_request_max",
        "gpu_cache_usage_perc",
        "prompt_tokens",  # '_total' suffix is only for reporting
        "generation_tokens",  # '_total' suffix is only for reporting
        "time_to_first_token_seconds",
        "time_per_output_token_seconds",
        "e2e_request_latency_seconds",
        "request_prompt_tokens",
        "request_generation_tokens",
        "request_success",  # '_total' suffix is only for reporting
    ]
    __unified = False

    @classmethod
    def unify_vllm_metrics(cls, max_num_seqs, stat_logger):
        if cls.__unified:
            return
        # if engine created is vLLM engine (AsyncLLMEngine), the prometheus metrics
        # have been constructed per vLLM specification and on default registry
        # (REGISTRY). This function is to modify the registered collectors to report
        # metrics in the unified specification (see below).
        # Note that this function may be integrated into the Nvidia vLLM fork once
        # it is ready, with the goal to unify the metrics setup in both runtimes:
        # setup via StatsLogger during engine initialization.
        for collector in copy.copy(REGISTRY._collector_to_names):
            # strip "vllm:" in name and only re-register collectors defined in
            # unified metrics
            if hasattr(collector, "_name") and "vllm" in collector._name:
                REGISTRY.unregister(collector)
                collector._name = collector._name.replace("vllm:", "")
                for supported_metric in NimMetrics.supported_metrics:
                    if supported_metric == collector._name:
                        REGISTRY.register(collector)
                        break
        # Inject static metrics for vLLM as it is not reported natively
        req_max = Gauge(
            name="num_request_max",
            documentation="Max number of concurrently running requests.",
            labelnames=list(stat_logger.keys()),
        )

        req_max.labels(**stat_logger).set(max_num_seqs)
        cls.__unified = True

    def __init__(self, labelnames: List[str], max_model_len: int, full_metrics: bool = True):
        # System stats
        #   Scheduler State
        self.gauge_scheduler_running = self._get_or_create_collector(
            Gauge,
            name="num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames,
        )
        self.gauge_scheduler_waiting = self._get_or_create_collector(
            Gauge,
            name="num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames,
        )
        self.gauge_scheduler_request_max = self._get_or_create_collector(
            Gauge,
            name="num_request_max",
            documentation="Max number of concurrently running requests.",
            labelnames=labelnames,
        )
        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage = self._get_or_create_collector(
            Gauge,
            name="gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
        )

        # Iteration stats
        self.counter_prompt_tokens = self._get_or_create_collector(
            Counter,
            name="prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
        )
        self.counter_generation_tokens = self._get_or_create_collector(
            Counter,
            name="generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
        )
        self.histogram_time_to_first_token = self._get_or_create_collector(
            Histogram,
            name="time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
        )
        self.histogram_time_per_output_token = self._get_or_create_collector(
            Histogram,
            name="time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5],
        )

        # Request stats
        #   Latency
        self.histogram_e2e_time_request = self._get_or_create_collector(
            Histogram,
            name="e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )

        if full_metrics:
            #   Metadata
            self.histogram_num_prompt_tokens_request = self._get_or_create_collector(
                Histogram,
                name="request_prompt_tokens",
                documentation="Number of prefill tokens processed.",
                labelnames=labelnames,
                buckets=build_1_2_5_buckets(max_model_len),
            )
            self.histogram_num_generation_tokens_request = self._get_or_create_collector(
                Histogram,
                name="request_generation_tokens",
                documentation="Number of generation tokens processed.",
                labelnames=labelnames,
                buckets=build_1_2_5_buckets(max_model_len),
            )
            self.counter_request_success = self._get_or_create_collector(
                Counter,
                name="request_success_total",
                documentation="Count of successfully processed requests.",
                labelnames=labelnames + [NimMetrics.labelname_finish_reason],
            )

    def _get_or_create_collector(self, collector_cls, name, **kwargs):
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        return collector_cls(name=name, **kwargs)


# end-metrics-definitions


def build_1_2_5_buckets(max_value: int):
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values (1, 2, 5) until the value exceeds the specified maximum.

    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    mantissa_lst = [1, 2, 5]
    exponent = 0
    buckets = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


@dataclass
class IterationStats:
    """Intermediate representation of Promethus stats at each iteration."""

    # System stats (should have _sys suffix)
    #   Scheduler State
    num_running_sys: int
    num_waiting_sys: int
    num_request_max_sys: int
    #   KV Cache Usage in %
    gpu_cache_usage_sys: float


@dataclass
class CompletionStats:
    """Intermediate representation of Promethus stats captures stats for completed requests."""

    # Iteration stats (should have _iter suffix)
    num_prompt_tokens_iter: int
    # number of generation token for the iteration will be aggregated based on
    # completion response. More accurate count can be obtained from
    # post-processed output as each output can have arbitrary number of
    # generation tokens
    num_generation_tokens_iter: int

    # Request stats (should have _requests suffix)
    #   Latency
    time_e2e_requests: List[float]
    #   Metadata
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    finished_reason_requests: List[str]
    time_to_first_tokens_iter: List[float]
    time_per_output_tokens_iter: List[float]


class StatLogger:
    """StatLogger is used to log iteration stats and request completion stats to Promethus."""

    def _expose_full_metrics(self) -> bool:
        # Expose full metrics if the vLLM version has the counterparts
        return True

    def __init__(self, labels: Dict[str, str], max_model_len: int) -> None:
        # Prometheus metrics
        self.labels = labels
        self.metrics = NimMetrics(
            labelnames=list(labels.keys()), max_model_len=max_model_len, full_metrics=self._expose_full_metrics()
        )

    def _log_iteration_prometheus(self, stats: IterationStats) -> None:
        # System state data
        self._log_gauge(self.metrics.gauge_scheduler_running, stats.num_running_sys)
        self._log_gauge(self.metrics.gauge_scheduler_waiting, stats.num_waiting_sys)
        self._log_gauge(self.metrics.gauge_scheduler_request_max, stats.num_request_max_sys)
        self._log_gauge(self.metrics.gauge_gpu_cache_usage, stats.gpu_cache_usage_sys)

    def _log_completion_prometheus(self, stats: CompletionStats):
        # Iteration level data
        self._log_counter(self.metrics.counter_prompt_tokens, stats.num_prompt_tokens_iter)
        self._log_counter(self.metrics.counter_generation_tokens, stats.num_generation_tokens_iter)
        # Request level data
        # Latency
        self._log_histogram(self.metrics.histogram_e2e_time_request, stats.time_e2e_requests)
        if self._expose_full_metrics():
            # Metadata
            finished_reason_counter = CollectionsCounter(stats.finished_reason_requests)
            self._log_counter_labels(
                self.metrics.counter_request_success, finished_reason_counter, NimMetrics.labelname_finish_reason
            )
            self._log_histogram(self.metrics.histogram_num_prompt_tokens_request, stats.num_prompt_tokens_requests)
            self._log_histogram(
                self.metrics.histogram_num_generation_tokens_request, stats.num_generation_tokens_requests
            )
        self._log_histogram(self.metrics.histogram_time_to_first_token, stats.time_to_first_tokens_iter)
        self._log_histogram(self.metrics.histogram_time_per_output_token, stats.time_per_output_tokens_iter)

    def _log_gauge(self, gauge: Gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter: Counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self.labels).inc(data)

    def _log_counter_labels(self, counter: Counter, data: CollectionsCounter, label_key: str) -> None:
        # Convenience function for collection counter of labels.
        for label, count in data.items():
            counter.labels(**{**self.labels, label_key: label}).inc(count)

    def _log_histogram(self, histogram: Histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.labels(**self.labels).observe(datum)

    def log_iteration(self, stats: IterationStats) -> None:
        """Called by LLMEngine.
        Logs to prometheus and tracked stats every iteration."""

        # Log to prometheus.
        self._log_iteration_prometheus(stats)

    def log_completion(self, stats: CompletionStats) -> None:
        """Called by LLMEngine.
        Logs to prometheus and tracked stats every completion."""

        # Log to prometheus.
        self._log_completion_prometheus(stats)
