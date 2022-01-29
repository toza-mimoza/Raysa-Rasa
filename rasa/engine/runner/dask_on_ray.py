from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Text

import dask

from rasa.engine.exceptions import GraphRunError
from rasa.engine.graph import ExecutionContext, GraphNode, GraphNodeHook, GraphSchema, serialize_graph_node, deserialize_graph_node
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.hooks import LoggingHook, TrainingHook, serializer_logging_hook, serializer_training_hook, deserializer_logging_hook, deserializer_training_hook

import ray
from ray.util.dask import ray_dask_get


logger = logging.getLogger(__name__)


class DaskOnRayGraphRunner(GraphRunner):
    """Dask implementation of a `GraphRunner`."""

    def __init__(
        self,
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> None:
        """Initializes a `DaskOnRayGraphRunner`.

        Args:
            graph_schema: The graph schema that will be run.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            execution_context: Information about the current graph run to be passed to
                each node.
            hooks: These are called before and after the execution of each node.
        """
        self._graph_schema = graph_schema
        self._instantiated_nodes: Dict[Text, GraphNode] = self._instantiate_nodes(
            graph_schema, model_storage, execution_context, None
        )
        self._execution_context: ExecutionContext = execution_context

        # set ray's scheduler
        # dask.config.set(scheduler=ray_dask_get)
        # ray.util.register_serializer(GraphNode, serializer=serialize_graph_node, deserializer=deserialize_graph_node)
        # ray.util.register_serializer(LoggingHook, serializer=serializer_logging_hook, deserializer=deserializer_logging_hook)
        ray.util.register_serializer(TrainingHook, serializer=serializer_training_hook, deserializer=deserializer_training_hook)

    @classmethod
    def create(
        cls,
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> DaskOnRayGraphRunner:
        """Creates the runner (see parent class for full docstring)."""
        return cls(graph_schema, model_storage, execution_context, hooks)

    @staticmethod
    def _instantiate_nodes(
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> Dict[Text, GraphNode]:
        return {
            node_name: GraphNode.from_schema_node(
                node_name, schema_node, model_storage, execution_context, hooks
            )
            for node_name, schema_node in graph_schema.nodes.items()
        }

    def _build_dask_graph(self, schema: GraphSchema) -> Dict[Text, Any]:
        """Builds a dask graph from the instantiated graph.

        For more information about dask graphs
        see: https://docs.dask.org/en/latest/spec.html
        """
        run_graph = {
            node_name: (
                self._instantiated_nodes[node_name],
                *schema_node.needs.values(),
            )
            for node_name, schema_node in schema.nodes.items()
        }
        return run_graph

    def run(
        self,
        inputs: Optional[Dict[Text, Any]] = None,
        targets: Optional[List[Text]] = None,
    ) -> Dict[Text, Any]:
        """Runs the graph (see parent class for full docstring)."""
        run_targets = targets if targets else self._graph_schema.target_names
        minimal_schema = self._graph_schema.minimal_graph_schema(run_targets)
        run_graph = self._build_dask_graph(minimal_schema)

        if inputs:
            self._add_inputs_to_graph(inputs, run_graph)

        logger.debug(
            f"Running graph with inputs: {inputs}, targets: {targets} "
            f"and {self._execution_context}."
        )

        try:
            # print(f"Run Graph: {run_graph}")
            # print(f"Run Targets: {run_targets}")

            # dask_result = dask.threaded.get(
            #     run_graph, run_targets
            # )
            # inspect_serializability(test_serializability(run_graph), name="test")
            # print(run_graph)
            # import sys
            #
            # original_stdout = (
            #     sys.stdout
            # )  # Save a reference to the original standard output
            #
            # with open("output_graph_schema.txt", "w") as f:
            #     sys.stdout = f  # Change the standard output to the file we created.
            #     print(self._graph_schema)
            #     sys.stdout = (
            #         original_stdout  # Reset the standard output to its original value
            #     )
            dask_result = ray_dask_get(run_graph, run_targets)
            return dict(dask_result)
        except RuntimeError as e:
            raise GraphRunError("Error running runner.") from e

    @staticmethod
    def _add_inputs_to_graph(inputs: Optional[Dict[Text, Any]], graph: Any) -> None:
        for input_name, input_value in inputs.items():
            if isinstance(input_value, str) and input_value in graph.keys():
                raise GraphRunError(
                    f"Input value '{input_value}' clashes with a node name. Make sure "
                    f"that none of the input names passed to the `run` method are the "
                    f"same as node names in the graph schema."
                )
            graph[input_name] = (input_name, input_value)
