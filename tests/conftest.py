import pytest

from graphrag_service.engine import Engine
from graphrag_service.graph.build import KnowledgeGraph


@pytest.fixture(scope="session")
def kg() -> KnowledgeGraph:
    return KnowledgeGraph.build()


@pytest.fixture(scope="session")
def engine() -> Engine:
    return Engine()
