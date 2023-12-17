"""Timestamp functionality."""

from datetime import datetime, timezone
import uuid
from dataclasses import dataclass, field


def utc_timestamp() -> str:
    """Generate an id based on the current timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-%S-%f")


@dataclass
class IdGenerator:
    """Generate a unique sequence of ids based on a namespace and a seed. Ids are unique but deterministic."""

    namespace: uuid.UUID = field(default_factory=uuid.uuid4)
    seed: str = field(default_factory=utc_timestamp)  # Seed is a string
    _counter: int = field(default=0, init=False)

    def __call__(self) -> uuid.UUID:
        unique_seed = f"{self.seed}_{self._counter}"
        self._counter += 1
        return uuid.uuid5(self.namespace, unique_seed)


def test_id_generator():
    """Test id generator."""
    fixed_namespace_uuid = uuid.uuid4()
    id_generator1 = IdGenerator(namespace=fixed_namespace_uuid, seed="2023-12-17")
    uuids1 = [id_generator1() for _ in range(5)]
    id_generator2 = IdGenerator(namespace=fixed_namespace_uuid, seed="2023-12-18")
    uuids2 = [id_generator2() for _ in range(5)]
    assert uuids1 != uuids2
    id_generator1_again = IdGenerator(namespace=fixed_namespace_uuid, seed="2023-12-17")
    uuids1_again = [id_generator1_again() for _ in range(5)]
    assert uuids1 == uuids1_again
