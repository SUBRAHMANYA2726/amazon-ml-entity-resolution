"""Entity consistency and singleton-audit interfaces."""

from business_entity_resolution.entity.consistency import (
    EntityConsistencyChecker,
    SingletonAuditor,
)

__all__ = ["EntityConsistencyChecker", "SingletonAuditor"]
