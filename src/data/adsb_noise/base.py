"""Common base classes shared by ADS-B noise pipeline components."""

from __future__ import annotations

import logging

from .config import MergerConfig


class PipelineComponent:
    """Provide shared configuration handling and logging for components."""

    def __init__(self, config: MergerConfig) -> None:
        """Initialise the component with configuration and a dedicated logger."""

        self.config: MergerConfig = config
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
