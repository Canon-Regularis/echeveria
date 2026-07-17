"""A post-model head that reports the physiology proxies on their own.

The drought-stage head already embeds these proxies, but reading a water-potential or
stomatal-conductance estimate then means running the whole drought-stage rule set and digging the
block out of its output. This head surfaces the same proxies directly, so ``analyze --physiology``
and ``/analyze?physiology=true`` report them without the stage classification, and the dashboard can
show them in their own panel. The proxies derive from the drought markers, so the two heads always
agree. Every value is a crude RGB proxy, never a measurement; the basis string carries that caveat
with the numbers.
"""

from __future__ import annotations

from phytovision.models.drought.rule_based import physiology_basis, physiology_proxies
from phytovision.types import PlantFeatures


class PhysiologyHead:
    name = "physiology"

    def run(self, features: PlantFeatures) -> dict[str, object]:
        return {**physiology_proxies(features.values), "basis": physiology_basis()}
