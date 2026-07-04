"""fal.ai integration — hosted image and video generation models.

Models are referenced by their fal id, e.g. ``fal-ai/flux/dev``,
``fal-ai/qwen-image``, ``fal-ai/ltx-video-13b-distilled``, ``fal-ai/wan-t2v``,
``fal-ai/hunyuan-video``.

    pyllm.configure(lambda c: setattr(c, "fal_api_key", "..."))
    img   = await pyllm.paint("a red panda", provider="fal", model="fal-ai/flux/dev")
    video = await pyllm.animate("a timelapse sunrise", model="fal-ai/ltx-video-13b-distilled")
"""

from __future__ import annotations

from ..protocols.fal import Fal as FalProtocol
from ..provider import Provider


class Fal(Provider):
    protocols = {"fal": FalProtocol}
    default_protocol_name = "fal"

    @property
    def api_base(self) -> str:
        return self.config.fal_api_base or "https://fal.run"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.config.fal_api_key}"}

    @classmethod
    def assumes_models_exist(cls) -> bool:
        return True

    @classmethod
    def configuration_options(cls) -> list[str]:
        return ["fal_api_key", "fal_api_base", "fal_queue_base"]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["fal_api_key"]
