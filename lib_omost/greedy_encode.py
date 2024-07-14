from __future__ import annotations
import logging
import torch
from typing import Callable, NamedTuple, TypedDict


class SpecialTokens(TypedDict):
    start: int
    end: int
    pad: int


class CLIPTokens(NamedTuple):
    clip_l_tokens: list[int]
    clip_g_tokens: list[int] | None = None

    @classmethod
    def empty_tokens(cls) -> CLIPTokens:
        return CLIPTokens(clip_l_tokens=[], clip_g_tokens=[])

    @property
    def length(self) -> int:
        return len(self.clip_l_tokens)

    def __repr__(self) -> str:
        return f"CLIPTokens(clip_l_tokens({len(self.clip_l_tokens)}), clip_g_tokens={len(self.clip_g_tokens) if self.clip_g_tokens else None})"

    def __add__(self, other: CLIPTokens) -> CLIPTokens:
        if self.clip_g_tokens is None or other.clip_g_tokens is None:
            clip_g_tokens = None
        else:
            clip_g_tokens = self.clip_g_tokens + other.clip_g_tokens

        return CLIPTokens(
            clip_l_tokens=self.clip_l_tokens + other.clip_l_tokens,
            clip_g_tokens=clip_g_tokens,
        )

    @staticmethod
    def _get_77_tokens(subprompt_inds: list[int]) -> list[int]:
        # Note that all subprompt are theoretically less than 75 tokens (without bos/eos)
        result = (
            [SPECIAL_TOKENS["start"]]
            + subprompt_inds[:75]
            + [SPECIAL_TOKENS["end"]]
            + [SPECIAL_TOKENS["pad"]] * 75
        )
        return result[:77]

    def clamp_to_77_tokens(self) -> CLIPTokens:
        return CLIPTokens(
            clip_l_tokens=self._get_77_tokens(self.clip_l_tokens),
            clip_g_tokens=(
                self._get_77_tokens(self.clip_g_tokens)
                if self.clip_g_tokens
                else None
            ),
        )


class EncoderOutput(NamedTuple):
    cond: torch.Tensor
    pooler: torch.Tensor


TokenizeFunc = Callable[[str], CLIPTokens]
EncodeFunc = Callable[[CLIPTokens], EncoderOutput]


# ComfyUI protocol. See sd1_clip.py/sdxl_clip.py for the actual implementation.
SPECIAL_TOKENS: SpecialTokens = {"start": 49406, "end": 49407, "pad": 49407}


def greedy_partition(items: list[CLIPTokens], max_sum: int) -> list[list[CLIPTokens]]:
    bags: list[list[CLIPTokens]] = []
    current_bag: list[CLIPTokens] = []
    current_sum: int = 0

    for item in items:
        num = item.length
        if current_sum + num > max_sum:
            if current_bag:
                bags.append(current_bag)
            current_bag = [item]
            current_sum = num
        else:
            current_bag.append(item)
            current_sum += num

    if current_bag:
        bags.append(current_bag)

    return bags


def encode_bag_of_subprompts_greedy(
    prefixes: list[str],
    suffixes: list[str],
    tokenize_func: TokenizeFunc,
    encode_func: EncodeFunc,
    logger: logging.Logger | None = None,
) -> EncoderOutput:
    """
    Note: tokenize_func is expected to clamp the tokens to 75 tokens.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Begin with tokenizing prefixes
    prefix_tokens: CLIPTokens = sum(
        [tokenize_func(prefix) for prefix in prefixes], CLIPTokens.empty_tokens()
    )
    logger.debug(f"Prefix tokens: {prefix_tokens}")

    # Then tokenizing suffixes
    allowed_suffix_length = 75 - prefix_tokens.length
    logger.debug(f"Allowed suffix length: {allowed_suffix_length}")
    suffix_targets: list[CLIPTokens] = [
        tokenize_func(subprompt) for subprompt in suffixes
    ]
    logger.debug(f"Suffix targets: {suffix_targets}")

    # Then merge prefix and suffix tokens
    suffix_targets = greedy_partition(suffix_targets, max_sum=allowed_suffix_length)
    targets = [
        sum([prefix_tokens, *b], CLIPTokens.empty_tokens()).clamp_to_77_tokens()
        for b in suffix_targets
    ]

    # Encode!
    encoded_embeds = [encode_func(target) for target in targets]
    conds_merged = torch.concat([embed.cond for embed in encoded_embeds], dim=1)
    poolers_merged = encoded_embeds[0].pooler
    logger.debug(f"merged conds: {conds_merged.shape}, pooler: {poolers_merged.shape}")

    return EncoderOutput(cond=conds_merged, pooler=poolers_merged)
