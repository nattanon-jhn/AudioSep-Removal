"""
LLM router utilities for AudioSep using the Google Gen AI Python SDK (google-genai).

Why this version
- Avoids manual REST payload quirks (400 Bad Request issues).
- Uses the same simple connection pattern you showed:
    from google import genai
    client = genai.Client()  # reads GEMINI_API_KEY from env
    response = client.models.generate_content(...)

What it outputs
A PromptRoute with:
- cls: positive | negative | negation | contrastive
- route: FG | BG
- positive_sentence
- negative_sentence

ENV
- GEMINI_API_KEY  (required to use Gemini)
Optional:
- GEMINI_MODEL (default: "gemini-2.5-flash")
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Optional dependency:
#   pip install -U google-genai
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover
    genai = None
    types = None


@dataclass
class PromptRoute:
    cls: str
    route: str
    positive_sentence: str
    negative_sentence: str

    def normalize(self) -> "PromptRoute":
        self.cls = (self.cls or "").strip().lower()
        self.route = (self.route or "").strip().upper()
        self.positive_sentence = (self.positive_sentence or "").strip()
        self.negative_sentence = (self.negative_sentence or "").strip()

        if self.route not in {"FG", "BG"}:
            self.route = "FG"

        if self.cls not in {"positive", "negative", "negation", "contrastive"}:
            if self.positive_sentence and self.negative_sentence:
                self.cls = "contrastive"
            elif self.negative_sentence and not self.positive_sentence:
                self.cls = "negative"
            else:
                self.cls = "positive"

        return self

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PromptRoute":
        return PromptRoute(
            cls=str(d.get("class", d.get("cls", ""))),
            route=str(d.get("route", "")),
            positive_sentence=str(d.get("Positive_sentence", d.get("positive_sentence", ""))),
            negative_sentence=str(d.get("Negative_sentence", d.get("negative_sentence", ""))),
        ).normalize()


DEFAULT_SYSTEM_PROMPT = """You are a strict prompt router for an audio source separation system.

Given a user prompt, you MUST output a single JSON object (no markdown, no extra text) with keys:
- "class": one of ["positive","negative","negation","contrastive"]
- "route": one of ["FG","BG"]
- "Positive_sentence": a short noun phrase representing what to keep/separate (or "" if none)
- "Negative_sentence": a short noun phrase representing what to remove/avoid (or "" if none)

Definitions:
- positive: user wants to KEEP some sound(s). route=FG. Positive_sentence=target, Negative_sentence=""
- negative: user wants to REMOVE some sound(s). route=BG. Positive_sentence="", Negative_sentence=target
- negation: user wants ALL sounds EXCEPT some sound(s). route=BG. Positive_sentence="", Negative_sentence=target
- contrastive: user wants KEEP A NOT B (or keep A, remove B). route=FG. Positive_sentence=A, Negative_sentence=B

Examples:
- keep dog barking -> {"class":"positive","route":"FG","Positive_sentence":"dog","Negative_sentence":""}
- remove dog barking -> {"class":"negative","route":"BG","Positive_sentence":"","Negative_sentence":"dog"}
- all sound except dog -> {"class":"negation","route":"BG","Positive_sentence":"","Negative_sentence":"dog"}
- keep dog not cat -> {"class":"contrastive","route":"FG","Positive_sentence":"dog","Negative_sentence":"cat"}
- keep thunder, remove water drop -> {"class":"contrastive","route":"FG","Positive_sentence":"thunder","Negative_sentence":"water drop"}
"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def rule_based_router(user_prompt: str) -> PromptRoute:
    p = (user_prompt or "").strip().lower()

    def clean(s: str) -> str:
        s = s.replace(",", " ").strip()
        s = re.sub(r"\b(sound|sounds|audio|noise|noises|the|a|an)\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        toks = [t for t in re.split(r"[\s;]+", s) if t]
        return " ".join(toks[:5]).strip()

    # keep/remove in either order
    if "keep" in p and "remove" in p:
        k_i = p.find("keep")
        r_i = p.find("remove")
        # Case 1: keep A remove B
        if k_i < r_i:
            tail = p.split("keep", 1)[1]
            if "remove" in tail:
                a, b = tail.split("remove", 1)
                return PromptRoute("contrastive", "FG", clean(a), clean(b)).normalize()
        # Case 2: remove B keep ... (often means 'remove B, keep the rest')
        else:
            after_remove = p.split("remove", 1)[1]
            if "keep" in after_remove:
                b, keep_tail = after_remove.split("keep", 1)
            else:
                b, keep_tail = after_remove, ""
            b_clean = clean(b)
            keep_tail_l = keep_tail.strip()
            # 'keep all/everything' -> background route: remove B, keep the rest
            if ("keep all" in p) or ("keep everything" in p) or ("all" in keep_tail_l.split()) or ("everything" in keep_tail_l.split()):
                return PromptRoute("negative", "BG", "", b_clean).normalize()
            keep_clean = clean(keep_tail)
            if keep_clean:
                return PromptRoute("contrastive", "FG", keep_clean, b_clean).normalize()
            return PromptRoute("negative", "BG", "", b_clean).normalize()

    if "except" in p:
        neg = clean(p.split("except", 1)[1])
        return PromptRoute("negation", "BG", "", neg).normalize()

    if any(k in p for k in ["remove", "delete", "suppress", "eliminate"]) and "keep" not in p:
        for kw in ["remove", "delete", "suppress", "eliminate"]:
            if kw in p:
                neg = clean(p.split(kw, 1)[1])
                return PromptRoute("negative", "BG", "", neg).normalize()

    if "keep" in p and ((" not " in f" {p} ") or (" without " in f" {p} ")):
        left = p.split("keep", 1)[1]
        if " not " in f" {left} ":
            a, b = re.split(r"\bnot\b", left, maxsplit=1)
        else:
            a, b = re.split(r"\bwithout\b", left, maxsplit=1)
        return PromptRoute("contrastive", "FG", clean(a), clean(b)).normalize()

    if "keep" in p:
        pos = clean(p.split("keep", 1)[1])
    else:
        pos = clean(p)
    return PromptRoute("positive", "FG", pos, "").normalize()


def gemini_router_prompt(
    user_prompt: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
) -> Tuple[PromptRoute, Any]:
    """
    Calls Gemini via google-genai SDK.

    - Reads GEMINI_API_KEY from env by default (like your snippet).
    - You can also pass api_key explicitly.
    """
    if genai is None or types is None:
        raise RuntimeError("google-genai is not installed. Run: pip install -U google-genai")

    model = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    # Create client (SDK supports passing api_key, but env is fine too)
    client = genai.Client(api_key=api_key)

    cfg = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=cfg,
    )

    text = getattr(resp, "text", None) or str(resp)
    obj = _extract_json_object(text) or {}
    pr = PromptRoute.from_dict(obj)
    return pr, resp


# Cache to reduce calls
_ROUTER_CACHE: Dict[str, PromptRoute] = {}


def route_prompt(
    user_prompt: str,
    enable_llm_router: bool = True,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    strict: bool = False,
    use_cache: bool = False,
) -> PromptRoute:
    """
    High-level router.
    - If enable_llm_router=True: try Gemini, fallback to rule-based (unless strict=True).
    - If enable_llm_router=False: rule-based.

    llm_kwargs are passed to gemini_router_prompt (model, api_key, system_prompt, temperature, max_output_tokens)
    """
    llm_kwargs = llm_kwargs or {}
    key = (user_prompt or "").strip().lower()

    if use_cache and key in _ROUTER_CACHE:
        print("[LLM ROUTER] cache hit")
        return _ROUTER_CACHE[key]

    if enable_llm_router:
        try:
            pr, _resp = gemini_router_prompt(user_prompt, **llm_kwargs)
            print("[LLM ROUTER] ✅ used Gemini SDK")
        except Exception as e:
            if strict:
                raise
            print(f"[LLM ROUTER] ❌ Gemini SDK failed ({type(e).__name__}: {e}) -> fallback rule-based")
            pr = rule_based_router(user_prompt)
    else:
        pr = rule_based_router(user_prompt)

    if use_cache:
        _ROUTER_CACHE[key] = pr
    return pr
