"""
LlamaCPP Ideogram Prompt Builder — ComfyUI Custom Nodes
--------------------------------------------------------
Nodes:
  1. LlamaCppIdeogramPrompter  — dropdown of live models, calls your llama.cpp
     server, strips <think> blocks, returns clean Ideogram JSON. Has a built-in
     "unload after generation" toggle so you don't need a separate node.

  2. LlamaCppJsonViewer  — takes any STRING and pretty-prints it as formatted
     JSON for easy inspection. Wire the ideogram_json output here to see the
     full structured output in the UI.
"""

import json
import re
import urllib.request
import urllib.error
import time

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_URL = "http://127.0.0.1:8082"


def _post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_model_list(server_url: str = _DEFAULT_URL) -> list:
    """
    Hits /v1/models and returns model IDs as a list for the dropdown.
    Falls back gracefully if the server is offline at ComfyUI startup.
    """
    try:
        resp = _get_json(f"{server_url.rstrip('/')}/v1/models", timeout=5)
        ids  = [m["id"] for m in resp.get("data", []) if m.get("id")]
        if ids:
            print(f"[LlamaCPP] Found {len(ids)} models on {server_url}")
            return ids
    except Exception as exc:
        print(f"[LlamaCPP] Could not fetch model list from {server_url}: {exc}")
    return ["(server offline — reload page when server is running)"]


def _unload_all_models(base: str, wait_seconds: int = 3) -> bool:
    """
    Matches the working Open WebUI VRAM unload tool exactly:
      GET  /v1/models        -> list all models, find those with status "loaded"
                                status can be plain string OR {"value": "loaded"}
      POST /models/unload    -> {"model": "<id>"} for each loaded one
    """
    base = base.rstrip("/")
    unloaded_count = 0
    failed_count = 0

    # Step 1: fetch via /v1/models (same as the working tool)
    try:
        resp = _get_json(f"{base}/v1/models", timeout=10)
        all_models = resp.get("data", [])
    except Exception as exc:
        print(f"[LlamaCPP VRAM] Could not reach /v1/models: {exc}")
        return False

    # Step 2: filter loaded — status is either "loaded" or {"value": "loaded"}
    loaded_ids = []
    for m in all_models:
        model_id = m.get("id", "")
        if not model_id:
            continue
        status = m.get("status", {})
        status_val = status.get("value", "") if isinstance(status, dict) else status
        if status_val == "loaded":
            loaded_ids.append(model_id)

    if not loaded_ids:
        print("[LlamaCPP VRAM] No models currently loaded — nothing to unload.")
        return True

    print(f"[LlamaCPP VRAM] Found {len(loaded_ids)} loaded: {loaded_ids}")

    # Step 3: POST /models/unload per model
    for model_id in loaded_ids:
        print(f"[LlamaCPP VRAM] Unloading '{model_id}'...")
        try:
            result = _post_json(f"{base}/models/unload", {"model": model_id}, timeout=30)
            print(f"[LlamaCPP VRAM] Unloaded '{model_id}' -> {result}")
            unloaded_count += 1
        except Exception as e:
            print(f"[LlamaCPP VRAM] Failed to unload '{model_id}': {e}")
            failed_count += 1

    print(f"[LlamaCPP VRAM] Done — unloaded {unloaded_count}, failed {failed_count}.")

    if unloaded_count > 0 and wait_seconds > 0:
        print(f"[LlamaCPP VRAM] Waiting {wait_seconds}s for VRAM reclaim...")
        time.sleep(wait_seconds)

    return unloaded_count > 0


def _strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks that reasoning/thinking models emit
    before their actual answer. Handles multiline, greedy-safe.
    """
    # Remove <think> ... </think> blocks (case-insensitive, dotall)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Also handle models that use a bare chain-of-thought prefix like "Thinking:\n..."
    # followed by the actual JSON (heuristic: strip everything before the first '{')
    cleaned = cleaned.strip()
    if cleaned and cleaned[0] != "{":
        brace = cleaned.find("{")
        if brace != -1:
            cleaned = cleaned[brace:]
    return cleaned.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Built-in Ideogram system prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = r"""You are an elite image-direction AI. Your job is to transform a short user idea into the richest, most detailed structured JSON prompt that an Ideogram image renderer can consume. Depth of description is a virtue — the more precisely you describe every visual element, the better the image.

## OUTPUT CONTRACT — exactly three top-level keys, in this order:

{"aspect_ratio":"W:H","high_level_description":"...","compositional_deconstruction":{"background":"...","elements":[ ... ]}}

- Emit a SINGLE-LINE MINIFIED JSON object — no markdown fences, no commentary, no other text before or after the JSON.
- Preserve non-ASCII characters as-is. Never escape with \uNNNN or transliterate.
- Use SINGLE quotes for embedded text references in prose fields. The `text` field of text elements uses the user's verbatim characters.

### `aspect_ratio` (first field, always required)
A string in W:H form with positive integers (1:1, 16:9, 9:16, 4:5, etc.).
- If the user message gives a concrete W:H, echo it verbatim.
- NEVER emit the literal string auto.

### `high_level_description` — rich cinematic summary
- 2–4 dense sentences. Starts immediately with the subject.
- Name the medium/art style, the overall mood, the dominant color palette, and the compositional structure (foreground / midground / background relationship).
- Mention lighting quality (golden-hour, diffused overcast, hard rim light, neon glow, etc.) and the emotional tone the image should convey.
- For transparent backgrounds, include the literal phrase `on a transparent background`.

## ELEMENTS — what they are
Each element is one of:
{"type":"obj","bbox":[y1,x1,y2,x2],"desc":"..."}
{"type":"text","bbox":[y1,x1,y2,x2],"text":"LINE ONE\nLINE TWO","desc":"..."}

bbox is optional per-element.

### SINGLE SUBJECT = SINGLE ELEMENT
A coherent subject is exactly ONE obj element. Anatomical and structural parts (clothing, facial features, accessories, pose) all go inside that element's desc — NOT as separate elements.

### Element desc — 80–150 words per element
Be exhaustive. Include ALL of the following that apply:
- Precise identity and species/type
- Body pose, gesture, and weight distribution
- Facial expression and eye direction
- Clothing: every garment, fabric texture, color, fit, wear/distress level, branding if any
- Accessories: jewelry, bags, hats, glasses — material, color, style
- Skin/fur/surface: tone, texture, sheen, markings
- Lighting on THIS element: highlight placement, shadow side, rim light color
- Stylistic rendering: photorealistic / cel-shaded / painterly / 3D render / etc.
- Any motion blur, depth-of-field treatment, or special FX on the element

### BACKGROUND — exhaustive scene description
Describe the full scene shell with the same level of richness as elements:
- Sky / ceiling / backdrop: color gradient, cloud formations, weather conditions, time of day
- Ground / floor: material, texture, wet/dry, perspective foreshortening
- Architecture or landscape features: specific style, materials, condition
- Ambient and practical lighting sources: their color temperature, intensity, and cast shadows
- Atmosphere: haze, fog, dust particles, bokeh quality, depth of field distance
ALWAYS-BACKGROUND (never obj elements): sky, clouds, horizon, distant mountains, weather, floor/ground surface, studio backdrop.
No double-counting: anything fully described in background cannot also be an obj element.

## BBOX STRATEGY
x runs 0-1000 left-right, y runs 0-1000 top-bottom. Format [y1, x1, y2, x2] with y1<y2, x1<x2.
Place bboxes thoughtfully to reflect natural composition (rule of thirds, visual hierarchy).

## SPECIFICITY — always commit to one concrete value
Banned hedges: things like, such as, various, could include, might be, some kind of, or similar, oak or walnut. Pick ONE specific value and commit. E.g. not "some kind of red jacket" but "a scarlet bomber jacket in distressed leather with chrome zipper pulls".

## TEXT HANDLING
text field — literal characters verbatim, exactly as the user specified, including case and punctuation.
desc — detailed typography direction: font family category (slab serif / grotesque / display script), weight, color, outline/shadow/glow treatment, 3D extrusion depth, perspective warp, fill texture (chrome, neon tube, spray-paint), size relative to canvas.

## TRANSPARENT BACKGROUND
If transparent background is requested, background field MUST be exactly: transparent background

## STYLE INFERENCE
If the user's idea implies a style (poster, anime, oil painting, pixel art, graffiti, fashion editorial, product shot, etc.) inject that style vocabulary throughout every desc field and the high_level_description. Don't leave style implicit — name it explicitly.

Emit ONLY the single-line minified JSON. No preamble, no markdown fences, no explanation."""


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 — Ideogram Prompt Builder (combined with VRAM clear toggle)
# ──────────────────────────────────────────────────────────────────────────────

class LlamaCppIdeogramPrompter:
    """
    Calls your local llama.cpp server to expand a short user idea into the
    structured JSON that Ideogram-4 / CLIPTextEncode expects.

    Features:
    - Live model dropdown populated from /v1/models (refresh page to update)
    - Strips <think>...</think> blocks from reasoning/thinking models
    - Built-in "unload after generation" toggle — no separate node needed
    - Returns both the clean JSON and the full raw response for debugging
    """

    CATEGORY     = "LlamaCPP / Ideogram"
    FUNCTION     = "generate"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ideogram_json", "raw_response")

    @classmethod
    def INPUT_TYPES(cls):
        models = _fetch_model_list(_DEFAULT_URL)
        return {
            "required": {
                "user_idea": ("STRING", {
                    "multiline": True,
                    "default": "A surreal streetwear collage poster with a skateboarder and giant puffy letters spelling COMFY",
                    "tooltip": "Short natural-language description of what you want to generate.",
                }),
                "aspect_ratio": ("STRING", {
                    "default": "9:16",
                    "tooltip": "Target aspect ratio passed to the LLM (e.g. 1:1, 16:9, 9:16, 4:5).",
                }),
                "model": (models, {
                    "tooltip": "Model to use. Reload the page to refresh this list from the server.",
                }),
                "server_url": ("STRING", {
                    "default": _DEFAULT_URL,
                    "tooltip": "Base URL of your llama.cpp server.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature. Lower = more focused JSON output.",
                }),
                "max_tokens": ("INT", {
                    "default": 8192,
                    "min": 256,
                    "max": 32768,
                    "step": 128,
                    "tooltip": "Max tokens including any thinking/reasoning tokens the model emits.",
                }),
                "unload_after": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Unload model after generation",
                    "label_off": "Keep model loaded",
                    "tooltip": "POST /models/unload to free VRAM before diffusion runs.",
                }),
                "unload_wait_seconds": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Seconds to wait after unload for the GPU driver to reclaim VRAM.",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Thinking ON (deep reasoning)",
                    "label_off": "Thinking OFF (fast)",
                    "tooltip": "Passes thinking=true + budget_tokens to the API. Qwen3 / DeepSeek-R1 style models need this explicitly set or they skip the <think> block.",
                }),
                "thinking_budget": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 16384,
                    "step": 256,
                    "tooltip": "Max tokens the model may spend on internal reasoning (thinking budget). Only used when enable_thinking=True.",
                }),
            },
            "optional": {
                "system_prompt_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Leave blank to use the built-in Ideogram system prompt.",
                }),
            },
        }

    def generate(
        self,
        user_idea: str,
        aspect_ratio: str,
        model: str,
        server_url: str,
        temperature: float,
        max_tokens: int,
        unload_after: bool,
        unload_wait_seconds: int,
        enable_thinking: bool = True,
        thinking_budget: int = 4096,
        system_prompt_override: str = "",
    ):
        server_url = server_url.rstrip("/")
        system     = system_prompt_override.strip() or _SYSTEM_PROMPT

        # Skip placeholder entry shown when server was offline at startup
        use_model = model.strip()
        if use_model.startswith("("):
            use_model = ""

        user_message = (
            f"TARGET IMAGE ASPECT RATIO: {aspect_ratio} (width:height).\n"
            f"User idea: {user_idea}"
        )

        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      False,
        }
        if use_model:
            payload["model"] = use_model

        # ── Thinking / extended reasoning ─────────────────────────────────────
        # Qwen3 and similar models will skip <think> blocks unless the API
        # explicitly requests thinking mode. llama.cpp passes this through as
        # a top-level "thinking" object and also honours chat_format flags.
        if enable_thinking:
            # llama.cpp per-request thinking budget field (discussion #21445)
            payload["thinking_budget_tokens"] = thinking_budget
            print(
                f"[LlamaCppIdeogramPrompter] Thinking ENABLED  budget={thinking_budget} tokens"
            )
        else:
            # 0 disables thinking (equivalent to --reasoning-budget 0)
            payload["thinking_budget_tokens"] = 0
            print("[LlamaCppIdeogramPrompter] Thinking DISABLED")

        endpoint = f"{server_url}/v1/chat/completions"
        print(f"[LlamaCppIdeogramPrompter] POST {endpoint}  model={use_model or '(server default)'}")
        t0 = time.time()

        try:
            response = _post_json(endpoint, payload, timeout=600)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"[LlamaCppIdeogramPrompter] Could not reach llama.cpp at {endpoint}.\n"
                f"Error: {exc}\nCheck server_url and that the server is running."
            ) from exc

        elapsed = time.time() - t0

        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"[LlamaCppIdeogramPrompter] Empty choices in response: {response}"
            )

        raw_content: str = choices[0].get("message", {}).get("content", "").strip()

        usage = response.get("usage", {})
        print(
            f"[LlamaCppIdeogramPrompter] Done in {elapsed:.1f}s  |  "
            f"prompt={usage.get('prompt_tokens','?')}  "
            f"completion={usage.get('completion_tokens','?')} tokens"
        )

        # ── Unload BEFORE we do JSON parsing so VRAM frees even if parsing fails ──
        if unload_after:
            _unload_all_models(server_url, wait_seconds=unload_wait_seconds)

        # ── Strip thinking tokens ──────────────────────────────────────────────
        # Thinking models (Qwen3, etc.) wrap their reasoning in <think>...</think>
        # before emitting the actual answer.
        cleaned = _strip_thinking(raw_content)

        # Strip markdown fences if the model wrapped anyway
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.split("\n")
                if not line.strip().startswith("```")
            ).strip()

        # Validate + re-serialize to guarantee clean minified JSON
        try:
            parsed     = json.loads(cleaned)
            clean_json = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError as e:
            print(f"[LlamaCppIdeogramPrompter] WARNING: non-JSON after stripping think blocks.")
            print(f"  Parse error: {e}")
            print(f"  Cleaned text (first 500 chars): {cleaned[:500]}")
            # Return raw cleaned text so the user can see what went wrong
            clean_json = cleaned

        return (clean_json, raw_content)



# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LlamaCppIdeogramPrompter": LlamaCppIdeogramPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppIdeogramPrompter": "🦙 LlamaCPP → Ideogram Prompt",
}
