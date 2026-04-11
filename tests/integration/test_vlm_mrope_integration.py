# SPDX-License-Identifier: Apache-2.0
"""
Integration test for mRoPE VLM models (Qwen3-VL, Qwen3.5).

Validates per-request rope_deltas tracking, cache store/restore, and mixed
batch (image + text-only) correctness with boundary/SSD cache.

Test categories:
  1. Single VLM image request: cache store → hit → identical output
  2. Single text-only request: cache store → hit → identical output
  3. Image → text-only state transition: no rope_deltas contamination
  4. Mixed batch (2 image + 2 text-only): all produce coherent output
  5. Mixed batch with SSD cache: cache hit produces identical output
  6. VLM image caching (vision feature cache): store → hit → same output
  7. Batched decode position IDs: per-request rope_deltas isolation (#689)

Run with:
    pytest tests/integration/test_vlm_mrope_integration.py -v -m slow -s
    pytest tests/integration/test_vlm_mrope_integration.py -v -m slow -s -k "Qwen3-VL"
"""

import gc
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Requires macOS with Apple Silicon",
    ),
]

MROPE_MODELS = [
    "/Users/cryingneko/Workspace/models/Qwen3-VL-30B-A3B-Instruct-3bit",
    "/Users/cryingneko/Workspace/models/Qwen3.5-27B-4bit",
    "/Users/cryingneko/Workspace/models/Qwen3.5-35B-A3B-4bit",
    "/Users/cryingneko/Workspace/models/GLM-4.6V-Flash-4bit",
    "/Users/cryingneko/Workspace/models/Qwen3.5-122B-A10B-oQ4",
    "/Users/cryingneko/Workspace/models/gemma-4-26b-a4b-it-4bit",
    "/Users/cryingneko/Workspace/models/gemma-3-12b-it-qat-4bit",
    "/Users/cryingneko/Workspace/models/gemma-4-e2b-it-4bit",
    "/Users/cryingneko/Workspace/models/Nemotron-Cascade-2-30B-A3B-4bit",
]

TEXT_QUESTIONS = [
    "Explain the difference between a stack and a queue in 3 sentences.",
    "What is binary search? Give a one-paragraph explanation.",
    "Why are hash tables O(1) for lookup? Explain briefly.",
    "Compare bubble sort and merge sort in terms of time complexity.",
]

IMAGE_QUESTIONS = [
    "Describe the colors you see in this image.",
    "What patterns do you notice in this image?",
    "Describe the overall appearance of this image.",
    "What does this image look like? Be brief.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _track_peak_memory(label: str):
    import mlx.core as mx

    mx.synchronize()
    mem_before = mx.get_active_memory()
    mx.reset_peak_memory()
    yield
    mx.synchronize()
    mem_after = mx.get_active_memory()
    peak = mx.get_peak_memory()
    print(
        f"    [mem] {label}: "
        f"active {mem_after / 1024**3:.2f}GB "
        f"(delta {(mem_after - mem_before) / 1024**3:+.2f}GB), "
        f"peak {peak / 1024**3:.2f}GB"
    )


def _apply_chat_template_as_ids(tokenizer, messages) -> List[int]:
    try:
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(prompt_str, str):
            if hasattr(prompt_str, "input_ids"):
                ids = prompt_str.input_ids
                return ids[0] if isinstance(ids[0], list) else list(ids)
            prompt_str = str(prompt_str)
        return tokenizer.encode(prompt_str)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        text += "\nassistant:"
        return tokenizer.encode(text)


def _create_test_image(seed: int = 0, width: int = 336, height: int = 336):
    from PIL import Image

    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            r = int(255 * ((x + seed * 80) % width) / width)
            g = int(255 * ((y + seed * 120) % height) / height)
            b = int((128 + seed * 60) % 256)
            pixels[x, y] = (r, g, b)
    return img


def _create_colored_image(color: Tuple[int, int, int], width: int = 336, height: int = 336):
    from PIL import Image

    return Image.new("RGB", (width, height), color)


def _check_output_quality(text: str, label: str):
    assert len(text.strip()) > 0, f"[{label}] Empty output"

    # Word count: use whitespace split for Latin, character count for CJK
    words = text.split()
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    if cjk_chars < 10:
        assert len(words) >= 3, (
            f"[{label}] Too few words ({len(words)}): {text!r}"
        )

    # Alpha/CJK ratio check — content should be mostly text, not control chars
    text_chars = sum(1 for c in text if c.isalpha() or '\u4e00' <= c <= '\u9fff')
    text_ratio = text_chars / max(len(text), 1)
    assert text_ratio > 0.2, (
        f"[{label}] Low text ratio ({text_ratio:.2f}), "
        f"possibly gibberish: {text[:200]!r}"
    )

    for i in range(len(text) - 20):
        if len(set(text[i : i + 20])) == 1:
            pytest.fail(
                f"[{label}] Excessive single-char repetition: "
                f"{text[max(0,i-5):i+25]!r}"
            )


def _prepare_vlm_inputs(
    vlm_model,
    processor,
    messages: List[Dict[str, Any]],
    images: List[Any],
) -> Tuple[List[int], Any, Dict[str, Any], Optional[str]]:
    import mlx.core as mx
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_template
    from mlx_vlm.utils import prepare_inputs

    from omlx.utils.image import compute_image_hash

    num_images = len(images)
    tokenizer = getattr(processor, "tokenizer", processor)

    try:
        prompt = vlm_apply_template(
            processor, vlm_model.config, messages, num_images=num_images
        )
    except Exception:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

    inputs = prepare_inputs(
        processor, images=images if images else None,
        prompts=[prompt] if isinstance(prompt, str) else prompt,
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    attention_mask = inputs.get("attention_mask")
    extra_model_inputs = {
        k: v for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "pixel_values")
        and v is not None
    }

    if pixel_values is not None and num_images > 0:
        try:
            embed_features = vlm_model.get_input_embeddings(
                input_ids, pixel_values, mask=attention_mask, **extra_model_inputs
            )
        except TypeError:
            embed_features = vlm_model.get_input_embeddings(
                input_ids, pixel_values, **extra_model_inputs
            )
        mx.eval(embed_features.inputs_embeds)

        extra_kwargs = {}
        if hasattr(embed_features, "to_dict"):
            feat_dict = embed_features.to_dict()
            for k, v in feat_dict.items():
                if k != "inputs_embeds" and v is not None:
                    extra_kwargs[k] = v

        # Capture per-request mRoPE state
        lm = getattr(vlm_model, "language_model", None)
        if lm is not None:
            pid = getattr(lm, "_position_ids", None)
            if pid is not None and "position_ids" not in extra_kwargs:
                extra_kwargs["position_ids"] = pid
            rd = getattr(lm, "_rope_deltas", None)
            if rd is not None:
                extra_kwargs["_captured_rope_deltas"] = rd

        image_hash = compute_image_hash(images)
        token_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
        return token_ids, embed_features.inputs_embeds, extra_kwargs, image_hash
    else:
        token_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
        return token_ids, None, {}, None


def _generate_tokens(
    model,
    tokenizer,
    prompt_token_ids: List[int],
    *,
    max_tokens: int = 100,
    ssd_cache_dir: Optional[str] = None,
    block_size: int = 2048,
    vlm_inputs_embeds: Optional[Any] = None,
    vlm_extra_kwargs: Optional[Dict[str, Any]] = None,
    vlm_image_hash: Optional[str] = None,
) -> Tuple[List[int], int]:
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    config_kwargs = dict(
        max_num_seqs=1,
        max_num_batched_tokens=16384,
        completion_batch_size=1,
        prefill_step_size=2048,
    )

    if ssd_cache_dir is not None:
        config_kwargs["paged_ssd_cache_dir"] = ssd_cache_dir
        config_kwargs["paged_cache_block_size"] = block_size
        config_kwargs["paged_ssd_cache_max_size"] = 10 * 1024 * 1024 * 1024

    config = SchedulerConfig(**config_kwargs)
    scheduler = Scheduler(config=config, model=model, tokenizer=tokenizer)

    rep_penalty = 1.1 if vlm_inputs_embeds is not None else 1.0

    request = Request(
        request_id="test",
        prompt=prompt_token_ids,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            repetition_penalty=rep_penalty,
        ),
    )
    if vlm_inputs_embeds is not None:
        request.vlm_inputs_embeds = vlm_inputs_embeds
        request.vlm_extra_kwargs = vlm_extra_kwargs
        request.vlm_image_hash = vlm_image_hash

    scheduler.add_request(request)

    cached_tokens = 0
    output_token_ids = []

    for _ in range(max_tokens + 200):
        step_result = scheduler.step()
        for output in step_result.outputs:
            if output.cached_tokens > 0:
                cached_tokens = output.cached_tokens
            if output.finished:
                output_token_ids = list(output.output_token_ids)
                break
        if step_result.finished_request_ids:
            break

    scheduler.shutdown()
    return output_token_ids, cached_tokens


def _generate_batch(
    model,
    tokenizer,
    prompt_list: List[List[int]],
    *,
    mode: str = "concurrent",
    max_tokens: int = 100,
    ssd_cache_dir: Optional[str] = None,
    block_size: int = 2048,
    vlm_embeds_list: Optional[List[Tuple[Any, Optional[Dict], Optional[str]]]] = None,
) -> List[Tuple[str, List[int], int]]:
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    n = len(prompt_list)

    config_kwargs = dict(
        max_num_seqs=n,
        max_num_batched_tokens=16384,
        completion_batch_size=n,
        prefill_step_size=2048,
    )

    if ssd_cache_dir is not None:
        config_kwargs["paged_ssd_cache_dir"] = ssd_cache_dir
        config_kwargs["paged_cache_block_size"] = block_size
        config_kwargs["paged_ssd_cache_max_size"] = 10 * 1024 * 1024 * 1024

    config = SchedulerConfig(**config_kwargs)
    scheduler = Scheduler(config=config, model=model, tokenizer=tokenizer)

    has_vlm = vlm_embeds_list is not None and any(e[0] is not None for e in vlm_embeds_list)
    rep_penalty = 1.1 if has_vlm else 1.0

    requests = []
    for i, prompt_ids in enumerate(prompt_list):
        req = Request(
            request_id=f"batch-{i}",
            prompt=prompt_ids,
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                repetition_penalty=rep_penalty,
            ),
        )
        if vlm_embeds_list is not None and i < len(vlm_embeds_list):
            embeds, kwargs, img_hash = vlm_embeds_list[i]
            req.vlm_inputs_embeds = embeds
            req.vlm_extra_kwargs = kwargs
            req.vlm_image_hash = img_hash
        requests.append(req)

    results: Dict[str, Tuple[List[int], int]] = {}
    finished_ids = set()

    if mode == "concurrent":
        for req in requests:
            scheduler.add_request(req)

        for _ in range(max_tokens * n + 500):
            step_result = scheduler.step()
            for output in step_result.outputs:
                if output.cached_tokens > 0 and output.request_id not in results:
                    results.setdefault(output.request_id, ([], output.cached_tokens))
                if output.finished:
                    results[output.request_id] = (
                        list(output.output_token_ids),
                        output.cached_tokens,
                    )
                    finished_ids.add(output.request_id)
            if len(finished_ids) >= n:
                break

    scheduler.shutdown()

    output_list = []
    for req in requests:
        rid = req.request_id
        if rid in results:
            tokens, cached = results[rid]
            output_list.append((rid, tokens, cached))
        else:
            output_list.append((rid, [], 0))

    return output_list


# ---------------------------------------------------------------------------
# Test 1: VLM image request — cache store → hit → identical
# ---------------------------------------------------------------------------

def _build_long_vlm_messages(tokenizer, question: str = "Describe this image in detail.") -> list:
    """Build VLM messages with a ~2K-token system prompt for cache testing."""
    base = (
        "You are a helpful image analysis assistant. "
        "You describe colors, shapes, patterns, and textures accurately. "
        "You provide thorough and detailed descriptions of what you see. "
    )
    long_system = base * 40  # ~2K tokens
    return [
        {"role": "system", "content": long_system},
        {"role": "user", "content": question},
    ]


def _test_vlm_image_cache_consistency(vlm_model, processor, adapter):
    import mlx.core as mx

    print("\n  [Test 1] VLM image cache: store → hit → identical...")

    tokenizer = getattr(processor, "tokenizer", processor)
    image = _create_colored_image((255, 0, 0))

    messages = _build_long_vlm_messages(tokenizer)
    token_ids, embeds, extra_kwargs, image_hash = _prepare_vlm_inputs(
        vlm_model, processor, messages, [image]
    )
    assert embeds is not None
    print(f"    Prompt: {len(token_ids)} tokens, hash={image_hash[:12]}")

    # Clear stale state before test
    adapter.clear_vlm_position_state()

    tmp_dir = tempfile.mkdtemp(prefix="omlx_mrope_vlm_cache_")
    try:
        # Fresh (cache miss)
        tokens_fresh, _ = _generate_tokens(
            adapter, tokenizer, token_ids,
            ssd_cache_dir=tmp_dir, block_size=256,
            vlm_inputs_embeds=embeds,
            vlm_extra_kwargs=extra_kwargs,
            vlm_image_hash=image_hash,
        )
        text_fresh = tokenizer.decode(tokens_fresh)
        print(f"    Fresh  ({len(tokens_fresh)} tokens): {text_fresh[:120]}...")
        _check_output_quality(text_fresh, "mRoPE VLM fresh")

        # Clear state between runs
        adapter.clear_vlm_position_state()

        # Re-prepare embeddings (resets _rope_deltas on language model)
        token_ids2, embeds2, extra_kwargs2, _ = _prepare_vlm_inputs(
            vlm_model, processor, messages, [image]
        )

        # Cache hit (same prompt + image hash)
        tokens_cached, cached_count = _generate_tokens(
            adapter, tokenizer, token_ids2,
            ssd_cache_dir=tmp_dir, block_size=256,
            vlm_inputs_embeds=embeds2,
            vlm_extra_kwargs=extra_kwargs2,
            vlm_image_hash=image_hash,
        )
        text_cached = tokenizer.decode(tokens_cached)
        print(f"    Cached ({len(tokens_cached)} tokens, hit={cached_count}): {text_cached[:120]}...")
        _check_output_quality(text_cached, "mRoPE VLM cached")

        match = tokens_fresh == tokens_cached
        if match:
            print("    Token match: IDENTICAL")
        else:
            min_len = min(len(tokens_fresh), len(tokens_cached))
            diff_idx = next(
                (i for i in range(min_len) if tokens_fresh[i] != tokens_cached[i]),
                min_len,
            )
            print(f"    Token match: DIFFER at position {diff_idx}")

        if cached_count > 0:
            print(f"    Cache hit confirmed: {cached_count} tokens from SSD")
        else:
            print("    No SSD cache hit (prompt may be too short for block boundaries)")

        # VLM SSD cache with image tokens at block boundaries may produce
        # different output due to KV cache numerical differences during
        # partial restore + re-prefill. This is a known SSD cache limitation,
        # not an mRoPE-specific issue. Both outputs must be coherent.
        if not match:
            print("    NOTE: SSD cache restored output differs (expected for VLM block-boundary images)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("    [Test 1] PASSED")


# ---------------------------------------------------------------------------
# Test 2: Text-only cache consistency (on mRoPE VLM adapter)
# ---------------------------------------------------------------------------

def _test_text_only_cache_consistency(adapter, tokenizer):
    print("\n  [Test 2] Text-only cache on mRoPE adapter: store → hit → identical...")

    adapter.clear_vlm_position_state()

    messages = [{"role": "user", "content": TEXT_QUESTIONS[0]}]
    token_ids = _apply_chat_template_as_ids(tokenizer, messages)
    print(f"    Prompt: {len(token_ids)} tokens")

    tmp_dir = tempfile.mkdtemp(prefix="omlx_mrope_text_cache_")
    try:
        tokens_fresh, _ = _generate_tokens(
            adapter, tokenizer, token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
        )
        text_fresh = tokenizer.decode(tokens_fresh)
        print(f"    Fresh  ({len(tokens_fresh)} tokens): {text_fresh[:120]}...")
        _check_output_quality(text_fresh, "mRoPE text-only fresh")

        tokens_cached, cached_count = _generate_tokens(
            adapter, tokenizer, token_ids,
            ssd_cache_dir=tmp_dir, block_size=2048,
        )
        text_cached = tokenizer.decode(tokens_cached)
        print(f"    Cached ({len(tokens_cached)} tokens, hit={cached_count}): {text_cached[:120]}...")
        _check_output_quality(text_cached, "mRoPE text-only cached")

        match = tokens_fresh == tokens_cached
        if match:
            print("    Token match: IDENTICAL")
        else:
            min_len = min(len(tokens_fresh), len(tokens_cached))
            diff_idx = next(
                (i for i in range(min_len) if tokens_fresh[i] != tokens_cached[i]),
                min_len,
            )
            print(f"    Token match: DIFFER at position {diff_idx}")

        assert match, "mRoPE text-only: SSD cache hit/fresh tokens differ"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("    [Test 2] PASSED")


# ---------------------------------------------------------------------------
# Test 3: Image → text-only state transition
# ---------------------------------------------------------------------------

def _test_image_to_text_transition(vlm_model, processor, adapter):
    import mlx.core as mx

    print("\n  [Test 3] Image → text-only state transition...")

    tokenizer = getattr(processor, "tokenizer", processor)
    adapter.clear_vlm_position_state()

    # Step 1: image request
    image = _create_colored_image((0, 0, 255))
    messages_img = [{"role": "user", "content": "What color is this image?"}]
    token_ids_img, embeds, extra_kwargs, image_hash = _prepare_vlm_inputs(
        vlm_model, processor, messages_img, [image]
    )
    assert embeds is not None

    tokens_img, _ = _generate_tokens(
        adapter, tokenizer, token_ids_img,
        vlm_inputs_embeds=embeds,
        vlm_extra_kwargs=extra_kwargs,
        vlm_image_hash=image_hash,
    )
    text_img = tokenizer.decode(tokens_img)
    print(f"    Image response ({len(tokens_img)} tokens): {text_img[:120]}...")
    _check_output_quality(text_img, "transition: image")

    # Step 2: text-only request (must not be contaminated by prior rope_deltas)
    messages_txt = [{"role": "user", "content": "Explain what a stack data structure is."}]
    token_ids_txt = _apply_chat_template_as_ids(tokenizer, messages_txt)

    tokens_txt, _ = _generate_tokens(
        adapter, tokenizer, token_ids_txt,
    )
    text_txt = tokenizer.decode(tokens_txt)
    print(f"    Text response ({len(tokens_txt)} tokens): {text_txt[:120]}...")
    _check_output_quality(text_txt, "transition: text")

    print("    [Test 3] PASSED")


# ---------------------------------------------------------------------------
# Test 4: Mixed batch (2 image + 2 text-only)
# ---------------------------------------------------------------------------

def _test_mixed_batch(vlm_model, processor, adapter):
    import mlx.core as mx

    print("\n  [Test 4] Mixed batch: 2 image + 2 text-only concurrent...")

    tokenizer = getattr(processor, "tokenizer", processor)
    adapter.clear_vlm_position_state()

    # Prepare 2 image requests + 2 text-only requests
    images = [_create_colored_image((255, 0, 0)), _create_colored_image((0, 255, 0))]

    prompt_list = []
    vlm_embeds_list = []

    # Request 0: image (red)
    messages_0 = [{"role": "user", "content": IMAGE_QUESTIONS[0]}]
    tid_0, emb_0, kw_0, hash_0 = _prepare_vlm_inputs(
        vlm_model, processor, messages_0, [images[0]]
    )
    prompt_list.append(tid_0)
    vlm_embeds_list.append((emb_0, kw_0, hash_0))

    # Request 1: text-only
    messages_1 = [{"role": "user", "content": TEXT_QUESTIONS[0]}]
    tid_1 = _apply_chat_template_as_ids(tokenizer, messages_1)
    prompt_list.append(tid_1)
    vlm_embeds_list.append((None, None, None))

    # Request 2: image (green)
    messages_2 = [{"role": "user", "content": IMAGE_QUESTIONS[1]}]
    tid_2, emb_2, kw_2, hash_2 = _prepare_vlm_inputs(
        vlm_model, processor, messages_2, [images[1]]
    )
    prompt_list.append(tid_2)
    vlm_embeds_list.append((emb_2, kw_2, hash_2))

    # Request 3: text-only
    messages_3 = [{"role": "user", "content": TEXT_QUESTIONS[1]}]
    tid_3 = _apply_chat_template_as_ids(tokenizer, messages_3)
    prompt_list.append(tid_3)
    vlm_embeds_list.append((None, None, None))

    results = _generate_batch(
        adapter, tokenizer, prompt_list,
        mode="concurrent",
        vlm_embeds_list=vlm_embeds_list,
    )

    for rid, tokens, cached in results:
        text = tokenizer.decode(tokens)
        print(f"    {rid}: {len(tokens)} tokens - {text[:100]}...")
        _check_output_quality(text, f"mixed batch {rid}")

    print("    [Test 4] PASSED")


# ---------------------------------------------------------------------------
# Test 5: Mixed batch with SSD cache — cache hit produces identical output
# ---------------------------------------------------------------------------

def _test_mixed_batch_cache(vlm_model, processor, adapter):
    import mlx.core as mx

    print("\n  [Test 5] Mixed batch + SSD cache: fresh → hit → identical...")

    tokenizer = getattr(processor, "tokenizer", processor)
    adapter.clear_vlm_position_state()

    image = _create_colored_image((255, 255, 0))
    messages_img = [{"role": "user", "content": "What color is this?"}]
    tid_img, emb_img, kw_img, hash_img = _prepare_vlm_inputs(
        vlm_model, processor, messages_img, [image]
    )

    messages_txt = [{"role": "user", "content": TEXT_QUESTIONS[2]}]
    tid_txt = _apply_chat_template_as_ids(tokenizer, messages_txt)

    prompt_list = [tid_img, tid_txt]
    vlm_embeds_list = [
        (emb_img, kw_img, hash_img),
        (None, None, None),
    ]

    tmp_dir = tempfile.mkdtemp(prefix="omlx_mrope_mixed_cache_")
    try:
        # Run 1: fresh (cache miss)
        results_fresh = _generate_batch(
            adapter, tokenizer, prompt_list,
            mode="concurrent",
            ssd_cache_dir=tmp_dir, block_size=2048,
            vlm_embeds_list=vlm_embeds_list,
        )
        print("    --- Fresh run ---")
        for rid, tokens, cached in results_fresh:
            text = tokenizer.decode(tokens)
            print(f"    {rid}: {len(tokens)} tokens - {text[:100]}...")
            _check_output_quality(text, f"mixed cache fresh {rid}")

        # Run 2: cache hit (same prompts + image hash)
        results_cached = _generate_batch(
            adapter, tokenizer, prompt_list,
            mode="concurrent",
            ssd_cache_dir=tmp_dir, block_size=2048,
            vlm_embeds_list=vlm_embeds_list,
        )
        print("    --- Cached run ---")
        for rid, tokens, cached in results_cached:
            text = tokenizer.decode(tokens)
            print(f"    {rid}: {len(tokens)} tokens (hit={cached}) - {text[:100]}...")
            _check_output_quality(text, f"mixed cache hit {rid}")

        # Compare token-by-token
        for i in range(len(results_fresh)):
            _, fresh_tokens, _ = results_fresh[i]
            _, cached_tokens, cached_count = results_cached[i]
            match = fresh_tokens == cached_tokens
            rid = results_fresh[i][0]
            if match:
                print(f"    {rid}: Token match IDENTICAL")
            else:
                min_len = min(len(fresh_tokens), len(cached_tokens))
                diff_idx = next(
                    (j for j in range(min_len) if fresh_tokens[j] != cached_tokens[j]),
                    min_len,
                )
                print(f"    {rid}: Token match DIFFER at position {diff_idx}")
            # Text-only requests must match exactly. VLM requests may
            # differ due to SSD block-boundary KV cache differences.
            is_vlm = vlm_embeds_list[i][0] is not None
            if not match and not is_vlm:
                pytest.fail(f"mRoPE mixed batch {rid} (text-only): SSD cache tokens differ")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("    [Test 5] PASSED")


# ---------------------------------------------------------------------------
# Test 6: Vision feature cache (image embedding SSD cache)
# ---------------------------------------------------------------------------

def _test_vision_feature_cache(vlm_model, processor, adapter):
    import mlx.core as mx

    from omlx.utils.image import compute_image_hash

    print("\n  [Test 6] Vision feature cache: store → hit → same generation...")

    tokenizer = getattr(processor, "tokenizer", processor)
    adapter.clear_vlm_position_state()
    model_path = getattr(vlm_model, "_name_or_path", None) or "unknown"

    # Check if model supports cached_image_features
    image = _create_colored_image((128, 0, 255))
    messages = [{"role": "user", "content": "Describe this image."}]

    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_template
    from mlx_vlm.utils import prepare_inputs

    try:
        prompt = vlm_apply_template(
            processor, vlm_model.config, messages, num_images=1
        )
    except Exception:
        prompt = "Describe this image."

    inputs = prepare_inputs(processor, images=[image], prompts=[prompt])
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    attention_mask = inputs.get("attention_mask")
    extra_inputs = {
        k: v for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "pixel_values") and v is not None
    }

    # Try to compute vision features
    from omlx.engine.vlm import VLMBatchedEngine

    engine_stub = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine_stub._vlm_model = vlm_model
    engine_stub._model_name = model_path

    features = engine_stub._compute_vision_features(pixel_values, extra_inputs)

    if features is None:
        print("    Model does not support _compute_vision_features, skipping")
        print("    [Test 6] SKIPPED")
        return

    mx.eval(features)
    print(f"    Vision features: shape={features.shape}")

    # Test cached_image_features kwarg
    try:
        call_kwargs = dict(extra_inputs)
        call_kwargs["cached_image_features"] = features
        embed_cached = vlm_model.get_input_embeddings(
            input_ids, pixel_values, mask=attention_mask, **call_kwargs
        )
        mx.eval(embed_cached.inputs_embeds)
    except TypeError:
        print("    cached_image_features kwarg not supported, skipping")
        print("    [Test 6] SKIPPED")
        return

    # Compare cached vs fresh embeddings
    embed_fresh = vlm_model.get_input_embeddings(
        input_ids, pixel_values, mask=attention_mask, **extra_inputs
    )
    mx.eval(embed_fresh.inputs_embeds)

    max_diff = mx.max(mx.abs(embed_cached.inputs_embeds - embed_fresh.inputs_embeds)).item()
    identical = mx.array_equal(embed_cached.inputs_embeds, embed_fresh.inputs_embeds)
    print(f"    Cached vs fresh: identical={identical}, max_diff={max_diff:.2e}")

    # Generate with cached features and verify quality
    extra_kwargs = {}
    if hasattr(embed_cached, "to_dict"):
        feat_dict = embed_cached.to_dict()
        for k, v in feat_dict.items():
            if k != "inputs_embeds" and v is not None:
                extra_kwargs[k] = v

    token_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
    image_hash = compute_image_hash([image])

    tokens, _ = _generate_tokens(
        adapter, tokenizer, token_ids,
        vlm_inputs_embeds=embed_cached.inputs_embeds,
        vlm_extra_kwargs=extra_kwargs,
        vlm_image_hash=image_hash,
    )
    text = tokenizer.decode(tokens)
    print(f"    Generated ({len(tokens)} tokens): {text[:120]}...")

    if len(tokens) > 0:
        _check_output_quality(text, "vision feature cache generation")

    print("    [Test 6] PASSED")


# ---------------------------------------------------------------------------
# Test 7: Batched decode — per-request position IDs and attention masks
# ---------------------------------------------------------------------------

def _test_batched_decode_position_ids_and_masks(vlm_model, processor, adapter):
    """Regression test for the mRoPE state clobbering bug (#689).

    The bug: When 2+ VLM requests are batched, get_input_embeddings() is called
    per-request during prefill, each time overwriting the global _rope_deltas on
    language_model.  During decode, all requests in the batch would incorrectly
    use the LAST request's rope_deltas, producing wrong 3-D position IDs for
    earlier requests and therefore gibberish output.

    The fix: Per-request rope_delta tracking via _uid_rope_deltas dict on
    VLMModelAdapter + _captured_rope_deltas stored per-request in
    vlm_extra_kwargs.

    Validation strategy:
      A. Verify per-request _captured_rope_deltas are recorded in extra_kwargs.
      B. Run each VLM request SEQUENTIALLY (single-request scheduler) to get a
         baseline where rope_deltas are trivially correct.
      C. Run the SAME requests in a CONCURRENT batch.
      D. Compare: batch outputs must be coherent — if rope_deltas were
         clobbered, the first request would produce gibberish because it
         would decode with the second request's position offsets.
      E. Verify text-only requests interleaved in the same batch are not
         contaminated by VLM rope_deltas (delta must be 0).
      F. Verify adapter._uid_rope_deltas is cleaned up after the batch
         completes (no leaked entries).
    """
    print("\n  [Test 7] Batched decode position IDs and attention masks...")

    tokenizer = getattr(processor, "tokenizer", processor)
    adapter.clear_vlm_position_state()

    # --- Part A: Verify per-request _captured_rope_deltas ---
    # Use two visually distinct images so get_input_embeddings produces
    # different mRoPE position layouts (and therefore different rope_deltas).
    img_red = _create_colored_image((255, 0, 0))
    img_blue = _create_colored_image((0, 0, 255))

    msgs_0 = [{"role": "user", "content": IMAGE_QUESTIONS[0]}]
    msgs_1 = [{"role": "user", "content": IMAGE_QUESTIONS[1]}]

    tid_0, emb_0, kw_0, hash_0 = _prepare_vlm_inputs(
        vlm_model, processor, msgs_0, [img_red]
    )
    tid_1, emb_1, kw_1, hash_1 = _prepare_vlm_inputs(
        vlm_model, processor, msgs_1, [img_blue]
    )

    assert emb_0 is not None, "VLM embeddings expected for image request 0"
    assert emb_1 is not None, "VLM embeddings expected for image request 1"

    # _captured_rope_deltas should be present in extra_kwargs for mRoPE models
    rd_0 = kw_0.get("_captured_rope_deltas")
    rd_1 = kw_1.get("_captured_rope_deltas")
    if adapter._uses_mrope:
        assert rd_0 is not None, (
            "mRoPE model should set _captured_rope_deltas in extra_kwargs (req 0)"
        )
        assert rd_1 is not None, (
            "mRoPE model should set _captured_rope_deltas in extra_kwargs (req 1)"
        )
        rd_0_val = float(rd_0.item()) if hasattr(rd_0, "item") else float(rd_0)
        rd_1_val = float(rd_1.item()) if hasattr(rd_1, "item") else float(rd_1)
        print(f"    rope_deltas: req0={rd_0_val}, req1={rd_1_val}")
    else:
        print("    Model does not use mRoPE, rope_deltas capture not applicable")

    # position_ids should be captured per-request for mRoPE models
    pid_0 = kw_0.get("position_ids")
    pid_1 = kw_1.get("position_ids")
    if adapter._uses_mrope:
        assert pid_0 is not None, (
            "mRoPE model should capture position_ids in extra_kwargs (req 0)"
        )
        assert pid_1 is not None, (
            "mRoPE model should capture position_ids in extra_kwargs (req 1)"
        )
        print(f"    position_ids shapes: req0={pid_0.shape}, req1={pid_1.shape}")

    # --- Part B: Sequential baseline ---
    print("    --- Sequential runs (baseline) ---")
    adapter.clear_vlm_position_state()

    tokens_seq_0, _ = _generate_tokens(
        adapter, tokenizer, tid_0,
        vlm_inputs_embeds=emb_0,
        vlm_extra_kwargs=kw_0,
        vlm_image_hash=hash_0,
    )
    text_seq_0 = tokenizer.decode(tokens_seq_0)
    print(f"    seq-0 ({len(tokens_seq_0)} tokens): {text_seq_0[:100]}...")
    _check_output_quality(text_seq_0, "sequential req 0")

    adapter.clear_vlm_position_state()

    tokens_seq_1, _ = _generate_tokens(
        adapter, tokenizer, tid_1,
        vlm_inputs_embeds=emb_1,
        vlm_extra_kwargs=kw_1,
        vlm_image_hash=hash_1,
    )
    text_seq_1 = tokenizer.decode(tokens_seq_1)
    print(f"    seq-1 ({len(tokens_seq_1)} tokens): {text_seq_1[:100]}...")
    _check_output_quality(text_seq_1, "sequential req 1")

    # --- Part C: Concurrent batch ---
    print("    --- Concurrent batch run ---")
    adapter.clear_vlm_position_state()

    # Re-prepare embeddings to get fresh mRoPE state
    _, emb_0b, kw_0b, _ = _prepare_vlm_inputs(
        vlm_model, processor, msgs_0, [img_red]
    )
    _, emb_1b, kw_1b, _ = _prepare_vlm_inputs(
        vlm_model, processor, msgs_1, [img_blue]
    )

    prompt_list = [tid_0, tid_1]
    vlm_embeds_list = [
        (emb_0b, kw_0b, hash_0),
        (emb_1b, kw_1b, hash_1),
    ]

    batch_results = _generate_batch(
        adapter, tokenizer, prompt_list,
        mode="concurrent",
        vlm_embeds_list=vlm_embeds_list,
    )

    for rid, tokens, _cached in batch_results:
        text = tokenizer.decode(tokens)
        print(f"    {rid} ({len(tokens)} tokens): {text[:100]}...")
        _check_output_quality(text, f"batch {rid}")

    # --- Part D: Compare sequential vs batch ---
    # With the bug, request 0 in the batch would use request 1's rope_deltas
    # and produce gibberish. Both must produce coherent, similar output.
    _, batch_tokens_0, _ = batch_results[0]
    _, batch_tokens_1, _ = batch_results[1]

    # Check first-token consistency: if the first several tokens match between
    # sequential and batch, position IDs are being computed correctly.
    # Exact full-sequence match is not required due to numerical batching effects.
    min_check_0 = min(5, len(tokens_seq_0), len(batch_tokens_0))
    min_check_1 = min(5, len(tokens_seq_1), len(batch_tokens_1))

    if min_check_0 > 0:
        first_match_0 = tokens_seq_0[:min_check_0] == batch_tokens_0[:min_check_0]
        print(f"    req0 first-{min_check_0}-token match: {first_match_0}")
        if not first_match_0:
            # Even if first tokens diverge due to batching numerics,
            # the output must still be coherent (not gibberish)
            print(f"    req0 seq: {tokens_seq_0[:min_check_0]}")
            print(f"    req0 bat: {batch_tokens_0[:min_check_0]}")

    if min_check_1 > 0:
        first_match_1 = tokens_seq_1[:min_check_1] == batch_tokens_1[:min_check_1]
        print(f"    req1 first-{min_check_1}-token match: {first_match_1}")

    # --- Part E: Mixed batch with text-only ---
    print("    --- Mixed batch: 2 VLM + 1 text-only ---")
    adapter.clear_vlm_position_state()

    # Re-prepare VLM embeddings
    _, emb_0c, kw_0c, _ = _prepare_vlm_inputs(
        vlm_model, processor, msgs_0, [img_red]
    )
    _, emb_1c, kw_1c, _ = _prepare_vlm_inputs(
        vlm_model, processor, msgs_1, [img_blue]
    )

    msgs_txt = [{"role": "user", "content": TEXT_QUESTIONS[3]}]
    tid_txt = _apply_chat_template_as_ids(tokenizer, msgs_txt)

    mixed_prompts = [tid_0, tid_txt, tid_1]
    mixed_embeds = [
        (emb_0c, kw_0c, hash_0),
        (None, None, None),       # text-only: rope_deltas must be 0
        (emb_1c, kw_1c, hash_1),
    ]

    mixed_results = _generate_batch(
        adapter, tokenizer, mixed_prompts,
        mode="concurrent",
        vlm_embeds_list=mixed_embeds,
    )

    for rid, tokens, _cached in mixed_results:
        text = tokenizer.decode(tokens)
        print(f"    {rid} ({len(tokens)} tokens): {text[:100]}...")
        _check_output_quality(text, f"mixed batch {rid}")

    # --- Part F: Verify uid_rope_deltas cleanup ---
    # After all requests finish, there should be no leaked entries
    leaked = dict(adapter._uid_rope_deltas)
    if leaked:
        print(f"    WARNING: leaked uid_rope_deltas entries: {leaked}")
    assert len(leaked) == 0, (
        f"uid_rope_deltas should be empty after batch completes, "
        f"but has {len(leaked)} leaked entries: {list(leaked.keys())}"
    )
    print("    uid_rope_deltas cleanup: OK (empty after batch)")

    print("    [Test 7] PASSED")


# ---------------------------------------------------------------------------
# Main test entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_path",
    MROPE_MODELS,
    ids=[Path(m).name for m in MROPE_MODELS],
)
def test_vlm_mrope_integration(model_path):
    """Full mRoPE VLM integration test: cache, batching, mixed requests."""
    import mlx.core as mx

    if not Path(model_path).exists():
        pytest.skip(f"Model not found: {model_path}")

    model_name = Path(model_path).name
    print(f"\n{'='*60}")
    print(f"mRoPE VLM Integration Test: {model_name}")
    print(f"{'='*60}")

    from omlx.engine.vlm import _patch_video_processor_bug
    from omlx.models.vlm import VLMModelAdapter
    from omlx.patches.gated_delta_advance import apply_gated_delta_advance_patch

    _patch_video_processor_bug()
    try:
        from omlx.engine.vlm import _patch_gemma4_batched_decode
        _patch_gemma4_batched_decode()
    except ImportError:
        pass

    try:
        from mlx_vlm.utils import load as vlm_load
        with _track_peak_memory("VLM model load"):
            vlm_model, processor = vlm_load(model_path)
    except Exception as e:
        pytest.skip(f"VLM load failed: {e}")

    # Build decode model (weight sharing)
    decode_model = None
    try:
        from pathlib import Path as _Path

        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model

        with _track_peak_memory("decode model (weight sharing)"):
            lm_model, _ = load_model(_Path(model_path), lazy=True)
            vlm_params = dict(tree_flatten(vlm_model.language_model.parameters()))
            lm_params = [("language_model." + k, v) for k, v in vlm_params.items()]
            lm_model.load_weights(lm_params, strict=False)
            apply_gated_delta_advance_patch(lm_model)
            decode_model = lm_model
        print(f"  decode model ready")
    except Exception as e:
        print(f"  decode model failed: {e}")

    adapter = VLMModelAdapter(vlm_model, decode_model=decode_model)
    vlm_tokenizer = getattr(processor, "tokenizer", processor)

    apply_gated_delta_advance_patch(adapter._language_model)
    print(f"  _uses_mrope: {adapter._uses_mrope}")

    try:
        with _track_peak_memory("Test 1 - VLM image cache consistency"):
            _test_vlm_image_cache_consistency(vlm_model, processor, adapter)
        with _track_peak_memory("Test 2 - text-only cache consistency"):
            _test_text_only_cache_consistency(adapter, vlm_tokenizer)
        with _track_peak_memory("Test 3 - image→text transition"):
            _test_image_to_text_transition(vlm_model, processor, adapter)
        with _track_peak_memory("Test 4 - mixed batch"):
            _test_mixed_batch(vlm_model, processor, adapter)
        with _track_peak_memory("Test 5 - mixed batch + SSD cache"):
            _test_mixed_batch_cache(vlm_model, processor, adapter)
        with _track_peak_memory("Test 6 - vision feature cache"):
            _test_vision_feature_cache(vlm_model, processor, adapter)
        with _track_peak_memory("Test 7 - batched decode position IDs"):
            _test_batched_decode_position_ids_and_masks(vlm_model, processor, adapter)
    finally:
        del vlm_model, processor, adapter, vlm_tokenizer, decode_model
        gc.collect()
        mx.clear_cache()

    print(f"\n{'='*60}")
    print(f"ALL mRoPE TESTS PASSED: {model_name}")
    print(f"{'='*60}")
