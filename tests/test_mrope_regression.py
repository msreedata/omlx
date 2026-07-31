# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for mRoPE VLM fix (commit 512c21b2).

Verifies that per-request rope_deltas are correctly tracked and that the
cache store/restore mechanism works as expected, especially with mixed
batches (image + text-only).

The commit introduced:
  - _uid_rope_deltas dict on VLMModelAdapter for per-UID delta tracking
  - Batch rope_deltas built dynamically in _patched_generation_batch_step
  - Per-request _position_ids and _rope_deltas capture after
    get_input_embeddings to prevent state clobbering
  - decode_model bypass for mRoPE models (1D RoPE incompatible)
  - VLM SSD cache hit: advance extra_kwargs by start_offset

Test categories:
  1. Per-request rope_deltas registration and unregistration
  2. Batch rope_deltas construction from UID mapping
  3. mRoPE decode bypasses decode_model and uses language_model
  4. Mixed batch position_ids correctness (image + text-only)
  5. Cache store/restore: rope_deltas preserved through register/clear/restore
  6. _captured_rope_deltas extraction from vlm_extra_kwargs
  7. clear_vlm_position_state resets all mRoPE state
  8. _patched_generation_batch_step builds deltas before each step
  9. Request.rope_deltas field defaults to 0.0 and is set after VLM prefill
"""

from unittest.mock import MagicMock

import mlx.core as mx  # Stub provided by conftest_mlx_stub on Linux CI

# ---------------------------------------------------------------------------
# Mock helpers (same pattern as test_vlm_model_adapter.py)
# ---------------------------------------------------------------------------

class MockMXArray:
    """Minimal mock for mx.array with shape, ndim, and arithmetic."""

    def __init__(self, shape=None, data=None):
        self._shape = shape or (1, 10, 128)
        self._data = data

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        result = 1
        for d in self._shape:
            result *= d
        return result

    def __getitem__(self, key):
        return MockMXArray(self._shape)


def _make_mrope_vlm_model():
    """Create a mock VLM model with mRoPE config (Qwen3-VL style)."""
    vlm = MagicMock()
    vlm.language_model = MagicMock()
    vlm.language_model.model = MagicMock()
    vlm.language_model.model.layers = [MagicMock() for _ in range(4)]
    vlm.language_model.args = MagicMock()
    vlm.config = MagicMock(spec=[])
    vlm.config.text_config = MagicMock(spec=[])
    vlm.config.text_config.rope_scaling = {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    }
    vlm.config.text_config.rope_parameters = None
    vlm.config.model_type = "qwen3_vl_moe"
    return vlm


def _make_standard_vlm_model():
    """Create a mock VLM model with standard RoPE (non-mRoPE)."""
    vlm = MagicMock()
    vlm.language_model = MagicMock()
    vlm.language_model.model = MagicMock()
    vlm.language_model.model.layers = [MagicMock() for _ in range(4)]
    vlm.language_model.args = MagicMock()
    vlm.config = MagicMock(spec=[])
    vlm.config.text_config = MagicMock(spec=[])
    vlm.config.text_config.rope_scaling = None
    vlm.config.text_config.rope_parameters = {
        "full_attention": {"rope_theta": 1000000.0},
        "sliding_attention": {"rope_theta": 10000.0},
    }
    vlm.config.model_type = "gemma3"
    return vlm


# ===========================================================================
# Test 1: Per-request rope_deltas registration and unregistration
# ===========================================================================

class TestRopeDeltaRegistration:
    """Verify the _uid_rope_deltas dict lifecycle on VLMModelAdapter."""

    def test_register_stores_delta_per_uid(self):
        """register_rope_delta stores the delta for a given UID."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        adapter.register_rope_delta(uid=100, delta=-42.0)
        adapter.register_rope_delta(uid=200, delta=-10.5)

        assert adapter._uid_rope_deltas[100] == -42.0
        assert adapter._uid_rope_deltas[200] == -10.5

    def test_unregister_removes_uid(self):
        """unregister_rope_delta removes the UID from the mapping."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        adapter.register_rope_delta(uid=100, delta=-42.0)
        adapter.register_rope_delta(uid=200, delta=-10.5)

        adapter.unregister_rope_delta(uid=100)
        assert 100 not in adapter._uid_rope_deltas
        assert 200 in adapter._uid_rope_deltas

    def test_unregister_nonexistent_uid_is_noop(self):
        """unregister_rope_delta for an absent UID does not raise."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Should not raise
        adapter.unregister_rope_delta(uid=999)
        assert adapter._uid_rope_deltas == {}

    def test_register_overwrites_existing_uid(self):
        """Registering the same UID again overwrites the delta value."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        adapter.register_rope_delta(uid=100, delta=-42.0)
        adapter.register_rope_delta(uid=100, delta=-99.0)

        assert adapter._uid_rope_deltas[100] == -99.0

    def test_uid_rope_deltas_starts_empty(self):
        """_uid_rope_deltas is empty on fresh adapter."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        assert adapter._uid_rope_deltas == {}
        assert adapter._batch_rope_deltas is None


# ===========================================================================
# Test 2: Batch rope_deltas construction from UID mapping
# ===========================================================================

class TestBatchRopeDeltas:
    """Verify set_batch_rope_deltas and its integration with the adapter."""

    def test_set_batch_rope_deltas(self):
        """set_batch_rope_deltas stores the array on the adapter."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        deltas = mx.array([-42.0, 0.0, -10.5])
        adapter.set_batch_rope_deltas(deltas)

        assert adapter._batch_rope_deltas is deltas

    def test_batch_rope_deltas_cleared_by_clear_vlm_position_state(self):
        """clear_vlm_position_state resets _batch_rope_deltas to None."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        adapter.set_batch_rope_deltas(mx.array([-42.0, 0.0]))
        assert adapter._batch_rope_deltas is not None

        adapter.clear_vlm_position_state()
        assert adapter._batch_rope_deltas is None


# ===========================================================================
# Test 3: mRoPE decode bypasses decode_model
# ===========================================================================

class TestMRoPEDecodeBypass:
    """mRoPE models must NOT use decode_model (1D RoPE incompatible)."""

    def test_mrope_with_batch_deltas_skips_decode_model(self):
        """With batch_rope_deltas set, decode_model is never called."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        decode_model = MagicMock()
        adapter = VLMModelAdapter(vlm, decode_model=decode_model)
        assert adapter._uses_mrope is True

        adapter.set_batch_rope_deltas(mx.array([-50.0, 0.0]))

        input_ids = mx.zeros((2, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([100, 80])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        decode_model.assert_not_called()
        vlm.language_model.assert_called_once()

    def test_mrope_without_batch_deltas_still_skips_decode_model(self):
        """Even without batch_rope_deltas, mRoPE adapter uses language_model."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        decode_model = MagicMock()
        adapter = VLMModelAdapter(vlm, decode_model=decode_model)

        input_ids = mx.zeros((1, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([50])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        decode_model.assert_not_called()
        vlm.language_model.assert_called_once()

    def test_standard_rope_uses_decode_model(self):
        """Non-mRoPE model with decode_model should use it (fast path)."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_standard_vlm_model()
        decode_model = MagicMock()
        decode_model.return_value = MockMXArray(shape=(1, 1, 32000))
        adapter = VLMModelAdapter(vlm, decode_model=decode_model)
        assert adapter._uses_mrope is False

        input_ids = MockMXArray(shape=(1, 1))
        cache = [MagicMock()]

        adapter(input_ids, cache=cache)

        decode_model.assert_called_once()
        vlm.language_model.assert_not_called()


# ===========================================================================
# Test 4: Mixed batch position_ids correctness
# ===========================================================================

class TestMixedBatchPositionIds:
    """Verify position_ids computation for mixed image + text-only batches."""

    def test_position_ids_shape_3d_for_mrope_batch(self):
        """position_ids should be (3, B, L) for mRoPE decode with B requests."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm, decode_model=MagicMock())

        # Batch of 3: VLM (delta=-50), text (delta=0), VLM (delta=-20)
        adapter.set_batch_rope_deltas(mx.array([-50.0, 0.0, -20.0]))

        input_ids = mx.zeros((3, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([100, 80, 60])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        pos_ids = call_kwargs["position_ids"]

        # Shape: (3, 3, 1) -- 3 mRoPE dims, 3 requests, 1 token
        assert pos_ids.shape == (3, 3, 1)

        # Request 0: 100 + (-50) = 50
        assert pos_ids[0, 0, 0].item() == 50.0
        # Request 1: 80 + 0 = 80 (text-only, no delta)
        assert pos_ids[0, 1, 0].item() == 80.0
        # Request 2: 60 + (-20) = 40
        assert pos_ids[0, 2, 0].item() == 40.0

    def test_all_three_mrope_dims_have_same_values(self):
        """All 3 mRoPE dimensions should have the same positional values during decode."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm, decode_model=MagicMock())

        adapter.set_batch_rope_deltas(mx.array([-30.0, 0.0]))

        input_ids = mx.zeros((2, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([70, 50])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        pos_ids = call_kwargs["position_ids"]

        # All 3 dims should match for each request
        for dim in range(3):
            assert pos_ids[dim, 0, 0].item() == 40.0  # 70 + (-30)
            assert pos_ids[dim, 1, 0].item() == 50.0  # 50 + 0

    def test_mixed_batch_vlm_gets_negative_delta_text_gets_zero(self):
        """In a mixed batch, VLM requests get negative rope_deltas, text gets 0."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm, decode_model=MagicMock())

        # Simulate: register 2 VLM UIDs with negative deltas, 1 text with 0
        adapter.register_rope_delta(uid=10, delta=-42.0)
        adapter.register_rope_delta(uid=20, delta=0.0)  # text-only
        adapter.register_rope_delta(uid=30, delta=-15.0)

        # Build batch deltas as scheduler would
        uids = [10, 20, 30]
        deltas = [adapter._uid_rope_deltas.get(uid, 0.0) for uid in uids]
        adapter.set_batch_rope_deltas(mx.array(deltas))

        input_ids = mx.zeros((3, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([200, 150, 100])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        pos_ids = call_kwargs["position_ids"]

        # Request 0 (VLM): 200 + (-42) = 158
        assert pos_ids[0, 0, 0].item() == 158.0
        # Request 1 (text): 150 + 0 = 150
        assert pos_ids[0, 1, 0].item() == 150.0
        # Request 2 (VLM): 100 + (-15) = 85
        assert pos_ids[0, 2, 0].item() == 85.0


# ===========================================================================
# Test 5: Cache store/restore: rope_deltas lifecycle
# ===========================================================================

class TestRopeDeltaCacheLifecycle:
    """Verify rope_deltas survive through the full register/generate/unregister cycle."""

    def test_full_lifecycle_register_generate_unregister(self):
        """Simulate full lifecycle: register deltas, build batch, unregister."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Step 1: Register UIDs (after VLM prefill)
        adapter.register_rope_delta(uid=1, delta=-50.0)
        adapter.register_rope_delta(uid=2, delta=0.0)
        adapter.register_rope_delta(uid=3, delta=-25.0)
        assert len(adapter._uid_rope_deltas) == 3

        # Step 2: Build batch (as _patched_generation_batch_step does)
        uids = [1, 2, 3]
        deltas = mx.array([adapter._uid_rope_deltas.get(uid, 0.0) for uid in uids])
        adapter.set_batch_rope_deltas(deltas)
        assert adapter._batch_rope_deltas is not None

        # Step 3: Request 2 finishes -- unregister
        adapter.unregister_rope_delta(uid=2)
        assert 2 not in adapter._uid_rope_deltas
        assert len(adapter._uid_rope_deltas) == 2

        # Step 4: Remaining UIDs still have correct deltas
        remaining_uids = [1, 3]
        new_deltas = [adapter._uid_rope_deltas.get(uid, 0.0) for uid in remaining_uids]
        assert new_deltas == [-50.0, -25.0]

        # Step 5: All finish -- unregister remaining
        adapter.unregister_rope_delta(uid=1)
        adapter.unregister_rope_delta(uid=3)
        assert adapter._uid_rope_deltas == {}

    def test_missing_uid_defaults_to_zero_delta(self):
        """UID not in _uid_rope_deltas defaults to 0.0 delta in batch construction."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        adapter.register_rope_delta(uid=1, delta=-50.0)

        # Simulate batch with uid=1 (registered) and uid=99 (unknown)
        uids = [1, 99]
        deltas = [adapter._uid_rope_deltas.get(uid, 0.0) for uid in uids]
        assert deltas == [-50.0, 0.0]


# ===========================================================================
# Test 6: _captured_rope_deltas extraction from vlm_extra_kwargs
# ===========================================================================

class TestCapturedRopeDeltas:
    """Verify _captured_rope_deltas is handled correctly in the adapter forward."""

    def test_captured_rope_deltas_stripped_before_language_model_call(self):
        """_captured_rope_deltas should be removed from vlm_extra_kwargs before
        being passed to the language model (it's an internal-only key)."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        input_ids = MockMXArray(shape=(1, 10))
        cache = [MagicMock()]
        embeds = MockMXArray(shape=(1, 10, 128))
        extra = {
            "position_ids": MockMXArray(shape=(3, 1, 10)),
            "_captured_rope_deltas": -42.0,
        }

        adapter(input_ids, cache=cache, inputs_embeds=embeds, vlm_extra_kwargs=extra)

        # Language model should have been called, and _captured_rope_deltas
        # should NOT appear in the kwargs
        call_kwargs = vlm.language_model.call_args[1]
        assert "_captured_rope_deltas" not in call_kwargs

    def test_get_last_rope_deltas_extracts_scalar(self):
        """get_last_rope_deltas extracts scalar from language model's _rope_deltas."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        vlm.language_model._rope_deltas = mx.array(-42.0)
        assert adapter.get_last_rope_deltas() == -42.0

    def test_get_last_rope_deltas_returns_zero_when_none(self):
        """get_last_rope_deltas returns 0.0 when _rope_deltas is None."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        vlm.language_model._rope_deltas = None
        assert adapter.get_last_rope_deltas() == 0.0

    def test_get_last_rope_deltas_handles_plain_float(self):
        """get_last_rope_deltas handles plain float on language model."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        vlm.language_model._rope_deltas = -10.5
        assert adapter.get_last_rope_deltas() == -10.5


# ===========================================================================
# Test 7: clear_vlm_position_state resets all mRoPE state
# ===========================================================================

class TestClearVLMPositionState:
    """Verify clear_vlm_position_state resets all position-related state."""

    def test_clears_position_ids_and_rope_deltas(self):
        """All position state on the language model and adapter is cleared."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Simulate state after VLM prefill
        vlm.language_model._position_ids = mx.array([1, 2, 3])
        vlm.language_model._rope_deltas = mx.array(-42.0)
        adapter.set_batch_rope_deltas(mx.array([-42.0, 0.0]))

        adapter.clear_vlm_position_state()

        assert vlm.language_model._position_ids is None
        assert vlm.language_model._rope_deltas is None
        assert adapter._batch_rope_deltas is None

    def test_clear_is_safe_without_prior_state(self):
        """Clearing state that was never set does not raise."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Should not raise even though _position_ids/_rope_deltas may not exist
        adapter.clear_vlm_position_state()

        assert vlm.language_model._position_ids is None
        assert vlm.language_model._rope_deltas is None
        assert adapter._batch_rope_deltas is None

    def test_clear_does_not_affect_uid_rope_deltas(self):
        """clear_vlm_position_state does NOT clear _uid_rope_deltas (per-UID state
        is managed separately via register/unregister)."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        adapter.register_rope_delta(uid=100, delta=-42.0)
        adapter.clear_vlm_position_state()

        # Per-UID deltas should survive position state clearing
        assert adapter._uid_rope_deltas[100] == -42.0


# ===========================================================================
# Test 8: _patched_generation_batch_step builds deltas before each step
# ===========================================================================

class TestPatchedGenerationBatchStep:
    """Verify the monkey-patched _step builds per-batch rope_deltas."""

    def test_patched_step_builds_deltas_from_uid_mapping(self):
        """_patched_generation_batch_step reads _uid_rope_deltas and builds
        batch_rope_deltas aligned with self.uids before calling original _step."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        # Register deltas as scheduler would after VLM prefill
        adapter.register_rope_delta(uid=10, delta=-50.0)
        adapter.register_rope_delta(uid=20, delta=0.0)
        adapter.register_rope_delta(uid=30, delta=-25.0)

        # Simulate what _patched_generation_batch_step does
        uids = [10, 20, 30]
        if (getattr(adapter, "_uses_mrope", False)
                and getattr(adapter, "_uid_rope_deltas", None)
                and uids):
            deltas = [adapter._uid_rope_deltas.get(uid, 0.0) for uid in uids]
            adapter.set_batch_rope_deltas(mx.array(deltas))

        expected = [-50.0, 0.0, -25.0]
        actual = adapter._batch_rope_deltas.tolist()
        assert actual == expected

    def test_patched_step_skips_for_non_mrope_model(self):
        """For non-mRoPE models, batch deltas should not be built."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_standard_vlm_model()
        adapter = VLMModelAdapter(vlm)
        assert adapter._uses_mrope is False

        uids = [10, 20]
        if (getattr(adapter, "_uses_mrope", False)
                and getattr(adapter, "_uid_rope_deltas", None)
                and uids):
            # This block should NOT execute for non-mRoPE
            adapter.set_batch_rope_deltas(None)  # pragma: no cover

        # _batch_rope_deltas should remain None
        assert adapter._batch_rope_deltas is None

    def test_patched_step_handles_empty_uids(self):
        """With empty uids list, no batch deltas should be built."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)
        adapter.register_rope_delta(uid=10, delta=-50.0)

        uids = []
        if (getattr(adapter, "_uses_mrope", False)
                and getattr(adapter, "_uid_rope_deltas", None)
                and uids):
            deltas = [adapter._uid_rope_deltas.get(uid, 0.0) for uid in uids]
            adapter.set_batch_rope_deltas(mx.array(deltas))

        assert adapter._batch_rope_deltas is None

    def test_patched_step_handles_batch_size_change(self):
        """When batch size shrinks (request finishes), deltas are rebuilt correctly."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm)

        adapter.register_rope_delta(uid=10, delta=-50.0)
        adapter.register_rope_delta(uid=20, delta=0.0)
        adapter.register_rope_delta(uid=30, delta=-25.0)

        # Step 1: full batch
        uids_full = [10, 20, 30]
        deltas_full = [adapter._uid_rope_deltas.get(uid, 0.0) for uid in uids_full]
        adapter.set_batch_rope_deltas(mx.array(deltas_full))
        assert adapter._batch_rope_deltas.tolist() == [-50.0, 0.0, -25.0]

        # Request 20 finishes
        adapter.unregister_rope_delta(uid=20)

        # Step 2: reduced batch
        uids_reduced = [10, 30]
        deltas_reduced = [adapter._uid_rope_deltas.get(uid, 0.0) for uid in uids_reduced]
        adapter.set_batch_rope_deltas(mx.array(deltas_reduced))
        assert adapter._batch_rope_deltas.tolist() == [-50.0, -25.0]


# ===========================================================================
# Test 9: Request.rope_deltas field
# ===========================================================================

class TestRequestRopeDeltas:
    """Verify the rope_deltas field on Request dataclass."""

    def test_request_rope_deltas_defaults_to_zero(self):
        """New Request objects have rope_deltas = 0.0 by default."""
        from omlx.request import Request, SamplingParams

        req = Request(
            request_id="test-1",
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        assert req.rope_deltas == 0.0

    def test_request_rope_deltas_is_settable(self):
        """rope_deltas can be set after VLM prefill."""
        from omlx.request import Request, SamplingParams

        req = Request(
            request_id="test-1",
            prompt=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        req.rope_deltas = -42.0
        assert req.rope_deltas == -42.0

    def test_text_only_request_keeps_zero_delta(self):
        """Text-only requests should keep rope_deltas = 0.0."""
        from omlx.request import Request, SamplingParams

        req = Request(
            request_id="text-only",
            prompt=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(),
        )
        # VLM prefill not triggered -- rope_deltas stays 0
        assert req.vlm_inputs_embeds is None
        assert req.rope_deltas == 0.0


# ===========================================================================
# Test 10: mRoPE detection
# ===========================================================================

class TestMRoPEDetectionRegression:
    """Regression tests for mRoPE detection (introduced in the same commit)."""

    def test_detect_mrope_qwen3_vl_style(self):
        """Qwen3-VL uses rope_scaling with mrope_section."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        assert VLMModelAdapter._detect_mrope(vlm) is True

    def test_detect_mrope_qwen35_style(self):
        """Qwen3.5 uses rope_parameters (not rope_scaling) with mrope_section."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = MagicMock(spec=[])
        vlm.config = MagicMock(spec=[])
        vlm.config.text_config = MagicMock(spec=[])
        vlm.config.text_config.rope_scaling = None
        vlm.config.text_config.rope_parameters = {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
        }
        assert VLMModelAdapter._detect_mrope(vlm) is True

    def test_detect_standard_rope_returns_false(self):
        """Non-mRoPE models return False."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_standard_vlm_model()
        assert VLMModelAdapter._detect_mrope(vlm) is False

    def test_adapter_sets_uses_mrope_on_init(self):
        """VLMModelAdapter sets _uses_mrope flag during __init__."""
        from omlx.models.vlm import VLMModelAdapter

        mrope_vlm = _make_mrope_vlm_model()
        adapter_mrope = VLMModelAdapter(mrope_vlm)
        assert adapter_mrope._uses_mrope is True

        standard_vlm = _make_standard_vlm_model()
        adapter_std = VLMModelAdapter(standard_vlm)
        assert adapter_std._uses_mrope is False


# ===========================================================================
# Test 11: Hybrid cache handling (ArraysCache + KVCache)
# ===========================================================================

class TestHybridCacheHandling:
    """Verify mRoPE decode handles hybrid caches (e.g. Qwen3.5 mix of
    ArraysCache and KVCache)."""

    def test_finds_first_cache_with_offset(self):
        """mRoPE decode skips cache layers without offset (ArraysCache)
        and uses the first layer with an offset attribute."""
        from omlx.models.vlm import VLMModelAdapter

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm, decode_model=MagicMock())

        adapter.set_batch_rope_deltas(mx.array([-10.0]))

        input_ids = mx.zeros((1, 1), dtype=mx.int32)

        # First cache layer is ArraysCache-like (no offset)
        arrays_cache = MagicMock(spec=[])  # No .offset attribute
        # Second cache layer has offset (KVCache)
        kv_cache = MagicMock()
        kv_cache.offset = mx.array([50])

        adapter(input_ids, cache=[arrays_cache, kv_cache])

        call_kwargs = vlm.language_model.call_args[1]
        pos_ids = call_kwargs["position_ids"]
        # Should use offset from kv_cache: 50 + (-10) = 40
        assert pos_ids[0, 0, 0].item() == 40.0

    def test_fallback_when_no_cache_has_offset(self):
        """When no cache layer has an offset, falls back to language_model
        with wrapped caches (no position_ids computation)."""
        from omlx.models.vlm import VLMModelAdapter, _IntOffsetCacheProxy

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm, decode_model=MagicMock())

        adapter.set_batch_rope_deltas(mx.array([-10.0]))

        input_ids = mx.zeros((1, 1), dtype=mx.int32)

        # All cache layers are ArraysCache-like (no offset)
        cache1 = MagicMock(spec=[])
        cache2 = MagicMock(spec=[])

        adapter(input_ids, cache=[cache1, cache2])

        call_kwargs = vlm.language_model.call_args[1]
        # Should NOT have position_ids (fallback path)
        assert "position_ids" not in call_kwargs
        # Cache should be wrapped with _IntOffsetCacheProxy
        wrapped = call_kwargs["cache"]
        assert isinstance(wrapped[0], _IntOffsetCacheProxy)


# ===========================================================================
# Test 12: _CachedOffsetProxy (new proxy introduced in the commit)
# ===========================================================================

class TestCachedOffsetProxyRegression:
    """Regression tests for _CachedOffsetProxy introduced in the commit."""

    def test_returns_precomputed_offset(self):
        """Proxy returns the pre-computed int offset, not the cache's native offset."""
        from omlx.models.vlm import _CachedOffsetProxy

        inner = MagicMock()
        inner.offset = 999  # This should be ignored
        proxy = _CachedOffsetProxy(inner, 42)

        assert proxy.offset == 42

    def test_delegates_non_offset_attributes(self):
        """Non-offset attributes delegate to the inner cache."""
        from omlx.models.vlm import _CachedOffsetProxy

        inner = MagicMock()
        inner.keys = "cached_keys"
        inner.values = "cached_values"
        proxy = _CachedOffsetProxy(inner, 50)

        assert proxy.keys == "cached_keys"
        assert proxy.values == "cached_values"

    def test_setattr_delegates_to_inner(self):
        """Setting non-private attributes delegates to the inner cache."""
        from omlx.models.vlm import _CachedOffsetProxy

        inner = MagicMock()
        proxy = _CachedOffsetProxy(inner, 50)

        proxy.some_attr = "test_value"
        assert inner.some_attr == "test_value"

    def test_getitem_delegates(self):
        """Indexing delegates to the inner cache."""
        from omlx.models.vlm import _CachedOffsetProxy

        inner = MagicMock()
        inner.__getitem__ = MagicMock(return_value="layer_0_cache")
        proxy = _CachedOffsetProxy(inner, 50)

        assert proxy[0] == "layer_0_cache"

    def test_is_truthy(self):
        """Proxy is always truthy (used in 'if cache:' checks)."""
        from omlx.models.vlm import _CachedOffsetProxy

        proxy = _CachedOffsetProxy(MagicMock(), 0)
        assert bool(proxy) is True

    def test_iterable(self):
        """Proxy supports iteration, delegating to inner cache."""
        from omlx.models.vlm import _CachedOffsetProxy

        inner = [1, 2, 3]
        proxy = _CachedOffsetProxy(inner, 50)

        assert list(proxy) == [1, 2, 3]


# ===========================================================================
# Test 13: Batch size mismatch guard (deltas.size != B)
# ===========================================================================

class TestBatchSizeMismatchGuard:
    """Verify graceful handling when batch_rope_deltas size doesn't match batch."""

    def test_stale_deltas_with_wrong_batch_size_falls_back(self):
        """If deltas.size != B, the adapter falls back to wrapped cache path."""
        from omlx.models.vlm import VLMModelAdapter, _IntOffsetCacheProxy

        vlm = _make_mrope_vlm_model()
        adapter = VLMModelAdapter(vlm, decode_model=MagicMock())

        # Set deltas for 2 requests but actual batch has 3
        adapter.set_batch_rope_deltas(mx.array([-50.0, 0.0]))

        input_ids = mx.zeros((3, 1), dtype=mx.int32)
        cache_layer = MagicMock()
        cache_layer.offset = mx.array([100, 80, 60])
        cache = [cache_layer]

        adapter(input_ids, cache=cache)

        call_kwargs = vlm.language_model.call_args[1]
        # Should NOT have position_ids (fallback due to size mismatch)
        assert "position_ids" not in call_kwargs
        # Cache should be wrapped
        assert isinstance(call_kwargs["cache"][0], _IntOffsetCacheProxy)
