"""
IIAS Category Handlers — 120 deterministic PHI-based app handlers.

Each of 12 categories has a factory function returning 10 handlers.
All handlers follow signature: (query: str, ctx: AppContext) -> dict[str, Any]

Mathematical Foundation:
- PHI = 1.618... governs all scaling
- LUCAS = [1,3,4,7,11,18,29,47,76,123,199,322] (840 states)
- BRAHIM_NUMBERS for mirror-pair balancing
- Energy conservation: E(x) = 2*PI for all x

Author: Elias Oulad Brahim
"""

from __future__ import annotations

import math
import re
from typing import Any

# ---------------------------------------------------------------------------
# Constants (from Brahim's Calculator)
# ---------------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2
OMEGA = 1 / PHI
BETA = 1 / PHI**3
GAMMA = 1 / PHI**4
LUCAS = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]
TOTAL_STATES = 840
BRAHIM = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)
MIRROR = 214
TWO_PI = 2 * math.pi
BW_NPU = 7.35   # GB/s measured
BW_GPU = 12.0
BW_RAM = 26.0
BW_SSD = 2.8


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_numeric(query: str, default: float = 100.0) -> float:
    """Extract first number from query string, or return default."""
    match = re.search(r"[\d]+(?:\.[\d]+)?", query)
    return float(match.group()) if match else default


def _phi_scale(value: float, dim: int) -> float:
    """Scale value by PHI^dim / TOTAL_STATES."""
    return value * (PHI ** dim) / TOTAL_STATES


def _lucas_weight(dim: int) -> float:
    """Lucas weight for dimension (1-indexed, clamped to 1..12)."""
    idx = max(0, min(dim - 1, 11))
    return LUCAS[idx] / TOTAL_STATES


def _phi_decay(value: float, step: int) -> float:
    """Exponential PHI decay: value * OMEGA^step."""
    return value * (OMEGA ** step)


def _saturate(n: float, bw_max: float = BW_NPU) -> float:
    """PHI saturation: bw_max * (1 - e^(-n/PHI))."""
    return bw_max * (1 - math.exp(-n / PHI))


def _mirror_pair(b_idx: int) -> tuple[int, int]:
    """Return (B[i], MIRROR - B[i]) pair. b_idx is 0-based."""
    b = BRAHIM[b_idx % len(BRAHIM)]
    return (b, MIRROR - b)


def _lucas_distribute(total: float) -> list[float]:
    """Distribute total across 12 dimensions by Lucas weights."""
    return [round(total * LUCAS[i] / TOTAL_STATES, 4) for i in range(12)]


# ---------------------------------------------------------------------------
# 1. INFRASTRUCTURE (D4, L=7, NPU — stability math)
# ---------------------------------------------------------------------------

def _infrastructure_handlers() -> dict[str, Any]:
    D, L = 4, 7

    def _auto_scaler(query: str, ctx: Any) -> dict[str, Any]:
        load = _parse_numeric(query, 100.0)
        replicas = max(1, math.ceil(load / (PHI**D * L)))
        headroom = (replicas * PHI**D * L - load) / max(load, 1e-9)
        return {
            "load": load,
            "optimal_replicas": replicas,
            "headroom_pct": round(headroom * 100, 2),
            "phi_factor": round(PHI**D, 4),
            "lucas_capacity": L,
            "energy": round(TWO_PI, 6),
        }

    def _load_balancer(query: str, ctx: Any) -> dict[str, Any]:
        total = _parse_numeric(query, 1000.0)
        dist = _lucas_distribute(total)
        return {
            "total_load": total,
            "distribution_12d": dist,
            "max_node": max(dist),
            "balance_ratio": round(min(dist) / max(dist), 4) if max(dist) else 0,
            "conservation": round(sum(dist), 2),
        }

    def _cost_optimizer(query: str, ctx: Any) -> dict[str, Any]:
        budget = _parse_numeric(query, 10000.0)
        npu_share = round(budget * _lucas_weight(D), 2)
        cpu_share = round(budget * _lucas_weight(5), 2)
        gpu_share = round(budget - npu_share - cpu_share, 2)
        return {
            "budget": budget,
            "npu_allocation": npu_share,
            "cpu_allocation": cpu_share,
            "gpu_allocation": gpu_share,
            "phi_efficiency": round(PHI**D / TOTAL_STATES, 6),
            "savings_pct": round((1 - OMEGA**D) * 100, 2),
        }

    def _api_gateway(query: str, ctx: Any) -> dict[str, Any]:
        rps = _parse_numeric(query, 5000.0)
        optimal_routes = max(1, math.ceil(rps / (PHI**D * L * 10)))
        latency_ms = round(1000 / (PHI**D * L), 4)
        return {
            "requests_per_sec": rps,
            "optimal_routes": optimal_routes,
            "estimated_latency_ms": latency_ms,
            "phi_throughput": round(PHI**D * L * 10, 2),
            "stability_index": round(L / TOTAL_STATES, 6),
        }

    def _training_scheduler(query: str, ctx: Any) -> dict[str, Any]:
        epochs = _parse_numeric(query, 100.0)
        batch_size = max(1, round(PHI**D * L))
        steps = math.ceil(epochs * 1000 / batch_size)
        lr_schedule = [round(0.001 * OMEGA**i, 8) for i in range(5)]
        return {
            "epochs": epochs,
            "optimal_batch_size": batch_size,
            "total_steps": steps,
            "lr_schedule": lr_schedule,
            "phi_warmup_steps": round(steps * OMEGA, 0),
        }

    def _cold_start_predictor(query: str, ctx: Any) -> dict[str, Any]:
        idle_sec = _parse_numeric(query, 300.0)
        cold_prob = round(1 - math.exp(-idle_sec / (PHI**D * 100)), 4)
        warmup_ms = round(PHI**D * L * 10, 2)
        return {
            "idle_seconds": idle_sec,
            "cold_start_probability": cold_prob,
            "predicted_warmup_ms": warmup_ms,
            "phi_decay_rate": round(OMEGA**D, 6),
            "lucas_buffer": L,
        }

    def _container_orchestrator(query: str, ctx: Any) -> dict[str, Any]:
        containers = int(_parse_numeric(query, 50.0))
        nodes = max(1, math.ceil(containers / (L * 3)))
        per_node = math.ceil(containers / nodes)
        utilization = round(containers / (nodes * L * 3), 4)
        return {
            "containers": containers,
            "optimal_nodes": nodes,
            "per_node": per_node,
            "utilization": utilization,
            "phi_packing_factor": round(PHI**D, 4),
            "lucas_slots": L * 3,
        }

    def _database_sharding(query: str, ctx: Any) -> dict[str, Any]:
        data_gb = _parse_numeric(query, 100.0)
        shards = max(1, round(data_gb / (PHI**D)))
        shard_size = round(data_gb / shards, 2)
        rebalance_threshold = round(shard_size * PHI, 2)
        return {
            "data_gb": data_gb,
            "optimal_shards": shards,
            "shard_size_gb": shard_size,
            "rebalance_threshold_gb": rebalance_threshold,
            "phi_split_ratio": round(OMEGA, 6),
            "lucas_replicas": L,
        }

    def _cdn_router(query: str, ctx: Any) -> dict[str, Any]:
        req_mb = _parse_numeric(query, 50.0)
        edges = max(1, math.ceil(req_mb / PHI**D))
        latency_ms = round(req_mb / (BW_NPU * edges) * 1000, 4)
        return {
            "request_mb": req_mb,
            "edge_nodes": edges,
            "estimated_latency_ms": latency_ms,
            "bandwidth_gbps": BW_NPU,
            "phi_fanout": round(PHI**D, 4),
        }

    def _queue_manager(query: str, ctx: Any) -> dict[str, Any]:
        depth = int(_parse_numeric(query, 1000.0))
        consumers = max(1, math.ceil(depth / (PHI**D * L)))
        drain_rate = round(PHI**D * L * consumers, 2)
        drain_time = round(depth / drain_rate, 4) if drain_rate else 0
        return {
            "queue_depth": depth,
            "optimal_consumers": consumers,
            "drain_rate_per_sec": drain_rate,
            "drain_time_sec": drain_time,
            "phi_prefetch": round(PHI**D, 4),
            "lucas_batch": L,
        }

    return {
        "auto_scaler": _auto_scaler,
        "load_balancer": _load_balancer,
        "cost_optimizer": _cost_optimizer,
        "api_gateway": _api_gateway,
        "training_scheduler": _training_scheduler,
        "cold_start_predictor": _cold_start_predictor,
        "container_orchestrator": _container_orchestrator,
        "database_sharding": _database_sharding,
        "cdn_router": _cdn_router,
        "queue_manager": _queue_manager,
    }


# ---------------------------------------------------------------------------
# 2. EDGE (D2, L=3, NPU — attention/decay math)
# ---------------------------------------------------------------------------

def _edge_handlers() -> dict[str, Any]:
    D, L = 2, 3

    def _edge_ai_router(query: str, ctx: Any) -> dict[str, Any]:
        model_mb = _parse_numeric(query, 50.0)
        fits_npu = model_mb <= BW_NPU * PHI**D
        latency_ms = round(model_mb / _saturate(L, BW_NPU) * 1000, 4)
        return {
            "model_mb": model_mb,
            "fits_npu": fits_npu,
            "npu_threshold_mb": round(BW_NPU * PHI**D, 2),
            "estimated_latency_ms": latency_ms,
            "phi_attention": round(PHI**D, 4),
        }

    def _hybrid_orchestrator(query: str, ctx: Any) -> dict[str, Any]:
        workload = _parse_numeric(query, 100.0)
        edge_share = round(workload * OMEGA**D, 2)
        cloud_share = round(workload - edge_share, 2)
        return {
            "total_workload": workload,
            "edge_share": edge_share,
            "cloud_share": cloud_share,
            "split_ratio": round(OMEGA**D, 4),
            "lucas_sync_slots": L,
        }

    def _battery_manager(query: str, ctx: Any) -> dict[str, Any]:
        capacity_mah = _parse_numeric(query, 5000.0)
        discharge = [
            round(capacity_mah * _phi_decay(1.0, i), 1) for i in range(10)
        ]
        runtime_h = round(capacity_mah / (PHI**D * 100), 2)
        return {
            "capacity_mah": capacity_mah,
            "discharge_curve": discharge,
            "estimated_runtime_h": runtime_h,
            "phi_efficiency": round(OMEGA**D, 4),
            "optimal_draw_ma": round(PHI**D * 100, 1),
        }

    def _offline_cache(query: str, ctx: Any) -> dict[str, Any]:
        cache_mb = _parse_numeric(query, 256.0)
        tiers = [
            {"tier": i + 1, "size_mb": round(cache_mb * _lucas_weight(i + 1), 2)}
            for i in range(L)
        ]
        hit_rate = round(1 - OMEGA**L, 4)
        return {
            "cache_mb": cache_mb,
            "tiers": tiers,
            "predicted_hit_rate": hit_rate,
            "eviction_threshold": round(cache_mb * OMEGA, 2),
        }

    def _realtime_pipeline(query: str, ctx: Any) -> dict[str, Any]:
        fps = _parse_numeric(query, 30.0)
        frame_budget_ms = round(1000 / fps, 4)
        phi_budget_ms = round(frame_budget_ms * OMEGA, 4)
        stages = L
        per_stage_ms = round(phi_budget_ms / stages, 4)
        return {
            "target_fps": fps,
            "frame_budget_ms": frame_budget_ms,
            "phi_budget_ms": phi_budget_ms,
            "stages": stages,
            "per_stage_ms": per_stage_ms,
        }

    def _privacy_isolator(query: str, ctx: Any) -> dict[str, Any]:
        data_items = int(_parse_numeric(query, 1000.0))
        buckets = max(1, math.ceil(data_items / (PHI**D * L)))
        noise_scale = round(BETA / PHI**D, 6)
        return {
            "data_items": data_items,
            "isolation_buckets": buckets,
            "noise_scale": noise_scale,
            "phi_privacy_budget": round(PHI**D, 4),
            "lucas_partitions": L,
        }

    def _thermal_manager(query: str, ctx: Any) -> dict[str, Any]:
        temp_c = _parse_numeric(query, 45.0)
        throttle_factor = round(max(0.0, 1 - (temp_c - 40) * OMEGA**D / 10), 4)
        target_c = round(40 + 10 * OMEGA, 1)
        cooling_curve = [
            round(temp_c * _phi_decay(1.0, i), 1) for i in range(6)
        ]
        return {
            "current_temp_c": temp_c,
            "throttle_factor": throttle_factor,
            "target_temp_c": target_c,
            "cooling_curve": cooling_curve,
            "phi_thermal_k": round(PHI**D, 4),
        }

    def _wake_controller(query: str, ctx: Any) -> dict[str, Any]:
        idle_sec = _parse_numeric(query, 60.0)
        sleep_prob = round(1 - math.exp(-idle_sec / (PHI**D * 30)), 4)
        wake_cost_mj = round(PHI**D * L * 0.5, 2)
        return {
            "idle_seconds": idle_sec,
            "sleep_probability": sleep_prob,
            "wake_cost_mj": wake_cost_mj,
            "phi_idle_threshold": round(PHI**D * 30, 2),
            "lucas_wake_stages": L,
        }

    def _sensor_fusion(query: str, ctx: Any) -> dict[str, Any]:
        sensors = int(_parse_numeric(query, 8.0))
        weights = [round(_lucas_weight(i + 1), 6) for i in range(min(sensors, 12))]
        confidence = round(1 - OMEGA**sensors, 4)
        return {
            "sensor_count": sensors,
            "fusion_weights": weights,
            "confidence": confidence,
            "phi_decay_per_sensor": round(OMEGA, 6),
        }

    def _local_model_selector(query: str, ctx: Any) -> dict[str, Any]:
        memory_mb = _parse_numeric(query, 512.0)
        max_params_m = round(memory_mb / (PHI**D * 4), 2)  # 4 bytes/param
        quantized_params_m = round(max_params_m * PHI, 2)   # INT8 gain
        return {
            "available_memory_mb": memory_mb,
            "max_params_millions": max_params_m,
            "quantized_params_millions": quantized_params_m,
            "phi_memory_factor": round(PHI**D * 4, 2),
            "lucas_model_tiers": L,
        }

    return {
        "edge_ai_router": _edge_ai_router,
        "hybrid_orchestrator": _hybrid_orchestrator,
        "battery_manager": _battery_manager,
        "offline_cache": _offline_cache,
        "realtime_pipeline": _realtime_pipeline,
        "privacy_isolator": _privacy_isolator,
        "thermal_manager": _thermal_manager,
        "wake_controller": _wake_controller,
        "sensor_fusion": _sensor_fusion,
        "local_model_selector": _local_model_selector,
    }


# ---------------------------------------------------------------------------
# 3. AI/ML (D8, L=47, CPU — prediction/quantization math)
# ---------------------------------------------------------------------------

def _ai_ml_handlers() -> dict[str, Any]:
    D, L = 8, 47

    def _inference_router(query: str, ctx: Any) -> dict[str, Any]:
        batch = int(_parse_numeric(query, 32.0))
        throughput = round(_saturate(batch, BW_RAM) * L, 2)
        latency_ms = round(batch / (PHI**D * 0.001), 4)
        return {
            "batch_size": batch,
            "throughput_tokens_s": throughput,
            "latency_ms": latency_ms,
            "phi_scaling": round(PHI**D, 4),
            "lucas_states": L,
        }

    def _model_quantizer(query: str, ctx: Any) -> dict[str, Any]:
        params_m = _parse_numeric(query, 7000.0)
        fp16_gb = round(params_m * 2 / 1024, 2)
        int8_gb = round(fp16_gb * OMEGA, 2)
        int4_gb = round(fp16_gb * OMEGA**2, 2)
        quality_loss = round((1 - OMEGA**3) * 100, 2)
        return {
            "params_millions": params_m,
            "fp16_gb": fp16_gb,
            "int8_gb": int8_gb,
            "int4_gb": int4_gb,
            "quality_retention_pct": quality_loss,
            "phi_compression": round(PHI, 4),
            "lucas_quant_levels": L,
        }

    def _attention_allocator(query: str, ctx: Any) -> dict[str, Any]:
        seq_len = int(_parse_numeric(query, 4096.0))
        heads = L
        head_dim = max(1, round(seq_len / (PHI**3)))
        memory_mb = round(seq_len * heads * head_dim * 4 / (1024**2), 2)
        return {
            "sequence_length": seq_len,
            "attention_heads": heads,
            "head_dimension": head_dim,
            "memory_mb": memory_mb,
            "phi_sparsity": round(OMEGA**3, 6),
        }

    def _context_manager(query: str, ctx: Any) -> dict[str, Any]:
        tokens = int(_parse_numeric(query, 8192.0))
        windows = max(1, math.ceil(tokens / (L * 100)))
        overlap = round(L * 100 * OMEGA, 0)
        effective = round(tokens * (1 + OMEGA * (windows - 1) / windows), 0)
        return {
            "total_tokens": tokens,
            "context_windows": windows,
            "overlap_tokens": int(overlap),
            "effective_tokens": int(effective),
            "phi_window_ratio": round(OMEGA, 6),
        }

    def _embedding_router(query: str, ctx: Any) -> dict[str, Any]:
        dim = int(_parse_numeric(query, 384.0))
        clusters = max(1, round(dim / PHI**3))
        quantized_dim = round(dim * OMEGA, 0)
        bandwidth = round(_saturate(clusters, BW_RAM), 4)
        return {
            "embedding_dim": dim,
            "phi_clusters": clusters,
            "quantized_dim": int(quantized_dim),
            "routing_bandwidth_gbps": bandwidth,
            "lucas_buckets": L,
        }

    def _finetune_scheduler(query: str, ctx: Any) -> dict[str, Any]:
        dataset_k = _parse_numeric(query, 100.0)
        epochs = max(1, round(math.log(dataset_k + 1) / math.log(PHI)))
        lr_init = round(1e-4 * OMEGA**(epochs - 1), 10)
        warmup = max(1, round(epochs * OMEGA))
        return {
            "dataset_size_k": dataset_k,
            "optimal_epochs": epochs,
            "learning_rate": lr_init,
            "warmup_epochs": warmup,
            "phi_schedule_decay": round(OMEGA, 6),
            "lucas_checkpoints": L,
        }

    def _prompt_optimizer(query: str, ctx: Any) -> dict[str, Any]:
        tokens = int(_parse_numeric(query, 500.0))
        compressed = max(1, round(tokens * OMEGA**2))
        savings_pct = round((1 - compressed / tokens) * 100, 2) if tokens else 0
        quality = round(1 - OMEGA**4, 4)
        return {
            "input_tokens": tokens,
            "optimized_tokens": compressed,
            "savings_pct": savings_pct,
            "quality_retention": quality,
            "phi_compression": round(PHI**2, 4),
        }

    def _multi_model_mixer(query: str, ctx: Any) -> dict[str, Any]:
        n_models = int(_parse_numeric(query, 3.0))
        weights = [round(OMEGA**i / sum(OMEGA**j for j in range(n_models)), 4)
                    for i in range(n_models)]
        ensemble_boost = round((1 - OMEGA**n_models) * 100, 2)
        return {
            "num_models": n_models,
            "mix_weights": weights,
            "ensemble_boost_pct": ensemble_boost,
            "phi_decay": round(OMEGA, 6),
            "lucas_voting_quorum": min(L, n_models),
        }

    def _rag_router(query: str, ctx: Any) -> dict[str, Any]:
        chunks = int(_parse_numeric(query, 100.0))
        top_k = max(1, round(chunks * OMEGA**2))
        rerank_k = max(1, round(top_k * OMEGA))
        relevance_threshold = round(OMEGA**3, 4)
        return {
            "total_chunks": chunks,
            "retrieval_top_k": top_k,
            "rerank_k": rerank_k,
            "relevance_threshold": relevance_threshold,
            "phi_retrieval_factor": round(PHI**2, 4),
            "lucas_index_shards": min(L, chunks),
        }

    def _agent_coordinator(query: str, ctx: Any) -> dict[str, Any]:
        agents = int(_parse_numeric(query, 5.0))
        agents = min(agents, 27)  # N-body ceiling
        rounds = max(1, math.ceil(math.log(agents + 1) / math.log(PHI)))
        budget_per = round(TOTAL_STATES / (agents * L), 4)
        return {
            "num_agents": agents,
            "coordination_rounds": rounds,
            "budget_per_agent": budget_per,
            "ceiling": 27,
            "phi_convergence": round(OMEGA**rounds, 6),
            "lucas_quorum": min(L, agents),
        }

    return {
        "inference_router": _inference_router,
        "model_quantizer": _model_quantizer,
        "attention_allocator": _attention_allocator,
        "context_manager": _context_manager,
        "embedding_router": _embedding_router,
        "finetune_scheduler": _finetune_scheduler,
        "prompt_optimizer": _prompt_optimizer,
        "multi_model_mixer": _multi_model_mixer,
        "rag_router": _rag_router,
        "agent_coordinator": _agent_coordinator,
    }


# ---------------------------------------------------------------------------
# 4. SECURITY (D3, L=4, NPU — BETA sensitivity scoring)
# ---------------------------------------------------------------------------

def _security_handlers() -> dict[str, Any]:
    D, L = 3, 4

    def _threat_classifier(query: str, ctx: Any) -> dict[str, Any]:
        severity = _parse_numeric(query, 5.0)
        score = round(severity * BETA * PHI**D, 4)
        risk_level = "critical" if score > 5 else "high" if score > 2 else "low"
        return {
            "input_severity": severity,
            "threat_score": score,
            "risk_level": risk_level,
            "beta_sensitivity": round(BETA, 6),
            "phi_amplification": round(PHI**D, 4),
            "lucas_classes": L,
        }

    def _access_controller(query: str, ctx: Any) -> dict[str, Any]:
        trust_level = _parse_numeric(query, 50.0)
        clearance = round(trust_level * OMEGA**D, 4)
        allowed = clearance > (BETA * 100)
        return {
            "trust_level": trust_level,
            "clearance_score": clearance,
            "access_allowed": allowed,
            "beta_threshold": round(BETA * 100, 4),
            "phi_trust_decay": round(OMEGA**D, 6),
        }

    def _encryption_router(query: str, ctx: Any) -> dict[str, Any]:
        data_mb = _parse_numeric(query, 10.0)
        key_bits = round(256 * PHI**D)
        throughput = round(_saturate(L, BW_NPU) * 1024, 2)
        encrypt_ms = round(data_mb * 1024 / throughput * 1000, 4) if throughput else 0
        return {
            "data_mb": data_mb,
            "key_strength_bits": key_bits,
            "throughput_mbps": throughput,
            "encrypt_time_ms": encrypt_ms,
            "phi_key_scaling": round(PHI**D, 4),
        }

    def _anomaly_detector(query: str, ctx: Any) -> dict[str, Any]:
        baseline = _parse_numeric(query, 100.0)
        threshold_high = round(baseline * PHI**D, 2)
        threshold_low = round(baseline * OMEGA**D, 2)
        sensitivity = round(BETA * L, 4)
        return {
            "baseline": baseline,
            "upper_threshold": threshold_high,
            "lower_threshold": threshold_low,
            "sensitivity": sensitivity,
            "beta_factor": round(BETA, 6),
            "lucas_window": L,
        }

    def _audit_logger(query: str, ctx: Any) -> dict[str, Any]:
        events_per_sec = _parse_numeric(query, 1000.0)
        buffer_size = max(1, round(events_per_sec * OMEGA**D))
        flush_interval_ms = round(1000 * OMEGA**D, 2)
        retention_days = round(PHI**D * L, 0)
        return {
            "events_per_sec": events_per_sec,
            "buffer_size": buffer_size,
            "flush_interval_ms": flush_interval_ms,
            "retention_days": int(retention_days),
            "phi_compression": round(OMEGA**D, 6),
        }

    def _zero_trust_gateway(query: str, ctx: Any) -> dict[str, Any]:
        sessions = int(_parse_numeric(query, 500.0))
        verify_interval_s = round(PHI**D * BETA * 10, 2)
        max_score = round(100 * (1 - OMEGA**L), 2)
        token_ttl_s = round(PHI**D * 60, 0)
        return {
            "active_sessions": sessions,
            "verify_interval_s": verify_interval_s,
            "max_trust_score": max_score,
            "token_ttl_s": int(token_ttl_s),
            "beta_decay": round(BETA, 6),
            "lucas_verification_layers": L,
        }

    def _rate_limiter(query: str, ctx: Any) -> dict[str, Any]:
        rps = _parse_numeric(query, 1000.0)
        burst = max(1, round(rps * PHI**D / 100))
        window_ms = round(1000 / (PHI**D), 2)
        refill_rate = round(rps * OMEGA, 2)
        return {
            "target_rps": rps,
            "burst_capacity": burst,
            "window_ms": window_ms,
            "refill_rate_per_s": refill_rate,
            "phi_burst_factor": round(PHI**D, 4),
        }

    def _firewall_router(query: str, ctx: Any) -> dict[str, Any]:
        rules = int(_parse_numeric(query, 100.0))
        tiers = L
        rules_per_tier = math.ceil(rules / tiers)
        eval_time_us = round(rules * OMEGA**D * 10, 4)
        return {
            "total_rules": rules,
            "tiers": tiers,
            "rules_per_tier": rules_per_tier,
            "eval_time_us": eval_time_us,
            "phi_priority_decay": round(OMEGA**D, 6),
        }

    def _secret_manager(query: str, ctx: Any) -> dict[str, Any]:
        secrets = int(_parse_numeric(query, 50.0))
        rotation_days = round(PHI**D * L * 7, 0)
        vault_size_kb = round(secrets * PHI**D * 0.5, 2)
        return {
            "total_secrets": secrets,
            "rotation_interval_days": int(rotation_days),
            "vault_size_kb": vault_size_kb,
            "phi_rotation_factor": round(PHI**D * L, 4),
            "beta_access_score": round(BETA, 6),
        }

    def _compliance_checker(query: str, ctx: Any) -> dict[str, Any]:
        policies = int(_parse_numeric(query, 20.0))
        scan_time_ms = round(policies * OMEGA**D * 100, 2)
        coverage = round((1 - OMEGA**policies) * 100, 2)
        risk_residual = round(OMEGA**policies * 100, 2)
        return {
            "policies": policies,
            "scan_time_ms": scan_time_ms,
            "coverage_pct": coverage,
            "residual_risk_pct": risk_residual,
            "beta_sensitivity": round(BETA, 6),
            "lucas_control_layers": L,
        }

    return {
        "threat_classifier": _threat_classifier,
        "access_controller": _access_controller,
        "encryption_router": _encryption_router,
        "anomaly_detector": _anomaly_detector,
        "audit_logger": _audit_logger,
        "zero_trust_gateway": _zero_trust_gateway,
        "rate_limiter": _rate_limiter,
        "firewall_router": _firewall_router,
        "secret_manager": _secret_manager,
        "compliance_checker": _compliance_checker,
    }


# ---------------------------------------------------------------------------
# 5. BUSINESS (D6, L=18, CPU — harmony/mirror-pair allocation)
# ---------------------------------------------------------------------------

def _business_handlers() -> dict[str, Any]:
    D, L = 6, 18

    def _resource_allocator(query: str, ctx: Any) -> dict[str, Any]:
        total = _parse_numeric(query, 1000.0)
        pairs = [_mirror_pair(i) for i in range(5)]
        alloc = [
            {"pair": i + 1, "left": p[0], "right": p[1],
             "allocation": round(total * (p[0] + p[1]) / (5 * MIRROR), 2)}
            for i, p in enumerate(pairs)
        ]
        return {
            "total_resource": total,
            "allocations": alloc,
            "mirror_sum": MIRROR,
            "phi_harmony": round(PHI**D, 4),
        }

    def _task_scheduler(query: str, ctx: Any) -> dict[str, Any]:
        tasks = int(_parse_numeric(query, 50.0))
        slots = L
        waves = max(1, math.ceil(tasks / slots))
        priority_decay = [round(OMEGA**i, 4) for i in range(min(tasks, 10))]
        return {
            "total_tasks": tasks,
            "parallel_slots": slots,
            "waves": waves,
            "priority_decay": priority_decay,
            "phi_scheduling": round(PHI**D, 4),
        }

    def _billing_calculator(query: str, ctx: Any) -> dict[str, Any]:
        usage = _parse_numeric(query, 1000.0)
        base_cost = round(usage * _lucas_weight(D), 4)
        phi_discount = round(base_cost * (1 - OMEGA), 4)
        final = round(base_cost - phi_discount, 4)
        return {
            "usage_units": usage,
            "base_cost": base_cost,
            "phi_discount": phi_discount,
            "final_cost": final,
            "lucas_weight": round(_lucas_weight(D), 6),
        }

    def _demand_forecaster(query: str, ctx: Any) -> dict[str, Any]:
        current = _parse_numeric(query, 500.0)
        forecast = [round(current * PHI**(i * 0.1), 2) for i in range(7)]
        growth_rate = round((PHI**0.1 - 1) * 100, 2)
        return {
            "current_demand": current,
            "forecast_7d": forecast,
            "daily_growth_pct": growth_rate,
            "phi_trend": round(PHI**0.1, 6),
            "lucas_seasonality": L,
        }

    def _capacity_planner(query: str, ctx: Any) -> dict[str, Any]:
        current_load = _parse_numeric(query, 70.0)
        headroom = round((100 - current_load) * OMEGA, 2)
        scale_trigger = round(100 - headroom, 2)
        recommended = max(1, math.ceil(current_load / (OMEGA * 100)))
        return {
            "current_load_pct": current_load,
            "phi_headroom_pct": headroom,
            "scale_trigger_pct": scale_trigger,
            "recommended_nodes": recommended,
            "lucas_capacity": L,
        }

    def _sla_monitor(query: str, ctx: Any) -> dict[str, Any]:
        target_pct = _parse_numeric(query, 99.9)
        budget_ms = round((100 - target_pct) * PHI**D * 10, 4)
        remaining_ms = round(budget_ms * OMEGA, 4)
        burn_rate = round(1 - OMEGA, 4)
        return {
            "sla_target_pct": target_pct,
            "error_budget_ms": budget_ms,
            "remaining_budget_ms": remaining_ms,
            "burn_rate": burn_rate,
            "phi_factor": round(PHI**D, 4),
        }

    def _workflow_optimizer(query: str, ctx: Any) -> dict[str, Any]:
        steps = int(_parse_numeric(query, 10.0))
        parallel = max(1, math.ceil(steps * OMEGA))
        serial = steps - parallel
        speedup = round(steps / (serial + parallel / L), 2) if (serial + parallel / L) else 1
        return {
            "total_steps": steps,
            "parallelizable": parallel,
            "serial": serial,
            "speedup_factor": speedup,
            "phi_parallel_ratio": round(OMEGA, 6),
            "lucas_workers": L,
        }

    def _priority_ranker(query: str, ctx: Any) -> dict[str, Any]:
        items = int(_parse_numeric(query, 20.0))
        ranks = [
            {"rank": i + 1, "score": round(OMEGA**i * 100, 2)}
            for i in range(min(items, 12))
        ]
        return {
            "items": items,
            "rankings": ranks,
            "phi_decay": round(OMEGA, 6),
            "top_concentration": round((1 - OMEGA**3) * 100, 2),
        }

    def _resource_forecaster(query: str, ctx: Any) -> dict[str, Any]:
        current = _parse_numeric(query, 60.0)
        projections = [round(current * PHI**(i * 0.05), 2) for i in range(12)]
        peak = max(projections)
        return {
            "current_usage_pct": current,
            "monthly_projection": projections,
            "peak_pct": peak,
            "phi_growth": round(PHI**0.05, 6),
            "lucas_buffer": L,
        }

    def _cost_allocator(query: str, ctx: Any) -> dict[str, Any]:
        total = _parse_numeric(query, 50000.0)
        dist = _lucas_distribute(total)
        top3 = sorted(dist, reverse=True)[:3]
        return {
            "total_cost": total,
            "distribution_12d": dist,
            "top_3_dimensions": top3,
            "conservation": round(sum(dist), 2),
            "phi_ratio": round(PHI, 6),
        }

    return {
        "resource_allocator": _resource_allocator,
        "task_scheduler": _task_scheduler,
        "billing_calculator": _billing_calculator,
        "demand_forecaster": _demand_forecaster,
        "capacity_planner": _capacity_planner,
        "sla_monitor": _sla_monitor,
        "workflow_optimizer": _workflow_optimizer,
        "priority_ranker": _priority_ranker,
        "resource_forecaster": _resource_forecaster,
        "cost_allocator": _cost_allocator,
    }


# ---------------------------------------------------------------------------
# 6. DATA (D5, L=11, CPU — PHI compression ratios)
# ---------------------------------------------------------------------------

def _data_handlers() -> dict[str, Any]:
    D, L = 5, 11

    def _data_tiering(query: str, ctx: Any) -> dict[str, Any]:
        data_gb = _parse_numeric(query, 500.0)
        hot = round(data_gb * _lucas_weight(12), 2)
        warm = round(data_gb * _lucas_weight(8), 2)
        cold = round(data_gb - hot - warm, 2)
        return {
            "total_gb": data_gb,
            "hot_gb": hot,
            "warm_gb": warm,
            "cold_gb": cold,
            "phi_tiering_ratio": round(PHI, 4),
            "lucas_tiers": L,
        }

    def _backup_scheduler(query: str, ctx: Any) -> dict[str, Any]:
        data_gb = _parse_numeric(query, 100.0)
        full_interval_h = round(PHI**D * L, 0)
        incr_interval_h = round(PHI**D, 0)
        backup_size = round(data_gb * OMEGA**D, 2)
        return {
            "data_gb": data_gb,
            "full_backup_interval_h": int(full_interval_h),
            "incremental_interval_h": int(incr_interval_h),
            "incremental_size_gb": backup_size,
            "phi_dedup_ratio": round(OMEGA**D, 6),
        }

    def _cache_invalidator(query: str, ctx: Any) -> dict[str, Any]:
        keys = int(_parse_numeric(query, 10000.0))
        ttl_s = round(PHI**D * 60, 0)
        evict_batch = max(1, round(keys * OMEGA**D))
        hit_rate = round(1 - OMEGA**L, 4)
        return {
            "total_keys": keys,
            "default_ttl_s": int(ttl_s),
            "eviction_batch": evict_batch,
            "predicted_hit_rate": hit_rate,
            "phi_ttl_scaling": round(PHI**D, 4),
        }

    def _replication_manager(query: str, ctx: Any) -> dict[str, Any]:
        data_gb = _parse_numeric(query, 50.0)
        replicas = max(2, round(PHI**2))
        total_gb = round(data_gb * replicas, 2)
        sync_lag_ms = round(data_gb / BW_RAM * 1000 * OMEGA, 4)
        return {
            "primary_gb": data_gb,
            "replicas": replicas,
            "total_storage_gb": total_gb,
            "sync_lag_ms": sync_lag_ms,
            "phi_replication_factor": round(PHI**2, 4),
        }

    def _data_compressor(query: str, ctx: Any) -> dict[str, Any]:
        size_mb = _parse_numeric(query, 1000.0)
        compressed = round(size_mb * OMEGA**D, 2)
        ratio = round(size_mb / compressed, 2) if compressed else 0
        throughput = round(_saturate(L, BW_RAM), 2)
        return {
            "original_mb": size_mb,
            "compressed_mb": compressed,
            "compression_ratio": ratio,
            "throughput_gbps": throughput,
            "phi_compression": round(PHI**D, 4),
        }

    def _migration_planner(query: str, ctx: Any) -> dict[str, Any]:
        data_gb = _parse_numeric(query, 200.0)
        phases = max(1, math.ceil(math.log(data_gb + 1) / math.log(PHI**D)))
        per_phase = round(data_gb / phases, 2)
        transfer_h = round(data_gb / (BW_SSD * 3.6), 2)
        return {
            "data_gb": data_gb,
            "migration_phases": phases,
            "per_phase_gb": per_phase,
            "estimated_transfer_h": transfer_h,
            "phi_phase_factor": round(PHI**D, 4),
        }

    def _retention_manager(query: str, ctx: Any) -> dict[str, Any]:
        age_days = _parse_numeric(query, 365.0)
        retain_prob = round(math.exp(-age_days / (PHI**D * 100)), 4)
        archive_threshold = round(PHI**D * 100, 0)
        purge_threshold = round(PHI**D * 365, 0)
        return {
            "data_age_days": age_days,
            "retention_probability": retain_prob,
            "archive_threshold_days": int(archive_threshold),
            "purge_threshold_days": int(purge_threshold),
            "phi_decay": round(OMEGA**D, 6),
        }

    def _indexer(query: str, ctx: Any) -> dict[str, Any]:
        docs = int(_parse_numeric(query, 100000.0))
        shards = max(1, math.ceil(docs / (PHI**D * L * 100)))
        index_mb = round(docs * OMEGA**D / 1000, 2)
        lookup_us = round(math.log(docs + 1) / math.log(PHI) * 10, 4)
        return {
            "documents": docs,
            "index_shards": shards,
            "index_size_mb": index_mb,
            "lookup_time_us": lookup_us,
            "phi_fanout": round(PHI**D, 4),
        }

    def _partitioner(query: str, ctx: Any) -> dict[str, Any]:
        rows_m = _parse_numeric(query, 100.0)
        partitions = max(1, round(rows_m / (PHI**D)))
        rows_per_part = round(rows_m / partitions, 2)
        scan_speedup = round(partitions * OMEGA, 2)
        return {
            "rows_millions": rows_m,
            "partitions": partitions,
            "rows_per_partition_m": rows_per_part,
            "scan_speedup_factor": scan_speedup,
            "phi_partition_size": round(PHI**D, 4),
        }

    def _archiver(query: str, ctx: Any) -> dict[str, Any]:
        data_gb = _parse_numeric(query, 500.0)
        archived = round(data_gb * OMEGA**D, 2)
        cost_reduction = round((1 - OMEGA**D) * 100, 2)
        retrieval_ms = round(PHI**D * L * 100, 0)
        return {
            "total_gb": data_gb,
            "archived_gb": archived,
            "cost_reduction_pct": cost_reduction,
            "retrieval_latency_ms": int(retrieval_ms),
            "phi_archival_depth": round(PHI**D, 4),
        }

    return {
        "data_tiering": _data_tiering,
        "backup_scheduler": _backup_scheduler,
        "cache_invalidator": _cache_invalidator,
        "replication_manager": _replication_manager,
        "data_compressor": _data_compressor,
        "migration_planner": _migration_planner,
        "retention_manager": _retention_manager,
        "indexer": _indexer,
        "partitioner": _partitioner,
        "archiver": _archiver,
    }


# ---------------------------------------------------------------------------
# 7. IOT (D2, L=3, NPU — INDS device routing, PHI power)
# ---------------------------------------------------------------------------

def _iot_handlers() -> dict[str, Any]:
    D, L = 2, 3

    def _device_router(query: str, ctx: Any) -> dict[str, Any]:
        devices = int(_parse_numeric(query, 100.0))
        dr = 1 + ((devices - 1) % 9) if devices else 9
        target = "NPU" if dr in (1, 4, 7) else "CPU" if dr in (2, 5, 8) else "GPU"
        groups = max(1, math.ceil(devices / (PHI**D * L)))
        return {
            "devices": devices,
            "inds_class": dr,
            "silicon_target": target,
            "device_groups": groups,
            "phi_routing": round(PHI**D, 4),
        }

    def _firmware_updater(query: str, ctx: Any) -> dict[str, Any]:
        fleet_size = int(_parse_numeric(query, 1000.0))
        batch = max(1, round(fleet_size * OMEGA**D))
        waves = max(1, math.ceil(fleet_size / batch))
        rollback_window_h = round(PHI**D * 24, 0)
        return {
            "fleet_size": fleet_size,
            "batch_size": batch,
            "update_waves": waves,
            "rollback_window_h": int(rollback_window_h),
            "phi_rollout_factor": round(OMEGA**D, 6),
        }

    def _telemetry_collector(query: str, ctx: Any) -> dict[str, Any]:
        metrics_per_sec = _parse_numeric(query, 10000.0)
        buffer = max(1, round(metrics_per_sec * OMEGA**D))
        flush_ms = round(1000 * OMEGA**D, 2)
        compression = round(OMEGA**D, 4)
        return {
            "metrics_per_sec": metrics_per_sec,
            "buffer_size": buffer,
            "flush_interval_ms": flush_ms,
            "compression_ratio": compression,
            "lucas_channels": L,
        }

    def _power_manager(query: str, ctx: Any) -> dict[str, Any]:
        watts = _parse_numeric(query, 5.0)
        optimal_w = round(watts * OMEGA**D, 4)
        savings_pct = round((1 - OMEGA**D) * 100, 2)
        sleep_threshold_w = round(watts * OMEGA**3, 4)
        return {
            "current_watts": watts,
            "optimal_watts": optimal_w,
            "savings_pct": savings_pct,
            "sleep_threshold_w": sleep_threshold_w,
            "phi_power_curve": round(PHI**D, 4),
        }

    def _protocol_adapter(query: str, ctx: Any) -> dict[str, Any]:
        protocols = int(_parse_numeric(query, 5.0))
        overhead_pct = round(protocols * OMEGA**D * 10, 2)
        translation_us = round(PHI**D * protocols * 10, 2)
        return {
            "protocols": protocols,
            "overhead_pct": overhead_pct,
            "translation_time_us": translation_us,
            "phi_adapter_cost": round(PHI**D, 4),
            "lucas_protocol_slots": L,
        }

    def _mesh_router(query: str, ctx: Any) -> dict[str, Any]:
        nodes = int(_parse_numeric(query, 50.0))
        hops = max(1, math.ceil(math.log(nodes + 1) / math.log(PHI**D)))
        fanout = max(1, round(PHI**D))
        latency_ms = round(hops * OMEGA**D * 10, 4)
        return {
            "mesh_nodes": nodes,
            "max_hops": hops,
            "fanout": fanout,
            "estimated_latency_ms": latency_ms,
            "phi_routing_depth": round(PHI**D, 4),
        }

    def _calibration_scheduler(query: str, ctx: Any) -> dict[str, Any]:
        sensors = int(_parse_numeric(query, 20.0))
        interval_h = round(PHI**D * L * 12, 0)
        drift_tolerance = round(OMEGA**D * 100, 4)
        batch = max(1, round(sensors * OMEGA))
        return {
            "sensors": sensors,
            "calibration_interval_h": int(interval_h),
            "drift_tolerance_pct": drift_tolerance,
            "calibration_batch": batch,
            "phi_precision": round(PHI**D, 4),
        }

    def _iot_anomaly_detector(query: str, ctx: Any) -> dict[str, Any]:
        readings = _parse_numeric(query, 1000.0)
        sigma_threshold = round(PHI**D * BETA, 4)
        window = max(1, round(readings * OMEGA**D))
        false_positive_pct = round(OMEGA**L * 100, 4)
        return {
            "readings": readings,
            "sigma_threshold": sigma_threshold,
            "sliding_window": window,
            "false_positive_pct": false_positive_pct,
            "beta_sensitivity": round(BETA, 6),
        }

    def _edge_aggregator(query: str, ctx: Any) -> dict[str, Any]:
        streams = int(_parse_numeric(query, 30.0))
        buckets = max(1, math.ceil(streams / (PHI**D)))
        agg_interval_ms = round(1000 * OMEGA**D, 2)
        throughput = round(_saturate(streams, BW_NPU), 4)
        return {
            "streams": streams,
            "aggregation_buckets": buckets,
            "agg_interval_ms": agg_interval_ms,
            "throughput_gbps": throughput,
            "phi_fanin": round(PHI**D, 4),
        }

    def _fleet_manager(query: str, ctx: Any) -> dict[str, Any]:
        fleet = int(_parse_numeric(query, 500.0))
        regions = max(1, math.ceil(fleet / (PHI**D * L * 10)))
        per_region = math.ceil(fleet / regions)
        health_interval_s = round(PHI**D * 30, 0)
        return {
            "fleet_size": fleet,
            "regions": regions,
            "devices_per_region": per_region,
            "health_check_interval_s": int(health_interval_s),
            "phi_scaling": round(PHI**D, 4),
            "lucas_tiers": L,
        }

    return {
        "device_router": _device_router,
        "firmware_updater": _firmware_updater,
        "telemetry_collector": _telemetry_collector,
        "power_manager": _power_manager,
        "protocol_adapter": _protocol_adapter,
        "mesh_router": _mesh_router,
        "calibration_scheduler": _calibration_scheduler,
        "iot_anomaly_detector": _iot_anomaly_detector,
        "edge_aggregator": _edge_aggregator,
        "fleet_manager": _fleet_manager,
    }


# ---------------------------------------------------------------------------
# 8. COMMUNICATION (D6, L=18, CPU — golden-ratio channel weighting)
# ---------------------------------------------------------------------------

def _communication_handlers() -> dict[str, Any]:
    D, L = 6, 18

    def _message_router(query: str, ctx: Any) -> dict[str, Any]:
        messages = int(_parse_numeric(query, 10000.0))
        channels = L
        per_channel = math.ceil(messages / channels)
        priority_weights = [round(OMEGA**i, 4) for i in range(min(channels, 12))]
        return {
            "messages": messages,
            "channels": channels,
            "per_channel": per_channel,
            "priority_weights": priority_weights,
            "phi_routing": round(PHI**D, 4),
        }

    def _protocol_selector(query: str, ctx: Any) -> dict[str, Any]:
        payload_kb = _parse_numeric(query, 64.0)
        overhead_pct = round(OMEGA**D * 100, 2)
        effective_kb = round(payload_kb * (1 - OMEGA**D), 2)
        latency_ms = round(payload_kb / (BW_RAM * 1024) * 1000, 4)
        return {
            "payload_kb": payload_kb,
            "protocol_overhead_pct": overhead_pct,
            "effective_payload_kb": effective_kb,
            "latency_ms": latency_ms,
            "phi_efficiency": round(1 - OMEGA**D, 4),
        }

    def _sync_manager(query: str, ctx: Any) -> dict[str, Any]:
        nodes = int(_parse_numeric(query, 10.0))
        sync_rounds = max(1, math.ceil(math.log(nodes + 1) / math.log(PHI)))
        conflict_prob = round(OMEGA**nodes, 6)
        gossip_fanout = max(1, round(PHI**2))
        return {
            "nodes": nodes,
            "sync_rounds": sync_rounds,
            "conflict_probability": conflict_prob,
            "gossip_fanout": gossip_fanout,
            "phi_convergence": round(PHI, 4),
        }

    def _compression_engine(query: str, ctx: Any) -> dict[str, Any]:
        size_kb = _parse_numeric(query, 1024.0)
        compressed = round(size_kb * OMEGA**D, 2)
        ratio = round(size_kb / compressed, 2) if compressed else 0
        throughput_mbps = round(_saturate(L, BW_RAM) * 1024, 2)
        return {
            "original_kb": size_kb,
            "compressed_kb": compressed,
            "ratio": ratio,
            "throughput_mbps": throughput_mbps,
            "phi_compression": round(PHI**D, 4),
        }

    def _notification_router(query: str, ctx: Any) -> dict[str, Any]:
        users = int(_parse_numeric(query, 10000.0))
        batches = max(1, math.ceil(users / (PHI**D * L)))
        delivery_ms = round(batches * OMEGA**D * 100, 2)
        priority_levels = L
        return {
            "target_users": users,
            "delivery_batches": batches,
            "estimated_delivery_ms": delivery_ms,
            "priority_levels": priority_levels,
            "phi_fanout": round(PHI**D * L, 4),
        }

    def _channel_selector(query: str, ctx: Any) -> dict[str, Any]:
        bandwidth_mbps = _parse_numeric(query, 100.0)
        channels = L
        per_channel = round(bandwidth_mbps / channels, 2)
        golden_split = round(bandwidth_mbps * OMEGA, 2)
        return {
            "bandwidth_mbps": bandwidth_mbps,
            "channels": channels,
            "per_channel_mbps": per_channel,
            "primary_channel_mbps": golden_split,
            "secondary_channel_mbps": round(bandwidth_mbps - golden_split, 2),
            "phi_split": round(OMEGA, 6),
        }

    def _presence_manager(query: str, ctx: Any) -> dict[str, Any]:
        users = int(_parse_numeric(query, 5000.0))
        heartbeat_s = round(PHI**D, 0)
        timeout_s = round(PHI**D * L, 0)
        memory_mb = round(users * OMEGA**D / 1000, 2)
        return {
            "online_users": users,
            "heartbeat_interval_s": int(heartbeat_s),
            "timeout_s": int(timeout_s),
            "memory_mb": memory_mb,
            "phi_presence_decay": round(OMEGA**D, 6),
        }

    def _translation_router(query: str, ctx: Any) -> dict[str, Any]:
        pairs = int(_parse_numeric(query, 10.0))
        cache_size = max(1, round(pairs * PHI**D))
        latency_ms = round(pairs * OMEGA**D * 50, 2)
        quality = round(1 - OMEGA**pairs, 4)
        return {
            "language_pairs": pairs,
            "cache_entries": cache_size,
            "avg_latency_ms": latency_ms,
            "quality_score": quality,
            "phi_cache_scaling": round(PHI**D, 4),
        }

    def _media_transcoder(query: str, ctx: Any) -> dict[str, Any]:
        size_mb = _parse_numeric(query, 100.0)
        output_mb = round(size_mb * OMEGA**D, 2)
        transcode_s = round(size_mb / (BW_GPU * OMEGA) * 10, 2)
        quality = round((1 - OMEGA**3) * 100, 2)
        return {
            "input_mb": size_mb,
            "output_mb": output_mb,
            "transcode_time_s": transcode_s,
            "quality_retention_pct": quality,
            "phi_compression": round(PHI**D, 4),
        }

    def _webhook_manager(query: str, ctx: Any) -> dict[str, Any]:
        endpoints = int(_parse_numeric(query, 50.0))
        retry_delays = [round(PHI**i, 2) for i in range(5)]
        timeout_s = round(PHI**D, 0)
        max_retries = max(1, round(math.log(endpoints + 1) / math.log(PHI)))
        return {
            "endpoints": endpoints,
            "retry_delays_s": retry_delays,
            "timeout_s": int(timeout_s),
            "max_retries": max_retries,
            "phi_backoff_base": round(PHI, 4),
        }

    return {
        "message_router": _message_router,
        "protocol_selector": _protocol_selector,
        "sync_manager": _sync_manager,
        "compression_engine": _compression_engine,
        "notification_router": _notification_router,
        "channel_selector": _channel_selector,
        "presence_manager": _presence_manager,
        "translation_router": _translation_router,
        "media_transcoder": _media_transcoder,
        "webhook_manager": _webhook_manager,
    }


# ---------------------------------------------------------------------------
# 9. DEVELOPER (D7, L=29, CPU — Lucas-29 scheduling)
# ---------------------------------------------------------------------------

def _developer_handlers() -> dict[str, Any]:
    D, L = 7, 29

    def _build_optimizer(query: str, ctx: Any) -> dict[str, Any]:
        targets = int(_parse_numeric(query, 20.0))
        parallel = min(L, targets)
        waves = max(1, math.ceil(targets / parallel))
        speedup = round(targets / (targets / parallel + (1 - 1 / parallel) * OMEGA), 2)
        return {
            "build_targets": targets,
            "parallel_jobs": parallel,
            "waves": waves,
            "speedup_factor": speedup,
            "lucas_workers": L,
        }

    def _test_scheduler(query: str, ctx: Any) -> dict[str, Any]:
        tests = int(_parse_numeric(query, 500.0))
        buckets = max(1, math.ceil(tests / L))
        priority = [round(OMEGA**i * 100, 2) for i in range(min(5, tests))]
        est_time_s = round(tests * OMEGA**D * 0.1, 2)
        return {
            "total_tests": tests,
            "parallel_buckets": buckets,
            "priority_scores": priority,
            "estimated_time_s": est_time_s,
            "lucas_parallelism": L,
        }

    def _feature_flagger(query: str, ctx: Any) -> dict[str, Any]:
        flags = int(_parse_numeric(query, 30.0))
        rollout_pct = round(OMEGA * 100, 2)
        ramp_schedule = [round(min(100, rollout_pct * PHI**i), 2) for i in range(5)]
        return {
            "total_flags": flags,
            "initial_rollout_pct": rollout_pct,
            "ramp_schedule": ramp_schedule,
            "phi_ramp_factor": round(PHI, 4),
            "lucas_cohorts": min(L, flags),
        }

    def _metric_collector(query: str, ctx: Any) -> dict[str, Any]:
        metrics = int(_parse_numeric(query, 200.0))
        sample_rate = round(OMEGA**D, 6)
        storage_mb_h = round(metrics * sample_rate * 0.001, 4)
        aggregation_s = round(PHI**D, 0)
        return {
            "metrics_count": metrics,
            "sample_rate": sample_rate,
            "storage_mb_per_h": storage_mb_h,
            "aggregation_interval_s": int(aggregation_s),
            "lucas_dimensions": L,
        }

    def _code_analyzer(query: str, ctx: Any) -> dict[str, Any]:
        loc = int(_parse_numeric(query, 10000.0))
        complexity = round(math.log(loc + 1) / math.log(PHI), 2)
        modules = max(1, round(loc / (PHI**D * 10)))
        coupling = round(OMEGA**modules, 4)
        return {
            "lines_of_code": loc,
            "phi_complexity": complexity,
            "suggested_modules": modules,
            "coupling_score": coupling,
            "lucas_review_layers": min(L, modules),
        }

    def _dependency_resolver(query: str, ctx: Any) -> dict[str, Any]:
        deps = int(_parse_numeric(query, 50.0))
        layers = max(1, math.ceil(math.log(deps + 1) / math.log(PHI**D)))
        conflict_risk = round(OMEGA**layers * 100, 2)
        parallel_installs = min(L, deps)
        return {
            "dependencies": deps,
            "resolution_layers": layers,
            "conflict_risk_pct": conflict_risk,
            "parallel_installs": parallel_installs,
            "phi_resolution_depth": round(PHI**D, 4),
        }

    def _version_manager(query: str, ctx: Any) -> dict[str, Any]:
        versions = int(_parse_numeric(query, 10.0))
        retention = max(1, round(versions * OMEGA))
        deprecation_schedule = [
            {"version": i + 1, "support_days": round(PHI**i * 30, 0)}
            for i in range(min(versions, 5))
        ]
        return {
            "total_versions": versions,
            "retained_versions": retention,
            "deprecation_schedule": deprecation_schedule,
            "phi_support_growth": round(PHI, 4),
        }

    def _ci_optimizer(query: str, ctx: Any) -> dict[str, Any]:
        pipeline_steps = int(_parse_numeric(query, 15.0))
        parallel = max(1, round(pipeline_steps * OMEGA))
        serial = pipeline_steps - parallel
        total_time = round(serial + parallel / L, 2)
        speedup = round(pipeline_steps / total_time, 2) if total_time else 1
        return {
            "pipeline_steps": pipeline_steps,
            "parallel_steps": parallel,
            "serial_steps": serial,
            "estimated_speedup": speedup,
            "lucas_runners": L,
        }

    def _doc_generator(query: str, ctx: Any) -> dict[str, Any]:
        modules = int(_parse_numeric(query, 25.0))
        pages_per_module = max(1, round(PHI**2))
        total_pages = modules * pages_per_module
        gen_time_s = round(total_pages * OMEGA**D * 0.5, 2)
        return {
            "modules": modules,
            "pages_per_module": pages_per_module,
            "total_pages": total_pages,
            "generation_time_s": gen_time_s,
            "phi_detail_factor": round(PHI**2, 4),
        }

    def _perf_profiler(query: str, ctx: Any) -> dict[str, Any]:
        samples = int(_parse_numeric(query, 10000.0))
        hotspots = max(1, round(samples * OMEGA**D * 0.01))
        overhead_pct = round(OMEGA**D * 100, 2)
        flamegraph_depth = max(1, round(math.log(samples + 1) / math.log(PHI)))
        return {
            "samples": samples,
            "detected_hotspots": hotspots,
            "profiling_overhead_pct": overhead_pct,
            "flamegraph_depth": flamegraph_depth,
            "phi_sampling_decay": round(OMEGA**D, 6),
        }

    return {
        "build_optimizer": _build_optimizer,
        "test_scheduler": _test_scheduler,
        "feature_flagger": _feature_flagger,
        "metric_collector": _metric_collector,
        "code_analyzer": _code_analyzer,
        "dependency_resolver": _dependency_resolver,
        "version_manager": _version_manager,
        "ci_optimizer": _ci_optimizer,
        "doc_generator": _doc_generator,
        "perf_profiler": _perf_profiler,
    }


# ---------------------------------------------------------------------------
# 10. SCIENTIFIC (D9, L=76, GPU — PHI^9 simulation scaling)
# ---------------------------------------------------------------------------

def _scientific_handlers() -> dict[str, Any]:
    D, L = 9, 76

    def _simulation_router(query: str, ctx: Any) -> dict[str, Any]:
        particles = int(_parse_numeric(query, 100000.0))
        gpu_blocks = max(1, math.ceil(particles / (PHI**D)))
        time_step = round(OMEGA**D, 10)
        throughput = round(_saturate(gpu_blocks, BW_GPU) * L, 2)
        return {
            "particles": particles,
            "gpu_blocks": gpu_blocks,
            "time_step": time_step,
            "throughput_gflops": throughput,
            "phi_scaling": round(PHI**D, 4),
            "lucas_states": L,
        }

    def _experiment_scheduler(query: str, ctx: Any) -> dict[str, Any]:
        experiments = int(_parse_numeric(query, 100.0))
        parallel = min(L, experiments)
        rounds = max(1, math.ceil(experiments / parallel))
        priority = [round(OMEGA**i * 100, 2) for i in range(min(5, experiments))]
        return {
            "experiments": experiments,
            "parallel_slots": parallel,
            "rounds": rounds,
            "priority_scores": priority,
            "phi_scheduling": round(PHI**D, 4),
        }

    def _hypothesis_ranker(query: str, ctx: Any) -> dict[str, Any]:
        hypotheses = int(_parse_numeric(query, 20.0))
        scores = [round(OMEGA**i * 100, 2) for i in range(min(hypotheses, 12))]
        confidence = round(1 - OMEGA**hypotheses, 4)
        bayesian_prior = round(1 / (PHI**D), 8)
        return {
            "hypotheses": hypotheses,
            "ranking_scores": scores,
            "top_confidence": confidence,
            "bayesian_prior": bayesian_prior,
            "phi_evidence_weight": round(PHI, 4),
        }

    def _dataset_sampler(query: str, ctx: Any) -> dict[str, Any]:
        total = int(_parse_numeric(query, 1000000.0))
        sample_size = max(1, round(total * OMEGA**D))
        strata = min(L, 12)
        per_stratum = math.ceil(sample_size / strata)
        return {
            "population_size": total,
            "sample_size": sample_size,
            "strata": strata,
            "per_stratum": per_stratum,
            "phi_sampling_rate": round(OMEGA**D, 8),
        }

    def _visualization_engine(query: str, ctx: Any) -> dict[str, Any]:
        data_points = int(_parse_numeric(query, 10000.0))
        render_ms = round(data_points / (PHI**D * L) * 1000, 2)
        lod_levels = max(1, math.ceil(math.log(data_points + 1) / math.log(PHI**3)))
        gpu_mem_mb = round(data_points * 4 * 3 / (1024**2) * PHI, 2)
        return {
            "data_points": data_points,
            "render_time_ms": render_ms,
            "lod_levels": lod_levels,
            "gpu_memory_mb": gpu_mem_mb,
            "phi_detail_scaling": round(PHI**3, 4),
        }

    def _statistical_analyzer(query: str, ctx: Any) -> dict[str, Any]:
        observations = int(_parse_numeric(query, 1000.0))
        dof = max(1, observations - round(PHI**2))
        chi_sq_threshold = round(dof * PHI, 4)
        p_value = round(OMEGA**math.sqrt(dof + 1), 8)
        return {
            "observations": observations,
            "degrees_of_freedom": dof,
            "chi_sq_threshold": chi_sq_threshold,
            "phi_p_value": p_value,
            "lucas_bins": min(L, observations),
        }

    def _model_validator(query: str, ctx: Any) -> dict[str, Any]:
        folds = int(_parse_numeric(query, 10.0))
        train_ratio = round(1 - 1 / (PHI * folds), 4)
        expected_var = round(OMEGA**folds, 6)
        confidence = round(1 - expected_var, 4)
        return {
            "cv_folds": folds,
            "train_ratio": train_ratio,
            "expected_variance": expected_var,
            "confidence": confidence,
            "phi_regularization": round(OMEGA, 6),
        }

    def _result_aggregator(query: str, ctx: Any) -> dict[str, Any]:
        results = int(_parse_numeric(query, 50.0))
        weights = [round(_lucas_weight(i + 1), 6) for i in range(min(results, 12))]
        consensus = round(1 - OMEGA**results, 4)
        meta_confidence = round(consensus * (1 - OMEGA**3), 4)
        return {
            "results": results,
            "aggregation_weights": weights,
            "consensus_score": consensus,
            "meta_confidence": meta_confidence,
            "phi_weighting": round(PHI, 4),
        }

    def _literature_scanner(query: str, ctx: Any) -> dict[str, Any]:
        papers = int(_parse_numeric(query, 1000.0))
        relevant = max(1, round(papers * OMEGA**3))
        scan_time_s = round(papers * OMEGA**D * 0.01, 2)
        citation_depth = max(1, round(math.log(papers + 1) / math.log(PHI)))
        return {
            "papers_scanned": papers,
            "relevant_papers": relevant,
            "scan_time_s": scan_time_s,
            "citation_depth": citation_depth,
            "phi_relevance_decay": round(OMEGA**3, 6),
        }

    def _citation_tracker(query: str, ctx: Any) -> dict[str, Any]:
        citations = int(_parse_numeric(query, 500.0))
        h_index = max(1, round(math.sqrt(citations) * OMEGA))
        impact = round(citations / (PHI**3), 2)
        growth_rate = round((PHI - 1) * 100, 2)
        return {
            "total_citations": citations,
            "phi_h_index": h_index,
            "impact_score": impact,
            "annual_growth_pct": growth_rate,
            "lucas_ranking_bins": min(L, citations),
        }

    return {
        "simulation_router": _simulation_router,
        "experiment_scheduler": _experiment_scheduler,
        "hypothesis_ranker": _hypothesis_ranker,
        "dataset_sampler": _dataset_sampler,
        "visualization_engine": _visualization_engine,
        "statistical_analyzer": _statistical_analyzer,
        "model_validator": _model_validator,
        "result_aggregator": _result_aggregator,
        "literature_scanner": _literature_scanner,
        "citation_tracker": _citation_tracker,
    }


# ---------------------------------------------------------------------------
# 11. PERSONAL (D10, L=123, GPU — PHI decay habit/focus curves)
# ---------------------------------------------------------------------------

def _personal_handlers() -> dict[str, Any]:
    D, L = 10, 123

    def _focus_manager(query: str, ctx: Any) -> dict[str, Any]:
        session_min = _parse_numeric(query, 25.0)
        optimal_min = round(session_min * PHI, 1)
        break_min = round(session_min * OMEGA**2, 1)
        focus_curve = [round(100 * _phi_decay(1.0, i), 1) for i in range(6)]
        return {
            "session_minutes": session_min,
            "optimal_session_min": optimal_min,
            "break_minutes": break_min,
            "focus_decay_curve": focus_curve,
            "phi_focus_ratio": round(PHI, 4),
            "dimension": D,
        }

    def _habit_tracker(query: str, ctx: Any) -> dict[str, Any]:
        streak_days = int(_parse_numeric(query, 30.0))
        strength = round(1 - OMEGA**streak_days, 4)
        next_milestone = math.ceil(streak_days * PHI)
        momentum = round(PHI**(streak_days * 0.01), 4)
        return {
            "streak_days": streak_days,
            "habit_strength": strength,
            "next_milestone": next_milestone,
            "momentum_score": momentum,
            "phi_growth": round(PHI, 4),
        }

    def _learning_planner(query: str, ctx: Any) -> dict[str, Any]:
        topics = int(_parse_numeric(query, 10.0))
        schedule = [
            {"topic": i + 1, "hours": round(PHI**(i * 0.2), 2),
             "review_day": round(PHI**i, 0)}
            for i in range(min(topics, 12))
        ]
        total_hours = round(sum(s["hours"] for s in schedule), 2)
        return {
            "topics": topics,
            "schedule": schedule,
            "total_hours": total_hours,
            "phi_spacing_ratio": round(PHI, 4),
        }

    def _goal_tracker(query: str, ctx: Any) -> dict[str, Any]:
        goals = int(_parse_numeric(query, 5.0))
        weights = [round(OMEGA**i, 4) for i in range(goals)]
        total_w = sum(weights)
        priorities = [round(w / total_w * 100, 2) for w in weights]
        completion_target = round(100 * (1 - OMEGA**goals), 2)
        return {
            "total_goals": goals,
            "priority_pct": priorities,
            "completion_target_pct": completion_target,
            "phi_focus_factor": round(PHI, 4),
        }

    def _time_optimizer(query: str, ctx: Any) -> dict[str, Any]:
        hours = _parse_numeric(query, 8.0)
        deep_work = round(hours * OMEGA, 2)
        shallow_work = round(hours * OMEGA**2, 2)
        breaks = round(hours - deep_work - shallow_work, 2)
        return {
            "available_hours": hours,
            "deep_work_hours": deep_work,
            "shallow_work_hours": shallow_work,
            "break_hours": breaks,
            "phi_split": round(OMEGA, 6),
            "lucas_time_blocks": min(L, round(hours * 4)),
        }

    def _energy_tracker(query: str, ctx: Any) -> dict[str, Any]:
        level = _parse_numeric(query, 70.0)
        curve = [round(level * _phi_decay(1.0, i), 1) for i in range(8)]
        recovery_h = round(abs(100 - level) / (PHI**2 * 10), 2)
        optimal_level = round(100 * OMEGA, 1)
        return {
            "current_level": level,
            "energy_curve": curve,
            "recovery_hours": recovery_h,
            "optimal_level": optimal_level,
            "phi_recovery_rate": round(PHI**2, 4),
        }

    def _mood_analyzer(query: str, ctx: Any) -> dict[str, Any]:
        score = _parse_numeric(query, 50.0)
        normalized = round(score / 100, 4)
        phi_sentiment = round(normalized * PHI - OMEGA, 4)
        stability = round(1 - abs(phi_sentiment) * OMEGA, 4)
        trend = "positive" if phi_sentiment > 0 else "neutral" if phi_sentiment == 0 else "negative"
        return {
            "mood_score": score,
            "phi_sentiment": phi_sentiment,
            "stability_index": stability,
            "trend": trend,
            "lucas_mood_bins": min(L, 10),
        }

    def _routine_optimizer(query: str, ctx: Any) -> dict[str, Any]:
        tasks = int(_parse_numeric(query, 12.0))
        time_blocks = [
            {"task": i + 1, "duration_min": round(60 * _lucas_weight(i + 1), 1)}
            for i in range(min(tasks, 12))
        ]
        total_min = round(sum(b["duration_min"] for b in time_blocks), 1)
        return {
            "daily_tasks": tasks,
            "time_blocks": time_blocks,
            "total_minutes": total_min,
            "phi_weighting": round(PHI, 4),
        }

    def _productivity_scorer(query: str, ctx: Any) -> dict[str, Any]:
        tasks_done = int(_parse_numeric(query, 8.0))
        score = round(min(100, tasks_done * PHI**2 * 3), 2)
        streak_bonus = round(PHI**(tasks_done * 0.1) - 1, 4)
        efficiency = round(score * (1 - OMEGA**tasks_done), 2)
        return {
            "tasks_completed": tasks_done,
            "productivity_score": score,
            "streak_bonus": streak_bonus,
            "efficiency": efficiency,
            "phi_multiplier": round(PHI**2, 4),
        }

    def _wellness_monitor(query: str, ctx: Any) -> dict[str, Any]:
        metrics = int(_parse_numeric(query, 5.0))
        composite = round(100 * (1 - OMEGA**metrics), 2)
        dimensions = [
            {"dim": i + 1, "weight": round(_lucas_weight(i + 1), 4)}
            for i in range(min(metrics, 12))
        ]
        balance = round(min(d["weight"] for d in dimensions) / max(d["weight"] for d in dimensions), 4)
        return {
            "health_metrics": metrics,
            "composite_score": composite,
            "dimensions": dimensions,
            "balance_index": balance,
            "phi_wellness": round(PHI, 4),
        }

    return {
        "focus_manager": _focus_manager,
        "habit_tracker": _habit_tracker,
        "learning_planner": _learning_planner,
        "goal_tracker": _goal_tracker,
        "time_optimizer": _time_optimizer,
        "energy_tracker": _energy_tracker,
        "mood_analyzer": _mood_analyzer,
        "routine_optimizer": _routine_optimizer,
        "productivity_scorer": _productivity_scorer,
        "wellness_monitor": _wellness_monitor,
    }


# ---------------------------------------------------------------------------
# 12. FINANCE (D8, L=47, CPU — mirror-pair portfolio balancing)
# ---------------------------------------------------------------------------

def _finance_handlers() -> dict[str, Any]:
    D, L = 8, 47

    def _portfolio_balancer(query: str, ctx: Any) -> dict[str, Any]:
        total = _parse_numeric(query, 100000.0)
        pairs = [_mirror_pair(i) for i in range(5)]
        allocation = [
            {"pair": i + 1, "left": p[0], "right": p[1],
             "amount": round(total * (p[0] + p[1]) / (5 * MIRROR), 2)}
            for i, p in enumerate(pairs)
        ]
        return {
            "total_value": total,
            "allocation": allocation,
            "mirror_sum": MIRROR,
            "phi_rebalance_threshold": round(OMEGA**2 * 100, 2),
        }

    def _risk_calculator(query: str, ctx: Any) -> dict[str, Any]:
        exposure = _parse_numeric(query, 50000.0)
        var_95 = round(exposure * OMEGA**D, 2)
        var_99 = round(exposure * OMEGA**(D + 2), 2)
        sharpe = round(PHI - 1, 4)
        max_drawdown = round(OMEGA**3 * 100, 2)
        return {
            "exposure": exposure,
            "var_95_pct": var_95,
            "var_99_pct": var_99,
            "phi_sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
        }

    def _budget_allocator(query: str, ctx: Any) -> dict[str, Any]:
        budget = _parse_numeric(query, 10000.0)
        dist = _lucas_distribute(budget)
        top_3 = sorted(dist, reverse=True)[:3]
        return {
            "total_budget": budget,
            "distribution_12d": dist,
            "top_3_allocations": top_3,
            "conservation": round(sum(dist), 2),
            "phi_ratio": round(PHI, 6),
        }

    def _trade_router(query: str, ctx: Any) -> dict[str, Any]:
        volume = _parse_numeric(query, 1000.0)
        slippage_pct = round(OMEGA**D * 100, 4)
        optimal_chunk = round(volume * OMEGA**2, 2)
        chunks = max(1, math.ceil(volume / optimal_chunk)) if optimal_chunk else 1
        return {
            "volume": volume,
            "slippage_pct": slippage_pct,
            "optimal_chunk_size": optimal_chunk,
            "num_chunks": chunks,
            "phi_timing_factor": round(PHI**D, 4),
        }

    def _tax_optimizer(query: str, ctx: Any) -> dict[str, Any]:
        income = _parse_numeric(query, 100000.0)
        deductions = round(income * OMEGA**3, 2)
        effective_rate = round((1 - OMEGA**3) * 100, 2)
        savings = round(income * OMEGA**D, 2)
        return {
            "gross_income": income,
            "optimal_deductions": deductions,
            "effective_rate_pct": effective_rate,
            "tax_savings": savings,
            "phi_bracket_scaling": round(PHI**3, 4),
        }

    def _expense_tracker(query: str, ctx: Any) -> dict[str, Any]:
        monthly = _parse_numeric(query, 5000.0)
        amounts = [round(monthly * _lucas_weight(i + 1), 2) for i in range(min(12, L))]
        categories = [
            {"category": f"cat_{i+1}", "amount": a}
            for i, a in enumerate(amounts)
        ]
        top_expense = max(amounts)
        savings_target = round(monthly * OMEGA, 2)
        return {
            "monthly_total": monthly,
            "breakdown": categories,
            "top_expense": top_expense,
            "savings_target": savings_target,
            "phi_savings_ratio": round(OMEGA, 6),
        }

    def _investment_analyzer(query: str, ctx: Any) -> dict[str, Any]:
        principal = _parse_numeric(query, 10000.0)
        growth = [round(principal * PHI**(i * 0.1), 2) for i in range(12)]
        cagr_pct = round((PHI**0.1 - 1) * 100, 2)
        doubling_periods = round(math.log(2) / math.log(PHI**0.1), 1)
        return {
            "principal": principal,
            "projected_growth": growth,
            "cagr_pct": cagr_pct,
            "doubling_periods": doubling_periods,
            "phi_growth_base": round(PHI**0.1, 6),
        }

    def _cashflow_predictor(query: str, ctx: Any) -> dict[str, Any]:
        current = _parse_numeric(query, 20000.0)
        forecast = [round(current * (1 + (OMEGA - 0.5) * 0.1)**i, 2) for i in range(12)]
        net_change = round(forecast[-1] - current, 2)
        monthly_rate = round((OMEGA - 0.5) * 10, 2)
        return {
            "current_cashflow": current,
            "forecast_12m": forecast,
            "net_change": net_change,
            "monthly_growth_pct": monthly_rate,
            "phi_trend_factor": round(OMEGA, 6),
        }

    def _fraud_detector(query: str, ctx: Any) -> dict[str, Any]:
        transactions = int(_parse_numeric(query, 10000.0))
        threshold = round(PHI**D * BETA, 4)
        flagged = max(0, round(transactions * OMEGA**D * 0.01))
        false_positive_pct = round(OMEGA**L * 100, 6)
        return {
            "transactions": transactions,
            "anomaly_threshold": threshold,
            "flagged_transactions": flagged,
            "false_positive_pct": false_positive_pct,
            "beta_sensitivity": round(BETA, 6),
        }

    def _compliance_monitor(query: str, ctx: Any) -> dict[str, Any]:
        regulations = int(_parse_numeric(query, 30.0))
        coverage = round((1 - OMEGA**regulations) * 100, 2)
        scan_interval_h = round(PHI**D * OMEGA, 0)
        risk_score = round(OMEGA**regulations * 100, 4)
        return {
            "regulations": regulations,
            "compliance_coverage_pct": coverage,
            "scan_interval_h": int(scan_interval_h),
            "residual_risk_pct": risk_score,
            "phi_monitoring_depth": round(PHI**D, 4),
        }

    return {
        "portfolio_balancer": _portfolio_balancer,
        "risk_calculator": _risk_calculator,
        "budget_allocator": _budget_allocator,
        "trade_router": _trade_router,
        "tax_optimizer": _tax_optimizer,
        "expense_tracker": _expense_tracker,
        "investment_analyzer": _investment_analyzer,
        "cashflow_predictor": _cashflow_predictor,
        "fraud_detector": _fraud_detector,
        "compliance_monitor": _compliance_monitor,
    }


# ---------------------------------------------------------------------------
# PUBLIC: merge all 12 category factories into one dict (120 handlers)
# ---------------------------------------------------------------------------

def build_category_handlers() -> dict[str, Any]:
    """Merge all 12 category factories into one dict (120 handlers)."""
    all_h: dict[str, Any] = {}
    for factory in [
        _infrastructure_handlers,
        _edge_handlers,
        _ai_ml_handlers,
        _security_handlers,
        _business_handlers,
        _data_handlers,
        _iot_handlers,
        _communication_handlers,
        _developer_handlers,
        _scientific_handlers,
        _personal_handlers,
        _finance_handlers,
    ]:
        all_h.update(factory())
    return all_h


CATEGORY_HANDLERS = build_category_handlers()
