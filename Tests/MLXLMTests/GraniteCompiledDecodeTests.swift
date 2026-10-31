// Copyright © 2026 Apple Inc.
//
// Granite's single-token decode runs as compiled segments split at the KV
// writes (`GraniteModelInner.decodeStep`). These tests pin what the compiled
// step must keep: the eager graph's tokens, the current weights, the general
// path for caches and dtypes it cannot take, and the model's lifetime.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class GraniteCompiledDecodeTests: XCTestCase {

    private func configuration(layers: Int = 3) throws -> GraniteConfiguration {
        let json = """
            {
                "hidden_size": 128, "num_hidden_layers": \(layers), "intermediate_size": 256,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5, "vocab_size": 64,
                "logits_scaling": 4.0, "attention_multiplier": 0.125,
                "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
                "max_position_embeddings": 256, "attention_bias": false, "mlp_bias": false,
                "rope_theta": 10000.0, "tie_word_embeddings": true
            }
            """
        return try JSONDecoder().decode(GraniteConfiguration.self, from: Data(json.utf8))
    }

    /// A model with deterministic weights in `dtype`.
    private func model(
        layers: Int = 3, seed: UInt64 = 7, dtype: DType = .float16
    ) throws -> GraniteModel {
        let configuration = try configuration(layers: layers)
        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = GraniteModel(configuration)
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            eval(model)
            // A `ModelContext` does this for every loaded model; the compiled step is eval-only.
            model.train(false)
            return model
        }
    }

    /// Prefills `prompt`, then greedy-decodes `steps` tokens one at a time. Returns every step's
    /// last-position logits and the tokens.
    private func decode(
        _ model: GraniteModel, prompt: [Int32] = [1, 2, 3, 4, 5], steps: Int = 8,
        cache: [KVCache]? = nil
    ) throws -> (logits: [[Float]], tokens: [Int]) {
        let cache = try cache ?? model.newCache(parameters: nil)
        var out = model(MLXArray(prompt).reshaped(1, -1), cache: cache)[0, -1]
        var logits: [[Float]] = []
        var tokens: [Int] = []
        for _ in 0 ..< steps {
            let token = argMax(out).item(Int.self)
            tokens.append(token)
            out = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)[0, -1]
            eval(out)
            logits.append(out.asType(.float32).asArray(Float.self))
        }
        return (logits, tokens)
    }

    private func withCompile<T>(_ enabled: Bool, _ body: () throws -> T) rethrows -> T {
        MLX.compile(enable: enabled)
        defer { MLX.compile(enable: true) }
        return try body()
    }

    // MARK: - Parity

    /// The compiled step picks the eager graph's tokens, and its logits stay within fp16 noise of
    /// them. Both models have the same weights; compile is switched off for the eager one.
    func testCompiledDecodeMatchesTheEagerGraph() throws {
        let compiled = try model()
        let eager = try model()
        let a = try decode(compiled)
        XCTAssertEqual(compiled.model.compiledDecodeSegmentCount, 4, "layers + 1 segments")
        let b = try withCompile(false) { try decode(eager) }
        XCTAssertEqual(a.tokens, b.tokens)
        for (x, y) in zip(a.logits, b.logits) {
            XCTAssertEqual(x.count, y.count)
            let worst = zip(x, y).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(worst, 0.05, "compiled logits drift from the eager graph")
        }
    }

    // MARK: - Fallbacks

    func testFloat32TakesTheGeneralPath() throws {
        let model = try model(dtype: .float32)
        _ = try decode(model)
        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    func testQuantizedKVCacheTakesTheGeneralPath() throws {
        let model = try model()
        let cache = (0 ..< 3).map { _ in QuantizedKVCache(groupSize: 32, bits: 8) as KVCache }
        _ = try decode(model, cache: cache)
        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    /// A prompt is not a decode step: more than one token never enters the segments.
    func testPrefillTakesTheGeneralPath() throws {
        let model = try model()
        _ = model(MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3), cache: try model.newCache(parameters: nil))
        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    // MARK: - Weights and lifetime

    /// The traces read the weights the model has now, not the ones they were compiled with.
    func testCompiledDecodeTracksWeightUpdates() throws {
        let warm = try model()
        let reference = try model()
        _ = try decode(warm)
        XCTAssertGreaterThan(warm.model.compiledDecodeSegmentCount, 0)

        let updated = warm.parameters().mapValues { $0 * 1.25 }
        warm.update(parameters: updated)
        reference.update(parameters: updated)

        let a = try decode(warm)
        let b = try decode(reference)
        XCTAssertEqual(a.logits, b.logits, "compiled decode used stale weights")
    }

    func testBlocksDeallocateAfterCompiledDecode() throws {
        var model: GraniteModel? = try model()
        _ = try decode(model!, steps: 3)
        weak var inner: GraniteModelInner? = model!.model
        weak var layer = model!.modules().compactMap { $0 as? GraniteTransformerBlock }.first
        XCTAssertNotNil(layer)
        model = nil
        XCTAssertNil(inner, "GraniteModelInner leaked after model release")
        XCTAssertNil(layer, "GraniteTransformerBlock leaked after model release")
    }
}
