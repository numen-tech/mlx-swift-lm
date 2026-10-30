// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// The one-step hold on ``MambaCache`` (pipelined decode): `retainsPreviousStep`,
/// `replaceState`, depth-1 `trim`.
final class MambaCacheHoldTests: XCTestCase {
    private func filled(_ value: Float) -> MLXArray {
        MLXArray.full([1, 3, 4], values: MLXArray(value))
    }

    private func step(_ cache: MambaCache, _ value: Float, tokens: Int = 1) {
        cache.replaceState(conv: filled(value), recurrent: filled(-value), advancedBy: tokens)
    }

    private func value(_ array: MLXArray?) -> Float? {
        array.map { $0[0, 0, 0].item(Float.self) }
    }

    func testHoldIsOffByDefault() {
        let cache = MambaCache()
        step(cache, 1)
        step(cache, 2)
        XCTAssertFalse(cache.canRewindHeldStep)
        XCTAssertFalse(cache.isTrimmable(after: 1))
        XCTAssertEqual(cache.trim(1), 0)
        XCTAssertEqual(value(cache[0]), 2, "a refused trim changes nothing")
    }

    func testTrimRestoresThePreviousStepAtDepthOne() {
        let cache = MambaCache()
        cache.retainsPreviousStep = true
        step(cache, 1)
        step(cache, 2)
        XCTAssertTrue(cache.canRewindHeldStep)
        XCTAssertEqual(value(cache[0]), 2)

        XCTAssertEqual(cache.trim(2), 0, "deeper rewinds are rejected")
        XCTAssertEqual(cache.trim(0), 0)
        XCTAssertEqual(value(cache[0]), 2, "a rejected trim changes nothing")
        XCTAssertTrue(cache.canRewindHeldStep)

        XCTAssertEqual(cache.trim(1), 1)
        XCTAssertEqual(value(cache[0]), 1)
        XCTAssertEqual(value(cache[1]), -1)
        XCTAssertFalse(cache.canRewindHeldStep, "depth 1 only: nothing is held after the rewind")
        XCTAssertEqual(cache.trim(1), 0)
    }

    func testTrimAfterTheFirstStepRestoresTheEmptyState() {
        let cache = MambaCache()
        cache.retainsPreviousStep = true
        step(cache, 1)
        XCTAssertEqual(cache.trim(1), 1)
        XCTAssertNil(cache[0])
        XCTAssertNil(cache[1])
    }

    func testPredictionAndWholeHistoryGate() {
        let cache = MambaCache()
        XCTAssertFalse(cache.isTrimmable(after: 1))
        cache.retainsPreviousStep = true
        XCTAssertTrue(cache.isTrimmable(after: 1), "the step about to be appended will be held")
        XCTAssertFalse(cache.isTrimmable(after: 0))
        XCTAssertFalse(cache.isTrimmable(after: 2))
        step(cache, 1)
        XCTAssertFalse(cache.isTrimmable, "a depth-1 hold is not whole-history trimmability")
        XCTAssertFalse(canTrimPromptCache([cache]))
    }

    func testAMultiTokenStepDropsTheHold() {
        let cache = MambaCache()
        cache.retainsPreviousStep = true
        step(cache, 1)
        XCTAssertTrue(cache.canRewindHeldStep)
        step(cache, 2, tokens: 5)
        XCTAssertFalse(cache.canRewindHeldStep, "a prefill chunk cannot be taken back a token at a time")
        XCTAssertEqual(cache.trim(1), 0)
    }

    func testOtherWritesAndDisablingReleaseTheHold() {
        let cache = MambaCache()
        cache.retainsPreviousStep = true
        step(cache, 1)
        step(cache, 2)
        cache[0] = filled(9)
        XCTAssertFalse(cache.canRewindHeldStep, "a slot write that bypassed replaceState")

        step(cache, 3)
        XCTAssertTrue(cache.canRewindHeldStep)
        cache.state = [filled(7), filled(8)]
        XCTAssertFalse(cache.canRewindHeldStep)

        step(cache, 4)
        cache.retainsPreviousStep = false
        XCTAssertFalse(cache.canRewindHeldStep, "turning the hold off releases it")

        cache.retainsPreviousStep = true
        step(cache, 5)
        cache.releaseHeldStep()
        XCTAssertFalse(cache.canRewindHeldStep)
        XCTAssertEqual(cache.trim(1), 0)
    }

    func testTrimRestoresOffsetAndLengthMetadata() {
        let cache = MambaCache(leftPadding: [2])
        cache.retainsPreviousStep = true
        cache.prepare(lengths: [6])
        cache.offset = 4
        step(cache, 1)
        let lengthsAfterOne = cache.currentLengths?.asArray(Int.self)
        let paddingAfterOne = cache.leftPaddingValues
        step(cache, 2)
        XCTAssertEqual(cache.currentLengths?.asArray(Int.self), (lengthsAfterOne ?? []).map { $0 - 1 })
        cache.offset = 5

        XCTAssertEqual(cache.trim(1), 1)
        XCTAssertEqual(cache.currentLengths?.asArray(Int.self), lengthsAfterOne)
        XCTAssertEqual(cache.leftPaddingValues, paddingAfterOne)
        XCTAssertEqual(cache.offset, 4)
    }

    func testCopyDoesNotCarryTheHold() {
        let cache = MambaCache()
        cache.retainsPreviousStep = true
        step(cache, 1)
        step(cache, 2)
        let copy = cache.copy() as! MambaCache
        XCTAssertFalse(copy.canRewindHeldStep)
        XCTAssertFalse(copy.retainsPreviousStep)
        XCTAssertTrue(cache.canRewindHeldStep)
    }

    func testSpeculativeCheckpointRestoreReleasesTheHold() {
        let cache = MambaCache()
        cache.retainsPreviousStep = true
        step(cache, 1)
        cache.saveSpeculativeCheckpoint(
            convState: filled(5), recurrentState: filled(6), advancedBy: 1)
        step(cache, 2)
        XCTAssertTrue(cache.canRewindHeldStep)
        XCTAssertTrue(cache.restoreSpeculativeCheckpoint())
        XCTAssertFalse(cache.canRewindHeldStep)
    }

    // MARK: - Through the model: eager and compiled GDN decode

    private let tokens: [Int32] = [1, 7, 3, 9, 2, 5]
    private let positions: [[Int32]] = [
        [0, 0, 0], [1, 2, 3], [2, 4, 5], [3, 5, 8], [4, 7, 9], [5, 9, 11],
    ]

    private func makeModel(seed: UInt64) throws -> Qwen35Language.Model {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 0,
                "num_experts_per_tok": 0,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16,
                "rope_parameters": {
                    "type": "default",
                    "mrope_section": [8, 4, 4],
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            }
            """
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self, from: Data(json.utf8))
        return withRandomState(MLXRandom.RandomState(seed: seed)) {
            let model = Qwen35Language.Model(configuration)
            model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
            model.train(false)
            return model
        }
    }

    private func forward(
        _ model: Qwen35Language.Model, _ cache: [KVCache?], index: Int, compiled: Bool
    ) -> MLXArray {
        let input = MLXArray([tokens[index]]).reshaped(1, 1)
        let positionIds = MLXArray(positions[index]).reshaped(3, 1, 1)
        let output = model.forward(
            input, cache: cache, positionIds: positionIds, useCompiledDecode: compiled)
        eval(output)
        return output
    }

    /// Feed `tokens`, rewind the last one, feed it again: the second output equals the first,
    /// bit for bit, and the cache ends where an uninterrupted run ends.
    private func assertRewindReplays(compiled: Bool) throws {
        let model = try makeModel(seed: 41)
        let cache: [KVCache?] = [MambaCache(), KVCacheSimple()]
        (cache[0] as? MambaCache)?.retainsPreviousStep = true

        var last: MLXArray?
        for index in tokens.indices { last = forward(model, cache, index: index, compiled: compiled) }
        let final = try XCTUnwrap(last)
        let stateBefore = cache.compactMap { $0 }.flatMap { $0.innerState() }.map { $0.asData(access: .copy).data }

        let mamba = try XCTUnwrap(cache[0] as? MambaCache)
        XCTAssertTrue(mamba.canRewindHeldStep)
        XCTAssertEqual(mamba.trim(1), 1)
        XCTAssertEqual(cache[1]?.trim(1), 1)

        let replayed = forward(model, cache, index: tokens.count - 1, compiled: compiled)
        XCTAssertEqual(
            replayed.asData(access: .copy).data, final.asData(access: .copy).data,
            "re-feeding the rewound token reproduces its output")
        let stateAfter = cache.compactMap { $0 }.flatMap { $0.innerState() }.map { $0.asData(access: .copy).data }
        XCTAssertEqual(stateAfter, stateBefore)
    }

    func testEagerDecodeRewindsOneToken() throws {
        try assertRewindReplays(compiled: false)
    }

    func testCompiledDecodeRewindsOneToken() throws {
        try assertRewindReplays(compiled: true)
    }

    /// A prefill chunk (more than one token) never leaves a hold, so a trim after prefill is
    /// refused rather than restoring the wrong state.
    func testPrefillLeavesNothingHeld() throws {
        let model = try makeModel(seed: 43)
        let cache: [KVCache?] = [MambaCache(), KVCacheSimple()]
        (cache[0] as? MambaCache)?.retainsPreviousStep = true
        let input = MLXArray(Array(tokens.prefix(4))).reshaped(1, 4)
        let positionIds = MLXArray(
            (0 ..< 3).flatMap { axis in positions.prefix(4).map { $0[axis] } }
        ).reshaped(3, 1, 4)
        eval(model.forward(input, cache: cache, positionIds: positionIds, useCompiledDecode: false))
        XCTAssertFalse((cache[0] as? MambaCache)?.canRewindHeldStep ?? true)
        _ = forward(model, cache, index: 4, compiled: false)
        XCTAssertTrue((cache[0] as? MambaCache)?.canRewindHeldStep ?? false)
    }
}
