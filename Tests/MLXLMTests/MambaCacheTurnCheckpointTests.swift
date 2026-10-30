// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// The turn checkpoint on ``MambaCache`` (multi-turn reuse): capture, restore, copy, install.
final class MambaCacheTurnCheckpointTests: XCTestCase {
    private func filled(_ value: Float, shape: [Int] = [1, 3, 4]) -> MLXArray {
        MLXArray.full(shape, values: MLXArray(value))
    }

    private func put(_ cache: MambaCache, _ value: Float) {
        cache[0] = filled(value)
        cache[1] = filled(-value, shape: [1, 2, 4, 4])
    }

    private func value(_ array: MLXArray?) -> Float? {
        array.map { $0.reshaped(-1)[0].item(Float.self) }
    }

    func testCaptureNeedsState() {
        let cache = MambaCache()
        XCTAssertFalse(cache.captureTurnCheckpoint(tokenOffset: 4))
        XCTAssertNil(cache.turnCheckpoint)
    }

    func testRestoreBringsBackTheCapturedState() throws {
        let cache = MambaCache()
        put(cache, 1)
        XCTAssertTrue(cache.captureTurnCheckpoint(tokenOffset: 7))
        put(cache, 2)
        put(cache, 3)
        XCTAssertEqual(value(cache[0]), 3)

        XCTAssertTrue(cache.restoreTurnCheckpoint())
        XCTAssertEqual(value(cache[0]), 1)
        XCTAssertEqual(value(cache[1]), -1)
        XCTAssertEqual(cache.turnCheckpoint?.tokenOffset, 7, "the checkpoint stays attached")

        // The live state moves on again; the checkpoint is still the captured one.
        put(cache, 5)
        XCTAssertTrue(cache.restoreTurnCheckpoint())
        XCTAssertEqual(value(cache[0]), 1)
    }

    func testCaptureReplacesTheEarlierCheckpoint() {
        let cache = MambaCache()
        put(cache, 1)
        cache.captureTurnCheckpoint(tokenOffset: 4)
        put(cache, 2)
        cache.captureTurnCheckpoint(tokenOffset: 9)
        XCTAssertEqual(cache.turnCheckpoint?.tokenOffset, 9)
        put(cache, 3)
        cache.restoreTurnCheckpoint()
        XCTAssertEqual(value(cache[0]), 2)
    }

    func testByteCountAndArraysCoverBothSlots() throws {
        let cache = MambaCache()
        put(cache, 1)
        cache.captureTurnCheckpoint(tokenOffset: 1)
        let checkpoint = try XCTUnwrap(cache.turnCheckpoint)
        XCTAssertEqual(checkpoint.arrays.count, 2)
        XCTAssertEqual(checkpoint.byteCount, (1 * 3 * 4 + 1 * 2 * 4 * 4) * 4)
    }

    func testCopyCarriesTheCheckpointAndRestoresIndependently() {
        let cache = MambaCache()
        put(cache, 1)
        cache.captureTurnCheckpoint(tokenOffset: 6)
        put(cache, 2)
        let copy = cache.copy() as! MambaCache
        XCTAssertEqual(copy.turnCheckpoint?.tokenOffset, 6)
        copy.restoreTurnCheckpoint()
        XCTAssertEqual(value(copy[0]), 1)
        XCTAssertEqual(value(cache[0]), 2, "restoring the copy leaves the original alone")
    }

    func testRestoreWithoutCheckpointChangesNothing() {
        let cache = MambaCache()
        put(cache, 4)
        XCTAssertFalse(cache.restoreTurnCheckpoint())
        XCTAssertEqual(value(cache[0]), 4)
    }

    func testDiscardAndInstall() throws {
        let cache = MambaCache()
        put(cache, 1)
        cache.captureTurnCheckpoint(tokenOffset: 3)
        let checkpoint = try XCTUnwrap(cache.turnCheckpoint)
        cache.discardTurnCheckpoint()
        XCTAssertNil(cache.turnCheckpoint)

        let other = MambaCache()
        other.installTurnCheckpoint(checkpoint)
        XCTAssertTrue(other.restoreTurnCheckpoint())
        XCTAssertEqual(value(other[0]), 1)
    }

    func testRestoreDropsThePendingSpeculativeCheckpoint() {
        let cache = MambaCache()
        put(cache, 1)
        cache.captureTurnCheckpoint(tokenOffset: 2)
        cache.saveSpeculativeCheckpoint(
            convState: filled(9), recurrentState: filled(-9, shape: [1, 2, 4, 4]), advancedBy: 1)
        XCTAssertTrue(cache.hasSpeculativeCheckpoint)
        cache.restoreTurnCheckpoint()
        XCTAssertFalse(cache.hasSpeculativeCheckpoint)
    }
}
