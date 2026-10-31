// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// A restore that puts the live state somewhere else must end an active rollback recording: the
/// recording's base and record describe the state it replaced, and a later commit would rebuild
/// from that stale base over the restored state.
final class MambaCacheRestoreDropsRecordingTests: XCTestCase {
    private func filled(_ value: Float) -> MLXArray {
        MLXArray.full([1, 3, 4], values: MLXArray(value))
    }

    private func value(_ array: MLXArray?) -> Float? {
        array.map { $0[0, 0, 0].item(Float.self) }
    }

    /// A cache with a turn checkpoint (value 1), then a recorded two-row pass (value 5).
    private func recordedPass() -> MambaCache {
        let cache = MambaCache()
        cache.replaceState(conv: filled(1), recurrent: filled(-1), advancedBy: 4)
        XCTAssertTrue(cache.captureTurnCheckpoint(tokenOffset: 4))
        cache.replaceState(conv: filled(3), recurrent: filled(-3), advancedBy: 1)
        cache.beginRollbackRecording()
        cache.replaceState(conv: filled(5), recurrent: filled(-5), advancedBy: 2)
        cache.recordRollback(rows: 2) { [self] _, _ in [self.filled(9), self.filled(-9)] }
        XCTAssertTrue(cache.canCommitRollback(keepingRows: 1))
        return cache
    }

    private func assertRecordingGone(_ cache: MambaCache, expecting expected: Float) {
        XCTAssertFalse(cache.isRecordingRollback)
        XCTAssertFalse(cache.canCommitRollback(keepingRows: 1))
        XCTAssertFalse(cache.commitRollback(keepingRows: 1))
        XCTAssertFalse(cache.restoreRollbackBase(), "nothing left to restore")
        XCTAssertEqual(value(cache[0]), expected, "the restored state must stand")
    }

    func testRestoreTurnCheckpointEndsTheRecording() {
        let cache = recordedPass()
        XCTAssertTrue(cache.restoreTurnCheckpoint())
        assertRecordingGone(cache, expecting: 1)
    }

    func testRestoreSpeculativeCheckpointEndsTheRecording() {
        let cache = MambaCache()
        cache.replaceState(conv: filled(1), recurrent: filled(-1), advancedBy: 4)
        cache.saveSpeculativeCheckpoint(
            convState: filled(2), recurrentState: filled(-2), advancedBy: 1)
        cache.beginRollbackRecording()
        cache.replaceState(conv: filled(5), recurrent: filled(-5), advancedBy: 2)
        cache.recordRollback(rows: 2) { [self] _, _ in [self.filled(9), self.filled(-9)] }
        XCTAssertTrue(cache.restoreSpeculativeCheckpoint())
        assertRecordingGone(cache, expecting: 2)
    }
}
