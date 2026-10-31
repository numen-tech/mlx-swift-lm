// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// The turn checkpoint of ``MambaCache`` must own its conv window: after a single-conversation
/// prefill chunk the window is a row-contiguous tail slice of the chunk's concatenated input, and
/// a long-lived checkpoint that merely referenced it would pin the whole chunk per recurrent layer
/// while `byteCount` reported only the window.
final class MambaCacheTurnCheckpointMemoryTests: XCTestCase {
    private func activeBytes() -> Int {
        Memory.clearCache()
        return Memory.activeMemory
    }

    func testCapturedWindowDoesNotPinTheChunkItWasSlicedFrom() {
        assertCaptureDoesNotPin(rows: (2048 - 3)..<2048)  // the tail of a prefill chunk
    }

    func testCapturedWindowDoesNotPinTheChunkWhenItIsAPrefixSlice() {
        assertCaptureDoesNotPin(rows: 0..<3)  // offset 0: nothing for a copy-if-needed op to fix
    }

    private func assertCaptureDoesNotPin(rows: Range<Int>) {
        let cache = MambaCache()
        let base = activeBytes()
        let chunkBytes = 2048 * 1024 * 4  // 8 MiB
        do {
            let chunk = MLXArray.ones([1, 2048, 1024], dtype: .float32)
            eval(chunk)
            // The row-contiguous tail of a batch-1 chunk: shares the chunk's storage.
            let window = chunk[0..., rows, 0...]
            eval(window)  // as in a decode loop: the cache holds an evaluated slice
            cache.replaceState(
                conv: window, recurrent: MLXArray.zeros([1, 2, 4, 4], dtype: .float32),
                advancedBy: 2048)
            XCTAssertTrue(cache.captureTurnCheckpoint(tokenOffset: 2048))
        }

        // The live state moves on; only the checkpoint can still reference the chunk.
        cache.replaceState(
            conv: MLXArray.zeros([1, 3, 1024], dtype: .float32),
            recurrent: MLXArray.zeros([1, 2, 4, 4], dtype: .float32), advancedBy: 1)
        eval(cache.turnCheckpoint!.arrays)

        // Metal releases a finished command buffer's inputs on its completion thread, a moment
        // after `eval` returns: poll for the settled figure instead of reading it once.
        var retained = activeBytes() - base
        for _ in 0 ..< 40 where retained >= chunkBytes / 8 {
            Thread.sleep(forTimeInterval: 0.05)
            retained = activeBytes() - base
        }
        XCTAssertLessThan(
            retained, chunkBytes / 8,
            "the checkpoint pins \(retained) bytes; its window is \(cache.turnCheckpoint!.byteCount)"
        )
    }
}
