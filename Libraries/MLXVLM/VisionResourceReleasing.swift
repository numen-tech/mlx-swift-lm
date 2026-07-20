// Copyright © 2026 Numen Technologies

import Foundation

/// A VLM model that can release its vision-tower weights back to a lazy, near-zero-resident
/// state and later reload them from a checkpoint directory.
///
/// ### Why
/// A vision tower's weights materialize into dirty, non-evictable buffers the first time a
/// vision forward pass runs (dtype-cast copies of the mmap-backed checkpoint arrays). In a
/// 27B-class on-device deployment this floor measured ~549 MB above the text-only baseline
/// after the *first* image turn (numen-tech/gemma4-qat task QA-9) — and it never comes back
/// down on its own, because the module keeps strong references to the materialized arrays
/// for as long as the model is loaded, while a chat session may go arbitrarily long between
/// image turns (or never see another one).
///
/// A conforming model exposes a narrow, host-driven release point: once a turn's vision
/// encode has produced its merged embeddings, the host (e.g. NumenKit's engine) can call
/// ``releaseVisionResources(reloadingFrom:)`` to drop the materialized buffers. The next
/// vision forward pass lazily re-materializes them from the checkpoint (mmap + cast, ~1-2 s
/// on the reference device) — an accepted, once-per-image-turn cost paid in exchange for
/// giving the memory back on every turn in between.
///
/// ### Contract
/// - **Atomic-or-throw.** Conforming implementations validate the replacement weights
///   completely (every current vision-tower parameter has a matching, same-shaped
///   checkpoint key, and the checkpoint isn't quantized in a way the release path can't
///   safely reproduce) *before* mutating any model state. A thrown error leaves the model
///   exactly as it was — a failed release must never leave the vision tower half-updated.
/// - **Lazy.** The reload must not force evaluation. It swaps in mmap-backed lazy arrays so
///   memory stays near-zero-resident until the next vision forward triggers `eval`.
/// - **An optimization, not a correctness requirement.** Callers should treat a thrown
///   error as "skip the release, the turn is otherwise fine" rather than aborting a working
///   generation.
public protocol VisionResourceReleasing {
    /// Release this model's materialized vision-tower weights and reload them lazily from
    /// `modelDirectory`'s checkpoint files, so the tower's memory footprint returns to
    /// ~zero until the next vision forward pass re-materializes it on demand.
    ///
    /// - Parameter modelDirectory: the directory this model was originally loaded from
    ///   (containing its `*.safetensors` weight files); the reload re-reads from disk.
    /// - Throws: if the checkpoint can't be reconciled with the current vision-tower
    ///   structure (missing/mismatched keys, or a quantized vision tower — unsupported by
    ///   this path). On throw, the model's vision tower is left untouched.
    func releaseVisionResources(reloadingFrom modelDirectory: URL) throws
}
