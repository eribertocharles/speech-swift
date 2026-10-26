import MLX
import MLXCommon
import MLXFast
import MLXNN

// MARK: - KV Cache Protocol

public protocol KVCache: AnyObject {
    var offset: Int { get }
    var keysArray: MLXArray? { get }
    var valuesArray: MLXArray? { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func replaceArrays(keys: MLXArray, values: MLXArray)
    func trim(_ count: Int)
}

// MARK: - Simple KV Cache (concatenation-based)

public final class KVCacheSimple: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public init() {}

    public var offset: Int { keys?.shape[2] ?? 0 }

    /// Read-only access to cache arrays (for compiled step functions).
    public var keysArray: MLXArray? { keys }
    public var valuesArray: MLXArray? { values }

    /// Replace cache arrays wholesale (for compiled step functions that return new arrays).
    public func replaceArrays(keys newK: MLXArray, values newV: MLXArray) {
        keys = newK
        values = newV
    }

    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        if let k = keys, let v = values {
            keys = concatenated([k, newK], axis: 2)
            values = concatenated([v, newV], axis: 2)
        } else {
            keys = newK
            values = newV
        }
        return (keys!, values!)
    }

    public func trim(_ count: Int) {
        if count >= offset {
            keys = nil
            values = nil
        } else if count > 0 {
            let newLen = offset - count
            if let k = keys, let v = values {
                keys = split(k, indices: [newLen], axis: 2)[0]
                values = split(v, indices: [newLen], axis: 2)[0]
            }
        }
    }
}

// MARK: - Pre-allocated KV Cache (fixed-capacity, O(1) per step)

/// Pre-allocates a fixed-size buffer on first use and writes new KV pairs
/// at the current offset via scatter. Avoids the O(n) concatenation cost
/// of KVCacheSimple, reducing total decode cost from O(n²) to O(n).
public final class KVCachePreAllocated: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private let capacity: Int
    private var currentOffset: Int = 0

    public init(capacity: Int) {
        self.capacity = capacity
    }

    public var offset: Int { currentOffset }

    public var keysArray: MLXArray? { keys }
    public var valuesArray: MLXArray? { values }

    public func replaceArrays(keys newK: MLXArray, values newV: MLXArray) {
        keys = newK
        values = newV
    }

    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        let t = newK.shape[2]

        if keys == nil {
            let b = newK.shape[0], h = newK.shape[1], d = newK.shape[3]
            keys = MLXArray.zeros([b, h, capacity, d], dtype: newK.dtype)
            values = MLXArray.zeros([b, h, capacity, d], dtype: newV.dtype)
        }

        // Scatter write at current position — O(1), no allocation
        let end = min(currentOffset + t, capacity)
        keys![0..., 0..., currentOffset..<end, 0...] = newK
        values![0..., 0..., currentOffset..<end, 0...] = newV
        currentOffset = end

        // Return valid slice (view, no copy)
        let k = keys![0..., 0..., 0..<currentOffset, 0...]
        let v = values![0..., 0..., 0..<currentOffset, 0...]
        return (k, v)
    }

    public func trim(_ count: Int) {
        if count >= currentOffset {
            keys = nil
            values = nil
            currentOffset = 0
        } else if count > 0 {
            currentOffset -= count
            // Shift remaining data to front
            if let k = keys, let v = values {
                keys = k[0..., 0..., count..<(currentOffset + count), 0...]
                values = v[0..., 0..., count..<(currentOffset + count), 0...]
            }
        }
    }
}

// MARK: - Ring KV Cache (fixed-size circular buffer for bounded context)

/// Fixed-capacity circular KV cache: once full, each new entry overwrites the oldest.
///
/// Mirrors `RingKVCache` in the PyTorch Moshi/PersonaPlex reference: memory and per-step
/// cost stay constant however long a stream runs, and attention covers at most the last
/// `capacity` positions. Keys are stored with RoPE already applied at their absolute
/// positions, so attention is correct regardless of where an entry sits in the ring.
///
/// Two ways to use it:
/// - `update(keys:values:)` writes entries and returns the valid ones in chronological
///   order (old entries first), for uncompiled and batched-prefill attention.
/// - Compiled single-step attention reads the full buffers (`keysArray`/`valuesArray`),
///   writes the new entry at `writeIndex`, masks unwritten slots with `stepMask(dtype:heads:)`,
///   then hands the updated buffers back via `replaceArrays` and calls `advance()`.
public final class RingKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    public let capacity: Int
    private var totalWritten: Int = 0
    private var fullMask: MLXArray?

    public init(capacity: Int) {
        precondition(capacity > 0, "RingKVCache capacity must be positive")
        self.capacity = capacity
    }

    /// Total entries ever written (the stream position), not the number retained.
    public var offset: Int { totalWritten }
    public var keysArray: MLXArray? { keys }
    public var valuesArray: MLXArray? { values }

    /// Number of valid entries currently retained.
    public var length: Int { min(totalWritten, capacity) }

    /// Ring slot the next entry will be written to.
    public var writeIndex: Int { totalWritten % capacity }

    public func replaceArrays(keys newK: MLXArray, values newV: MLXArray) {
        keys = newK
        values = newV
    }

    /// Records that one entry was written at `writeIndex` outside `update` (compiled path).
    public func advance() {
        totalWritten += 1
    }

    /// Additive attention mask over the full buffer for a single query whose own entry has
    /// just been written at `writeIndex`: 0 for written slots, a large negative value for
    /// slots not yet written. Shape `[1, heads, 1, capacity]`, the exact shape of the
    /// attention scores: a mask that needs broadcasting would bake the capacity into a
    /// shapeless-compiled graph, breaking later calls with a different capacity.
    public func stepMask(dtype: DType, heads: Int) -> MLXArray {
        let valid = min(totalWritten + 1, capacity)
        if valid == capacity, let full = fullMask, full.dtype == dtype, full.dim(1) == heads {
            return full  // every slot is written from here on
        }
        var mask = [Float](repeating: -1e9, count: capacity)
        for i in 0..<valid { mask[i] = 0 }
        let row = MLXArray(mask).reshaped([1, 1, 1, capacity]).asType(dtype)
        let result = broadcast(row, to: [1, heads, 1, capacity])
        eval(result)
        if valid == capacity { fullMask = result }
        return result
    }

    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        let t = newK.shape[2]

        if keys == nil {
            let b = newK.shape[0], h = newK.shape[1], d = newK.shape[3]
            keys = MLXArray.zeros([b, h, capacity, d], dtype: newK.dtype)
            values = MLXArray.zeros([b, h, capacity, d], dtype: newV.dtype)
        }

        // Attention covers the most recent `capacity` positions, the new entries included,
        // as in the reference (write first, then attend over the ring). Older retained
        // entries that the new ones displace are left out.
        let attendK: MLXArray
        let attendV: MLXArray
        let keptPast = max(0, capacity - t)
        if keptPast > 0, let (pastK, pastV) = orderedEntries(last: keptPast) {
            attendK = concatenated([pastK, newK], axis: 2)
            attendV = concatenated([pastV, newV], axis: 2)
        } else {
            attendK = newK
            attendV = newV
        }

        // Store the newest `capacity` entries, in at most two contiguous runs per wrap.
        var source = max(0, t - capacity)
        var position = (totalWritten + source) % capacity
        while source < t {
            let run = min(t - source, capacity - position)
            keys![0..., 0..., position..<(position + run), 0...] = newK[0..., 0..., source..<(source + run), 0...]
            values![0..., 0..., position..<(position + run), 0...] = newV[0..., 0..., source..<(source + run), 0...]
            source += run
            position = (position + run) % capacity
        }
        totalWritten += t

        return (attendK, attendV)
    }

    /// The newest `last` retained entries (fewer if fewer exist), oldest first; nil when empty.
    private func orderedEntries(last: Int) -> (MLXArray, MLXArray)? {
        guard let k = keys, let v = values else { return nil }
        let count = min(last, length)
        guard count > 0 else { return nil }
        // The newest entry sits just before `writeIndex`; walk back `count` slots.
        let start = (totalWritten - count) % capacity
        if start + count <= capacity {
            return (k[0..., 0..., start..<(start + count), 0...], v[0..., 0..., start..<(start + count), 0...])
        }
        let tail = capacity - start
        return (
            concatenated([k[0..., 0..., start..<capacity, 0...], k[0..., 0..., 0..<(count - tail), 0...]], axis: 2),
            concatenated([v[0..., 0..., start..<capacity, 0...], v[0..., 0..., 0..<(count - tail), 0...]], axis: 2)
        )
    }

    /// Only a full reset is supported: entries older than the ring can't be restored.
    public func trim(_ count: Int) {
        precondition(count >= totalWritten, "RingKVCache supports only a full reset")
        keys = nil
        values = nil
        totalWritten = 0
    }
}

// MARK: - Attention Helpers

public func createAttentionMask(h: MLXArray, cache: (any KVCache)?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let T = h.shape[1]
    if T <= 1, (cache?.offset ?? 0) > 0 {
        return .none
    }

    let offset = cache?.offset ?? 0
    let totalLen = offset + T
    let dtype = h.dtype

    if offset > 0 {
        let causal = MLXArray.tri(T, m: totalLen, k: offset, type: Float.self) * 1e9 - 1e9
        return .array(causal.reshaped([1, 1, T, totalLen]).asType(dtype))
    }

    let causal = MLXArray.tri(T, m: T, k: 0, type: Float.self) * 1e9 - 1e9
    return .array(causal.reshaped([1, 1, T, T]).asType(dtype))
}

// MARK: - Linear/QuantizedLinear Helper

@inline(__always)
public func applyLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
    switch module {
    case let l as Linear: return l(x)
    case let q as QuantizedLinear: return q(x)
    default: fatalError("Expected Linear or QuantizedLinear, got \(type(of: module))")
    }
}

public func makeLinear(_ inputDims: Int, _ outputDims: Int, bias: Bool, groupSize: Int? = nil, bits: Int? = nil) -> Module {
    if let gs = groupSize, let b = bits, b < 16 {
        return QuantizedLinear.preQuantized(inputDims, outputDims, bias: bias, groupSize: gs, bits: b)
    }
    return Linear(inputDims, outputDims, bias: bias)
}
