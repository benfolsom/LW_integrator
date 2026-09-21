// Study-only benchmark for mixed-precision Metal retarded-root proposals.
//
// The production integrator is not connected to this executable.  A Metal
// thread proposes one float32 bracket and root per observer event.  The host
// certifies each bracket with the original float64 endpoint residuals, falls
// back to a complete float64 binary search when needed, and executes the
// strict float64 root solve on the CPU.  GPU root proposals are diagnostic and
// never become accepted physical results directly.

import Foundation
import Metal

#if !arch(arm64)
fputs("The explicit Metal root study requires Darwin arm64.\n", stderr)
exit(2)
#endif

let cMMNS = 299.792458
let bundleMagic = Data("LWMTLR02".utf8)

struct Parameters {
    var knotCount: UInt32
    var eventCount: UInt32
    var proposalIterations: UInt32
    var padding: UInt32 = 0
}

struct Bundle {
    let knotCount: Int
    let eventCount: Int
    let rootToleranceMM: Double
    let maxRootIterations: Int
    let sourceTimes: [Double]
    let sourcePositions: [Double]
    let segmentDurations: [Double]
    let positionCoefficients: [Double]
    let observerTimes: [Double]
    let observerPositions: [Double]
    let expectedStatus: [Int64]
    let expectedRetardedTimes: [Double]
}

struct RootResult {
    let status: Int32
    let segment: Int32
    let timeNS: Double
}

struct SampleDouble {
    let x: Double
    let y: Double
    let z: Double
    let betaX: Double
    let betaY: Double
    let betaZ: Double
}

struct HybridResult {
    let roots: [RootResult]
    let acceptedProposals: Int
    let cpuFallbacks: Int
}

final class BundleCursor {
    private let data: Data
    private var offset: Int

    init(data: Data, offset: Int) {
        self.data = data
        self.offset = offset
    }

    private func require(_ byteCount: Int) throws -> Range<Int> {
        let end = offset + byteCount
        guard byteCount >= 0 && end <= data.count else {
            throw NSError(
                domain: "MetalRetardedRoots",
                code: 10,
                userInfo: [NSLocalizedDescriptionKey: "truncated benchmark bundle"]
            )
        }
        let range = offset..<end
        offset = end
        return range
    }

    func readUInt64() throws -> UInt64 {
        let range = try require(MemoryLayout<UInt64>.stride)
        return data.withUnsafeBytes { bytes in
            UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: range.lowerBound, as: UInt64.self))
        }
    }

    func readDouble() throws -> Double {
        return Double(bitPattern: try readUInt64())
    }

    func readDoubles(_ count: Int) throws -> [Double] {
        let byteCount = count * MemoryLayout<Double>.stride
        let range = try require(byteCount)
        var result = [Double](repeating: 0.0, count: count)
        result.withUnsafeMutableBytes { destination in
            _ = data.copyBytes(to: destination, from: range)
        }
        return result
    }

    func readInt64s(_ count: Int) throws -> [Int64] {
        let byteCount = count * MemoryLayout<Int64>.stride
        let range = try require(byteCount)
        var result = [Int64](repeating: 0, count: count)
        result.withUnsafeMutableBytes { destination in
            _ = data.copyBytes(to: destination, from: range)
        }
        return result
    }

    var isAtEnd: Bool { offset == data.count }
}

func loadBundle(_ path: String) throws -> Bundle {
    let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
    guard data.count >= bundleMagic.count && data.prefix(bundleMagic.count) == bundleMagic else {
        throw NSError(
            domain: "MetalRetardedRoots",
            code: 11,
            userInfo: [NSLocalizedDescriptionKey: "unrecognized benchmark bundle"]
        )
    }
    let cursor = BundleCursor(data: data, offset: bundleMagic.count)
    let knotCount = Int(try cursor.readUInt64())
    let eventCount = Int(try cursor.readUInt64())
    let rootTolerance = try cursor.readDouble()
    let maxIterations = Int(try cursor.readUInt64())
    guard knotCount >= 2 && eventCount >= 1 && maxIterations >= 1 else {
        throw NSError(
            domain: "MetalRetardedRoots",
            code: 12,
            userInfo: [NSLocalizedDescriptionKey: "invalid benchmark bundle dimensions"]
        )
    }
    let bundle = Bundle(
        knotCount: knotCount,
        eventCount: eventCount,
        rootToleranceMM: rootTolerance,
        maxRootIterations: maxIterations,
        sourceTimes: try cursor.readDoubles(knotCount),
        sourcePositions: try cursor.readDoubles(3 * knotCount),
        segmentDurations: try cursor.readDoubles(knotCount - 1),
        positionCoefficients: try cursor.readDoubles((knotCount - 1) * 18),
        observerTimes: try cursor.readDoubles(eventCount),
        observerPositions: try cursor.readDoubles(3 * eventCount),
        expectedStatus: try cursor.readInt64s(eventCount),
        expectedRetardedTimes: try cursor.readDoubles(eventCount)
    )
    guard cursor.isAtEnd else {
        throw NSError(
            domain: "MetalRetardedRoots",
            code: 13,
            userInfo: [NSLocalizedDescriptionKey: "unexpected trailing benchmark data"]
        )
    }
    return bundle
}

let shaderSource = #"""
#include <metal_stdlib>
using namespace metal;

struct Parameters {
    uint knotCount;
    uint eventCount;
    uint proposalIterations;
    uint padding;
};

struct SampleFloat {
    float3 position;
    float3 beta;
};

inline float norm3(float x, float y, float z) {
    return sqrt(x * x + y * y + z * z);
}

inline float knot_residual(
    device const float* sourceTimes,
    device const float* sourcePositions,
    device const float* observerTimes,
    device const float* observerPositions,
    uint eventIndex,
    uint knotIndex
) {
    uint sourceVector = 3 * knotIndex;
    uint observerVector = 3 * eventIndex;
    float dx = observerPositions[observerVector] - sourcePositions[sourceVector];
    float dy = observerPositions[observerVector + 1] - sourcePositions[sourceVector + 1];
    float dz = observerPositions[observerVector + 2] - sourcePositions[sourceVector + 2];
    return 299.792458f * (observerTimes[eventIndex] - sourceTimes[knotIndex])
        - norm3(dx, dy, dz);
}

inline SampleFloat quintic_sample(
    device const float* sourceTimes,
    device const float* segmentDurations,
    device const float* coefficients,
    uint segmentIndex,
    float sampleTime
) {
    float duration = segmentDurations[segmentIndex];
    float tau = clamp(
        (sampleTime - sourceTimes[segmentIndex]) / duration,
        0.0f,
        1.0f
    );
    float tau2 = tau * tau;
    float tau3 = tau2 * tau;
    float tau4 = tau3 * tau;
    float tau5 = tau4 * tau;
    float3 position = float3(0.0f);
    float3 beta = float3(0.0f);
    uint base = segmentIndex * 18;
    for (uint axis = 0; axis < 3; ++axis) {
        float c0 = coefficients[base + axis];
        float c1 = coefficients[base + 3 + axis];
        float c2 = coefficients[base + 6 + axis];
        float c3 = coefficients[base + 9 + axis];
        float c4 = coefficients[base + 12 + axis];
        float c5 = coefficients[base + 15 + axis];
        position[axis] = c0 + c1 * tau + c2 * tau2 + c3 * tau3
            + c4 * tau4 + c5 * tau5;
        beta[axis] = (
            c1 + 2.0f * c2 * tau + 3.0f * c3 * tau2
            + 4.0f * c4 * tau3 + 5.0f * c5 * tau4
        ) / duration / 299.792458f;
    }
    return SampleFloat{position, beta};
}

kernel void propose_retarded_roots(
    device const float* sourceTimes [[buffer(0)]],
    device const float* sourcePositions [[buffer(1)]],
    device const float* segmentDurations [[buffer(2)]],
    device const float* coefficients [[buffer(3)]],
    device const float* observerTimes [[buffer(4)]],
    device const float* observerPositions [[buffer(5)]],
    device int* outputSegments [[buffer(6)]],
    device float* outputTimes [[buffer(7)]],
    device int* outputStatus [[buffer(8)]],
    constant Parameters& parameters [[buffer(9)]],
    uint eventIndex [[thread_position_in_grid]]
) {
    if (eventIndex >= parameters.eventCount) {
        return;
    }
    uint lowerIndex = 0;
    uint upperIndex = parameters.knotCount - 1;
    float lowerResidual = knot_residual(
        sourceTimes, sourcePositions, observerTimes, observerPositions,
        eventIndex, lowerIndex
    );
    float upperResidual = knot_residual(
        sourceTimes, sourcePositions, observerTimes, observerPositions,
        eventIndex, upperIndex
    );
    if (lowerResidual < 0.0f || upperResidual > 0.0f) {
        outputStatus[eventIndex] = 1;
        outputSegments[eventIndex] = -1;
        outputTimes[eventIndex] = NAN;
        return;
    }
    while (upperIndex - lowerIndex > 1) {
        uint middleIndex = lowerIndex + (upperIndex - lowerIndex) / 2;
        float middleResidual = knot_residual(
            sourceTimes, sourcePositions, observerTimes, observerPositions,
            eventIndex, middleIndex
        );
        if (middleResidual >= 0.0f) {
            lowerIndex = middleIndex;
        } else {
            upperIndex = middleIndex;
        }
    }

    float lowerTime = sourceTimes[lowerIndex];
    float upperTime = sourceTimes[lowerIndex + 1];
    float trialTime = 0.5f * (lowerTime + upperTime);
    uint observerVector = 3 * eventIndex;
    for (uint iteration = 0; iteration < parameters.proposalIterations; ++iteration) {
        SampleFloat sample = quintic_sample(
            sourceTimes, segmentDurations, coefficients, lowerIndex, trialTime
        );
        float dx = observerPositions[observerVector] - sample.position.x;
        float dy = observerPositions[observerVector + 1] - sample.position.y;
        float dz = observerPositions[observerVector + 2] - sample.position.z;
        float separation = norm3(dx, dy, dz);
        float residual = 299.792458f * (observerTimes[eventIndex] - trialTime)
            - separation;
        if (residual > 0.0f) {
            lowerTime = trialTime;
        } else {
            upperTime = trialTime;
        }
        float derivative = NAN;
        if (separation > 0.0f) {
            float projection = dx / separation * sample.beta.x
                + dy / separation * sample.beta.y
                + dz / separation * sample.beta.z;
            derivative = -299.792458f * (1.0f - projection);
        }
        float available = residual > 0.0f
            ? upperTime - trialTime
            : trialTime - lowerTime;
        float maximumNewtonResidual = -derivative * available;
        float nextTrial = NAN;
        if (isfinite(derivative) && derivative < 0.0f
            && isfinite(maximumNewtonResidual)
            && fabs(residual) < maximumNewtonResidual) {
            nextTrial = trialTime - residual / derivative;
        }
        if (!isfinite(nextTrial) || nextTrial <= lowerTime || nextTrial >= upperTime) {
            trialTime = 0.5f * (lowerTime + upperTime);
        } else {
            trialTime = nextTrial;
        }
    }
    outputStatus[eventIndex] = 0;
    outputSegments[eventIndex] = int(lowerIndex);
    outputTimes[eventIndex] = 0.5f * (lowerTime + upperTime);
}
"""#

@inline(__always)
func norm3(_ x: Double, _ y: Double, _ z: Double) -> Double {
    return sqrt(x * x + y * y + z * z)
}

@inline(__always)
func knotResidual(_ bundle: Bundle, _ eventIndex: Int, _ knotIndex: Int) -> Double {
    let sourceVector = 3 * knotIndex
    let observerVector = 3 * eventIndex
    let dx = bundle.observerPositions[observerVector] - bundle.sourcePositions[sourceVector]
    let dy = bundle.observerPositions[observerVector + 1] - bundle.sourcePositions[sourceVector + 1]
    let dz = bundle.observerPositions[observerVector + 2] - bundle.sourcePositions[sourceVector + 2]
    return cMMNS * (bundle.observerTimes[eventIndex] - bundle.sourceTimes[knotIndex])
        - norm3(dx, dy, dz)
}

@inline(__always)
func findBracket(_ bundle: Bundle, _ eventIndex: Int) -> Int {
    var lower = 0
    var upper = bundle.knotCount - 1
    if knotResidual(bundle, eventIndex, lower) < 0.0
        || knotResidual(bundle, eventIndex, upper) > 0.0 {
        return -1
    }
    while upper - lower > 1 {
        let middle = lower + (upper - lower) / 2
        if knotResidual(bundle, eventIndex, middle) >= 0.0 {
            lower = middle
        } else {
            upper = middle
        }
    }
    return lower
}

@inline(__always)
func polynomial(
    _ bundle: Bundle,
    _ base: Int,
    _ axis: Int,
    _ tau: Double,
    _ tau2: Double,
    _ tau3: Double,
    _ tau4: Double,
    _ tau5: Double,
    _ duration: Double
) -> (Double, Double) {
    let c0 = bundle.positionCoefficients[base + axis]
    let c1 = bundle.positionCoefficients[base + 3 + axis]
    let c2 = bundle.positionCoefficients[base + 6 + axis]
    let c3 = bundle.positionCoefficients[base + 9 + axis]
    let c4 = bundle.positionCoefficients[base + 12 + axis]
    let c5 = bundle.positionCoefficients[base + 15 + axis]
    let position = c0 + c1 * tau + c2 * tau2 + c3 * tau3
        + c4 * tau4 + c5 * tau5
    let beta = (
        c1 + 2.0 * c2 * tau + 3.0 * c3 * tau2
            + 4.0 * c4 * tau3 + 5.0 * c5 * tau4
    ) / duration / cMMNS
    return (position, beta)
}

@inline(__always)
func quinticSample(_ bundle: Bundle, _ segment: Int, _ timeNS: Double) -> SampleDouble {
    let duration = bundle.segmentDurations[segment]
    let rawTau = (timeNS - bundle.sourceTimes[segment]) / duration
    let tau = min(max(rawTau, 0.0), 1.0)
    let tau2 = tau * tau
    let tau3 = tau2 * tau
    let tau4 = tau3 * tau
    let tau5 = tau4 * tau
    let base = segment * 18
    let x = polynomial(bundle, base, 0, tau, tau2, tau3, tau4, tau5, duration)
    let y = polynomial(bundle, base, 1, tau, tau2, tau3, tau4, tau5, duration)
    let z = polynomial(bundle, base, 2, tau, tau2, tau3, tau4, tau5, duration)
    return SampleDouble(
        x: x.0, y: y.0, z: z.0,
        betaX: x.1, betaY: y.1, betaZ: z.1
    )
}

@inline(__always)
func solveStrictRoot(_ bundle: Bundle, _ eventIndex: Int, _ segment: Int) -> RootResult {
    if segment < 0 {
        return RootResult(status: 1, segment: -1, timeNS: Double.nan)
    }
    var lowerTime = bundle.sourceTimes[segment]
    var upperTime = bundle.sourceTimes[segment + 1]
    var trialTime = 0.5 * (lowerTime + upperTime)
    let observerVector = 3 * eventIndex
    for _ in 0..<bundle.maxRootIterations {
        let sample = quinticSample(bundle, segment, trialTime)
        let dx = bundle.observerPositions[observerVector] - sample.x
        let dy = bundle.observerPositions[observerVector + 1] - sample.y
        let dz = bundle.observerPositions[observerVector + 2] - sample.z
        let separation = norm3(dx, dy, dz)
        let residual = cMMNS * (bundle.observerTimes[eventIndex] - trialTime) - separation
        if abs(residual) <= bundle.rootToleranceMM {
            lowerTime = trialTime
            upperTime = trialTime
            break
        }
        if residual > 0.0 {
            lowerTime = trialTime
        } else {
            upperTime = trialTime
        }
        if lowerTime.nextUp >= upperTime {
            break
        }

        var derivative = Double.nan
        if separation > 0.0 {
            let projection = dx / separation * sample.betaX
                + dy / separation * sample.betaY
                + dz / separation * sample.betaZ
            derivative = -cMMNS * (1.0 - projection)
        }
        let availableTime = residual > 0.0
            ? upperTime - trialTime
            : trialTime - lowerTime
        let maximumNewtonResidual = -derivative * availableTime
        var nextTrial = Double.nan
        if derivative.isFinite && derivative < 0.0
            && maximumNewtonResidual.isFinite
            && abs(residual) < maximumNewtonResidual {
            nextTrial = trialTime - residual / derivative
        }
        if nextTrial == trialTime {
            trialTime = residual > 0.0 ? trialTime.nextUp : trialTime.nextDown
        } else if !nextTrial.isFinite || nextTrial <= lowerTime || nextTrial >= upperTime {
            trialTime = 0.5 * (lowerTime + upperTime)
        } else {
            trialTime = nextTrial
        }
    }
    return RootResult(
        status: 0,
        segment: Int32(segment),
        timeNS: 0.5 * (lowerTime + upperTime)
    )
}

func sourceHistoryIsStrictlyTimelike(_ bundle: Bundle) -> Bool {
    let relativeMargin = 64.0 * Double.ulpOfOne
    for index in 0..<(bundle.knotCount - 1) {
        let first = 3 * index
        let second = first + 3
        let dx = bundle.sourcePositions[second] - bundle.sourcePositions[first]
        let dy = bundle.sourcePositions[second + 1] - bundle.sourcePositions[first + 1]
        let dz = bundle.sourcePositions[second + 2] - bundle.sourcePositions[first + 2]
        let lightDistance = cMMNS * (bundle.sourceTimes[index + 1] - bundle.sourceTimes[index])
        let chordDistance = norm3(dx, dy, dz)
        let margin = relativeMargin * max(abs(lightDistance), abs(chordDistance))
        if !(lightDistance.isFinite && chordDistance + margin < lightDistance) {
            return false
        }
    }
    return true
}

func strictRoots(_ bundle: Bundle, _ count: Int) -> [RootResult] {
    var roots = [RootResult]()
    roots.reserveCapacity(count)
    for eventIndex in 0..<count {
        roots.append(solveStrictRoot(bundle, eventIndex, findBracket(bundle, eventIndex)))
    }
    return roots
}

func hybridRoots(
    _ bundle: Bundle,
    _ count: Int,
    _ proposalSegments: UnsafePointer<Int32>,
    timelikeCertified: Bool
) -> HybridResult {
    var roots = [RootResult]()
    roots.reserveCapacity(count)
    var accepted = 0
    var fallbacks = 0
    for eventIndex in 0..<count {
        let proposal = Int(proposalSegments[eventIndex])
        var segment = -1
        var proposalCertified = false
        if timelikeCertified && proposal >= 0 && proposal + 1 < bundle.knotCount {
            let lowerResidual = knotResidual(bundle, eventIndex, proposal)
            let upperResidual = knotResidual(bundle, eventIndex, proposal + 1)
            proposalCertified = lowerResidual >= 0.0 && upperResidual <= 0.0
                && (upperResidual < 0.0 || proposal + 1 == bundle.knotCount - 1)
        }
        if proposalCertified {
            segment = proposal
            accepted += 1
        } else {
            segment = findBracket(bundle, eventIndex)
            fallbacks += 1
        }
        roots.append(solveStrictRoot(bundle, eventIndex, segment))
    }
    return HybridResult(roots: roots, acceptedProposals: accepted, cpuFallbacks: fallbacks)
}

func rootsBitwiseEqual(_ first: [RootResult], _ second: [RootResult]) -> Bool {
    guard first.count == second.count else { return false }
    for index in first.indices {
        if first[index].status != second[index].status
            || first[index].segment != second[index].segment
            || first[index].timeNS.bitPattern != second[index].timeNS.bitPattern {
            return false
        }
    }
    return true
}

func numbaParity(_ bundle: Bundle, _ roots: [RootResult]) -> [String: Any] {
    var statusMismatches = 0
    var bitwiseTimeMismatches = 0
    var maximumAbsoluteDifference = 0.0
    for eventIndex in roots.indices {
        if Int64(roots[eventIndex].status) != bundle.expectedStatus[eventIndex] {
            statusMismatches += 1
        }
        let expected = bundle.expectedRetardedTimes[eventIndex]
        let actual = roots[eventIndex].timeNS
        if expected.bitPattern != actual.bitPattern {
            bitwiseTimeMismatches += 1
        }
        if expected.isFinite && actual.isFinite {
            maximumAbsoluteDifference = max(maximumAbsoluteDifference, abs(expected - actual))
        }
    }
    return [
        "status_mismatches": statusMismatches,
        "bitwise_retarded_time_mismatches": bitwiseTimeMismatches,
        "maximum_absolute_retarded_time_difference_ns": maximumAbsoluteDifference,
        "bitwise_equal": statusMismatches == 0 && bitwiseTimeMismatches == 0,
    ]
}

func median(_ values: [Double]) -> Double {
    let ordered = values.sorted()
    let middle = ordered.count / 2
    if ordered.count % 2 == 0 {
        return 0.5 * (ordered[middle - 1] + ordered[middle])
    }
    return ordered[middle]
}

func elapsedMS(_ start: UInt64, _ end: UInt64) -> Double {
    return Double(end - start) / 1.0e6
}

func makeBuffer<T>(_ values: [T], _ device: MTLDevice) throws -> MTLBuffer {
    return try values.withUnsafeBytes { bytes in
        guard let base = bytes.baseAddress,
              let buffer = device.makeBuffer(
                bytes: base,
                length: bytes.count,
                options: .storageModeShared
              ) else {
            throw NSError(
                domain: "MetalRetardedRoots",
                code: 20,
                userInfo: [NSLocalizedDescriptionKey: "shared Metal buffer allocation failed"]
            )
        }
        return buffer
    }
}

func copyPrefix<T>(_ values: [T], count: Int, to buffer: MTLBuffer) {
    values.withUnsafeBytes { source in
        let bytes = count * MemoryLayout<T>.stride
        buffer.contents().copyMemory(from: source.baseAddress!, byteCount: bytes)
    }
}

func parseArguments() throws -> (String, [Int], Int) {
    var input: String?
    var counts: [Int]?
    var timingTarget = 5000
    var index = 1
    while index < CommandLine.arguments.count {
        let argument = CommandLine.arguments[index]
        guard index + 1 < CommandLine.arguments.count else {
            throw NSError(
                domain: "MetalRetardedRoots", code: 30,
                userInfo: [NSLocalizedDescriptionKey: "missing value for \(argument)"]
            )
        }
        let value = CommandLine.arguments[index + 1]
        if argument == "--input" {
            input = value
        } else if argument == "--counts" {
            counts = value.split(separator: ",").compactMap { Int($0) }
        } else if argument == "--timing-event-target" {
            timingTarget = Int(value) ?? 0
        } else {
            throw NSError(
                domain: "MetalRetardedRoots", code: 31,
                userInfo: [NSLocalizedDescriptionKey: "unknown argument \(argument)"]
            )
        }
        index += 2
    }
    guard let inputPath = input, let eventCounts = counts,
          !eventCounts.isEmpty, eventCounts.allSatisfy({ $0 > 0 }), timingTarget > 0 else {
        throw NSError(
            domain: "MetalRetardedRoots", code: 32,
            userInfo: [NSLocalizedDescriptionKey: "invalid benchmark arguments"]
        )
    }
    return (inputPath, eventCounts, timingTarget)
}

do {
    let (inputPath, counts, timingEventTarget) = try parseArguments()
    let bundle = try loadBundle(inputPath)
    guard let maximumCount = counts.max(), maximumCount <= bundle.eventCount else {
        throw NSError(
            domain: "MetalRetardedRoots", code: 33,
            userInfo: [NSLocalizedDescriptionKey: "requested event count exceeds bundle"]
        )
    }
    guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory else {
        throw NSError(
            domain: "MetalRetardedRoots", code: 34,
            userInfo: [NSLocalizedDescriptionKey: "a unified-memory Metal device is required"]
        )
    }
    guard let queue = device.makeCommandQueue() else {
        throw NSError(
            domain: "MetalRetardedRoots", code: 35,
            userInfo: [NSLocalizedDescriptionKey: "Metal command queue creation failed"]
        )
    }

    let compileOptions = MTLCompileOptions()
    compileOptions.mathMode = .safe
    let libraryStart = DispatchTime.now().uptimeNanoseconds
    let library = try device.makeLibrary(source: shaderSource, options: compileOptions)
    let libraryEnd = DispatchTime.now().uptimeNanoseconds
    guard let function = library.makeFunction(name: "propose_retarded_roots") else {
        throw NSError(
            domain: "MetalRetardedRoots", code: 36,
            userInfo: [NSLocalizedDescriptionKey: "Metal root function is missing"]
        )
    }
    let pipelineStart = DispatchTime.now().uptimeNanoseconds
    let pipeline = try device.makeComputePipelineState(function: function)
    let pipelineEnd = DispatchTime.now().uptimeNanoseconds

    let conversionStart = DispatchTime.now().uptimeNanoseconds
    let sourceTimesFloat = bundle.sourceTimes.map(Float.init)
    let sourcePositionsFloat = bundle.sourcePositions.map(Float.init)
    let segmentDurationsFloat = bundle.segmentDurations.map(Float.init)
    let coefficientsFloat = bundle.positionCoefficients.map(Float.init)
    let observerTimesFloat = bundle.observerTimes.map(Float.init)
    let observerPositionsFloat = bundle.observerPositions.map(Float.init)
    let conversionEnd = DispatchTime.now().uptimeNanoseconds

    let bufferStart = DispatchTime.now().uptimeNanoseconds
    let sourceTimesBuffer = try makeBuffer(sourceTimesFloat, device)
    let sourcePositionsBuffer = try makeBuffer(sourcePositionsFloat, device)
    let segmentDurationsBuffer = try makeBuffer(segmentDurationsFloat, device)
    let coefficientsBuffer = try makeBuffer(coefficientsFloat, device)
    let observerTimesBuffer = try makeBuffer(observerTimesFloat, device)
    let observerPositionsBuffer = try makeBuffer(observerPositionsFloat, device)
    guard let outputSegmentsBuffer = device.makeBuffer(
            length: maximumCount * MemoryLayout<Int32>.stride,
            options: .storageModeShared
          ),
          let outputTimesBuffer = device.makeBuffer(
            length: maximumCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
          ),
          let outputStatusBuffer = device.makeBuffer(
            length: maximumCount * MemoryLayout<Int32>.stride,
            options: .storageModeShared
          ) else {
        throw NSError(
            domain: "MetalRetardedRoots", code: 37,
            userInfo: [NSLocalizedDescriptionKey: "Metal output buffer allocation failed"]
        )
    }
    let bufferEnd = DispatchTime.now().uptimeNanoseconds
    let mutableProposalSegments = outputSegmentsBuffer.contents().bindMemory(
        to: Int32.self,
        capacity: maximumCount
    )
    let proposalSegments = UnsafePointer(mutableProposalSegments)
    let proposalTimes = UnsafePointer(
        outputTimesBuffer.contents().bindMemory(to: Float.self, capacity: maximumCount)
    )

    func dispatch(
        _ count: Int,
        copyObservers: Bool,
        proposalIterations: UInt32 = 0
    ) throws {
        if copyObservers {
            copyPrefix(observerTimesFloat, count: count, to: observerTimesBuffer)
            copyPrefix(observerPositionsFloat, count: 3 * count, to: observerPositionsBuffer)
        }
        var parameters = Parameters(
            knotCount: UInt32(bundle.knotCount),
            eventCount: UInt32(count),
            proposalIterations: proposalIterations
        )
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NSError(
                domain: "MetalRetardedRoots", code: 38,
                userInfo: [NSLocalizedDescriptionKey: "Metal command encoding failed"]
            )
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(sourceTimesBuffer, offset: 0, index: 0)
        encoder.setBuffer(sourcePositionsBuffer, offset: 0, index: 1)
        encoder.setBuffer(segmentDurationsBuffer, offset: 0, index: 2)
        encoder.setBuffer(coefficientsBuffer, offset: 0, index: 3)
        encoder.setBuffer(observerTimesBuffer, offset: 0, index: 4)
        encoder.setBuffer(observerPositionsBuffer, offset: 0, index: 5)
        encoder.setBuffer(outputSegmentsBuffer, offset: 0, index: 6)
        encoder.setBuffer(outputTimesBuffer, offset: 0, index: 7)
        encoder.setBuffer(outputStatusBuffer, offset: 0, index: 8)
        encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 9)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            throw commandBuffer.error ?? NSError(
                domain: "MetalRetardedRoots", code: 39,
                userInfo: [NSLocalizedDescriptionKey: "Metal command failed"]
            )
        }
    }

    let timelikeCertified = sourceHistoryIsStrictlyTimelike(bundle)
    let selfTestCount = min(129, maximumCount)
    try dispatch(selfTestCount, copyObservers: true)
    let selfTestStrict = strictRoots(bundle, selfTestCount)
    let selfTestHybrid = hybridRoots(
        bundle,
        selfTestCount,
        proposalSegments,
        timelikeCertified: timelikeCertified
    )
    let selfTestPassed = timelikeCertified
        && rootsBitwiseEqual(selfTestStrict, selfTestHybrid.roots)
    guard selfTestPassed else {
        throw NSError(
            domain: "MetalRetardedRoots", code: 40,
            userInfo: [NSLocalizedDescriptionKey: "Metal startup self-test failed"]
        )
    }
    // Exercise the safety path independently of whether this smooth capture
    // snapshot happens to expose a natural float32 bracket disagreement.
    let originalProposal = mutableProposalSegments[0]
    mutableProposalSegments[0] = originalProposal + 1
    let injectedFallback = hybridRoots(
        bundle,
        1,
        proposalSegments,
        timelikeCertified: timelikeCertified
    )
    let injectedFallbackPassed = injectedFallback.cpuFallbacks == 1
        && rootsBitwiseEqual(Array(selfTestStrict.prefix(1)), injectedFallback.roots)
    mutableProposalSegments[0] = originalProposal
    guard injectedFallbackPassed else {
        throw NSError(
            domain: "MetalRetardedRoots", code: 41,
            userInfo: [NSLocalizedDescriptionKey: "float64 fallback self-test failed"]
        )
    }

    var scenarioReports = [[String: Any]]()
    for count in counts {
        let repeats = max(5, min(41, Int(ceil(Double(timingEventTarget) / Double(count)))))
        for _ in 0..<3 {
            try dispatch(count, copyObservers: true)
            _ = hybridRoots(bundle, count, proposalSegments, timelikeCertified: timelikeCertified)
            _ = strictRoots(bundle, count)
        }
        try dispatch(count, copyObservers: true)
        let strictReference = strictRoots(bundle, count)
        let hybridReference = hybridRoots(
            bundle, count, proposalSegments, timelikeCertified: timelikeCertified
        )
        // Root proposals are inventoried separately.  The exact hybrid needs
        // only a bracket proposal and should not pay for unused float32 root
        // iterations.
        try dispatch(count, copyObservers: true, proposalIterations: 8)
        var proposalSegmentMismatches = 0
        var maximumProposalRootErrorNS = 0.0
        for eventIndex in 0..<count {
            if proposalSegments[eventIndex] != strictReference[eventIndex].segment {
                proposalSegmentMismatches += 1
            }
            let proposal = Double(proposalTimes[eventIndex])
            let exact = strictReference[eventIndex].timeNS
            if proposal.isFinite && exact.isFinite {
                maximumProposalRootErrorNS = max(
                    maximumProposalRootErrorNS,
                    abs(proposal - exact)
                )
            }
        }

        var dispatchTimes = [Double]()
        var rootProposalDispatchTimes = [Double]()
        var transferDispatchTimes = [Double]()
        var hybridTimes = [Double]()
        var strictTimes = [Double]()
        for _ in 0..<repeats {
            var start = DispatchTime.now().uptimeNanoseconds
            try dispatch(count, copyObservers: false)
            var end = DispatchTime.now().uptimeNanoseconds
            dispatchTimes.append(elapsedMS(start, end))

            start = DispatchTime.now().uptimeNanoseconds
            try dispatch(count, copyObservers: false, proposalIterations: 8)
            end = DispatchTime.now().uptimeNanoseconds
            rootProposalDispatchTimes.append(elapsedMS(start, end))

            start = DispatchTime.now().uptimeNanoseconds
            try dispatch(count, copyObservers: true)
            end = DispatchTime.now().uptimeNanoseconds
            transferDispatchTimes.append(elapsedMS(start, end))

            start = DispatchTime.now().uptimeNanoseconds
            try dispatch(count, copyObservers: true)
            _ = hybridRoots(bundle, count, proposalSegments, timelikeCertified: timelikeCertified)
            end = DispatchTime.now().uptimeNanoseconds
            hybridTimes.append(elapsedMS(start, end))

            start = DispatchTime.now().uptimeNanoseconds
            _ = strictRoots(bundle, count)
            end = DispatchTime.now().uptimeNanoseconds
            strictTimes.append(elapsedMS(start, end))
        }
        let hybridMedian = median(hybridTimes)
        let strictMedian = median(strictTimes)
        scenarioReports.append([
            "events": count,
            "history_knots": bundle.knotCount,
            "timing_repeats": repeats,
            "metal_dispatch_wait_median_ms": median(dispatchTimes),
            "float32_root_proposal_dispatch_wait_median_ms": median(
                rootProposalDispatchTimes
            ),
            "observer_transfer_dispatch_wait_median_ms": median(transferDispatchTimes),
            "certified_hybrid_median_ms": hybridMedian,
            "strict_swift_cpu_median_ms": strictMedian,
            "strict_swift_cpu_over_hybrid": strictMedian / hybridMedian,
            "accepted_float32_bracket_proposals": hybridReference.acceptedProposals,
            "float64_binary_search_fallbacks": hybridReference.cpuFallbacks,
            "float32_segment_mismatches_before_certification": proposalSegmentMismatches,
            "maximum_float32_root_proposal_error_ns": maximumProposalRootErrorNS,
            "hybrid_vs_strict_swift_bitwise_equal": rootsBitwiseEqual(
                strictReference, hybridReference.roots
            ),
            "strict_swift_vs_numba": numbaParity(bundle, strictReference),
        ])
    }

    let report: [String: Any] = [
        "device": [
            "name": device.name,
            "has_unified_memory": device.hasUnifiedMemory,
            "recommended_max_working_set_bytes": device.recommendedMaxWorkingSetSize,
            "operating_system": ProcessInfo.processInfo.operatingSystemVersionString,
        ],
        "capability_gate": [
            "required_platform": "Darwin arm64",
            "explicit_selection_only": true,
            "startup_self_test_passed": selfTestPassed,
            "source_history_strictly_timelike_float64": timelikeCertified,
            "self_test_events": selfTestCount,
            "self_test_cpu_fallbacks": selfTestHybrid.cpuFallbacks,
            "injected_bad_proposal_fallback_passed": injectedFallbackPassed,
        ],
        "numeric_contract": [
            "metal_type": "float32 proposals only",
            "metal_math_mode": "safe",
            "float64_endpoint_certification": true,
            "float64_complete_binary_search_fallback": true,
            "accepted_root_solver": "strict float64 CPU",
            "metal_root_proposal_is_authoritative": false,
        ],
        "startup": [
            "runtime_metal_library_compile_ms": elapsedMS(libraryStart, libraryEnd),
            "compute_pipeline_compile_ms": elapsedMS(pipelineStart, pipelineEnd),
            "float64_to_float32_history_conversion_ms": elapsedMS(conversionStart, conversionEnd),
            "persistent_shared_buffer_setup_ms": elapsedMS(bufferStart, bufferEnd),
        ],
        "scenarios": scenarioReports,
    ]
    let json = try JSONSerialization.data(
        withJSONObject: report,
        options: [.prettyPrinted, .sortedKeys]
    )
    FileHandle.standardOutput.write(json)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fputs("Metal retarded-root benchmark failed: \(error)\n", stderr)
    exit(1)
}
