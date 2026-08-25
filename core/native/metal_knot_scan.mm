#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>

static const char *kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Parameters {
    uint knotCount;
    uint sourceCount;
    uint eventCount;
    uint padding;
};

inline float residual_at(
    device const float *sourceTimes,
    device const float *sourcePositions,
    device const float *observerTimes,
    device const float *observerPositions,
    uint sourceCount,
    uint eventIndex,
    uint sourceIndex,
    uint knotIndex
) {
    uint sourceScalar = knotIndex * sourceCount + sourceIndex;
    uint sourceVector = 3 * sourceScalar;
    uint observerVector = 3 * eventIndex;
    float dx = observerPositions[observerVector] - sourcePositions[sourceVector];
    float dy = observerPositions[observerVector + 1] - sourcePositions[sourceVector + 1];
    float dz = observerPositions[observerVector + 2] - sourcePositions[sourceVector + 2];
    float separation = sqrt(dx * dx + dy * dy + dz * dz);
    return 299.792458f * (observerTimes[eventIndex] - sourceTimes[sourceScalar])
        - separation;
}

kernel void propose_segments(
    device const float *sourceTimes [[buffer(0)]],
    device const float *sourcePositions [[buffer(1)]],
    device const int *aliveCounts [[buffer(2)]],
    device const float *observerTimes [[buffer(3)]],
    device const float *observerPositions [[buffer(4)]],
    device int *outputSegments [[buffer(5)]],
    constant Parameters &parameters [[buffer(6)]],
    uint pairIndex [[thread_position_in_grid]]
) {
    uint pairCount = parameters.eventCount * parameters.sourceCount;
    if (pairIndex >= pairCount) {
        return;
    }
    uint eventIndex = pairIndex / parameters.sourceCount;
    uint sourceIndex = pairIndex - eventIndex * parameters.sourceCount;
    int alive = aliveCounts[sourceIndex];
    if (alive < 2) {
        outputSegments[pairIndex] = -1;
        return;
    }
    uint lower = 0;
    uint upper = uint(alive - 1);
    float lowerResidual = residual_at(
        sourceTimes, sourcePositions, observerTimes, observerPositions,
        parameters.sourceCount, eventIndex, sourceIndex, lower
    );
    float upperResidual = residual_at(
        sourceTimes, sourcePositions, observerTimes, observerPositions,
        parameters.sourceCount, eventIndex, sourceIndex, upper
    );
    if (lowerResidual < 0.0f || upperResidual > 0.0f) {
        outputSegments[pairIndex] = -1;
        return;
    }
    while (upper - lower > 1) {
        uint middle = lower + (upper - lower) / 2;
        float middleResidual = residual_at(
            sourceTimes, sourcePositions, observerTimes, observerPositions,
            parameters.sourceCount, eventIndex, sourceIndex, middle
        );
        if (middleResidual >= 0.0f) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    outputSegments[pairIndex] = int(lower);
}
)METAL";

struct Parameters {
    uint32_t knotCount;
    uint32_t sourceCount;
    uint32_t eventCount;
    uint32_t padding;
};

static void copy_error(char *buffer, size_t capacity, NSString *message) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }
    const char *utf8 = message == nil ? "unknown Metal error" : message.UTF8String;
    std::strncpy(buffer, utf8, capacity - 1);
    buffer[capacity - 1] = '\0';
}

@interface LWMetalKnotScanContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@end

@implementation LWMetalKnotScanContext
@end

extern "C" void *lw_metal_knot_scan_create(char *errorBuffer, size_t errorCapacity) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil || !device.hasUnifiedMemory) {
            copy_error(errorBuffer, errorCapacity, @"a unified-memory Metal device is required");
            return nullptr;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            copy_error(errorBuffer, errorCapacity, @"Metal command queue creation failed");
            return nullptr;
        }
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = NO;
        NSError *libraryError = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(kShaderSource)
                                                      options:options
                                                        error:&libraryError];
        if (library == nil) {
            copy_error(errorBuffer, errorCapacity, libraryError.localizedDescription);
            return nullptr;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"propose_segments"];
        if (function == nil) {
            copy_error(errorBuffer, errorCapacity, @"Metal proposal function is missing");
            return nullptr;
        }
        NSError *pipelineError = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&pipelineError];
        if (pipeline == nil) {
            copy_error(errorBuffer, errorCapacity, pipelineError.localizedDescription);
            return nullptr;
        }
        LWMetalKnotScanContext *context = [[LWMetalKnotScanContext alloc] init];
        context.device = device;
        context.queue = queue;
        context.pipeline = pipeline;
        return (__bridge_retained void *)context;
    }
}

extern "C" void lw_metal_knot_scan_destroy(void *opaqueContext) {
    if (opaqueContext != nullptr) {
        CFBridgingRelease(opaqueContext);
    }
}

extern "C" const char *lw_metal_knot_scan_device_name(void *opaqueContext) {
    if (opaqueContext == nullptr) {
        return "unavailable";
    }
    LWMetalKnotScanContext *context = (__bridge LWMetalKnotScanContext *)opaqueContext;
    return context.device.name.UTF8String;
}

extern "C" int lw_metal_knot_scan_candidates(
    void *opaqueContext,
    const float *observerTimes,
    const float *observerPositions,
    uint32_t eventCount,
    const float *sourceTimes,
    const float *sourcePositions,
    uint32_t knotCount,
    uint32_t sourceCount,
    const int32_t *aliveCounts,
    int32_t *outputSegments,
    char *errorBuffer,
    size_t errorCapacity
) {
    @autoreleasepool {
        if (opaqueContext == nullptr || eventCount == 0 || sourceCount == 0 || knotCount == 0) {
            copy_error(errorBuffer, errorCapacity, @"invalid Metal knot-scan arguments");
            return 1;
        }
        LWMetalKnotScanContext *context = (__bridge LWMetalKnotScanContext *)opaqueContext;
        NSUInteger pairCount = NSUInteger(eventCount) * NSUInteger(sourceCount);
        id<MTLBuffer> sourceTimesBuffer = [context.device
            newBufferWithBytes:sourceTimes
                        length:NSUInteger(knotCount) * NSUInteger(sourceCount) * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> sourcePositionsBuffer = [context.device
            newBufferWithBytes:sourcePositions
                        length:3 * NSUInteger(knotCount) * NSUInteger(sourceCount) * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> aliveCountsBuffer = [context.device
            newBufferWithBytes:aliveCounts
                        length:NSUInteger(sourceCount) * sizeof(int32_t)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> observerTimesBuffer = [context.device
            newBufferWithBytes:observerTimes
                        length:NSUInteger(eventCount) * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> observerPositionsBuffer = [context.device
            newBufferWithBytes:observerPositions
                        length:3 * NSUInteger(eventCount) * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [context.device
            newBufferWithLength:pairCount * sizeof(int32_t)
                        options:MTLResourceStorageModeShared];
        if (sourceTimesBuffer == nil || sourcePositionsBuffer == nil || aliveCountsBuffer == nil
            || observerTimesBuffer == nil || observerPositionsBuffer == nil || outputBuffer == nil) {
            copy_error(errorBuffer, errorCapacity, @"Metal shared-buffer allocation failed");
            return 2;
        }
        Parameters parameters{ knotCount, sourceCount, eventCount, 0 };
        id<MTLCommandBuffer> commandBuffer = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            copy_error(errorBuffer, errorCapacity, @"Metal command encoding failed");
            return 3;
        }
        [encoder setComputePipelineState:context.pipeline];
        [encoder setBuffer:sourceTimesBuffer offset:0 atIndex:0];
        [encoder setBuffer:sourcePositionsBuffer offset:0 atIndex:1];
        [encoder setBuffer:aliveCountsBuffer offset:0 atIndex:2];
        [encoder setBuffer:observerTimesBuffer offset:0 atIndex:3];
        [encoder setBuffer:observerPositionsBuffer offset:0 atIndex:4];
        [encoder setBuffer:outputBuffer offset:0 atIndex:5];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:6];
        NSUInteger width = MIN(context.pipeline.maxTotalThreadsPerThreadgroup, NSUInteger(256));
        [encoder dispatchThreads:MTLSizeMake(pairCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            copy_error(errorBuffer, errorCapacity, commandBuffer.error.localizedDescription);
            return 4;
        }
        std::memcpy(outputSegments, outputBuffer.contents, pairCount * sizeof(int32_t));
        return 0;
    }
}
