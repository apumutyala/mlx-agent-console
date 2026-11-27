// [agent-console patch C] Inkling 2 names for the model-agnostic expert streaming in
// MLXLMCommon/ExpertStreaming.swift (kept so callers and Inkling2.swift read as before).

import MLXLMCommon

public typealias Inkling2ExpertStreaming = ExpertStreaming
public typealias Inkling2ExpertStreamingOptions = ExpertStreamingOptions
public typealias Inkling2ExpertStreamingSnapshot = ExpertStreamingSnapshot

typealias Inkling2ExpertStreamingError = ExpertStreamingError
typealias Inkling2ExpertManifest = ExpertStreamManifest
typealias Inkling2ExpertStreamRuntime = ExpertStreamRuntime
typealias Inkling2ExpertSlotCache = ExpertSlotCache
typealias Inkling2ExpertStreamLayer = ExpertStreamLayer
typealias Inkling2RoutedExperts = RoutedExperts
