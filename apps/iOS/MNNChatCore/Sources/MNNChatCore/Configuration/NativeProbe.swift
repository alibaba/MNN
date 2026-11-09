//
//  NativeProbe.swift
//  MNNChatCore
//
//  App-shell-facing shim for the `--native-probe` launch argument, so app
//  targets never need to import the ObjC++ CMNNInference module directly.
//

import CMNNInference
import Foundation

public enum NativeProbe {
    /// Runs the native Metal pipeline probe (DEBUG harness). Blocks; call from
    /// the app entry init before any view is created.
    public static func run() {
        #if DEBUG
        LLMInferenceEngineWrapper.runNativeProbe()
        #endif
    }
}
