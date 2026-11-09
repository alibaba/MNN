import CoreML

public struct Weights {
    enum WeightsError: LocalizedError {
        case notSupported(message: String)
        case invalidFile

        public var errorDescription: String? {
            switch self {
            case let .notSupported(message):
                String(localized: "The weight format '\(message)' is not supported by this application.", comment: "Error when weight format is not supported")
            case .invalidFile:
                String(localized: "The weights file is invalid or corrupted.", comment: "Error when weight file is invalid")
            }
        }
    }

    private let dictionary: [String: MLMultiArray]

    init(_ dictionary: [String: MLMultiArray]) {
        self.dictionary = dictionary
    }

    subscript(key: String) -> MLMultiArray? { dictionary[key] }

    public static func from(fileURL: URL) throws -> Weights {
        guard ["safetensors", "gguf", "mlx"].contains(fileURL.pathExtension)
        else { throw WeightsError.notSupported(message: "\(fileURL.pathExtension)") }

        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        switch ([UInt8](data.subdata(in: 0..<4)), [UInt8](data.subdata(in: 4..<6))) {
        case ([0x47, 0x47, 0x55, 0x46], _): throw WeightsError.notSupported(message: "gguf")
        case ([0x93, 0x4E, 0x55, 0x4D], [0x50, 0x59]): throw WeightsError.notSupported(message: "mlx")
        default: return try Safetensor.from(data: data)
        }
    }
}

struct Safetensor {
    typealias Error = Weights.WeightsError

    struct Header {
        struct Offset: Decodable {
            let dataOffsets: [Int]?
            let dtype: String?
            let shape: [Int]?

            /// Unsupported: "I8", "U8", "I16", "U16", "BF16"
            var dataType: MLMultiArrayDataType? {
                get throws {
                    switch dtype {
                    case "I32", "U32": .int32
                    case "F16": .float16
                    case "F32": .float32
                    case "F64", "U64": .float64
                    default: throw Error.notSupported(message: "\(dtype ?? "empty")")
                    }
                }
            }
        }

        static func from(data: Data) throws -> [String: Offset?] {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode([String: Offset?].self, from: data)
        }
    }

    static func from(data: Data) throws -> Weights {
        let headerSize: Int = data.subdata(in: 0..<8).withUnsafeBytes { $0.load(as: Int.self) }
        guard headerSize < data.count else { throw Error.invalidFile }
        let header = try Header.from(data: data.subdata(in: 8..<(headerSize + 8)))

        var dict = [String: MLMultiArray]()
        for (key, point) in header {
            guard let offsets = point?.dataOffsets, offsets.count == 2,
                  let shape = point?.shape as? [NSNumber],
                  let dType = try point?.dataType
            else { continue }

            let strides = shape.dropFirst().reversed().reduce(into: [1]) { acc, a in
                acc.insert(acc[0].intValue * a.intValue as NSNumber, at: 0)
            }
            let start = 8 + offsets[0] + headerSize
            let end = 8 + offsets[1] + headerSize
            let tensorData = data.subdata(in: start..<end) as NSData
            let ptr = UnsafeMutableRawPointer(mutating: tensorData.bytes)
            dict[key] = try MLMultiArray(dataPointer: ptr, shape: shape, dataType: dType, strides: strides)
        }

        return Weights(dict)
    }
}
