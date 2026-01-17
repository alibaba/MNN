export interface ScheduleConfig {
    type?: ForwardType;
    numThread?: number;
    backupType?: ForwardType;
    mode?: number;
}

export interface BackendConfig {
    precision?: number;
    power?: number;
    memory?: number;
}

export enum ForwardType {
    CPU = 0,
    METAL = 1,
    OPENCL = 2,
    OPENGL = 3,
    VULKAN = 4,
    NN = 5,
    CUDA = 6,
    HIAI = 7,
}

export enum ErrorCode {
    NO_ERROR = 0,
    OUT_OF_MEMORY = 1,
    NOT_SUPPORT = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION = 4,
}

export enum DataType {
    FLOAT = 0,
    INT32 = 1,
    INT64 = 2,
    UINT8 = 3,
}

export class Session {
    private constructor();
}

export class Tensor {
    getShape(): number[];
    getDataType(): DataType;
    getData(): Float32Array | Int32Array | Uint8Array;
    copyFrom(data: Float32Array | Int32Array | Uint8Array): void;
    getHost(): bigint;
    getElementSize(): number;
}

export class Interpreter {
    static createFromFile(path: string): Interpreter;
    static createFromBuffer(buffer: Buffer | ArrayBuffer | TypedArray): Interpreter;

    createSession(config?: ScheduleConfig): Session;
    resizeSession(session: Session): void;
    runSession(session: Session): ErrorCode;

    getSessionInput(session: Session, name?: string): Tensor | null;
    getSessionOutput(session: Session, name?: string): Tensor | null;

    getModelVersion(): string;
    release(): void;
}

export class Llm {
    load(): boolean;
    response(query: string, history?: boolean): string;
    generate(input_ids: number[]): number[];
    applyChatTemplate(content: string): string;
    setConfig(config: string): void;
}

export namespace llm {
    function create(configPath: string): Llm;
}

export const version: string;
