# Model Tags System Analysis

## 1. Data Loading Mechanism

The system employs a multi-layered approach to load model tags from various sources, orchestrated by `ModelListManager` and `ModelMarketCache`.

### Sources

#### 1. Builtin Models
*   **Storage Location**: Application private storage under `.mnnmodels/builtin/[modelName]/`.
*   **Configuration File**: `market_config.json`.
*   **Loading Mechanism**:
    *   `ModelMarketCache` recursively scans `context.filesDir` for files named `market_config.json`.
    *   Detected configs are parsed into `ModelMarketItem` objects and stored in the memory cache.

#### 2. Local Testing Models (`/data/local/tmp`)
*   **Storage Location**: `/data/local/tmp/mnn_models/[model_dir]/`.
*   **Configuration File**: `market_config.json` (must be placed alongside `config.json`).
*   **Loading Mechanism**:
    *   `ModelUtils.localModelList` scans `/data/local/tmp/mnn_models/` for directories containing `config.json` to identify available local models.
    *   When `ModelListManager` processes these models, it attempts to read `market_config.json` from the same directory via `ModelMarketCache.readMarketConfigFromLocal`.
    *   The system creates a `ModelMarketItem` from this file, which includes the `tags`.

#### 3. Downloaded Models
*   **Storage Location**: `[filesDir]/configs/[safeModelId]/market_config.json`.
*   **Source**: Originally fetched from `ModelRepository` (Network or Assets).
*   **Loading Mechanism**:
    *   `ModelRepository` fetches the global `model_market.json` from the network or loads it from assets.
    *   `ModelMarketCache` subscribes to these updates.
    *   When a model is downloaded or market data updates, the specific model's metadata (including tags) is persisted to the local filesystem for offline access.

### Key Components

*   **`ModelListManager`**: The central coordinator that initializes the model list. It merges data from `ChatDataManager` (downloaded models), `ModelUtils` (local models), and `ModelMarketCache` (metadata/tags). It exposes the `getModelTags(modelId)` API.
*   **`ModelMarketCache`**: A singleton that maintains an in-memory `ConcurrentHashMap` of `ModelMarketItem`s. It handles the low-level file I/O for reading `market_config.json` from all sources.
*   **`ModelRepository`**: Manages the retrieval of the global model market catalog from the Alibaba CDN or local assets.

## 2. Tag Processing & Localization

The raw tags read from JSON are processed to support internationalization.

*   **`ModelMarketItem`**: Contains the raw list of tag strings (e.g., `["chat", "qwen"]`).
*   **`TagMapper`**: A singleton responsible for translating tag keys.
    *   It initializes mappings from `ModelMarketConfig.tagTranslations`.
    *   `getTag(stringTag)` returns a `Tag` object.
*   **`Tag` Object**: Holds the `key` (original string) and `ch` (Chinese translation).
    *   `getDisplayText()`: Checks `DeviceUtils.isChinese`. If true, returns the Chinese translation; otherwise, returns the original key.

## 3. UI Display

The display logic is encapsulated in the RecyclerView adapter's ViewHolder and a custom layout view.

### `ModelItemHolder`
*   **Binding**: In the `bind()` method, it retrieves the tags for the current model.
*   **Filtering**: It enforces a strict limit on the number of tags displayed:
    ```kotlin
    modelItem.getDisplayTags(context).take(3)
    ```
    This ensures that at most 3 tags are passed to the UI.

### `TagsLayout`
*   **Rendering**: A custom `LinearLayout` subclass designed for rendering tags.
*   **Dynamic View Creation**: It iterates through the list of tag strings and creates a `TextView` for each.
*   **Styling**:
    *   **Text Color**: Uses the theme's `colorPrimary`.
    *   **Background**: `R.drawable.shape_tag_view` (rounded corners).
    *   **Typography**: `R.dimen.h4` text size, single line.
*   **Layout Logic**: Implements a custom `onLayout` method (specifically `performFlexWrapLayout`) to handle wrapping. It calculates the width of each tag and determines if it fits in the current row. If a tag exceeds the available width, it (and subsequent tags) may be hidden to prevent layout issues.

## 4. Summary Flow

1.  **Discovery**: `ModelListManager` identifies a model (Local, Builtin, or Downloaded).
2.  **Metadata Load**: `ModelMarketCache` reads the corresponding `market_config.json` to get the `ModelMarketItem`.
3.  **Data Binding**: `ModelItemHolder` asks the `ModelItem` for its display tags.
4.  **Translation**: The system translates the raw tag keys using `TagMapper`.
5.  **Filtering**: The list is truncated to the first 3 tags.
6.  **Rendering**: `TagsLayout` dynamically generates styled `TextViews` for these tags.
