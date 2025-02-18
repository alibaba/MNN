// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;

import com.alibaba.mls.api.HfApiException;
import com.alibaba.mls.api.HfFileMetadata;

import java.io.IOException;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class HfFileMetadataUtils {

    private static final String HUGGINGFACE_HEADER_X_REPO_COMMIT = "x-repo-commit";
    private static final String HUGGINGFACE_HEADER_X_LINKED_ETAG = "x-linked-etag";
    private static final String HUGGINGFACE_HEADER_X_LINKED_SIZE = "x-linked-size";
    private static long parseContentLength(String contentLength) {
        if (contentLength != null && !contentLength.isEmpty()) {
            try {
                return Long.parseLong(contentLength);
            } catch (NumberFormatException e) {
                return 0;
            }
        }
        return 0;
    }

    // Helper method to normalize ETag
     private static String normalizeETag(String etag) {
        if (etag != null && etag.startsWith("\"") && etag.endsWith("\"")) {
            return etag.substring(1, etag.length() - 1);
        }
        return etag;
    }

    public static HfFileMetadata getFileMetadata(OkHttpClient client, String url) throws HfApiException {
        Request request = new Request.Builder()
                .url(url)
                .head()
                .header("Accept-Encoding", "identity")
                .build();

        HfFileMetadata metadata = new HfFileMetadata();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful() && response.code() != 302) {
                throw new HfApiException("Failed to fetch metadata status " + response.code());
            }

            metadata.location = url;
            if (response.code() == 302) {
                String location = response.header("Location");
                if (location != null) {
                    metadata.location = location;
                }
            }

            String linkedEtag = response.header(HUGGINGFACE_HEADER_X_LINKED_ETAG);
            String etag = response.header("ETag");
            metadata.etag = normalizeETag(linkedEtag != null && !linkedEtag.isEmpty() ? linkedEtag : etag);

            String linkedSize = response.header(HUGGINGFACE_HEADER_X_LINKED_SIZE);
            String contentLength = response.header("Content-Length");
            metadata.size = parseContentLength(linkedSize != null && !linkedSize.isEmpty() ? linkedSize : contentLength);

            metadata.commitHash = response.header(HUGGINGFACE_HEADER_X_REPO_COMMIT);

        } catch (IOException e) {
            throw new HfApiException("GetFileMetadata IOException: " + e.getMessage());
        }

        return metadata;
    }

}
