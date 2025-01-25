// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;

import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

public class HfFileUtils {

    public static final String TAG = "FileUtils";

    public static String repoFolderName(String repoId, String repoType) {
        if (repoId == null || repoType == null) {
            return "";
        }
        String[] repoParts = repoId.split("/");
        List<String> parts = new ArrayList<>();
        parts.add(repoType + "s"); // e.g., "models"
        for (String part : repoParts) {
            if (part != null && !part.isEmpty()) {
                parts.add(part);
            }
        }
        // Join parts with "--" separator
        return String.join("--", parts);
    }

    public static boolean deleteDirectoryRecursively2(File dir) {
        if (dir == null || !dir.exists()) {
            return false;
        }

        Path dirPath = dir.toPath();
        try {
            Files.walkFileTree(dirPath, new SimpleFileVisitor<Path>() {

                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    Files.delete(file);
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult postVisitDirectory(Path directory, IOException exc) throws IOException {
                    Files.delete(directory);
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException {
                    return FileVisitResult.TERMINATE;
                }
            });
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    public static File getPointerPath(File storageFolder, String commitHash, String relativePath) {
        File commitFolder = new File(storageFolder, "snapshots/" + commitHash);
        return new File(commitFolder, relativePath);
    }

    public static File getPointerPathParent(File storageFolder, String sha) {
        return new File(storageFolder, "snapshots/" + sha);
    }

    public static String getLastFileName(String path) {
        if (path == null || path.isEmpty()) {
            return path;
        }
        int pos = path.lastIndexOf('/');
        return (pos == -1) ? path : path.substring(pos + 1);
    }

    public static void createSymlink(String target, String linkPath)  {
        Path targetPath = Paths.get(target);
        Path link = Paths.get(linkPath);
        try {
            Files.createSymbolicLink(link, targetPath);
        } catch (IOException e) {
            Log.e(TAG, "createSymlink error", e);
        }
    }

    public static void createSymlink(Path target, Path linkPath) throws IOException {
        Files.createSymbolicLink(linkPath, target);
    }

    public static void moveWithPermissions(File src, File dest)  {
        try {
            Files.move(src.toPath(), dest.toPath(), StandardCopyOption.REPLACE_EXISTING);
            dest.setReadable(true, true);
            dest.setWritable(true, true);
            dest.setExecutable(false, false);
        } catch (IOException e) {
            Log.e(TAG, "moveWithPermissions Failed", e);
        }
    }
}
