package com.mnn.llm;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class FileUtils {
    public static String findConfigDir(Context context, String ModelDir) throws IOException {
        File directory = new File(ModelDir);
        String modelDir = "";
        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles();
            boolean configFileExists = false;
            for (File file : files) {
                if (file.isDirectory()) {
                    File configFile = new File(file, "config.json");
                    if (configFile.exists() && configFile.isFile()) {
                        modelDir = file.getAbsolutePath();
                        configFileExists = true;
                        break;
                    }
                }
            }
            if(!configFileExists){
                return "";
            }
        }

        return modelDir+"/";
    }
    public static void copyAssetsRecursively(Context context, String assetDir, String targetDir) throws IOException {
        AssetManager assetManager = context.getAssets();
        String[] files = assetManager.list(assetDir);
        if (files == null) return;
        File targetDirectory = new File(targetDir);
        if (!targetDirectory.exists()) {
            if (targetDirectory.mkdirs()) {
                System.out.println("Directories created successfully!");
            } else {
                System.out.println("Failed to create directories or they already exist.");
            }
        }
        for (String file : files) {
            String assetPath = assetDir.isEmpty() ? file : assetDir + File.separator + file;
            String targetPath = targetDir + File.separator + file;
            File targetFile = new File(targetPath);
            if (targetFile.exists()) {
                //File exists, skip
                continue;
            }
            if (assetManager.list(assetPath).length > 0) {
                copyAssetsRecursively(context, assetPath, targetPath);
            } else {
                try (InputStream in = assetManager.open(assetPath);
                     OutputStream out = new FileOutputStream(targetPath)) {
                    byte[] buffer = new byte[1024];
                    int length;
                    while ((length = in.read(buffer)) > 0) {
                        out.write(buffer, 0, length);
                    }
                }
            }
        }
    }
    public static String copyFile(String srcPath, String dstDir) throws IOException {
        File src = new File(srcPath);
        if (src == null) return null;
        File targetDirectory = new File(dstDir);
        if (!targetDirectory.exists()) {
            if (targetDirectory.mkdirs()) {
                System.out.println("Directories created successfully!");
            } else {
                System.out.println("Failed to create directories or they already exist.");
            }
        }
        String fileName = src.getName();
        File dst = new File(dstDir+"/"+fileName);
        try (FileInputStream in = new FileInputStream(src);
             FileOutputStream out = new FileOutputStream(dst)) {
            byte[] buffer = new byte[1024];
            int length;
            while ((length = in.read(buffer)) > 0) {
                out.write(buffer, 0, length);
            }
        }
        return dstDir+"/"+fileName;
    }
}
