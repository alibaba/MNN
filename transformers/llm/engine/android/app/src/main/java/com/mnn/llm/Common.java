package com.mnn.llm;

import android.app.Activity;
import android.content.res.AssetManager;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;


public class Common {
    public static String copyAssetResource2File(Activity activity, String assetsDir) throws IOException, InterruptedException  {
        AssetManager assetManager = activity.getBaseContext().getAssets();
        // make output dir
        String outDir = activity.getCacheDir() + "/" + assetsDir;
        File outPath = new File(outDir);
        if (!outPath.exists()) {
            outPath.mkdirs();
        }
        // visit input files and copy
        String[] files = assetManager.list(assetsDir);
        int nums = files.length;
        CopyThread [] threads = new CopyThread[nums];
        for (int i = 0; i < nums; i++) {
            System.out.println("MNN_DEBUG: " + files[i]);
            String assetsFile = files[i];
            if (new File(outDir + '/' + assetsFile).exists()) {
                continue;
            }
            threads[i] = new CopyThread(assetManager, assetsDir + '/' + assetsFile, outDir + '/' + assetsFile);
            threads[i].start();
        }
        for (int i = 0; i < nums; i++) {
            if (threads[i] != null) {
                threads[i].join();
            }
        }
        return outDir;
    }
}

class CopyThread extends Thread {
    private AssetManager mAsset;
    private String mSrcPath;
    private String mDstPath;
    public CopyThread(AssetManager asset, String src, String dst) {
        mAsset = asset;
        mSrcPath = src;
        mDstPath = dst;
    }
    public void run() {
        try {
            InputStream inS = mAsset.open(mSrcPath);
            File outF = new File(mDstPath);
            FileOutputStream outS = new FileOutputStream(outF);
            int byteCount;
            byte[] buffer = new byte[1024];
            while ((byteCount = inS.read(buffer)) != -1) {
                outS.write(buffer, 0, byteCount);
            }
            outS.flush();
            inS.close();
            outS.close();
            outF.setReadable(true);
        } catch (Exception e) {}
    }
}
