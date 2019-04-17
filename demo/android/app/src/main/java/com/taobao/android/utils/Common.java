package com.taobao.android.utils;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class Common {
    public static String TAG = "MNNDemo";

    public static void copyAssetResource2File(Context context, String assetsFile, String outFile) throws IOException {
        InputStream is = context.getAssets().open(assetsFile);
        File outF = new File(outFile);
        FileOutputStream fos = new FileOutputStream(outF);

        int byteCount;
        byte[] buffer = new byte[1024];
        while ((byteCount = is.read(buffer)) != -1) {
            fos.write(buffer, 0, byteCount);
        }
        fos.flush();
        is.close();
        fos.close();
        outF.setReadable(true);
    }
}
