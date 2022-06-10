package com.taobao.android.utils;

import android.content.Context;
import android.text.TextUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class TxtFileReader {
    private static class ImageUrlProvider {
        private BufferedReader reader;
        private boolean visitEnd = false;

        ImageUrlProvider(Context context, String fileName) throws IOException {
            InputStream in = context.getAssets().open(fileName);
            reader = new BufferedReader(new InputStreamReader(in));
        }

        synchronized String getLine() {
            if (!visitEnd) {
                try {
                    String url = reader.readLine();
                    if (url == null) {
                        visitEnd = true;
                    } else {
                        return url;
                    }
                } catch (Throwable t) {
                    t.printStackTrace();
                }
            }
            return null;
        }

        void close() {
            if (reader != null) {
                try {
                    reader.close();
                } catch (Throwable t) {
                    t.printStackTrace();
                }
            }
        }
    }

    public static List<String> getUniqueUrls(Context context, String fileName, int count) throws IOException {
        List<String> rets = new ArrayList<String>();
        ImageUrlProvider provider = new ImageUrlProvider(context, fileName);
        while (rets.size() < count) {
            String url = provider.getLine();
            if (TextUtils.isEmpty(url)) {
                break;
            }
            //if (!rets.contains(url)) {
            rets.add(url);
            //}
        }
        provider.close();
        return rets;
    }
}
