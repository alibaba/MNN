package com.mnn.llm;

import android.util.Log;

import java.io.Serializable;

public class Chat implements Serializable {
    public native boolean Init(String modelDir);
    public native String Submit(String input);
    public native byte[] Response();
    public native float Done();
    public native void Reset();

    static {
        try {
            System.loadLibrary("llm");
        } catch (UnsatisfiedLinkError e) {
            Log.e("JNI", "Failed to load library: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
