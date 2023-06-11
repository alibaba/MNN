package com.taobao.android.mnn;

import android.content.Context;
import android.util.Log;

public class MNNNetInstance {
    private static final String TAG = "MNNDemo";

    public static MNNNetInstance createFromFile(String fileName) {
        long instance = MNNNetNative.nativeCreateNetFromFile(fileName);
        if (0 == instance) {
            Log.e(TAG, "Create Net Failed from file " + fileName);
            return null;
        }

        return new MNNNetInstance(instance);
    }

    public static MNNNetInstance createFromBuffer(byte[] buffer) {
        long instance = MNNNetNative.nativeCreateNetFromBuffer(buffer);
        if (0 == instance) {
            Log.e(TAG, "Create Net Failed from buffer, buffer maybe null or invalide");
            return null;
        }
        return new MNNNetInstance(instance);
    }

    public static class Config {
        public int forwardType = MNNForwardType.FORWARD_CPU.type;
        public int numThread = 4;
        public String[] saveTensors = null;
        public String[] outputTensors = null;

    }

    public class Session {
        public class Tensor {
            private Tensor(long ptr) {
                mTensorInstance = ptr;
            }

            protected long instance() {
                return mTensorInstance;
            }

            public void reshape(int[] dims) {
                MNNNetNative.nativeReshapeTensor(mNetInstance, mTensorInstance, dims);
                mData = null;
            }

            public void setInputIntData(int[] data) {
                MNNNetNative.nativeSetInputIntData(mNetInstance, mTensorInstance, data);
                mData = null;
            }

            public void setInputFloatData(float[] data) {
                MNNNetNative.nativeSetInputFloatData(mNetInstance, mTensorInstance, data);
                mData = null;
            }

            public int[] getDimensions() {
                return MNNNetNative.nativeTensorGetDimensions(mTensorInstance);
            }

            public float[] getFloatData() {
                getData();
                return mData;
            }

            public int[] getIntData() {
                if (null == mIntData) {
                    int size = MNNNetNative.nativeTensorGetIntData(mTensorInstance, null);
                    mIntData = new int[size];
                }
                MNNNetNative.nativeTensorGetIntData(mTensorInstance, mIntData);

                return mIntData;
            }

            public void getData() {
                if (null == mData) {
                    int size = MNNNetNative.nativeTensorGetData(mTensorInstance, null);
                    mData = new float[size];
                }
                MNNNetNative.nativeTensorGetData(mTensorInstance, mData);
            }

            public byte[] getUINT8Data() {
                if (null == mUINT8Data) {
                    int size = MNNNetNative.nativeTensorGetUINT8Data(mTensorInstance, null);
                    mUINT8Data = new byte[size];
                }
                MNNNetNative.nativeTensorGetUINT8Data(mTensorInstance, mUINT8Data);

                return mUINT8Data;
            }

            private float[] mData = null;
            private int[] mIntData = null;
            private byte[] mUINT8Data = null;
            private long mTensorInstance;
        }


        private Session(long ptr) {
            mSessionInstance = ptr;
        }

        //After all input tensors' reshape, call this method
        public void reshape() {
            MNNNetNative.nativeReshapeSession(mNetInstance, mSessionInstance);
        }

        public void run() {
            MNNNetNative.nativeRunSession(mNetInstance, mSessionInstance);
        }

        public Tensor[] runWithCallback(String[] names) {
            long[] tensorPtr = new long[names.length];

            Tensor[] tensorReturnArray = new Tensor[names.length];
            MNNNetNative.nativeRunSessionWithCallback(mNetInstance, mSessionInstance, names, tensorPtr);
            for (int i = 0; i < names.length; i++) {
                tensorReturnArray[i] = new Tensor(tensorPtr[i]);
            }
            return tensorReturnArray;
        }

        public Tensor getInput(String name) {
            long tensorPtr = MNNNetNative.nativeGetSessionInput(mNetInstance, mSessionInstance, name);
            if (0 == tensorPtr) {
                Log.e(TAG, "Can't find seesion input: " + name);
                return null;
            }
            return new Tensor(tensorPtr);
        }

        public Tensor getOutput(String name) {
            long tensorPtr = MNNNetNative.nativeGetSessionOutput(mNetInstance, mSessionInstance, name);
            if (0 == tensorPtr) {
                Log.e(TAG, "Can't find seesion output: " + name);
                return null;
            }
            return new Tensor(tensorPtr);
        }

        //Release the session from net instance, it's not needed if you call net.release()
        public void release() {
            checkValid();
            MNNNetNative.nativeReleaseSession(mNetInstance, mSessionInstance);
            mSessionInstance = 0;
        }

        private long mSessionInstance = 0;
    }

    public Session createSession(Config config) {
        checkValid();

        if (null == config) {
            config = new Config();
        }

        long sessionId = MNNNetNative.nativeCreateSession(mNetInstance, config.forwardType, config.numThread, config.saveTensors, config.outputTensors);
        if (0 == sessionId) {
            Log.e(TAG, "Create Session Error");
            return null;
        }
        return new Session(sessionId);
    }

    private void checkValid() {
        if (mNetInstance == 0) {
            throw new RuntimeException("MNNNetInstance native pointer is null, it may has been released");
        }
    }


    public void release() {
        checkValid();
        MNNNetNative.nativeReleaseNet(mNetInstance);
        mNetInstance = 0;
    }

    private MNNNetInstance(long instance) {
        mNetInstance = instance;
    }

    private long mNetInstance;
}
