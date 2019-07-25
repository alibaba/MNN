package com.taobao.android.opengl;

import android.content.Context;
import android.opengl.GLES20;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.HashMap;

public class Shader {
    private int mProgram = 0;
    private int mShaderVertex = 0;
    private int mShaderFragment = 0;
    private String vertexSource;
    private String fragmentSource;

    //hashmap for storing uniform/attribute handles
    private final HashMap<String, Integer> mShaderHandleMap = new HashMap<String, Integer>();


    public Shader() {
        // TODO Auto-generated constructor stub
    }

    public void setProgram(int vertexShader, int fragmentShader, Context context)
            throws Exception {
        vertexSource = loadRawString(vertexShader, context);
        fragmentSource = loadRawString(fragmentShader, context);

        mShaderVertex = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        mShaderFragment = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);

        int program = GLES20.glCreateProgram();
        if(program != 0){
            GLES20.glAttachShader(program, mShaderVertex);
            GLES20.glAttachShader(program, mShaderFragment);
            GLES20.glLinkProgram(program);
            int[] linkStatus = new int[1];
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
            if(linkStatus[0] != GLES20.GL_TRUE){
                String error = GLES20.glGetProgramInfoLog(program);
                deleteProgram();
                throw new Exception(error);
            }
        }

        mProgram = program;
        mShaderHandleMap.clear();

    }

    public void useProgram(){
        GLES20.glUseProgram(mProgram);
    }

    public void deleteProgram(){
        GLES20.glDeleteShader(mShaderVertex);
        GLES20.glDeleteShader(mShaderFragment);
        GLES20.glDeleteProgram(mProgram);
        mProgram = mShaderVertex = mShaderFragment = 0;
    }

    public int programHandle(){
        return mProgram;
    }

    public int getHandle(String name){
        if(mShaderHandleMap.containsKey(name)){
            return mShaderHandleMap.get(name);
        }

        int handle = GLES20.glGetAttribLocation(mProgram, name);
        if(handle == -1){
            handle = GLES20.glGetUniformLocation(mProgram, name);
        }
        if(handle == -1){
            Log.d("GLSL shader", "Could not get attrib location for " + name);
        }else{
            mShaderHandleMap.put(name, handle);
        }

        return handle;
    }

    public int[] getHandles(String... names){
        int[] res = new int[names.length];
        for(int i = 0; i < names.length; ++i){
            res[i] = getHandle(names[i]);
        }

        return res;
    }

    private int loadShader(int shaderType, String source)throws Exception {
        int shader = GLES20.glCreateShader(shaderType);
        if(shader != 0){
            GLES20.glShaderSource(shader, source);
            GLES20.glCompileShader(shader);
            int[] compiled = new int[1];
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);

            if(compiled[0] == 0){
                String error = GLES20.glGetShaderInfoLog(shader);
                GLES20.glDeleteShader(shader);
                throw new Exception(error);
            }
        }

        return shader;
    }

    private String loadRawString(int rawId, Context context) throws Exception {
        InputStream is = context.getResources().openRawResource(rawId);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buf = new byte[1024];
        int len;
        while((len = is.read(buf))!= -1){
            baos.write(buf, 0, len);
        }
        return baos.toString();
    }


}
