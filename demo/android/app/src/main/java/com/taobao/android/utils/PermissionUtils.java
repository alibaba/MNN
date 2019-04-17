
package com.taobao.android.utils;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.v4.app.ActivityCompat;

/**
 * Description:
 */
public class PermissionUtils {


    public static void askPermission(Activity context, String[] permissions, int req, Runnable
            runnable) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            int result = ActivityCompat.checkSelfPermission(context, permissions[0]);
            if (result == PackageManager.PERMISSION_GRANTED) {
                runnable.run();
            } else {
                ActivityCompat.requestPermissions(context, new String[]{Manifest.permission.CAMERA
                        , Manifest.permission.WRITE_EXTERNAL_STORAGE}, req);
            }
        } else {
            runnable.run();
        }
    }

    public static void onRequestPermissionsResult(boolean isReq, int[] grantResults, Runnable
            okRun, Runnable deniRun) {
        if (isReq) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                okRun.run();
            } else {
                deniRun.run();
            }
        }
    }

}
