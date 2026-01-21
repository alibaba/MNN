//! Safe wrapper for MNN Tensor

use crate::ffi;
use std::ffi::c_void;

/// MNN Tensor wrapper
pub struct Tensor {
    pub(crate) ptr: *mut ffi::MnnTensor,
}

impl Tensor {
    /// Create a new host tensor
    pub fn new(shape: &[i32], type_: i32, data: Option<&mut [u8]>) -> Self {
        unsafe {
            let data_ptr = data.map(|d| d.as_mut_ptr() as *mut c_void).unwrap_or(std::ptr::null_mut());
            let ptr = ffi::mnn_tensor_create(
                shape.as_ptr(),
                shape.len() as i32,
                type_,
                data_ptr,
            );
            Tensor { ptr }
        }
    }

    /// Create a new device tensor
    pub fn new_device(shape: &[i32], type_: i32) -> Self {
        unsafe {
            let ptr = ffi::mnn_tensor_create_device(
                shape.as_ptr(),
                shape.len() as i32,
                type_,
            );
            Tensor { ptr }
        }
    }

    /// Create a host tensor from device tensor
    pub fn from_device(device_tensor: &Tensor, copy_data: bool) -> Option<Self> {
        unsafe {
            let ptr = ffi::mnn_tensor_create_host_from_device(device_tensor.ptr, copy_data);
            if ptr.is_null() {
                None
            } else {
                Some(Tensor { ptr })
            }
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let mut shape = [0i32; 6]; // Max dims usually 4-6
            let size = ffi::mnn_tensor_get_shape(self.ptr, shape.as_mut_ptr(), 6);
            shape[..size as usize].to_vec()
        }
    }

    /// Get raw data pointer (host only)
    pub fn data(&self) -> *mut c_void {
        unsafe { ffi::mnn_tensor_get_data(self.ptr) }
    }

    /// Get data as slice (unsafe)
    pub unsafe fn data_as_slice<T>(&self) -> &[T] {
        let ptr = self.data() as *const T;
        let size = self.size() as usize / std::mem::size_of::<T>();
        std::slice::from_raw_parts(ptr, size)
    }

    /// Get data as mutable slice (unsafe)
    pub unsafe fn data_as_mut_slice<T>(&mut self) -> &mut [T] {
        let ptr = self.data() as *mut T;
        let size = self.size() as usize / std::mem::size_of::<T>();
        std::slice::from_raw_parts_mut(ptr, size)
    }

    /// Get total size in bytes
    pub fn size(&self) -> i32 {
        unsafe { ffi::mnn_tensor_get_size(self.ptr) }
    }

    /// Copy data from host tensor to device tensor (self)
    pub fn copy_from_host(&mut self, host_tensor: &Tensor) -> bool {
        unsafe { ffi::mnn_tensor_copy_from_host(self.ptr, host_tensor.ptr) }
    }

    /// Copy data from device tensor (self) to host tensor
    pub fn copy_to_host(&self, host_tensor: &mut Tensor) -> bool {
        unsafe { ffi::mnn_tensor_copy_to_host(self.ptr, host_tensor.ptr) }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { ffi::mnn_tensor_destroy(self.ptr) };
    }
}

// Wrap existing pointer (e.g. from Session)
pub(crate) unsafe fn wrap_tensor(ptr: *mut ffi::MnnTensor) -> Tensor {
    Tensor { ptr }
}
