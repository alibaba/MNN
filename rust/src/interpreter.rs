//! Safe wrapper for MNN Interpreter and Session

use crate::ffi;
use crate::tensor::{wrap_tensor, Tensor};
use std::ffi::CString;
use std::path::Path;

/// MNN Interpreter wrapper
pub struct Interpreter {
    ptr: *mut ffi::MnnInterpreter,
}

/// MNN Session wrapper
pub struct Session {
    pub(crate) ptr: *mut ffi::MnnSession,
}

impl Interpreter {
    /// Create interpreter from model file
    pub fn create_from_file<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| crate::error::MnnError::InitError("Invalid path".to_string()))?;
        let c_path = CString::new(path_str).map_err(|e| {
            crate::error::MnnError::InitError(format!("Invalid path string: {}", e))
        })?;

        unsafe {
            let ptr = ffi::mnn_interpreter_create_from_file(c_path.as_ptr());
            if ptr.is_null() {
                Err(crate::error::MnnError::InitError(
                    "Failed to create interpreter".to_string(),
                ))
            } else {
                Ok(Interpreter { ptr })
            }
        }
    }

    /// Create a new session
    pub fn create_session(&mut self, num_threads: i32) -> crate::Result<Session> {
        unsafe {
            let ptr = ffi::mnn_interpreter_create_session(self.ptr, num_threads);
            if ptr.is_null() {
                Err(crate::error::MnnError::RuntimeError(
                    "Failed to create session".to_string(),
                ))
            } else {
                Ok(Session { ptr })
            }
        }
    }

    /// Get session input tensor
    pub fn get_session_input(&self, session: &Session, name: Option<&str>) -> Option<Tensor> {
        unsafe {
            let c_name = name.and_then(|n| CString::new(n).ok());
            let name_ptr = c_name
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null());

            let ptr = ffi::mnn_interpreter_get_session_input(self.ptr, session.ptr, name_ptr);
            if ptr.is_null() {
                None
            } else {
                Some(wrap_tensor(ptr))
            }
        }
    }

    /// Get session output tensor
    pub fn get_session_output(&self, session: &Session, name: Option<&str>) -> Option<Tensor> {
        unsafe {
            let c_name = name.and_then(|n| CString::new(n).ok());
            let name_ptr = c_name
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null());

            let ptr = ffi::mnn_interpreter_get_session_output(self.ptr, session.ptr, name_ptr);
            if ptr.is_null() {
                None
            } else {
                Some(wrap_tensor(ptr))
            }
        }
    }

    /// Run session
    pub fn run_session(&self, session: &Session) -> crate::Result<()> {
        unsafe {
            let ret = ffi::mnn_interpreter_run_session(self.ptr, session.ptr);
            if ret != 0 {
                Err(crate::error::MnnError::RuntimeError(format!(
                    "Session run failed with code: {}",
                    ret
                )))
            } else {
                Ok(())
            }
        }
    }

    /// Resize session (call after changing input shape)
    pub fn resize_session(&self, session: &Session) {
        unsafe {
            ffi::mnn_interpreter_resize_session(self.ptr, session.ptr);
        }
    }
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        unsafe { ffi::mnn_interpreter_destroy(self.ptr) };
    }
}

// Session does not need Drop because it is managed by Interpreter/Net in MNN C++ API usually?
// Actually in MNN C++, Session is owned by Interpreter but we need to verify if we need to explicitly release it.
// The `releaseSession` API exists in C++.
// For simplicity in this binding, we rely on Interpreter destruction to clean up sessions unless we expose releaseSession.
// But `mnn_c.cpp` doesn't expose `releaseSession` yet, and `mnn_interpreter_destroy` calls `MNN::Interpreter::destroy`.
// `MNN::Interpreter` destructor releases all sessions. So `Session` struct here is just a handle.
