message(WARNING
    "MNN_NEUROPILOT is not supported in this open-source distribution. "
    "The option will be disabled. Contact MediaTek and the MNN team for access and integration support."
)

set(MNN_NEUROPILOT OFF CACHE BOOL "Build the NeuroPilot backend" FORCE)
set(MNN_NEUROPILOT OFF PARENT_SCOPE)
