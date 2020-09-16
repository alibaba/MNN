#ifndef MNNConvertDef_H
#define MNNConvertDef_H

#if defined(_MSC_VER)
#if defined(MNNConvertDeps_EXPORTS)
#define MNNConvertDeps_PUBLIC __declspec(dllexport)
#elif defined(USING_MNNConvertDeps)
#define MNNConvertDeps_PUBLIC __declspec(dllimport)
#else
#define MNNConvertDeps_PUBLIC
#endif
#else
#define MNNConvertDeps_PUBLIC __attribute__((visibility("default")))
#endif

#endif