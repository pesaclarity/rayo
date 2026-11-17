/* rayo/export.h: symbol visibility for the public API. */
#ifndef RAYO_EXPORT_H
#define RAYO_EXPORT_H

#if defined(_WIN32)
#define RAYO_API __declspec(dllexport)
#elif defined(__GNUC__)
#define RAYO_API __attribute__((visibility("default")))
#else
#define RAYO_API
#endif

#endif
