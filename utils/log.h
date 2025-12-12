
#pragma once

#include <cstdio>

#define FL_LOG_DEBUG(format, ...) fprintf(stderr, "\033[34m[DEBUG] %s:%d: " format "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define FL_LOG_INFO(format, ...) fprintf(stderr, "\033[32m[INFO] %s:%d: " format "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define FL_LOG_WARNING(format, ...) fprintf(stderr, "\033[33m[WARNING] %s:%d: " format "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define FL_LOG_ERROR(format, ...) fprintf(stderr, "\033[31m[ERROR] %s:%d: " format "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__)