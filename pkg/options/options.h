#ifndef BINDING_OPTIONS_H
#define BINDING_OPTIONS_H
#include "../util/util.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct go_llama_params {
    char * model;
    bool use_mmap;
    void * antiprompt;
} go_llama_params;

#ifdef __cplusplus
}
#endif
#endif //BINDING_OPTIONS_H
