#ifndef BINDING_OPTIONS_H
#define BINDING_OPTIONS_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct go_llama_params {
    char * model;
    bool use_mmap;
} go_llama_params;
#ifdef __cplusplus
}
#endif
#endif //BINDING_OPTIONS_H
