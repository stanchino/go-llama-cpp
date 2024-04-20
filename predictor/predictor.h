#ifndef BINDING_PREDICTOR_H
#define BINDING_PREDICTOR_H
#ifdef __cplusplus
#include <fstream>
static bool file_exists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string &path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}
extern "C" {
#endif
typedef const char cchar_t;
extern void tokenCallback(cchar_t *);
int go_llama_predict(void * state_ptr, const char * prompt);
#ifdef __cplusplus
}
#endif
#endif //BINDING_PREDICTOR_H