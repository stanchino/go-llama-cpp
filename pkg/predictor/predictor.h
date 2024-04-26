#ifndef BINDING_PREDICTOR_H
#define BINDING_PREDICTOR_H
#ifdef __cplusplus

#include <fstream>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

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
typedef struct go_llama_predict_state {
    struct gpt_params * params;
    struct llama_context *ctx_guidance;
    struct llama_sampling_context *ctx_sampling;
    unsigned int n_ctx;
    unsigned int n_total_consumed ;
    int n_past;
    int n_remain;
    int n_past_guidance;
    unsigned int guidance_offset;
    unsigned int original_prompt_len;
    bool is_anti_prompt;
    bool is_interacting;
    bool input_echo;
    bool display;
    void * embeddings;
} go_llama_predict_state;

typedef const char cchar_t;
struct go_llama_predict_state * go_llama_init_predict_state(struct go_llama_state * o_state);
void go_llama_set_prompt(struct go_llama_state * state, struct go_llama_predict_state * p_state, const char * prompt);
int go_llama_predict(struct go_llama_state *state, struct go_llama_predict_state *p_state);
extern char *predictorInputCallback(struct go_llama_predict_state *state);
extern void predictorOutputCallback(struct go_llama_predict_state *state, cchar_t *);
extern void predictorEndOutputCallback(struct go_llama_predict_state *state);
#ifdef __cplusplus
}
#endif
#endif //BINDING_PREDICTOR_H
