// clef-demo: use a Clef decision model from the command line.
//
// Runs a real joint-head decision: the prompt is split into spans
// (one QUESTION span + one OPTION span per option), each span is tagged
// with a llama_decision_order, the whole batch is encoded in one forward
// pass, and one score per option is read back from the embeddings output.
// Scores -> softmax -> choice + confidence.
//
//   clef-demo --demo                      built-in example decision
//   clef-demo --demo privacy              named demo (db|privacy|privacy-reversed|privacy-subtle)
//   clef-demo --question "..." --option "A" --option "B" [--type choice|score|noul]
//   clef-demo --download                  fetch GGUF with `hf download`
//
// NOTE: consecutive same-order tokens merge into ONE span, so option text
// must be separated by NONE-order text (the "Option X:" labels below) —
// the GGUF's decision template emits label text between options for the
// same reason.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "llama.h"

static const char * kDefaultRepo = "ggml-org/Clef-Flash-GGUF";
static const char * kDefaultFile = "Clef-Flash-Q4_K_M.gguf";
static const char * kOurDir      = "models/llamacpp/clef-flash";

static bool file_exists(const std::string & p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

static void usage(const char * argv0) {
    std::printf("usage:\n"
                "  %s --demo [db|privacy|privacy-reversed|privacy-subtle] [--download] [model.gguf]\n"
                "  %s --question \"...\" --option \"A\" --option \"B\" [--type choice|score|noul] [model.gguf]\n"
                "\n"
                "Model resolution: argv path > $CLEF_MODEL > %s/%s\n",
                argv0, argv0, kOurDir, kDefaultFile);
}

static std::string hf_download(const std::string & repo, const std::string & file) {
    // Create the destination with the filesystem API (mkdir -p is not a cmd
    // builtin on Windows), then shell out to the Hugging Face CLI.
    std::error_code ec;
    std::filesystem::create_directories(kOurDir, ec);
    std::string cmd = "hf download \"" + repo + "\" \"" + file + "\" --local-dir " + kOurDir;
    std::printf("[download] %s\n", cmd.c_str());
    if (std::system(cmd.c_str()) != 0) {
        return "";
    }
    std::string direct = std::string(kOurDir) + "/" + file;
    if (file_exists(direct)) {
        return direct;
    }
    // hf --local-dir can nest the file under the repo path; walk for it.
    for (const auto & entry : std::filesystem::recursive_directory_iterator(kOurDir, ec)) {
        if (entry.path().filename() == file) {
            return entry.path().string();
        }
    }
    return "";
}

struct Piece {
    std::string text;
    int32_t     order; // llama_decision_order value
};

static int run_decision(const std::string & model_path, const std::string & qtype,
                        const std::vector<std::string> & options, const std::string & question) {
    const int64_t t_start = llama_time_us();

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 99;
    llama_model * model        = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::printf("LOAD FAILED: %s\n", model_path.c_str());
        return 1;
    }
    char arch[64] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 8192;
    cparams.n_batch              = 2048;
    cparams.n_ubatch             = 512;
    cparams.embeddings           = true;
    cparams.pooling_type         = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::printf("CONTEXT INIT FAILED\n");
        llama_model_free(model);
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           add_bos = llama_vocab_get_add_bos(vocab);

    int32_t qorder = 2; // QUESTION_CHOICE
    if (qtype == "score") {
        qorder = 3;
    } else if (qtype == "noul") {
        qorder = 1;
    }
    std::vector<Piece> pieces;
    pieces.push_back({question, qorder});
    for (size_t i = 0; i < options.size(); i++) {
        char label[32];
        std::snprintf(label, sizeof(label), "\nOption %c:\n", (char)('A' + i));
        pieces.push_back({label, 0});      // NONE separator between option spans
        pieces.push_back({options[i], 4}); // OPTION
    }
    pieces.push_back({"\n", 0});

    std::vector<llama_token> toks;
    std::vector<int32_t>     orders;
    for (size_t p = 0; p < pieces.size(); p++) {
        const std::string & txt = pieces[p].text;
        std::vector<llama_token> pt(txt.size() + 8);
        int32_t n = llama_tokenize(vocab, txt.c_str(), (int32_t)txt.size(), pt.data(), (int32_t)pt.size(),
                                   p == 0 ? add_bos : false, true);
        if (n < 0) {
            pt.resize(-n);
            n = llama_tokenize(vocab, txt.c_str(), (int32_t)txt.size(), pt.data(), (int32_t)pt.size(),
                               p == 0 ? add_bos : false, true);
        }
        if (n <= 0) {
            std::printf("EMPTY PIECE (question/options must not be empty)\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        pt.resize(n);
        toks.insert(toks.end(), pt.begin(), pt.end());
        orders.insert(orders.end(), (size_t)n, pieces[p].order);
    }

    llama_batch batch   = llama_batch_init((int32_t)toks.size(), 0, 1);
    batch.n_tokens      = (int32_t)toks.size();
    for (size_t i = 0; i < toks.size(); i++) {
        batch.token[i]          = toks[i];
        batch.pos[i]            = (llama_pos)i;
        batch.n_seq_id[i]       = 1;
        batch.seq_id[i][0]      = 0;
        batch.logits[i]         = 1; // all tokens are outputs (embeddings, pooling NONE)
        if (!llama_batch_set_decision_order(&batch, (int32_t)i,
                                            (llama_decision_order)orders[i])) {
            std::printf("FAILED to set decision order at %d\n", (int)i);
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    if (llama_encode(ctx, batch) != 0) {
        std::printf("ENCODE FAILED\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const size_t n_opt = options.size();
    std::vector<float> scores;
    bool               ok = true;
    for (size_t i = 0; i < n_opt; i++) {
        const float * e = llama_get_embeddings_ith(ctx, (int32_t)i);
        if (!e) {
            ok = false;
            break;
        }
        scores.push_back(e[0]);
        if (!std::isfinite(e[0])) {
            ok = false;
        }
    }

    const double secs = (llama_time_us() - t_start) / 1000000.0;

    std::printf("================ INPUT ================\n");
    std::printf("model    : %s\n", model_path.c_str());
    std::printf("arch     : %s\n", arch);
    std::printf("type     : %s\n", qtype.c_str());
    std::printf("question : %s\n", question.c_str());
    for (size_t i = 0; i < options.size(); i++) {
        std::printf("option %c : %s\n", (char)('A' + i), options[i].c_str());
    }
    std::printf("tokens   : %d\n", (int)toks.size());
    std::printf("================ OUTPUT ===============\n");
    if (!ok) {
        std::printf("scores   : <non-finite - the head could not use the decision order>\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    double smax = scores[0];
    for (float s : scores) {
        smax = std::max(smax, (double)s);
    }
    std::vector<double> probs(n_opt, 0.0);
    double              sum = 0.0;
    for (size_t i = 0; i < n_opt; i++) {
        probs[i] = std::exp(scores[i] - smax);
        sum += probs[i];
    }
    for (double & p : probs) {
        p /= sum;
    }
    const size_t best    = std::max_element(probs.begin(), probs.end()) - probs.begin();
    const double uniform = 1.0 / n_opt;
    const double conf    = n_opt < 2 ? 1.0 : std::max(0.0, (probs[best] - uniform) / (1.0 - uniform));

    for (size_t i = 0; i < n_opt; i++) {
        std::printf("  %c  score %+9.4f  prob %.4f%s\n", (char)('A' + i), scores[i], probs[i],
                    i == best ? "  <-- choice" : "");
    }
    std::printf("choice     : %c (%s)\n", (char)('A' + best), options[best].c_str());
    std::printf("confidence : %.4f\n", conf);
    std::printf("time       : %.1fs\n", secs);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

struct DemoCase {
    const char *              name;
    const char *              type;
    const char *              question;
    std::vector<const char *> options;
};

static const DemoCase kDemos[] = {
    {
        "db", "choice",
        "The user asks: 'I need an embedded vector database for on-device "
        "search on a phone. No server, no cloud, no network.' "
        "Which option best satisfies the request?",
        {
            "sqlite-vec: vector search running inside SQLite, fully embedded and on-device.",
            "Postgres with pgvector: server database that needs hosted infrastructure.",
            "Pinecone: managed cloud vector database that needs network access and an API key.",
        },
    },
    {
        "privacy", "choice",
        "A journalist wants to analyze leaked documents on a laptop that may be "
        "monitored by the network operator. Which AI setup keeps the analysis "
        "unseeable by any third party?",
        {
            "On-device model via the RunAnywhere SDK, airplane mode, no network calls, results only in local storage",
            "Cloud AI API over Tor, so the network operator cannot see which service is being used",
            "Hosted cloud chat service with a VPN and cloud history cleared after each session",
            "Local model but with SDK analytics/telemetry left enabled for product improvement",
            "Web-based AI app over HTTPS, documents encrypted at rest in the cloud",
        },
    },
    {
        "privacy-reversed", "choice",
        "A journalist wants to analyze leaked documents on a laptop that may be "
        "monitored by the network operator. Which AI setup keeps the analysis "
        "unseeable by any third party?",
        {
            "Web-based AI app over HTTPS, documents encrypted at rest in the cloud",
            "Local model but with SDK analytics/telemetry left enabled for product improvement",
            "Hosted cloud chat service with a VPN and cloud history cleared after each session",
            "Cloud AI API over Tor, so the network operator cannot see which service is being used",
            "On-device model via the RunAnywhere SDK, airplane mode, no network calls, results only in local storage",
        },
    },
    {
        "privacy-subtle", "choice",
        "Which AI setup keeps sensitive legal documents unseeable by any third party during analysis?",
        {
            "On-device model, fully offline, but every session is backup-synced end-to-end-encrypted to the vendor cloud",
            "On-device model, fully offline, backups stay on an external disk the user controls",
        },
    },
};

static int run_demo(const std::string & model_path, const DemoCase & d) {
    std::vector<std::string> options;
    for (const char * o : d.options) {
        options.push_back(o);
    }
    return run_decision(model_path, d.type, options, d.question);
}

int main(int argc, char ** argv) {
    bool        demo      = false;
    bool        want_dl   = false;
    std::string demo_name = "db";
    std::string model_path;
    std::string qtype;
    std::string question;
    std::vector<std::string> options;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--demo") {
            demo = true;
            // an argument here is a demo name only if it matches one of the
            // known cases; anything else (e.g. a model path) is left to be
            // parsed as a positional argument
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                bool is_known = false;
                for (const auto & d : kDemos) {
                    if (demo_name.empty() && std::strcmp(argv[i + 1], d.name) == 0) {
                        is_known = true;
                        break;
                    }
                }
                if (is_known) {
                    demo_name = argv[++i];
                }
            }
        } else if (a == "--download") {
            want_dl = true;
        } else if (a == "--question" && i + 1 < argc) {
            question = argv[++i];
        } else if (a == "--option" && i + 1 < argc) {
            options.push_back(argv[++i]);
        } else if (a == "--type" && i + 1 < argc) {
            qtype = argv[++i];
        } else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::printf("unknown flag: %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        } else {
            model_path = a;
        }
    }

    if (model_path.empty()) {
        if (const char * e = std::getenv("CLEF_MODEL")) {
            model_path = e;
        }
    }
    if (model_path.empty()) {
        std::string p = std::string(kOurDir) + "/" + kDefaultFile;
        if (file_exists(p)) {
            model_path = p;
        }
    }
    if (want_dl || model_path.empty()) {
        std::string got = hf_download(kDefaultRepo, kDefaultFile);
        if (!got.empty() && file_exists(got)) {
            model_path = got;
        }
    }
    if (model_path.empty() || !file_exists(model_path)) {
        std::printf("model not found. Pass a path, set $CLEF_MODEL, or re-run with --download.\n");
        return 1;
    }

    if (demo) {
        const DemoCase * chosen = nullptr;
        for (const auto & d : kDemos) {
            if (demo_name == d.name) {
                chosen = &d;
                break;
            }
        }
        if (!chosen) {
            std::printf("unknown demo '%s'. available:", demo_name.c_str());
            for (const auto & d : kDemos) {
                std::printf(" %s", d.name);
            }
            std::printf("\n");
            return 1;
        }
        return run_demo(model_path, *chosen);
    }

    if (qtype.empty()) {
        qtype = "choice";
    }
    if (question.empty() || options.size() < 2) {
        std::printf("need --question plus at least two --option (or use --demo).\n");
        usage(argv[0]);
        return 1;
    }
    if (options.size() > 26) {
        std::printf("max 26 options in this demo.\n");
        return 1;
    }

    return run_decision(model_path, qtype, options, question);
}