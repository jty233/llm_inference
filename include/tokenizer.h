#pragma once
#include "bpe/bpe.h"
#include "re2/re2.h"
#include <string>
#include <vector>

struct Tokenizer {
    void init(const std::string& merge_path, const std::string& vocab_path) {
        merges.open(merge_path, std::ios::in);
        load_merge_rules(merges, &bpe_ranks);
        vocab_txt.open(vocab_path, std::ios::in);
        load_vocab(vocab_txt, &t2i, &i2t);

        bytes_to_unicode(&b2u, &u2b);
    }

    std::vector<int> encode(const std::string& str) {
        std::vector<int> ids;
        ::encode(str, re, bpe_ranks, b2u, t2i, &ids);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        return ::decode(ids, u2b, i2t);
    }

    std::string decode(int id) {
        return ::decode({id}, u2b, i2t);
    }

    RE2 re = RE2(("('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
        "?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)"));
    BPERanks bpe_ranks;
    std::fstream merges;
    std::unordered_map<uint8_t, wchar_t> b2u;
    std::unordered_map<wchar_t, uint8_t> u2b;
    std::unordered_map<std::string, int> t2i;
    std::unordered_map<int, std::string> i2t;
    std::fstream vocab_txt;
};