#pragma once

#include "json.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <codecvt>
#include <locale>
#include <regex>
struct Qwen3Tokenizer {
    Qwen3Tokenizer(const std::string& path) {
        std::ifstream fin(path);
        fin >> vocab;
        id2Str.resize(vocab.size());
        for (auto& p : vocab.items()) {
            int id = p.value();
            std::string str = normalizeToken(p.key());
            id2Str[id] = str;
            token2Id[str] = id;
        }
    }

    std::string normalizeToken(const std::string& token) {
        static std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        std::u32string u32token = converter.from_bytes(token);
        std::string result;

        for (char32_t ch : u32token) {
            switch (ch) {
                case 0x0120: result += ' '; break;     // 'Ġ' => space
                case 0x010A: result += '\n'; break;     // line feed
                case 0x010B: result += '\t'; break;     // tab
                case 0x000A: result += '\n'; break;     // actual '\n'
                case 0x0009: result += '\t'; break;     // actual '\t'
                case 0x000D: result += '\r'; break;     // carriage return
                case 0x00A0: result += ' '; break;      // non-breaking space
                default:
                    if (ch >= 0x20 && ch <= 0x7E) {
                        result += static_cast<char>(ch);
                    } else {
                        result += converter.to_bytes(ch);
                    }
            }
        }

        return result;
    }

    std::string decode(int idx) {
        return id2Str[idx];
    }

    std::vector<int32_t> encode(const std::string &text) const {
        std::vector<std::string> words;

        std::string str = text;
        std::regex re(pat_);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }

        // find the longest tokens that form the words:
        std::vector<int32_t> tokens;
        for (const auto & word : words)
        {
            if (word.size() == 0) continue;

            int i = 0;
            int n = word.size();
            while (i < n) {
                int j = n;
                while (j > i)
                {
                    auto it = token2Id.find(word.substr(i, j-i));
                    if (it != token2Id.end()) {
                        tokens.push_back(it->second);
                        i = j;
                        break;
                    }
                    --j;
                }
                if (i == n)
                    break;
                if (j == i)
                {
                    auto sub = word.substr(i, 1);
                    if (token2Id.find(sub) != token2Id.end())
                        tokens.push_back(token2Id.at(sub));
                    else
                        fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                    ++i;
                }
            }
        }

        return tokens;
    }

    nlohmann::json vocab;
    std::vector<std::string> id2Str;
    std::map<std::string, int> token2Id;
    const std::string pat_ = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
};