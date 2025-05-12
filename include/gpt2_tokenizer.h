#pragma once

#include "json.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <codecvt>
#include <locale>
struct GPT2Tokenizer {
    GPT2Tokenizer(const std::string& path) {
        std::ifstream fin(path);
        fin >> vocab;
        id2Str.resize(vocab.size());
        for (auto& p : vocab.items()) {
            int id = p.value();
            std::string str = p.key();
            id2Str[id] = str;
        }
    }

    std::string decode(int idx) {
        std::string token = id2Str[idx];
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

    nlohmann::json vocab;
    std::vector<std::string> id2Str;
};