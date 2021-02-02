#include <cctype>
#include <cstring>
#include <string>
#include <memory>
#include <unordered_map>
#include <ostream>
#include <variant>
#include <vector>

#include <arbor/arbexcept.hpp>

#include "util/strprintf.hpp"
#include "s_expr.hpp"
namespace arb {

inline bool is_alphanumeric(char c) {
    return std::isdigit(c) || std::isalpha(c);
}
inline bool is_plusminus(char c) {
    return (c=='-' || c=='+');
}
inline bool is_valid_symbol_char(char c) {
    switch (c) {
        case '+':
        case '-':
        case '*':
        case '/':
        case '@':
        case '$':
        case '%':
        case '^':
        case '&':
        case '_':
        case '=':
        case '<':
        case '>':
        case '~':
        case '.':
            return true;
        default:
            return is_alphanumeric(c);
    }
}

std::ostream& operator<<(std::ostream& o, const src_location& l) {
    return o << l.line << ":" << l.column;
}

std::ostream& operator<<(std::ostream& o, const tok& t) {
    switch (t) {
        case tok::nil:    return o << "nil";
        case tok::lparen: return o << "lparen";
        case tok::rparen: return o << "rparen";
        case tok::real:   return o << "real";
        case tok::integer:return o << "integer";
        case tok::symbol: return o << "symbol";
        case tok::string: return o << "string";
        case tok::eof:    return o << "eof";
        case tok::error:  return o << "error";
    }
    return o << "<unknown>";
}

std::ostream& operator<<(std::ostream& o, const token& t) {
    if (t.kind==tok::string) {
        return o << util::pprintf("\"{}\"", t.spelling);
    }
    return o << util::pprintf("{}", t.spelling);
}

//
// lexer
//

struct s_expr_lexer_error: public arb::arbor_internal_error {
    s_expr_lexer_error(const std::string& msg, src_location l):
        arbor_internal_error(util::pprintf("s-expression internal error at {}: {}", l, msg))
    {}
};

static std::unordered_map<tok, const char*> tok_to_keyword = {
    {tok::nil,    "nil"},
};

static std::unordered_map<std::string, tok> keyword_to_tok = {
    {"nil",    tok::nil},
};

class lexer {
    transmogrifier line_start_;
    transmogrifier stream_;
    unsigned line_;
    token token_;

public:

    lexer(transmogrifier begin):
        line_start_(begin), stream_(begin), line_(0)
    {
        // Prime the first token.
        parse();
    }

    // Return the current token in the stream.
    const token& current() {
        return token_;
    }

    const token& next() {
        parse();
        return token_;
    }

private:

    src_location loc() const {
        return src_location(line_+1, stream_-line_start_+1);
    }

    bool empty() const {
        return *stream_ == '\0';
    }

    // Consume and return the next token in the stream.
    void parse() {
        using namespace std::string_literals;

        while (!empty()) {
            switch (*stream_) {
                // end of file
                case 0      :       // end of string
                    token_ = {loc(), tok::eof, "eof"s};
                    return;

                // new line
                case '\n'   :
                    line_++;
                    ++stream_;
                    line_start_ = stream_;
                    continue;

                // white space
                case ' '    :
                case '\t'   :
                case '\v'   :
                case '\f'   :
                    character();
                    continue;   // skip to next character

                case ';':
                    eat_comment();
                    continue;
                case '(':
                    token_ = {loc(), tok::lparen, {character()}};
                    return;
                case ')':
                    token_ = {loc(), tok::rparen, {character()}};
                    return;
                case 'a' ... 'z':
                case 'A' ... 'Z':
                    token_ = symbol();
                    return;
                case '0' ... '9':
                    token_ = number();
                    return;
                case '"':
                    token_ = string();
                    return;
                case '-':
                case '+':
                case '.':
                    {
                        if (empty()) {
                            token_ = {loc(), tok::error, "Unexpected end of input."};
                        }
                        char c = stream_.peek(1);
                        if (std::isdigit(c) or c=='.') {
                            token_ = number();
                            return;
                        }
                    }
                    token_ = {loc(), tok::error,
                        util::pprintf("Unexpected character '{}'.", character())};
                    return;

                default:
                    token_ = {loc(), tok::error,
                        util::pprintf("Unexpected character '{}'.", character())};
                    return;
            }
        }

        if (!empty()) {
            // todo: handle error: should never hit this
        }
        token_ = {loc(), tok::eof, "eof"s};
        return;
    }

    // Consumes characters in the stream until end of stream or a new line.
    // Assumes that the current location is the `;` that starts the comment.
    void eat_comment() {
        while (!empty() && *stream_!='\n') {
            ++stream_;
        }
    }

    // Parse alphanumeric sequence that starts with an alphabet character,
    // and my contain alphabet, numeric or one of {+ -  *  /  @  $  %  ^  &  _  =  <  >  ~ .}
    //
    // This definition follows the symbol naming conventions of common lisp, without the
    // use of pipes || to define symbols with arbitrary strings.
    //
    // Valid symbols:
    //    sub_dendrite
    //    sub-dendrite
    //    sub-dendrite:
    //    foo@3.2/lower
    //    temp_
    //    branch3
    //    A
    // Invalid symbols:
    //    _cat          ; can't start with underscore
    //    -cat          ; can't start with hyphen
    //    2ndvar        ; can't start with numeric character
    //
    // Returns the appropriate token kind if symbol is a keyword.
    token symbol() {
        auto start = loc();
        std::string symbol;
        char c = *stream_;

        // Assert that current position is at the start of an identifier
        if( !(std::isalpha(c)) ) {
            throw s_expr_lexer_error(
                "Lexer attempting to read identifier when none is available", loc());
        }

        symbol += c;
        ++stream_;
        while(1) {
            c = *stream_;

            if(is_valid_symbol_char(c)) {
                symbol += c;
                ++stream_;
            }
            else {
                break;
            }
        }

        // test if the symbol matches a keyword
        auto it = keyword_to_tok.find(symbol.c_str());
        if (it!=keyword_to_tok.end()) {
            return {start, it->second, std::move(symbol)};
        }
        return {start, tok::symbol, std::move(symbol)};
    }

    token string() {
        using namespace std::string_literals;
        if (*stream_ != '"') {
            s_expr_lexer_error(
                "Lexer attempting to read string without opening \"", loc());
        }

        auto start = loc();
        ++stream_;
        std::string str;
        while (!empty() && *stream_!='"') {
            str.push_back(*stream_);
            ++stream_;
        }
        if (empty()) return {start, tok::error, "string missing closing \""};
        ++stream_; // gobble the closing "

        return {start, tok::string, str};
    }

    token number() {
        using namespace std::string_literals;

        auto start = loc();
        std::string str;
        char c = *stream_;

        // Start counting the number of points in the number.
        auto num_point = (c=='.' ? 1 : 0);
        auto uses_scientific_notation = 0;

        str += c;
        ++stream_;
        while(1) {
            c = *stream_;
            if (std::isdigit(c)) {
                str += c;
                ++stream_;
            }
            else if (c=='.') {
                if (++num_point>1) {
                    // Can't have more than one '.' in a number
                    return {start, tok::error, "unexpected '.'"s};
                }
                str += c;
                ++stream_;
                if (uses_scientific_notation) {
                    // Can't have a '.' in the mantissa
                    return {start, tok::error, "unexpected '.'"s};
                }
            }
            else if (!uses_scientific_notation && (c=='e' || c=='E')) {
                if ( std::isdigit(stream_.peek(1)) ||
                    (is_plusminus(stream_.peek(1)) && std::isdigit(stream_.peek(2))))
                {
                    uses_scientific_notation++;
                    str += c;
                    stream_++;
                    // Consume the next char if +/-
                    if (is_plusminus(*stream_)) {
                        str += *stream_++;
                    }
                }
                else {
                    // the 'e' or 'E' is the beginning of a new token
                    break;
                }
            }
            else {
                break;
            }
        }

        const bool is_real = uses_scientific_notation || num_point>0;
        return {start, (is_real? tok::real: tok::integer), std::move(str)};
    }

    char character() {
        return *stream_++;
    }
};

//
// s expression members
//

bool s_expr::is_atom() const {
    return state.index()==0;
}

const token& s_expr::atom() const {
    return std::get<0>(state);
}

const s_expr& s_expr::head() const {
    return std::get<1>(state).head.get();
}

const s_expr& s_expr::tail() const {
    return std::get<1>(state).tail.get();
}

s_expr& s_expr::head() {
    return std::get<1>(state).head.get();
}

s_expr& s_expr::tail() {
    return std::get<1>(state).tail.get();
}

s_expr::operator bool() const {
    return !(is_atom() && atom().kind==tok::nil);
}

std::ostream& operator<<(std::ostream& o, const s_expr& x) {
    if (x.is_atom()) return o << x.atom();
#if 1
    o << "(";
    bool first = true;
    for (auto& e: x) {
        o << (first? "": " ") << e;
        first = false;
    }
    return o << ")";
#else
    return o << "(" << x.head() << " . " << x.tail() << ")";
#endif
}

std::size_t length(const s_expr& l) {
    // The length of an atom is 1.
    if (l.is_atom() && l) {
        return 1;
    }
    // nil marks the end of a list.
    if (!l) {
        return 0u;
    }
    return 1+length(l.tail());
}

src_location location(const s_expr& l) {
    if (l.is_atom()) return l.atom().loc;
    return location(l.head());
}

//
// parsing s expressions
//

namespace impl {

// If there is a parsing error, then an atom with kind==tok::error is returned
// with the error string in its spelling.
s_expr parse(lexer& L) {
    using namespace std::string_literals;

    s_expr node;
    auto t = L.current();

    if (t.kind==tok::lparen) {
        t = L.next();
        s_expr* n = &node;
        while (true) {
            if (t.kind == tok::eof) {
                return token{t.loc, tok::error,
                    "Unexpected end of input. Missing a closing parenthesis ')'."};
            }
            if (t.kind == tok::error) {
                return t;
            }
            else if (t.kind == tok::rparen) {
                *n = token{t.loc, tok::nil, "nil"};
                t = L.next();
                break;
            }
            else if (t.kind == tok::lparen) {
                auto e = parse(L);
                if (e.is_atom() && e.atom().kind==tok::error) return e;
                *n = {std::move(e), {}};
                t = L.current();
            }
            else {
                *n = {s_expr(t), {}};
                t = L.next();
            }

            n = &n->tail();
        }
    }
    else if (t.kind==tok::eof) {
        return token{t.loc, tok::error, "Empty expression."};
    }
    else if (t.kind==tok::rparen) {
        return token{t.loc, tok::error, "Missing opening parenthesis'('."};
    }
    // an atom or an error
    else {
        L.next(); // advance the lexer to the next token
        return t;
    }

    return node;
}

}

s_expr parse_s_expr(transmogrifier begin) {
    lexer l(begin);
    s_expr result = impl::parse(l);
    const bool err = result.is_atom()? result.atom().kind==tok::error: false;
    if (!err) {
        auto t = l.current();
        if (t.kind!=tok::eof) {
            return token{t.loc, tok::error,
                         util::pprintf("Unexpected '{}' at the end of input.", t)};
        }
    }
    return result;
}

s_expr parse_s_expr(const std::string& in) {
    return parse_s_expr(transmogrifier{in});
}

// For parsing a file with multiple high level s expressions.
// Returns a vector of the expressions.
// If an error occured, terminate early and the last expression will be an error.
std::vector<s_expr> parse_multi_s_expr(transmogrifier begin) {
    std::vector<s_expr> result;
    lexer l(begin);
    bool error = false;
    while (!error && l.current().kind!=tok::eof) {
        result.push_back(impl::parse(l));
        const auto& e = result.back();
        error = e.is_atom() && e.atom().kind==tok::error;
    }

    return result;
}


} // namespace arb
