#pragma once
#include "model.hpp"
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <unordered_map>

namespace edu {
inline std::vector<std::string> tokenize(const std::string& text) {
    std::istringstream stream(text); stream.imbue(std::locale::classic());
    std::vector<std::string> words;
    std::string word;
    while (stream>>word) words.push_back(word);
    return words;
}
struct Vocab {
    enum { PAD=0, BOS=1, EOS=2, UNK=3 };
    std::vector<std::string> words{"<pad>","<bos>","<eos>","<unk>"};
    std::unordered_map<std::string,int> index;
    Vocab() { rebuild(); }
    void rebuild() {
        index.clear();
        for (size_t i=0;i<words.size();++i) require(index.emplace(words[i],static_cast<int>(i)).second,"duplicate vocabulary entry");
    }
    void learn(const std::vector<std::string>& tokens) {
        for (auto& w:tokens) if (!index.count(w)) {
            int id=static_cast<int>(words.size()); words.push_back(w); index[w]=id;
        }
    }
    std::vector<int> encode(const std::vector<std::string>& tokens) const {
        std::vector<int> ids;
        for (auto& w:tokens) { auto it=index.find(w); ids.push_back(it==index.end() ? UNK:it->second); }
        return ids;
    }
    std::vector<std::string> decode(const std::vector<int>& ids) const {
        std::vector<std::string> tokens;
        for (int id:ids) {
            if (id==EOS) break;
            if (id!=PAD && id!=BOS) tokens.push_back(words.at(id));
        }
        return tokens;
    }
};
struct Pair { std::vector<std::string> source,target; };
inline std::vector<Pair> read_data(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    require(static_cast<bool>(f),"cannot open dataset");
    std::vector<Pair> pairs;
    std::string line;
    size_t number=0;
    while (std::getline(f,line)) {
        ++number;
        if (number==1 && line.substr(0,3)=="\xEF\xBB\xBF") line.erase(0,3);
        if (!line.empty() && line.back()=='\r') line.pop_back();
        if (line.empty()) continue;
        size_t tab=line.find('\t');
        if (tab==std::string::npos || line.find('\t',tab+1)!=std::string::npos)
            throw std::runtime_error("Expected source<TAB>target on line "+std::to_string(number));
        Pair pair{tokenize(line.substr(0,tab)),tokenize(line.substr(tab+1))};
        require(!pair.source.empty() && !pair.target.empty(),"empty source or target");
        for (const auto* side:{&pair.source,&pair.target}) for (auto& w:*side)
            require(w!="<pad>" && w!="<bos>" && w!="<eos>" && w!="<unk>","reserved token in dataset");
        pairs.push_back(std::move(pair));
    }
    require(!pairs.empty(),"empty dataset"); return pairs;
}
struct Example { std::vector<int> source,input,labels; };
inline Example encode_pair(const Pair& pair,const Vocab& source,const Vocab& target,int max_len) {
    Example e;
    e.source=source.encode(pair.source); e.source.push_back(Vocab::EOS);
    e.labels=target.encode(pair.target); e.input={Vocab::BOS};
    e.input.insert(e.input.end(),e.labels.begin(),e.labels.end());
    e.labels.push_back(Vocab::EOS);
    require(e.source.size()<=static_cast<size_t>(max_len) && e.input.size()<=static_cast<size_t>(max_len),"dataset sequence too long: increase --max-len or preprocess data");
    return e;
}
inline std::string join(const std::vector<std::string>& words) {
    std::string out;
    for (auto& w:words) { if (!out.empty()) out+=' '; out+=w; }
    return out;
}
// Portable, human-readable checkpoint; 17 digits round-trip double precision.
// Contains weights and vocabularies, not Adam state; training resume is not exposed.
inline void save(const std::string& path,const Transformer& model,const Vocab& src,const Vocab& tgt) {
    std::ofstream out(path);
    require(static_cast<bool>(out),"cannot create checkpoint (parent folder must exist)");
    out.imbue(std::locale::classic()); out<<std::setprecision(17);
    auto c=model.config;
    out<<"EDU_TRANSFORMER_V1\n"<<c.d<<' '<<c.heads<<' '<<c.ff<<' '<<c.layers<<' '<<c.max_len<<'\n';
    for (auto* v:{&src,&tgt}) {
        out<<v->words.size()<<'\n'; for (auto& w:v->words) out<<std::quoted(w)<<'\n';
    }
    out<<model.parameters.all.size()<<'\n';
    for (auto& p:model.parameters.all) {
        out<<p.name<<' '<<p.value->rows<<' '<<p.value->cols<<'\n';
        for (double v:p.value->data) out<<v<<' ';
        out<<'\n';
    }
    out.flush(); require(static_cast<bool>(out),"checkpoint write failed");
}
struct Loaded { std::unique_ptr<Transformer> model; Vocab source,target; };
inline Loaded load(const std::string& path) {
    std::ifstream in(path); in.imbue(std::locale::classic());
    require(static_cast<bool>(in),"cannot open checkpoint");
    std::string magic; in>>magic; require(magic=="EDU_TRANSFORMER_V1","unsupported checkpoint");
    Config c; in>>c.d>>c.heads>>c.ff>>c.layers>>c.max_len; c.validate();
    require(c.d<=512 && c.ff<=2048 && c.layers<=12 && c.max_len<=4096,"checkpoint exceeds educational size limits");
    Loaded result;
    for (auto* v:{&result.source,&result.target}) {
        size_t n=0; in>>n; require(n>=4 && n<=100000,"invalid checkpoint vocabulary size");
        v->words.resize(n);
        for (auto& w:v->words) { in>>std::quoted(w); require(!w.empty(),"invalid vocabulary token"); }
        require(v->words[0]=="<pad>" && v->words[1]=="<bos>" && v->words[2]=="<eos>" && v->words[3]=="<unk>","invalid special tokens");
        v->rebuild();
    }
    require(static_cast<bool>(in),"truncated checkpoint header");
    result.model=std::make_unique<Transformer>(c,static_cast<int>(result.source.words.size()),static_cast<int>(result.target.words.size()));
    size_t count=0; in>>count; require(count==result.model->parameters.all.size(),"parameter count mismatch");
    for (auto& p:result.model->parameters.all) {
        std::string name; int r=0,col=0; in>>name>>r>>col;
        require(name==p.name && r==p.value->rows && col==p.value->cols,"parameter shape/name mismatch");
        for (double& v:p.value->data) { in>>v; require(static_cast<bool>(in) && std::isfinite(v),"invalid checkpoint value"); }
    }
    std::string trailing; require(!(in>>trailing),"unexpected checkpoint content");
    return result;
}
} // namespace edu
