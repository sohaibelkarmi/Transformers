#pragma once
#include "data.hpp"
#include <array>
#include <map>

namespace edu {
// Single-reference, corpus-level, case-sensitive, whitespace-tokenized BLEU.
// Clip counts separately for each sentence, then aggregate across the corpus.
struct Bleu {
    std::array<size_t,4> matched{},total{};
    size_t hypothesis_length=0,reference_length=0;
    using Counts=std::map<std::vector<std::string>,size_t>;
    static Counts ngrams(const std::vector<std::string>& words,size_t n) {
        Counts counts;
        for (size_t i=0;i+n<=words.size();++i)
            ++counts[std::vector<std::string>(words.begin()+i,words.begin()+i+n)];
        return counts;
    }
    void add(const std::vector<std::string>& hypothesis,const std::vector<std::string>& reference) {
        hypothesis_length+=hypothesis.size(); reference_length+=reference.size();
        for (size_t n=1;n<=4;++n) {
            auto h=ngrams(hypothesis,n),r=ngrams(reference,n);
            for (auto& entry:h) {
                total[n-1]+=entry.second;
                auto it=r.find(entry.first);
                if (it!=r.end()) matched[n-1]+=std::min(entry.second,it->second);
            }
        }
    }
    double score(bool effective_order=false) const {
        if (!hypothesis_length) return 0;
        double log_precision=0; int orders=0;
        for (int n=0;n<4;++n) {
            if (!total[n] && effective_order) continue;
            if (!total[n] || !matched[n]) return 0; // unsmoothed
            log_precision+=std::log(static_cast<double>(matched[n])/total[n]); ++orders;
        }
        if (!orders) return 0;
        double penalty=std::min(0.0,1.0-static_cast<double>(reference_length)/hypothesis_length);
        return 100.0*std::exp(penalty+log_precision/orders);
    }
};
struct Metrics {
    double negative_log_likelihood=0;
    size_t tokens=0,correct=0,sentences=0,exact=0,ended=0,unknown_source=0,source_tokens=0,unknown_target=0,target_tokens=0;
    Bleu bleu;
    double ce() const { return negative_log_likelihood/tokens; }
    void print(std::ostream& out) const {
        out<<"CE="<<ce()<<" PPL="<<std::exp(ce())
           <<" teacher_token_acc="<<100.0*correct/tokens<<"%"
           <<" generated_exact="<<100.0*exact/sentences<<"%"
           <<" BLEU4="<<bleu.score()<<" BLEU_effective="<<bleu.score(true)
           <<" EOS_rate="<<100.0*ended/sentences<<"%"
           <<" source_UNK="<<100.0*unknown_source/source_tokens<<"%"
           <<" target_UNK="<<100.0*unknown_target/target_tokens<<"%\n";
    }
};
inline Metrics evaluate(const Transformer& model,const Vocab& source,const Vocab& target,const std::vector<Pair>& data) {
    NoGrad guard;
    Metrics m;
    for (const auto& pair:data) {
        auto e=encode_pair(pair,source,target,model.config.max_len);
        auto logits=model(e.source,e.input);
        m.negative_log_likelihood+=cross_entropy(logits,e.labels)->data[0]*e.labels.size();
        m.tokens+=e.labels.size();
        for (int i=0;i<logits->rows;++i) {
            auto begin=logits->data.begin()+static_cast<size_t>(i)*logits->cols;
            int best=static_cast<int>(std::max_element(begin,begin+logits->cols)-begin);
            m.correct+=best==e.labels[i];
        }
        auto generated=model.generate(e.source);
        auto words=target.decode(generated);
        // Original reference words are retained: unknown words are not collapsed for BLEU.
        m.bleu.add(words,pair.target);
        bool ended=!generated.empty() && generated.back()==Vocab::EOS;
        m.ended+=ended; m.exact+=(ended && words==pair.target); ++m.sentences;
        m.source_tokens+=pair.source.size(); m.target_tokens+=pair.target.size();
        m.unknown_source+=std::count(e.source.begin(),e.source.end(),Vocab::UNK);
        m.unknown_target+=std::count(e.labels.begin(),e.labels.end(),Vocab::UNK);
    }
    return m;
}
} // namespace edu
