#pragma once
#include "tensor.hpp"
#include <iostream>
#include <random>
#include <string>

namespace edu {
struct Config {
    int d=32, heads=4, ff=64, layers=1, max_len=64;
    void validate() const {
        require(d>=2 && heads>0 && d%heads==0,"d must be >=2 and divisible by heads");
        require(ff>0 && layers>0 && max_len>=2,"invalid model configuration");
    }
};
struct Parameter { std::string name; T value; };
struct Parameters {
    std::mt19937 rng;
    std::vector<Parameter> all;
    explicit Parameters(unsigned seed):rng(seed) {}
    T make(const std::string& name,int r,int c,const std::string& init="xavier") {
        auto t=tensor(r,c,true);
        if (init=="one") std::fill(t->data.begin(),t->data.end(),1.0);
        else if (init!="zero") {
            double bound=std::sqrt(6.0/(r+c));
            std::uniform_real_distribution<double> dist(-bound,bound);
            for (double& v:t->data) v=dist(rng);
        }
        all.push_back({name,t}); return t;
    }
    void zero_grad() { for (auto& p:all) std::fill(p.value->grad.begin(),p.value->grad.end(),0.0); }
    size_t count() const { size_t n=0; for (auto& p:all) n+=p.value->data.size(); return n; }
};
struct Linear {
    T w,b;
    Linear(Parameters& p,const std::string& name,int in,int out)
        :w(p.make(name+".weight",in,out)),b(p.make(name+".bias",1,out,"zero")) {}
    T operator()(const T& x) const { return add(matmul(x,w),b); }
};
struct Norm {
    T gamma,beta;
    Norm(Parameters& p,const std::string& name,int d)
        :gamma(p.make(name+".gamma",1,d,"one")),beta(p.make(name+".beta",1,d,"zero")) {}
    T operator()(const T& x) const { return layer_norm(x,gamma,beta); }
};
struct Attention {
    int heads,dk;
    Linear q,k,v,output;
    Attention(Parameters& p,const std::string& name,int d,int h)
        :heads(h),dk(d/h),q(p,name+".q",d,d),k(p,name+".k",d,d),
         v(p,name+".v",d,d),output(p,name+".out",d,d) {}
    T operator()(const T& queries,const T& memory,bool causal=false) const {
        auto Q=q(queries), K=k(memory), V=v(memory);
        std::vector<T> pieces;
        for (int h=0;h<heads;++h) {
            auto qh=columns(Q,h*dk,dk), kh=columns(K,h*dk,dk), vh=columns(V,h*dk,dk);
            // [Tq,dk] [dk,Tk] -> [Tq,Tk]. Each row is one query's scores.
            auto scores=scale(matmul(qh,transpose(kh)),1.0/std::sqrt(static_cast<double>(dk)));
            auto weights=softmax(scores,causal);
            pieces.push_back(matmul(weights,vh)); // [Tq,Tk] [Tk,dk] -> [Tq,dk]
        }
        return output(concat(pieces));
    }
};
struct FeedForward {
    Linear first,second;
    FeedForward(Parameters& p,const std::string& n,int d,int ff):first(p,n+".up",d,ff),second(p,n+".down",ff,d) {}
    T operator()(const T& x) const { return second(relu(first(x))); }
};
struct EncoderBlock {
    Norm n1,n2;
    Attention attention;
    FeedForward ff;
    EncoderBlock(Parameters& p,const std::string& n,const Config& c)
        :n1(p,n+".n1",c.d),n2(p,n+".n2",c.d),attention(p,n+".attention",c.d,c.heads),ff(p,n+".ff",c.d,c.ff) {}
    T operator()(T x) const {
        auto normalized=n1(x);
        x=add(x,attention(normalized,normalized));
        return add(x,ff(n2(x)));
    }
};
struct DecoderBlock {
    Norm n1,n2,n3;
    Attention self,cross;
    FeedForward ff;
    DecoderBlock(Parameters& p,const std::string& n,const Config& c)
        :n1(p,n+".n1",c.d),n2(p,n+".n2",c.d),n3(p,n+".n3",c.d),
         self(p,n+".self",c.d,c.heads),cross(p,n+".cross",c.d,c.heads),ff(p,n+".ff",c.d,c.ff) {}
    T operator()(T x,const T& memory) const {
        auto normalized=n1(x);
        x=add(x,self(normalized,normalized,true));
        x=add(x,cross(n2(x),memory));
        return add(x,ff(n3(x)));
    }
};
inline T positional(int length,int d) {
    auto p=tensor(length,d);
    for (int pos=0;pos<length;++pos) for (int j=0;j<d;++j) {
        double angle=pos/std::pow(10000.0,2.0*(j/2)/d);
        p->at(pos,j)=j%2==0 ? std::sin(angle):std::cos(angle);
    }
    return p;
}
struct Transformer {
    Config config;
    Parameters parameters;
    T source_embedding,target_embedding;
    std::vector<EncoderBlock> encoders;
    std::vector<DecoderBlock> decoders;
    Norm encoder_norm,decoder_norm;
    Linear vocabulary;
    static Config checked(Config c) { c.validate(); return c; }
    Transformer(Config c,int source_vocab,int target_vocab,unsigned seed=42)
        :config(checked(c)),parameters(seed),
         source_embedding(parameters.make("source_embedding",source_vocab,c.d)),
         target_embedding(parameters.make("target_embedding",target_vocab,c.d)),
         encoder_norm(parameters,"encoder_final_norm",c.d),decoder_norm(parameters,"decoder_final_norm",c.d),
         vocabulary(parameters,"vocabulary",c.d,target_vocab) {
        for (int i=0;i<c.layers;++i) {
            encoders.emplace_back(parameters,"encoder."+std::to_string(i),c);
            decoders.emplace_back(parameters,"decoder."+std::to_string(i),c);
        }
    }
    T input(const T& table,const std::vector<int>& ids) const {
        require(!ids.empty() && ids.size()<=static_cast<size_t>(config.max_len),"sequence exceeds max_len (including special tokens)");
        return add(scale(embedding(table,ids),std::sqrt(static_cast<double>(config.d))),positional(static_cast<int>(ids.size()),config.d));
    }
    T encode(const std::vector<int>& source) const {
        auto x=input(source_embedding,source);
        for (auto& block:encoders) x=block(x);
        return encoder_norm(x);
    }
    T decode(const std::vector<int>& prefix,const T& memory) const {
        auto x=input(target_embedding,prefix);
        for (auto& block:decoders) x=block(x,memory);
        return vocabulary(decoder_norm(x));
    }
    T operator()(const std::vector<int>& source,const std::vector<int>& prefix) const { return decode(prefix,encode(source)); }
    // Returns generated IDs, including EOS when generated; BOS is input only.
    std::vector<int> generate(const std::vector<int>& source,int limit=0) const {
        NoGrad guard;
        if (limit<=0) limit=config.max_len;
        limit=std::min(limit,config.max_len);
        auto memory=encode(source);
        std::vector<int> prefix{1},result; // fixed BOS ID
        for (int step=0;step<limit;++step) {
            auto logits=decode(prefix,memory);
            int last=logits->rows-1, best=2; // permit EOS and UNK, forbid PAD and BOS
            for (int j=3;j<logits->cols;++j) if (logits->at(last,j)>logits->at(last,best)) best=j;
            result.push_back(best);
            if (best==2) break;
            prefix.push_back(best);
        }
        return result;
    }
};
struct Adam {
    std::vector<std::vector<double>> first,second;
    size_t step=0;
    explicit Adam(const Parameters& p) {
        for (auto& x:p.all) { first.emplace_back(x.value->data.size(),0.0); second.emplace_back(x.value->data.size(),0.0); }
    }
    double update(Parameters& p,double lr,double clip=1.0) {
        double norm=0;
        for (auto& x:p.all) for (double g:x.value->grad) { require(std::isfinite(g),"nonfinite gradient"); norm=std::hypot(norm,g); }
        double multiplier=norm>clip ? clip/norm:1.0;
        ++step;
        double correction1=1-std::pow(0.9,static_cast<double>(step));
        double correction2=1-std::pow(0.999,static_cast<double>(step));
        for (size_t k=0;k<p.all.size();++k) {
            auto& x=*p.all[k].value;
            for (size_t j=0;j<x.data.size();++j) {
                double g=x.grad[j]*multiplier;
                first[k][j]=0.9*first[k][j]+0.1*g;
                second[k][j]=0.999*second[k][j]+0.001*g*g;
                x.data[j]-=lr*(first[k][j]/correction1)/(std::sqrt(second[k][j]/correction2)+1e-8);
            }
        }
        return norm;
    }
};
} // namespace edu
