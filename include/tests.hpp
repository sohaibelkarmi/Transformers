#pragma once
#include "metrics.hpp"
#include <filesystem>

namespace edu {
inline void close(double got,double expected,double tolerance,const char* message) {
    if (!std::isfinite(got) || std::abs(got-expected)>tolerance)
        throw std::runtime_error(std::string(message)+": got "+std::to_string(got)+", expected "+std::to_string(expected));
}
inline void gradcheck(const std::string& name,const std::vector<T>& leaves,
                      const std::function<T()>& objective,size_t per_tensor=100000) {
    for (auto& p:leaves) std::fill(p->grad.begin(),p->grad.end(),0.0);
    backward(objective());
    size_t checks=0;
    for (auto& p:leaves) {
        auto analytic=p->grad;
        size_t n=std::min(per_tensor,p->data.size());
        for (size_t sample=0;sample<n;++sample) {
            size_t i=n==1 ? 0:sample*(p->data.size()-1)/(n-1);
            double original=p->data[i],plus,minus;
            { NoGrad guard;
              p->data[i]=original+1e-5; plus=objective()->data[0];
              p->data[i]=original-1e-5; minus=objective()->data[0];
              p->data[i]=original; }
            double numerical=(plus-minus)/2e-5;
            double tolerance=2e-5+2e-4*std::max(std::abs(analytic[i]),std::abs(numerical));
            close(analytic[i],numerical,tolerance,(name+" gradient").c_str()); ++checks;
        }
    }
    std::cout<<"PASS "<<name<<" ("<<checks<<" finite differences)\n";
}
inline T test_tensor(int r,int c,double offset=0.0) {
    auto x=tensor(r,c,true);
    for (size_t i=0;i<x->data.size();++i) x->data[i]=std::sin(static_cast<double>(i)+0.3)+offset;
    return x;
}
inline void tests() {
    auto a=test_tensor(2,3), b=test_tensor(3,4), bias=test_tensor(1,4);
    gradcheck("matmul + bias + cross entropy",{a,b,bias},[&] { return cross_entropy(add(matmul(a,b),bias),{1,3}); });
    auto gamma=test_tensor(1,3,1.0),beta=test_tensor(1,3);
    gradcheck("layer norm",{a,gamma,beta},[&] { return cross_entropy(layer_norm(a,gamma,beta),{0,2}); });
    gradcheck("ReLU",{a},[&] { return cross_entropy(relu(a),{0,2}); });
    auto s=test_tensor(3,3);
    gradcheck("masked softmax",{s},[&] { return cross_entropy(softmax(s,true),{0,1,2}); });
    auto table=test_tensor(5,3);
    gradcheck("repeated embedding IDs",{table},[&] { return cross_entropy(embedding(table,{2,2,4}),{0,1,2}); });
    gradcheck("transpose + slice + concat",{a},[&] {
        auto t=transpose(a); return cross_entropy(concat({columns(t,1,1),columns(t,0,1)}),{0,1,0});
    });
    auto stable=tensor(1,3); stable->data={1000,1000,1000};
    close(cross_entropy(stable,{1})->data[0],std::log(3.0),1e-12,"stable CE");
    auto probabilities=softmax(s,true);
    for (int i=0;i<3;++i) {
        double sum=0;
        for (int j=0;j<3;++j) { sum+=probabilities->at(i,j); if (j>i) close(probabilities->at(i,j),0,0,"mask"); }
        close(sum,1,1e-12,"softmax row sum");
    }
    auto shared=test_tensor(1,3);
    gradcheck("shared DAG residual",{shared},[&] { auto x=scale(shared,2); return cross_entropy(add(x,x),{2}); });
    auto loss=cross_entropy(shared,{0}); std::fill(shared->grad.begin(),shared->grad.end(),0);
    backward(loss); auto once=shared->grad; backward(loss);
    for (size_t i=0;i<once.size();++i) close(shared->grad[i],2*once[i],1e-12,"gradient accumulation");
    Config c; c.d=4;c.heads=2;c.ff=7;c.layers=2;c.max_len=8;
    Transformer model(c,8,9,123);
    std::vector<T> leaves; for (auto& p:model.parameters.all) leaves.push_back(p.value);
    gradcheck("complete two-layer Transformer",leaves,[&] { return cross_entropy(model({4,5,2},{1,6,7}),{6,7,2}); },8);
    {
        NoGrad guard;
        auto before=model({4,5,2},{1,6,7}),after=model({4,5,2},{1,6,8});
        for (int i=0;i<2;++i) for (int j=0;j<9;++j) close(before->at(i,j),after->at(i,j),1e-12,"future token leakage");
        auto changed_source=model({5,5,2},{1,6,7});
        double difference=0;
        for (size_t i=0;i<before->data.size();++i) difference+=std::abs(before->data[i]-changed_source->data[i]);
        require(difference>1e-6,"decoder ignores encoder");
        require(!before->requires_grad && before->parents.empty(),"NoGrad retained graph");
    }
    Bleu exact; exact.add(tokenize("a b c d"),tokenize("a b c d")); close(exact.score(),100,1e-10,"exact BLEU");
    Bleu short_exact; short_exact.add(tokenize("a b"),tokenize("a b"));
    close(short_exact.score(),0,0,"strict short BLEU"); close(short_exact.score(true),100,1e-10,"effective BLEU");
    Bleu clipped; clipped.add(tokenize("a a a"),tokenize("a a"));
    require(clipped.matched[0]==2 && clipped.total[0]==3 && clipped.matched[1]==1,"BLEU clipping");
    close(clipped.score(true),0,0,"zero trigram precision");
    Bleu brevity; brevity.add(tokenize("a b"),tokenize("a b c d")); close(brevity.score(true),100/std::exp(1.0),1e-10,"brevity penalty");
    Bleu disjoint; disjoint.add(tokenize("a b c d"),tokenize("w x y z")); close(disjoint.score(),0,0,"disjoint BLEU");
    Bleu aggregate; aggregate.add(tokenize("a b c d"),tokenize("a b c d")); aggregate.add(tokenize("x"),tokenize("y"));
    close(aggregate.score(),100*std::pow(4.0/5.0,0.25),1e-10,"corpus BLEU aggregation");
    close(Bleu{}.score(),0,0,"empty BLEU");
    Vocab src,tgt;src.learn(tokenize("i like cats dogs"));tgt.learn(tokenize("je aime les chats chiens"));
    auto e=encode_pair({tokenize("i like cats"),tokenize("je aime chats")},src,tgt,8);
    require(e.input.front()==Vocab::BOS && e.labels.back()==Vocab::EOS,"target shift endpoints");
    for (size_t i=1;i<e.input.size();++i) require(e.input[i]==e.labels[i-1],"target shift");
    require(src.encode({"never_seen"})[0]==Vocab::UNK,"unknown token");
    auto file=std::filesystem::temp_directory_path()/("edu-transformer-test-"+std::to_string(std::random_device{}())+".txt");
    save(file.string(),model,src,tgt);
    auto restored=load(file.string()); std::filesystem::remove(file);
    for (size_t i=0;i<model.parameters.all.size();++i)
        require(model.parameters.all[i].value->data==restored.model->parameters.all[i].value->data,"checkpoint round trip");
    require(restored.source.words==src.words && restored.target.words==tgt.words,"vocabulary round trip");
    Parameters p(1);auto weight=p.make("test",1,1,"zero");weight->grad[0]=2;Adam optimizer(p);
    optimizer.update(p,0.01);close(weight->data[0],-0.01,1e-8,"Adam first update + clipping");
    std::cout<<"PASS causality, source dependence, stable CE, metrics, target shift, checkpoint, Adam\n";
    // A real learning test: all layers must cooperate to reduce sequence loss.
    Config tiny;tiny.d=8;tiny.heads=2;tiny.ff=16;tiny.max_len=6;
    Transformer learner(tiny,6,6,42);Adam adam(learner.parameters);
    double initial;
    { NoGrad guard; initial=cross_entropy(learner({4,2},{1,5}),{5,2})->data[0]; }
    for (int step=0;step<80;++step) {
        learner.parameters.zero_grad();backward(cross_entropy(learner({4,2},{1,5}),{5,2}));adam.update(learner.parameters,0.01);
    }
    NoGrad guard;double final=cross_entropy(learner({4,2},{1,5}),{5,2})->data[0];
    require(final<0.1 && final<initial/10,"learning smoke test did not converge");
    require(learner.generate({4,2})==std::vector<int>({5,2}),"greedy decoding did not learn toy target");
    std::cout<<"PASS learning: CE "<<initial<<" -> "<<final<<"\nAll tests passed.\n";
}
} // namespace edu
