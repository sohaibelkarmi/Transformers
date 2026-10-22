#include "tests.hpp"
#include <limits>
#include <set>

using namespace edu;
struct Args {
    std::map<std::string,std::string> values;
    Args(int argc,char** argv,const std::set<std::string>& allowed) {
        for (int i=2;i<argc;i+=2) {
            std::string key=argv[i];
            require(allowed.count(key)!=0,"unknown option (run without arguments for help)");
            require(i+1<argc,"missing option value");
            require(values.emplace(key,argv[i+1]).second,"duplicate option");
        }
    }
    std::string get(const std::string& key,const std::string& fallback="") const {
        auto it=values.find(key);return it==values.end() ? fallback:it->second;
    }
    int integer(const std::string& key,int fallback) const {
        std::string s=get(key,std::to_string(fallback));size_t used=0;int n=std::stoi(s,&used);
        require(used==s.size() && n>0,"integer options must be positive");return n;
    }
    double real(const std::string& key,double fallback) const {
        std::string s=get(key,std::to_string(fallback));size_t used=0;double n=std::stod(s,&used);
        require(used==s.size() && std::isfinite(n) && n>0,"real options must be finite and positive");return n;
    }
};
void print_matrix(const std::string& label,const T& x) {
    std::cout<<label<<" ["<<x->rows<<", "<<x->cols<<"]\n";
    for (int i=0;i<x->rows;++i) { for (int j=0;j<x->cols;++j) std::cout<<std::setw(10)<<x->at(i,j);std::cout<<'\n'; }
}
void trace() {
    NoGrad guard;
    auto q=tensor(3,2),k=tensor(3,2),v=tensor(3,2);
    q->data={1,0, 0,1, 1,1}; k->data=q->data; v->data={10,0, 0,10, 5,5};
    print_matrix("Queries Q: what each position looks for",q);
    print_matrix("Keys K: what each position offers",k);
    print_matrix("Values V: content to mix",v);
    auto scores=scale(matmul(q,transpose(k)),1/std::sqrt(2.0));
    print_matrix("Scores Q K^T / sqrt(2)",scores);
    auto full=softmax(scores),masked=softmax(scores,true);
    print_matrix("Encoder attention: row probabilities sum to 1",full);
    print_matrix("Decoder attention: future positions are zero",masked);
    print_matrix("Encoder weighted values A V",matmul(full,v));
    print_matrix("Decoder weighted values A V",matmul(masked,v));
    std::cout<<"At decoder position 0 only V[0] is visible, so its output is [10,0].\n"
             <<"Real heads learn Q=XWq+bq, K=XWk+bk, V=XWv+bv.\n"
             <<"Cross-attention uses decoder queries and encoder keys/values.\n";
    Vocab source,target;source.learn(tokenize("i like cats"));target.learn(tokenize("j aime les chats"));
    Config c;Transformer model(c,static_cast<int>(source.words.size()),static_cast<int>(target.words.size()));
    auto e=encode_pair({tokenize("i like cats"),tokenize("j aime les chats")},source,target,c.max_len);
    auto memory=model.encode(e.source);auto logits=model.decode(e.input,memory);
    std::cout<<"\nFull model shape walkthrough (random, untrained weights):\n"
             <<"source: i like cats <eos> -> [4,32] embeddings + positions -> [4,32] encoder memory\n"
             <<"decoder input: <bos> j aime les chats -> [5,32]\n"
             <<"labels:         j aime les chats <eos>\n"
             <<"4 self-attention heads: scores [5,5]; 4 cross-attention heads: scores [5,4]\n"
             <<"feed-forward: [5,32] -> [5,64] -> [5,32]\n"
             <<"output logits: ["<<logits->rows<<","<<logits->cols<<"], one row per next-token prediction\n"
             <<"mean cross entropy: "<<cross_entropy(logits,e.labels)->data[0]<<"\n"
             <<"trainable parameters: "<<model.parameters.count()<<"\n";
}
void train(const Args& args) {
    auto training=read_data(args.get("--train"));
    auto validation_path=args.get("--valid");
    std::vector<Pair> validation;
    if (!validation_path.empty()) validation=read_data(validation_path);
    Vocab source,target;
    for (auto& p:training) { source.learn(p.source);target.learn(p.target); }
    Config c;
    c.d=args.integer("--d",32);c.heads=args.integer("--heads",4);c.ff=args.integer("--ff",64);
    c.layers=args.integer("--layers",1);c.max_len=args.integer("--max-len",64);c.validate();
    require(c.d<=512 && c.ff<=2048 && c.layers<=12 && c.max_len<=4096,"configuration exceeds educational size limits");
    require(source.words.size()<=100000 && target.words.size()<=100000,"vocabulary exceeds educational size limit");
    int epochs=args.integer("--epochs",100),batch=args.integer("--batch",8),every=args.integer("--eval-every",10);
    unsigned seed=static_cast<unsigned>(args.integer("--seed",42));
    double lr=args.real("--lr",0.003);
    std::string checkpoint=args.get("--save","model.txt");
    std::vector<Example> encoded;
    for (auto& p:training) encoded.push_back(encode_pair(p,source,target,c.max_len));
    for (auto& p:validation) (void)encode_pair(p,source,target,c.max_len);
    Transformer model(c,static_cast<int>(source.words.size()),static_cast<int>(target.words.size()),seed);
    Adam optimizer(model.parameters);std::mt19937 shuffle_rng(seed);
    std::vector<size_t> order(training.size());std::iota(order.begin(),order.end(),0);
    std::cout<<"Training "<<training.size()<<" pairs, source vocab="<<source.words.size()<<", target vocab="<<target.words.size()
             <<", parameters="<<model.parameters.count()<<"\n";
    if (validation.empty()) std::cout<<"No validation supplied: metrics measure memorization; saving final weights.\n";
    else std::cout<<"Validation pairs="<<validation.size()<<"; saving the best validation CE checkpoint.\n";
    double best=std::numeric_limits<double>::infinity();
    for (int epoch=1;epoch<=epochs;++epoch) {
        std::shuffle(order.begin(),order.end(),shuffle_rng);
        double epoch_nll=0;size_t epoch_tokens=0;
        for (size_t begin=0;begin<order.size();begin+=batch) {
            size_t end=std::min(order.size(),begin+static_cast<size_t>(batch)),batch_tokens=0;
            for (size_t i=begin;i<end;++i) batch_tokens+=encoded[order[i]].labels.size();
            model.parameters.zero_grad();
            for (size_t i=begin;i<end;++i) {
                auto& e=encoded[order[i]];
                auto loss=cross_entropy(model(e.source,e.input),e.labels);
                require(std::isfinite(loss->data[0]),"nonfinite training loss");
                epoch_nll+=loss->data[0]*e.labels.size();epoch_tokens+=e.labels.size();
                // Equivalent to one mean over every target token in the batch.
                backward(scale(loss,static_cast<double>(e.labels.size())/batch_tokens));
            }
            optimizer.update(model.parameters,lr);
        }
        if (epoch==1 || epoch%every==0 || epoch==epochs) {
            std::cout<<"epoch="<<epoch<<" online_train_CE="<<epoch_nll/epoch_tokens<<'\n';
            std::cout<<(validation.empty() ? "train ":"valid ");
            auto metrics=evaluate(model,source,target,validation.empty() ? training:validation);
            metrics.print(std::cout);
            if (!validation.empty() && metrics.ce()<best) {
                best=metrics.ce();save(checkpoint,model,source,target);std::cout<<"saved best: "<<checkpoint<<'\n';
            }
        }
    }
    if (validation.empty()) { save(checkpoint,model,source,target);std::cout<<"saved final: "<<checkpoint<<'\n'; }
}
void help() {
    std::cout<<R"(Educational encoder-decoder Transformer, C++17, standard library only.
Commands:
  transformer trace
  transformer test
  transformer train --train data/toy_train.tsv --save toy.model --epochs 150
    [--valid data/valid.tsv] [--d 32] [--heads 4] [--ff 64] [--layers 1]
    [--max-len 64] [--batch 8] [--lr 0.003] [--seed 42] [--eval-every 10]
  transformer translate --model toy.model --text "i like cats"
    [--max-new 64]
  transformer evaluate --model toy.model --data data/toy_train.tsv
TSV format: source sentence<TAB>target sentence, no header, UTF-8.
Tokenization: case-sensitive whitespace; separate punctuation yourself.
)";
}
int main(int argc,char** argv) {
    try {
        std::cout<<std::fixed<<std::setprecision(6);
        if (argc<2) { help();return 0; }
        std::string command=argv[1];
        if (command=="trace" || command=="test") {
            require(argc==2,"trace and test accept no options");
            if (command=="trace") trace();else tests();
        } else if (command=="train") {
            Args args(argc,argv,{"--train","--valid","--save","--epochs","--d","--heads","--ff","--layers","--max-len","--batch","--lr","--seed","--eval-every"});train(args);
        } else if (command=="translate") {
            Args args(argc,argv,{"--model","--text","--max-new"});auto loaded=load(args.get("--model"));
            auto words=tokenize(args.get("--text"));require(!words.empty(),"provide nonempty --text");
            auto ids=loaded.source.encode(words);ids.push_back(Vocab::EOS);
            auto output=loaded.model->generate(ids,args.integer("--max-new",loaded.model->config.max_len));
            std::cout<<join(loaded.target.decode(output))<<'\n';
            if (output.empty() || output.back()!=Vocab::EOS) std::cerr<<"Generation reached its length limit before EOS.\n";
        } else if (command=="evaluate") {
            Args args(argc,argv,{"--model","--data"});auto loaded=load(args.get("--model"));
            auto data=read_data(args.get("--data"));evaluate(*loaded.model,loaded.source,loaded.target,data).print(std::cout);
        } else { help();return 1; }
    } catch (const std::exception& e) { std::cerr<<"Error: "<<e.what()<<'\n';return 1; }
    return 0;
}
