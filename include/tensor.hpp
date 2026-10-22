#pragma once
// A tiny reverse-mode differentiation engine. Every tensor is a 2D matrix.
// No BLAS, Eigen, neural-network library, or external autograd is used.
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace edu {
inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Tensor;
using T = std::shared_ptr<Tensor>;
inline thread_local bool recording = true;
struct NoGrad {
    bool previous = recording;
    NoGrad() { recording = false; }
    ~NoGrad() { recording = previous; }
    NoGrad(const NoGrad&) = delete;
    NoGrad& operator=(const NoGrad&) = delete;
};
struct Tensor {
    int rows, cols;
    std::vector<double> data, grad;
    bool requires_grad;
    std::vector<T> parents;
    // Receives this node by reference: no shared_ptr cycle through its closure.
    std::function<void(const Tensor&)> propagate;
    Tensor(int r, int c, bool trainable = false)
        : rows(r), cols(c), requires_grad(trainable) {
        require(r > 0 && c > 0, "Tensor dimensions must be positive");
        data.resize(static_cast<size_t>(r) * c);
        if (trainable) grad.resize(data.size());
    }
    double& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    double at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
};
inline T tensor(int r, int c, bool trainable = false) {
    return std::make_shared<Tensor>(r, c, trainable);
}
inline T node(int r, int c, const std::vector<T>& parents) {
    bool track = recording && std::any_of(parents.begin(), parents.end(),
                                        [](const T& x) { return x->requires_grad; });
    auto out = tensor(r, c, track);
    if (track) out->parents = parents;
    return out;
}
inline void accumulate(const T& x, size_t i, double g) {
    if (x->requires_grad) x->grad[i] += g;
}
inline void backward(const T& loss) {
    require(loss->data.size() == 1 && loss->requires_grad, "backward needs a tracked scalar");
    // Explicit DFS avoids recursion depth depending on the graph size.
    std::vector<T> order;
    std::unordered_set<Tensor*> seen;
    std::vector<std::pair<T, bool>> stack{{loss, false}};
    while (!stack.empty()) {
        auto item = stack.back(); stack.pop_back();
        if (item.second) { order.push_back(item.first); continue; }
        if (!seen.insert(item.first.get()).second) continue;
        stack.push_back({item.first, true});
        for (const auto& p : item.first->parents) stack.push_back({p, false});
    }
    // Parameter (leaf) gradients accumulate across sentences in a batch.
    // Intermediate gradients must start fresh on every backward call.
    for (auto& x : order)
        if (!x->parents.empty()) std::fill(x->grad.begin(), x->grad.end(), 0.0);
    loss->grad[0] = 1.0;
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if ((*it)->propagate) (*it)->propagate(**it);
}
inline T add(const T& a, const T& b) {
    require(a->cols == b->cols && (a->rows == b->rows || b->rows == 1), "add shape mismatch");
    auto out = node(a->rows, a->cols, {a, b});
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = a->data[i] + b->data[i % b->data.size()];
    if (out->requires_grad) out->propagate = [a,b](const Tensor& o) {
        for (size_t i = 0; i < o.data.size(); ++i) {
            accumulate(a, i, o.grad[i]); accumulate(b, i % b->data.size(), o.grad[i]);
        }
    };
    return out;
}
inline T scale(const T& a, double factor) {
    auto out = node(a->rows, a->cols, {a});
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = factor * a->data[i];
    if (out->requires_grad) out->propagate = [a,factor](const Tensor& o) {
        for (size_t i = 0; i < o.data.size(); ++i) accumulate(a, i, factor * o.grad[i]);
    };
    return out;
}
inline T matmul(const T& a, const T& b) {
    require(a->cols == b->rows, "matmul shape mismatch");
    auto out = node(a->rows, b->cols, {a,b});
    for (int i = 0; i < a->rows; ++i)
        for (int k = 0; k < a->cols; ++k)
            for (int j = 0; j < b->cols; ++j) out->at(i,j) += a->at(i,k) * b->at(k,j);
    if (out->requires_grad) out->propagate = [a,b](const Tensor& o) {
        // dA = dOut B^T; dB = A^T dOut.
        for (int i = 0; i < a->rows; ++i)
            for (int k = 0; k < a->cols; ++k)
                for (int j = 0; j < b->cols; ++j) {
                    double g = o.grad[static_cast<size_t>(i)*o.cols+j];
                    accumulate(a, static_cast<size_t>(i)*a->cols+k, g*b->at(k,j));
                    accumulate(b, static_cast<size_t>(k)*b->cols+j, g*a->at(i,k));
                }
    };
    return out;
}
inline T transpose(const T& a) {
    auto out = node(a->cols, a->rows, {a});
    for (int i=0;i<a->rows;++i) for (int j=0;j<a->cols;++j) out->at(j,i)=a->at(i,j);
    if (out->requires_grad) out->propagate=[a](const Tensor& o) {
        for (int i=0;i<a->rows;++i) for (int j=0;j<a->cols;++j)
            accumulate(a,static_cast<size_t>(i)*a->cols+j,o.grad[static_cast<size_t>(j)*o.cols+i]);
    };
    return out;
}
inline T columns(const T& a, int first, int count) {
    require(first>=0 && count>0 && first+count<=a->cols,"columns out of range");
    auto out=node(a->rows,count,{a});
    for (int i=0;i<a->rows;++i) for (int j=0;j<count;++j) out->at(i,j)=a->at(i,first+j);
    if (out->requires_grad) out->propagate=[a,first,count](const Tensor& o) {
        for (int i=0;i<a->rows;++i) for (int j=0;j<count;++j)
            accumulate(a,static_cast<size_t>(i)*a->cols+first+j,o.grad[static_cast<size_t>(i)*count+j]);
    };
    return out;
}
inline T concat(const std::vector<T>& xs) {
    require(!xs.empty(),"empty concat");
    int width=0;
    for (auto& x:xs) { require(x->rows==xs[0]->rows,"concat shape mismatch"); width+=x->cols; }
    auto out=node(xs[0]->rows,width,xs);
    int offset=0;
    for (auto& x:xs) {
        for (int i=0;i<x->rows;++i) for (int j=0;j<x->cols;++j) out->at(i,offset+j)=x->at(i,j);
        offset+=x->cols;
    }
    if (out->requires_grad) out->propagate=[xs](const Tensor& o) {
        int off=0;
        for (auto& x:xs) {
            for (int i=0;i<x->rows;++i) for (int j=0;j<x->cols;++j)
                accumulate(x,static_cast<size_t>(i)*x->cols+j,o.grad[static_cast<size_t>(i)*o.cols+off+j]);
            off+=x->cols;
        }
    };
    return out;
}
inline T relu(const T& a) {
    auto out=node(a->rows,a->cols,{a});
    for (size_t i=0;i<a->data.size();++i) out->data[i]=std::max(0.0,a->data[i]);
    if (out->requires_grad) out->propagate=[a](const Tensor& o) {
        for (size_t i=0;i<o.data.size();++i) accumulate(a,i,a->data[i]>0 ? o.grad[i]:0.0);
    };
    return out;
}
inline T softmax(const T& a, bool causal=false) {
    require(!causal || a->rows==a->cols,"causal softmax must be square");
    auto out=node(a->rows,a->cols,{a});
    for (int i=0;i<a->rows;++i) {
        int end=causal ? i+1:a->cols;
        double maximum=a->at(i,0), total=0;
        for (int j=1;j<end;++j) maximum=std::max(maximum,a->at(i,j));
        for (int j=0;j<end;++j) total+=(out->at(i,j)=std::exp(a->at(i,j)-maximum));
        for (int j=0;j<end;++j) out->at(i,j)/=total;
        // Masked entries stay exactly zero, with exactly zero derivative.
    }
    if (out->requires_grad) out->propagate=[a](const Tensor& o) {
        for (int i=0;i<o.rows;++i) {
            double dot=0;
            for (int j=0;j<o.cols;++j) { size_t k=static_cast<size_t>(i)*o.cols+j; dot+=o.grad[k]*o.data[k]; }
            for (int j=0;j<o.cols;++j) { size_t k=static_cast<size_t>(i)*o.cols+j; accumulate(a,k,o.data[k]*(o.grad[k]-dot)); }
        }
    };
    return out;
}
inline T embedding(const T& weights, const std::vector<int>& ids) {
    require(!ids.empty(),"empty embedding input");
    auto out=node(static_cast<int>(ids.size()),weights->cols,{weights});
    for (size_t i=0;i<ids.size();++i) {
        require(ids[i]>=0 && ids[i]<weights->rows,"token ID out of range");
        for (int j=0;j<weights->cols;++j) out->at(static_cast<int>(i),j)=weights->at(ids[i],j);
    }
    if (out->requires_grad) out->propagate=[weights,ids](const Tensor& o) {
        for (size_t i=0;i<ids.size();++i) for (int j=0;j<weights->cols;++j)
            accumulate(weights,static_cast<size_t>(ids[i])*weights->cols+j,o.grad[i*o.cols+j]);
    };
    return out;
}
inline T layer_norm(const T& x, const T& gamma, const T& beta) {
    require(gamma->rows==1 && beta->rows==1 && gamma->cols==x->cols && beta->cols==x->cols,"layer norm shape");
    auto out=node(x->rows,x->cols,{x,gamma,beta});
    std::vector<double> normalized(x->data.size()), inverse(x->rows);
    for (int i=0;i<x->rows;++i) {
        double mean=0,var=0;
        for (int j=0;j<x->cols;++j) mean+=x->at(i,j);
        mean/=x->cols;
        for (int j=0;j<x->cols;++j) var+=std::pow(x->at(i,j)-mean,2);
        inverse[i]=1.0/std::sqrt(var/x->cols+1e-5);
        for (int j=0;j<x->cols;++j) {
            size_t k=static_cast<size_t>(i)*x->cols+j;
            normalized[k]=(x->data[k]-mean)*inverse[i];
            out->data[k]=normalized[k]*gamma->data[j]+beta->data[j];
        }
    }
    if (out->requires_grad) out->propagate=[x,gamma,beta,normalized,inverse](const Tensor& o) {
        for (int i=0;i<x->rows;++i) {
            double sum=0,projection=0;
            for (int j=0;j<x->cols;++j) {
                size_t k=static_cast<size_t>(i)*x->cols+j;
                double g=o.grad[k]*gamma->data[j]; sum+=g; projection+=g*normalized[k];
                accumulate(gamma,j,o.grad[k]*normalized[k]); accumulate(beta,j,o.grad[k]);
            }
            for (int j=0;j<x->cols;++j) {
                size_t k=static_cast<size_t>(i)*x->cols+j;
                accumulate(x,k,inverse[i]*(o.grad[k]*gamma->data[j]-(sum+normalized[k]*projection)/x->cols));
            }
        }
    };
    return out;
}
// Mean negative log-likelihood, fused with log-softmax for numerical stability.
inline T cross_entropy(const T& logits, const std::vector<int>& labels) {
    require(static_cast<int>(labels.size())==logits->rows,"label count mismatch");
    auto out=node(1,1,{logits});
    std::vector<double> probabilities(logits->data.size());
    for (int i=0;i<logits->rows;++i) {
        require(labels[i]>=0 && labels[i]<logits->cols,"label out of range");
        double mx=logits->at(i,0),total=0;
        for (int j=1;j<logits->cols;++j) mx=std::max(mx,logits->at(i,j));
        for (int j=0;j<logits->cols;++j) total+=(probabilities[static_cast<size_t>(i)*logits->cols+j]=std::exp(logits->at(i,j)-mx));
        out->data[0]+=(mx-logits->at(i,labels[i])+std::log(total))/logits->rows;
        for (int j=0;j<logits->cols;++j) probabilities[static_cast<size_t>(i)*logits->cols+j]/=total;
    }
    if (out->requires_grad) out->propagate=[logits,labels,probabilities](const Tensor& o) {
        for (int i=0;i<logits->rows;++i) for (int j=0;j<logits->cols;++j) {
            size_t k=static_cast<size_t>(i)*logits->cols+j;
            accumulate(logits,k,o.grad[0]*(probabilities[k]-(j==labels[i] ? 1.0:0.0))/logits->rows);
        }
    };
    return out;
}
} // namespace edu
