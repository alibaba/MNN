#ifndef CORE_REFCOUNT_H
#define CORE_REFCOUNT_H
#include <stdlib.h>
namespace NNR {

#define NNR_SAFE_UNREF(x)\
    if (NULL!=(x)) {(x)->decRef();}
#define SAFE_REF(x)\
    if (NULL!=(x)) (x)->addRef();
#define SAFE_ASSIGN(dst, src) \
    {\
        if (src!=NULL)\
        {\
            src->addRef();\
        }\
        if (dst!=NULL)\
        {\
            dst->decRef();\
        }\
        dst = src;\
    }
class RefCount
{
public:
    void addRef()
    {
        mNum++;
    }
    void decRef()
    {
        --mNum;
        if (0 >= mNum)
        {
            delete this;
        }
    }
    inline int count() const{return mNum;}
protected:
    RefCount():mNum(1){}
    RefCount(const RefCount& f):mNum(f.mNum){}
    void operator=(const RefCount& f)
    {
        if (this != &f)
        {
            mNum = f.mNum;
        }
    }
    virtual ~RefCount(){}
private:
    int mNum;
};
#define NNR_SAFE_UNREF(x)\
if (NULL!=(x)) {(x)->decRef();}
#define NNR_SAFE_REF(x)\
if (NULL!=(x)) (x)->addRef();
#define NNR_SAFE_ASSIGN(dst, src) \
{\
if (src!=NULL)\
{\
src->addRef();\
}\
if (dst!=NULL)\
{\
dst->decRef();\
}\
dst = src;\
}
template <typename T>
class SharedPtr {
public:
    SharedPtr() : mT(NULL) {}
    SharedPtr(T* obj) : mT(obj) {}
    SharedPtr(const SharedPtr& o) : mT(o.mT) { NNR_SAFE_REF(mT); }
    ~SharedPtr() { NNR_SAFE_UNREF(mT); }
    
    SharedPtr& operator=(const SharedPtr& rp) {
        NNR_SAFE_ASSIGN(mT, rp.mT);
        return *this;
    }
    SharedPtr& operator=(T* obj) {
        NNR_SAFE_UNREF(mT);
        mT = obj;
        return *this;
    }
    
    T* get() const { return mT; }
    T& operator*() const { return *mT; }
    T* operator->() const { return mT; }
    
private:
    T* mT;
};
};
#endif
