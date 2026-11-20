#ifndef HexagonCommandStatus_hpp
#define HexagonCommandStatus_hpp

namespace MNN {

class HexagonCommandStatus {
public:
    void latch(int status) {
        if (mStatus == 0 && status != 0) {
            mStatus = status;
        }
    }

    int peek() const {
        return mStatus;
    }

    int consume() {
        const int status = mStatus;
        mStatus = 0;
        return status;
    }

private:
    int mStatus = 0;
};

} // namespace MNN

#endif
