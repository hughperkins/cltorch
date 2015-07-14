#pragma once

#ifdef __cplusplus

class OpBase {
public:
};

class HasScalars : public OpBase {
public:
    virtual int getNumScalars() const = 0;
    virtual float getScalar(int index) const = 0;
};

class HasOperator1 : public OpBase {
public:
    virtual std::string operator1() const = 0;
};

class HasOperator2 : public OpBase {
public:
    virtual std::string operator2() const = 0;
};

class HasOperator3 : public OpBase {
public:
    virtual std::string operator3() const = 0;
};

class HasGlobalTensors {
public:
  virtual int getNumGlobalTensors() const = 0;
  virtual THClTensor *getTensor(int index) const = 0;
  virtual std::string getTensorName(int index) const = 0;
};

class HasPointTensors {
public:
  virtual int getNumPointTensors() const = 0;
  virtual const THClTensor *getPointTensor(int index) const = 0;
};

#endif // __cplusplus

