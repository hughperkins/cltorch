#pragma once

class TensorGenOp : public HasOperator1, public HasOperator2 {
public:
  std::string cfun;
  TensorGenOp( std::string cfun ) {
     this->cfun = cfun;
  }
  std::string operator1() const {
    return "*out =" + cfun + "( *out )";
  }
  std::string operator2() const {
    return "*out = " + cfun + "( *in1 )";
  }
};

class TensorGenOpFullInline1 : public HasOperator1 {
public:
  std::string cfun;
  TensorGenOpFullInline1( std::string cfun ) {
     this->cfun = cfun;
  }
  std::string operator1() const {
    return cfun;
  }
};

class TensorGenOpFullInline2 : public HasOperator2 {
public:
  std::string cfun;
  TensorGenOpFullInline2( std::string cfun ) {
     this->cfun = cfun;
  }
  std::string operator2() const {
    return cfun;
  }
};

class TensorGenOpFullInline3 : public HasOperator3 {
public:
  std::string cfun;
  TensorGenOpFullInline3( std::string cfun ) {
     this->cfun = cfun;
  }
  std::string operator3() const {
    return cfun;
  }
};

// used for maxall etc
class MaxOp : public HasOperator2, public HasOperator3 {
public:
    std::string operator2() const {
        return "*out = fmax(*out, *in1)";
    }
    std::string operator3() const {
        return "*out = fmax(*in1, *in2)";
    }
};

// used for minall etc
class MinOp : public HasOperator2, public HasOperator3 {
public:
    std::string operator2() const {
        return "*out = fmin(*out, *in1)";
    }
    std::string operator3() const {
        return "*out = fmin(*in1, *in2)";
    }
};

class TensorAddOp : public HasOperator2, public HasOperator3 {
public:
    std::string operator2() const {
        return "*out += *in1";
    }
    std::string operator3() const {
        return "*out = *in1 + *in2";
    }
};

class TensorCAddOp : public HasOperator2, public HasOperator3, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar(int index) const { return val; }
  TensorCAddOp(float v) : val(v) {}
    std::string operator2() const {
        return "*out += val1 * *in1";
    }
    std::string operator3() const {
        return "*out = *in1 + val1 * *in2";
    }
  float val;
};

class TensorSubOp : public HasOperator2, public HasOperator3 {
public:
    std::string operator2() const {
        return "*out -= *in1";
    }
    std::string operator3() const {
        return "*out = *in1 - *in2";
    }
};

class TensorCSubOp : public HasOperator2, public HasOperator3, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar(int index) const { return val; }
  TensorCSubOp(float v) : val(v) {}
    std::string operator2() const {
        return "*out -= val1 * *in1";
    }
    std::string operator3() const {
        return "*out = *in1 - val1 * *in2";
    }
  float val;
};

class TensorMulOp : public HasOperator2, public HasOperator3 {
public:
    std::string operator2() const {
        return "*out *= *in1";
    }
    std::string operator3() const {
        return "*out = (*in1) * (*in2)";
    }
};



